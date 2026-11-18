"""Runs one trip through the team's pipeline.

Request flow for POST /predict:
  1. load_trip           read the trip files from the evaluation folder
  2. hide the reference  make a copy of the trip with route=None
  3. build_candidates    candidate road arcs per GPS point   (no ground truth)
  4. score               trained model adds p_match           (no ground truth)
  5. decode              HMM / Viterbi picks one arc per point (no ground truth)
  6. metrics             only now is the reference route used, on the server,
                         to label candidates and compute accuracy

Steps 3-5 only ever see the copy without the route, so ground truth cannot
leak into the prediction. Nothing from the reference route is returned.
"""
import copy
import dataclasses
import importlib
import json
import logging
import time

import numpy as np

from api import config

log = logging.getLogger("pathsnap")

# contract name -> module that owns it
CONTRACT = {
    "load_trip": "data_processing.loader",
    "build_candidates": "features.candidates",
    "add_labels": "features.labels",
    "score": "models.inference",
    "decode": "hmm.viterbi",
    "point_metrics": "evaluation.metrics",
    "route_continuity": "evaluation.metrics",
}


class Pipeline:
    def __init__(self):
        self.funcs = {}
        self.sources = {}
        missing = []
        for name, module in CONTRACT.items():
            try:
                self.funcs[name] = getattr(importlib.import_module(module), name)
                self.sources[name] = module
            except (ImportError, AttributeError) as exc:
                missing.append(f"{module}.{name} ({exc})")

        if missing:
            if not config.placeholder_allowed():
                raise RuntimeError(
                    "Pipeline functions missing:\n  " + "\n  ".join(missing)
                    + "\nPull the latest code, or set PATHSNAP_PLACEHOLDER=1 for development only."
                )
            from api import placeholder
            for name in CONTRACT:
                if name not in self.funcs:
                    self.funcs[name] = getattr(placeholder, name)
                    self.sources[name] = "api.placeholder  <-- PLACEHOLDER"

        self.uses_placeholder = any("PLACEHOLDER" in s for s in self.sources.values())
        self.hmm_config = self._load_hmm_config()

    def _load_hmm_config(self):
        path = config.HMM_CONFIG_PATH
        if path.exists():
            return json.loads(path.read_text())
        if self.uses_placeholder:
            return {}
        raise RuntimeError(f"{path} not found. Shriram exports it from hmm/.")

    def describe(self):
        lines = [f"  {name:<17} from {src}" for name, src in self.sources.items()]
        for label, path in [("model", config.MODEL_PATH), ("model config", config.MODEL_CONFIG_PATH),
                            ("hmm config", config.HMM_CONFIG_PATH)]:
            state = f"{path.stat().st_size / 1e6:.2f} MB" if path.exists() else "MISSING"
            lines.append(f"  {label:<17} {path.relative_to(config.REPO_ROOT)} ({state})")
        return "\n".join(lines)

    def warm_up(self, trip_dir):
        """Score a few candidates at startup so the model file is loaded before the first request.
        No metrics are computed here and nothing is cached."""
        t0 = time.perf_counter()
        trip = _without_route(self.funcs["load_trip"](trip_dir))
        candidates = self.funcs["build_candidates"](trip).head(50)
        self.funcs["score"](candidates.drop(columns=["label"], errors="ignore"))
        log.info("warm-up: scored %d candidates in %.0f ms", len(candidates), _ms(t0))

    def predict(self, trip_dir, log_prefix=""):
        f = self.funcs
        timings = {}

        start = time.perf_counter()
        trip = f["load_trip"](trip_dir)
        timings["load"] = _ms(start)

        reference_route = trip.route
        blind_trip = _without_route(trip)

        t0 = time.perf_counter()
        candidates = f["build_candidates"](blind_trip)
        timings["candidates"] = _ms(t0)

        t0 = time.perf_counter()
        # Defensive: candidates built from blind_trip have no reference columns,
        # but make sure none could ever reach the model.
        scored = f["score"](candidates.drop(columns=["ref_arc", "label"], errors="ignore"))
        timings["score"] = _ms(t0)

        t0 = time.perf_counter()
        pred = f["decode"](scored, blind_trip, self.hmm_config)
        timings["decode"] = _ms(t0)
        inference_ms = _ms(start)

        metrics = {"top1_accuracy": None, "route_continuity": None, "macro_f1": None,
                   "candidate_miss_rate": None, "inference_ms": round(inference_ms, 1)}
        metrics["route_continuity"] = _num(f["route_continuity"](pred, blind_trip))
        if reference_route is not None:
            t0 = time.perf_counter()
            labelled = f["add_labels"](candidates, trip)
            point = f["point_metrics"](pred, labelled)
            metrics["top1_accuracy"] = _num(point.get("top1_accuracy"))
            metrics["macro_f1"] = _num(point.get("macro_f1"))
            metrics["candidate_miss_rate"] = _num(point.get("candidate_miss_rate"))
            timings["metrics"] = _ms(t0)

        lonlat, obs_ids = _track_points(trip)
        nodes = np.asarray(trip.nodes, dtype=float)
        arcs = np.asarray(trip.arcs, dtype=np.int64)
        # One prediction per GPS point, in the same order as "gps", so the UI can
        # pair gps[i] with predicted[i]. Points the decoder skipped get -1.
        arc_by_obs = dict(zip(np.asarray(pred["obs_idx"], dtype=np.int64), np.asarray(pred["arc_id"], dtype=np.int64)))
        pred_obs = obs_ids
        pred_arcs = np.array([arc_by_obs.get(o, -1) for o in obs_ids], dtype=np.int64)

        response = {
            "trip_id": str(trip.trip_id),
            "gps": [[round(float(lat), 7), round(float(lon), 7)] for lon, lat in lonlat],
            "predicted": [{"obs_idx": int(o), "arc_id": int(a)} for o, a in zip(pred_obs, pred_arcs)],
            "route": _route_polyline(pred_arcs, nodes, arcs),
            "metrics": metrics,
        }

        assigned = int((pred_arcs >= 0).sum())
        log.info("%s trip %s | %d GPS points, %d candidates, %d/%d points matched",
                 log_prefix, response["trip_id"], len(lonlat), len(candidates), assigned, len(pred_arcs))
        log.info("%s trip %s | times ms: %s", log_prefix, response["trip_id"],
                 ", ".join(f"{k} {v:.0f}" for k, v in timings.items()))
        log.info("%s trip %s | metrics: %s", log_prefix, response["trip_id"], metrics)
        return response


def _track_points(trip):
    """GPS points as (lon, lat) plus their obs_idx, in track order."""
    track = trip.track
    if hasattr(track, "columns") and {"lon", "lat"} <= set(track.columns):
        lonlat = track[["lon", "lat"]].to_numpy(dtype=float)
        obs = track["obs_idx"].to_numpy(dtype=np.int64) if "obs_idx" in track.columns else np.arange(len(track))
        return lonlat, obs
    arr = np.asarray(track, dtype=float)
    return arr[:, :2], np.arange(len(arr))


def _without_route(trip):
    if hasattr(trip, "without_route"):
        return trip.without_route()  # Shriram's loader provides this
    if dataclasses.is_dataclass(trip):
        return dataclasses.replace(trip, route=None)
    blind = copy.copy(trip)
    blind.route = None
    return blind


def _route_polyline(pred_arcs, nodes, arcs):
    """Turn the predicted arc sequence into [lat, lon] points for the map."""
    seq = pred_arcs[pred_arcs >= 0]
    if len(seq) == 0:
        return []
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]  # drop repeats of the same arc
    points = []
    for arc in seq:
        for node in arcs[arc]:
            lon, lat = nodes[node]
            pt = [round(float(lat), 7), round(float(lon), 7)]
            if not points or points[-1] != pt:
                points.append(pt)
    return points


def _ms(start):
    return (time.perf_counter() - start) * 1000


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)
