"""
############################################################################
#  PLACEHOLDER PIPELINE - DEVELOPMENT ONLY - DELETE THIS FILE BEFORE THE DEMO
#
#  Stand-ins for Shriram's and Sharanya's contract functions so the API and
#  UI can be built and tested before their code exists. Nothing here is
#  trained. It is a plain nearest-road rule:
#    candidates = road arcs within RADIUS_M of each GPS point
#    score      = Gaussian of distance (no model)
#    decode     = highest score per point (no HMM)
#    labels     = nearest arc on the reference route
#  The API only uses a function from here if the real one cannot be
#  imported AND PATHSNAP_PLACEHOLDER=1 is set. Every response made with it
#  is flagged in the console and in the UI.
#  Never report any number produced with this file.
############################################################################
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6371000.0
RADIUS_M = 50.0
MAX_CANDIDATES = 8
SIGMA_M = 10.0
STEP_M = 10.0  # spacing of sample points along each road segment


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame  # lon, lat, t
    nodes: np.ndarray    # (n_nodes, 2) lon, lat
    arcs: np.ndarray     # (n_arcs, 2) from_node, to_node
    route: np.ndarray | None


def load_trip(trip_dir):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = pd.read_csv(trip_dir / f"{tid}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route_file = trip_dir / f"{tid}.route"
    route = np.loadtxt(route_file, dtype=np.int64, ndmin=1) if route_file.exists() else None
    return Trip(tid, track, nodes, arcs, route)


def _to_metres(lon, lat, lon0, lat0):
    x = EARTH_R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_R * np.radians(lat - lat0)
    return np.column_stack([x, y])


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom == 0, 1.0, denom)
    u = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0, 1)
    c = a + u[:, None] * ab
    return np.linalg.norm(p - c, axis=1)


def _geometry(trip, arc_ids=None):
    track = np.asarray(trip.track, dtype=float)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    p = _to_metres(track[:, 0], track[:, 1], lon0, lat0)
    nodes_m = _to_metres(trip.nodes[:, 0], trip.nodes[:, 1], lon0, lat0)
    arcs = trip.arcs if arc_ids is None else trip.arcs[arc_ids]
    return track, p, nodes_m[arcs[:, 0]], nodes_m[arcs[:, 1]]


def _nearby_pairs(p, a, b, radius):
    """Return (point_idx, segment_idx, distance) for segments within radius."""
    seg_len = np.linalg.norm(b - a, axis=1)
    n_samples = np.maximum(1, np.ceil(seg_len / STEP_M)).astype(int) + 1
    seg_of_sample = np.repeat(np.arange(len(a)), n_samples)
    frac = np.concatenate([np.linspace(0, 1, n) for n in n_samples])
    samples = a[seg_of_sample] + frac[:, None] * (b - a)[seg_of_sample]
    tree = cKDTree(samples)
    hits = tree.query_ball_point(p, r=radius + STEP_M)
    pi = np.repeat(np.arange(len(p)), [len(h) for h in hits])
    si = seg_of_sample[np.concatenate(hits).astype(int)] if len(pi) else np.array([], int)
    pairs = np.unique(np.column_stack([pi, si]), axis=0) if len(pi) else np.empty((0, 2), int)
    d = _point_segment_dist(p[pairs[:, 0]], a[pairs[:, 1]], b[pairs[:, 1]])
    keep = d <= radius
    return pairs[keep, 0], pairs[keep, 1], d[keep]


def build_candidates(trip):
    track, p, a, b = _geometry(trip)
    pi, si, d = _nearby_pairs(p, a, b, RADIUS_M)
    df = pd.DataFrame({"obs_idx": pi, "arc_id": si, "f_dist_m": d})
    df = df.sort_values(["obs_idx", "f_dist_m"]).groupby("obs_idx").head(MAX_CANDIDATES)
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(2, "t", track[df["obs_idx"], 2])
    df.insert(3, "gps_lon", track[df["obs_idx"], 0])
    df.insert(4, "gps_lat", track[df["obs_idx"], 1])
    return df.reset_index(drop=True)


def add_labels(candidates, trip):
    route = np.unique(trip.route)
    _, p, a, b = _geometry(trip, route)
    d = np.stack([_point_segment_dist(p, np.repeat(a[i:i + 1], len(p), 0), np.repeat(b[i:i + 1], len(p), 0))
                  for i in range(len(route))], axis=1)
    ref_arc = route[d.argmin(axis=1)]
    out = candidates.copy()
    out["label"] = (out["arc_id"].to_numpy() == ref_arc[out["obs_idx"].to_numpy()]).astype(int)
    out.attrs["ref_arc"] = ref_arc
    return out


def score(candidates):
    out = candidates.copy()
    out["p_match"] = np.exp(-0.5 * (out["f_dist_m"] / SIGMA_M) ** 2)
    return out


def decode(scored, trip, config):
    n = len(trip.track)
    best = scored.loc[scored.groupby("obs_idx")["p_match"].idxmax(), ["obs_idx", "arc_id"]]
    pred = pd.DataFrame({"obs_idx": np.arange(n), "arc_id": -1})
    pred.loc[best["obs_idx"].to_numpy(), "arc_id"] = best["arc_id"].to_numpy()
    return pred


def point_metrics(pred, labelled):
    ref_arc = labelled.attrs["ref_arc"]
    hit = pred["arc_id"].to_numpy() == ref_arc[pred["obs_idx"].to_numpy()]
    covered = labelled.groupby("obs_idx")["label"].max().reindex(range(len(ref_arc)), fill_value=0)
    return {"top1_accuracy": float(hit.mean()), "candidate_miss_rate": float(1 - covered.mean())}


def route_continuity(pred, trip):
    arcs = np.asarray(pred["arc_id"])
    arcs = arcs[arcs >= 0]
    if len(arcs) < 2:
        return 0.0
    prev, nxt = arcs[:-1], arcs[1:]
    ok = (prev == nxt) | (trip.arcs[prev, 1] == trip.arcs[nxt, 0])
    return float(ok.mean())
