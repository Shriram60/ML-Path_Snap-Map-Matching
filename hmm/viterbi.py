"""HMM route reconstruction with Viterbi decoding.

Hidden state: one candidate arc for one GPS point.
Emission: log of the classifier's p_match, multiplied by emission_weight.
Transition: compares the driving distance between two candidates with the GPS
distance between their points. Similar distances are likely. Candidates with no
drivable connection within the search radius get a fixed penalty, so decoding
always finishes and a single bad link cannot break a whole trip.

Viterbi recurrence:
    V[t][j] = emission[t][j] + max_i(V[t-1][i] + transition[t-1][i][j])

Points with no candidates are skipped and returned as arc -1. The next transition
then spans the skipped time. Very long jumps start a fresh route segment.
"""
from dataclasses import asdict, dataclass
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).parent / "hmm_config.json"


@dataclass(frozen=True)
class DecoderConfig:
    emission_weight: float = 1.0
    transition_scale_m: float = 15.0
    unreachable_penalty: float = 25.0
    search_radius_m: float = 300.0
    route_slack_factor: float = 2.0
    break_distance_m: float = 1500.0
    backward_tolerance_m: float = 15.0
    min_probability: float = 1e-6


def load_config(path=CONFIG_PATH):
    values = json.loads(Path(path).read_text(encoding="utf-8"))["parameters"]
    return DecoderConfig(**values)


def save_config(config, source, path=CONFIG_PATH):
    payload = {"source": source, "parameters": asdict(config)}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class RoadRouter:
    """Bounded shortest driving distances between road nodes."""

    def __init__(self, trip):
        start = trip.node_xy[trip.arcs[:, 0]]
        end = trip.node_xy[trip.arcs[:, 1]]
        lengths = np.linalg.norm(end - start, axis=1)
        usable = np.flatnonzero(lengths > 0)
        order = usable[np.argsort(trip.arcs[usable, 0], kind="stable")]
        self.origins = trip.arcs[order, 0]
        self.targets = trip.arcs[order, 1]
        self.lengths = lengths[order]
        self.cache = {}

    def _neighbours(self, node):
        low = np.searchsorted(self.origins, node, side="left")
        high = np.searchsorted(self.origins, node, side="right")
        return zip(self.targets[low:high].tolist(), self.lengths[low:high].tolist())

    def reach(self, source, radius):
        cached = self.cache.get(source)
        if cached is not None and cached[0] >= radius:
            return cached[1]
        best = {source: 0.0}
        done = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if node in done:
                continue
            done[node] = distance
            for target, length in self._neighbours(node):
                total = distance + length
                if total <= radius and total < best.get(target, np.inf):
                    best[target] = total
                    heapq.heappush(queue, (total, target))
        self.cache[source] = (radius, done)
        return done


def _transition_scores(previous, current, gps_distance, gap, trip, router, config):
    prev_arc, prev_s, prev_len = previous
    next_arc, next_s, _ = current
    i, j = np.meshgrid(np.arange(len(prev_arc)), np.arange(len(next_arc)), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, b = prev_arc[i], next_arc[j]
    n_nodes = len(trip.nodes)
    radius = max(config.search_radius_m, gps_distance * config.route_slack_factor + 100.0)

    route = np.full(len(a), np.inf)
    extra = np.zeros(len(a))
    same = a == b
    moved = next_s[j[same]] - prev_s[i[same]]
    route[same] = np.maximum(moved, 0.0)
    extra[same] = np.maximum(-moved - config.backward_tolerance_m, 0.0)

    other = np.flatnonzero(~same)
    if len(other):
        keys = trip.arcs[a[other], 1].astype(np.int64) * n_nodes + trip.arcs[b[other], 0]
        unique, inverse = np.unique(keys, return_inverse=True)
        network = np.array(
            [
                router.reach(int(key // n_nodes), radius).get(int(key % n_nodes), np.inf)
                for key in unique
            ]
        )
        route[other] = (
            prev_len[i[other]] - prev_s[i[other]] + network[inverse] + next_s[j[other]]
        )

    scale = config.transition_scale_m * np.sqrt(max(gap, 1.0))
    with np.errstate(invalid="ignore"):
        score = -(np.abs(route - gps_distance) + extra) / scale
    score[~np.isfinite(route)] = -np.inf
    score = np.maximum(score, -config.unreachable_penalty)
    return score.reshape(len(prev_arc), len(next_arc))


def _empty_result(trip):
    return pd.DataFrame({"obs_idx": trip.track["obs_idx"].to_numpy(), "arc_id": -1})


def _candidate_groups(scored, trip):
    ordered = scored.sort_values(["obs_idx", "arc_id"], kind="stable")
    obs = ordered["obs_idx"].to_numpy()
    arcs = ordered["arc_id"].to_numpy().astype(np.int64)
    start_xy = trip.node_xy[trip.arcs[arcs, 0]]
    end_xy = trip.node_xy[trip.arcs[arcs, 1]]
    lengths = np.linalg.norm(end_xy - start_xy, axis=1)
    along = ordered["f_along_arc"].to_numpy(float) * lengths
    prob = ordered["p_match"].to_numpy(float)
    breaks = np.flatnonzero(obs[1:] != obs[:-1]) + 1
    edges = np.r_[0, breaks, len(obs)]
    return obs[edges[:-1]], edges, arcs, along, lengths, prob


def decode(scored, trip, config=None):
    """Return obs_idx and decoded arc_id for every valid GPS point."""
    config = config or load_config()
    result = _empty_result(trip)
    if scored.empty:
        return result
    obs, edges, arcs, along, lengths, prob = _candidate_groups(scored, trip)
    positions = np.searchsorted(trip.track["obs_idx"].to_numpy(), obs)
    points = trip.track_xy[positions]
    times = trip.track["t"].to_numpy()[positions]
    emission = config.emission_weight * np.log(np.clip(prob, config.min_probability, 1.0))
    router = RoadRouter(trip)

    def group(k):
        low, high = edges[k], edges[k + 1]
        return arcs[low:high], along[low:high], lengths[low:high]

    best = emission[edges[0]:edges[1]].copy()
    back = [None]
    previous = group(0)
    for k in range(1, len(obs)):
        current = group(k)
        gps_distance = float(np.linalg.norm(points[k] - points[k - 1]))
        gap = float(times[k] - times[k - 1])
        step_emission = emission[edges[k]:edges[k + 1]]
        if gps_distance > config.break_distance_m:
            origin = int(np.argmax(best))
            back.append(np.full(len(step_emission), origin))
            best = np.full(len(step_emission), best[origin]) + step_emission
        else:
            transition = _transition_scores(previous, current, gps_distance, gap, trip, router, config)
            totals = best[:, None] + transition
            pointer = np.argmax(totals, axis=0)
            back.append(pointer)
            best = totals[pointer, np.arange(totals.shape[1])] + step_emission
        previous = current

    chosen = np.empty(len(obs), dtype=np.int64)
    state = int(np.argmax(best))
    for k in range(len(obs) - 1, -1, -1):
        chosen[k] = arcs[edges[k] + state]
        if k:
            state = int(back[k][state])
    decoded = result["arc_id"].to_numpy().copy()
    decoded[positions] = chosen
    result["arc_id"] = decoded
    return result


def classifier_only(scored, trip):
    """Pick the highest-scoring candidate independently for each GPS point."""
    result = _empty_result(trip)
    if scored.empty:
        return result
    best = scored.loc[scored.groupby("obs_idx", sort=False)["p_match"].idxmax()]
    arc_by_obs = pd.Series(best["arc_id"].to_numpy(), index=best["obs_idx"].to_numpy())
    result["arc_id"] = result["obs_idx"].map(arc_by_obs).fillna(-1).astype(np.int64)
    return result
