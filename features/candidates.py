"""Generate candidate road arcs and their features for every GPS point.

Nothing here reads trip.route, so the same code runs at inference. Arcs are
directed: the two directions of a road are separate candidates, and heading is
what separates them.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.geometry import point_segment

ID_COLUMNS = ["trip_id", "obs_idx", "t", "gps_lon", "gps_lat", "arc_id"]
FEATURE_COLUMNS = [
    "f_dist_m",
    "f_dist_rank",
    "f_dist_gap_to_best_m",
    "f_dist_ratio_to_best",
    "f_signed_offset_m",
    "f_along_arc",
    "f_dist_to_arc_end_m",
    "f_arc_length_log",
    "f_heading_cos",
    "f_heading_known",
    "f_heading_rank",
    "f_heading_cos_carried",
    "f_heading_cos_next",
    "f_straightness",
    "f_net_speed_mps",
    "f_context_dist_mean_m",
    "f_context_dist_max_m",
    "f_anchor_match",
    "f_anchor_support",
    "f_connected_prev",
    "f_connected_next",
    "f_from_out_degree",
    "f_to_out_degree",
    "f_has_reverse",
    "f_speed_mps",
    "f_step_prev_m",
    "f_dt_prev_log",
    "f_dt_next_log",
    "f_n_candidates",
]


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 60.0
    max_candidates: int = 24
    sample_spacing_m: float = 10.0
    heading_window: int = 2
    min_heading_move_m: float = 5.0
    steady_window: int = 5
    steady_min_straightness: float = 0.5
    steady_min_move_m: float = 15.0
    context_offsets: tuple = (-8, -3, 3, 8)
    context_cap_m: float = 200.0
    anchor_sigma_m: float = 10.0
    anchor_heading_weight: float = 2.0
    support_window: int = 5


def find_nearby_arcs(trip, radius_m, max_candidates=None, spacing_m=10.0, points=None):
    """Return (point position, arc, distance, along, closest) sorted by point then distance."""
    points = trip.track_xy if points is None else points
    start = trip.node_xy[trip.arcs[:, 0]]
    end = trip.node_xy[trip.arcs[:, 1]]
    has_length = np.linalg.norm(end - start, axis=1) > 0
    usable = (trip.canonical_arc == np.arange(len(trip.arcs))) & has_length
    reach = radius_m + spacing_m
    low = points.min(axis=0) - reach
    high = points.max(axis=0) + reach
    inside = (np.maximum(start, end) >= low).all(axis=1) & (np.minimum(start, end) <= high).all(axis=1)
    arc_ids = np.flatnonzero(usable & inside)
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0), np.empty((0, 2)))
    if len(arc_ids) == 0:
        return empty

    lengths = np.linalg.norm(end[arc_ids] - start[arc_ids], axis=1)
    counts = np.ceil(lengths / spacing_m).astype(int) + 1
    owner = np.repeat(np.arange(len(arc_ids)), counts)
    first = np.r_[0, np.cumsum(counts)[:-1]]
    along = (np.arange(len(owner)) - first[owner]) / (counts[owner] - 1)
    samples = start[arc_ids[owner]] + along[:, None] * (end - start)[arc_ids[owner]]
    hits = cKDTree(samples).query_ball_point(points, radius_m + spacing_m / 2)

    sizes = np.fromiter(map(len, hits), int, len(hits))
    if sizes.sum() == 0:
        return empty
    position = np.repeat(np.arange(len(points)), sizes)
    sample = np.fromiter((s for hit in hits for s in hit), np.int64, sizes.sum())
    keys = np.unique(position.astype(np.int64) * len(arc_ids) + owner[sample])
    position = (keys // len(arc_ids)).astype(int)
    arc = arc_ids[keys % len(arc_ids)]

    distance, along_arc, closest = point_segment(points[position], start[arc], end[arc])
    keep = distance <= radius_m
    position, arc, distance, along_arc, closest = (
        position[keep], arc[keep], distance[keep], along_arc[keep], closest[keep]
    )
    order = np.lexsort((arc, distance, position))
    position, arc, distance, along_arc, closest = (
        position[order], arc[order], distance[order], along_arc[order], closest[order]
    )
    if max_candidates is not None:
        rank = _group_rank(position)
        keep = rank < max_candidates
        position, arc, distance, along_arc, closest = (
            position[keep], arc[keep], distance[keep], along_arc[keep], closest[keep]
        )
    return position, arc, distance, along_arc, closest


def _group_rank(sorted_groups):
    starts = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    first = np.flatnonzero(starts)[np.cumsum(starts) - 1]
    return np.arange(len(sorted_groups)) - first


def _rank_within(groups, values):
    order = np.lexsort((-values, groups))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = _group_rank(groups[order])
    return ranks


def _motion(points, window, min_move):
    index = np.arange(len(points))
    vector = points[np.minimum(index + window, len(points) - 1)] - points[np.maximum(index - window, 0)]
    moved = np.linalg.norm(vector, axis=1)
    known = moved >= min_move
    unit = np.zeros_like(vector)
    unit[known] = vector[known] / moved[known, None]
    return unit, known


def motion_summary(points, times, config=CandidateConfig()):
    """Return net movement, straightness, elapsed time and a steady-movement flag.

    A stopped vehicle's points bounce around, so the path is long while the net
    movement is small. GPS jitter at a stop therefore does not count as steady.
    """
    n = len(points)
    index = np.arange(n)
    low = np.maximum(index - config.steady_window, 0)
    high = np.minimum(index + config.steady_window, n - 1)
    net_vector = points[high] - points[low]
    net = np.linalg.norm(net_vector, axis=1)
    travelled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    path = travelled[high] - travelled[low]
    straightness = np.divide(net, path, out=np.zeros(n), where=path > 0)
    elapsed = np.maximum(times[high] - times[low], 1.0)
    steady = (straightness >= config.steady_min_straightness) & (net >= config.steady_min_move_m)
    return net_vector, net, straightness, elapsed, steady


def _steady_motion(points, times, config):
    """Carry the heading across stops from the nearest steady movement."""
    net_vector, net, straightness, elapsed, steady = motion_summary(points, times, config)
    n = len(points)
    index = np.arange(n)
    unit = np.zeros((n, 2))
    unit[steady] = net_vector[steady] / net[steady, None]
    last = np.maximum.accumulate(np.where(steady, index, -1))
    following = np.minimum.accumulate(np.where(steady, index, n)[::-1])[::-1]
    carried = np.zeros((n, 2))
    upcoming = np.zeros((n, 2))
    carried[last >= 0] = unit[last[last >= 0]]
    upcoming[following < n] = unit[following[following < n]]
    no_earlier = last < 0
    carried[no_earlier] = upcoming[no_earlier]
    return carried, upcoming, straightness, net / elapsed


def _empty_table():
    table = pd.DataFrame(columns=ID_COLUMNS + FEATURE_COLUMNS)
    return table.astype({"obs_idx": np.int32, "arc_id": np.int32})


def build_candidates(trip, config=CandidateConfig()):
    """Return one row per GPS point and candidate arc, without labels."""
    pos, arc, dist, along, closest = find_nearby_arcs(
        trip, config.radius_m, config.max_candidates, config.sample_spacing_m
    )
    if len(pos) == 0:
        return _empty_table()

    points = trip.track_xy
    n_points = len(points)
    n_arcs = len(trip.arcs)
    track = trip.track
    times = track["t"].to_numpy()
    from_node, to_node = trip.arcs[:, 0], trip.arcs[:, 1]
    start, end = trip.node_xy[from_node[arc]], trip.node_xy[to_node[arc]]
    segment = end - start
    length = np.linalg.norm(segment, axis=1)
    direction = segment / length[:, None]
    from_gps = points[pos] - closest

    best_dist = np.full(n_points, np.nan)
    first_rows = np.flatnonzero(_group_rank(pos) == 0)
    best_dist[pos[first_rows]] = dist[first_rows]
    n_candidates = np.bincount(pos, minlength=n_points)

    motion, known = _motion(points, config.heading_window, config.min_heading_move_m)
    heading_cos = np.sum(motion[pos] * direction, axis=1)
    carried, upcoming, straightness, net_speed = _steady_motion(points, times, config)
    carried_cos = np.sum(carried[pos] * direction, axis=1)
    next_cos = np.sum(upcoming[pos] * direction, axis=1)

    context = [
        np.minimum(
            point_segment(points[np.clip(pos + offset, 0, n_points - 1)], start, end)[0],
            config.context_cap_m,
        )
        for offset in config.context_offsets
    ]
    context = np.vstack(context)

    # The anchor is each point's most plausible arc by simple geometry. It gives
    # every candidate context about its neighbours without using any labels.
    anchor_score = (
        -0.5 * (dist / config.anchor_sigma_m) ** 2 + config.anchor_heading_weight * carried_cos
    )
    anchor_rows = np.flatnonzero(_rank_within(pos, anchor_score) == 0)
    anchor = np.full(n_points, -1, dtype=np.int64)
    anchor[pos[anchor_rows]] = arc[anchor_rows]
    anchor_keys = np.flatnonzero(anchor >= 0) * n_arcs + anchor[anchor >= 0]

    window = [o for o in range(-config.support_window, config.support_window + 1) if o != 0]
    support = np.zeros(len(pos))
    for offset in window:
        support += np.isin((pos + offset) * n_arcs + arc, anchor_keys)
    support /= len(window)

    def connected(neighbour, forward):
        has = (neighbour >= 0) & (neighbour < n_points)
        other = np.full(len(pos), -1, dtype=np.int64)
        other[has] = anchor[neighbour[has]]
        valid = other >= 0
        result = np.zeros(len(pos), dtype=bool)
        same = other[valid] == arc[valid]
        if forward:
            linked = to_node[arc[valid]] == from_node[other[valid]]
        else:
            linked = to_node[other[valid]] == from_node[arc[valid]]
        result[valid] = same | linked
        return result

    out_degree = np.bincount(from_node, minlength=len(trip.nodes))
    arc_keys = from_node.astype(np.int64) * len(trip.nodes) + to_node
    reverse_keys = to_node[arc].astype(np.int64) * len(trip.nodes) + from_node[arc]

    steps = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1)]
    dt_prev = np.r_[1.0, np.diff(times)]
    dt_next = np.r_[np.diff(times), 1.0]
    best = best_dist[pos]

    features = {
        "f_dist_m": dist,
        "f_dist_rank": _group_rank(pos),
        "f_dist_gap_to_best_m": dist - best,
        "f_dist_ratio_to_best": (dist + 1.0) / (best + 1.0),
        "f_signed_offset_m": direction[:, 0] * from_gps[:, 1] - direction[:, 1] * from_gps[:, 0],
        "f_along_arc": along,
        "f_dist_to_arc_end_m": length * np.minimum(along, 1.0 - along),
        "f_arc_length_log": np.log1p(length),
        "f_heading_cos": heading_cos,
        "f_heading_known": known[pos],
        "f_heading_rank": _rank_within(pos, carried_cos),
        "f_heading_cos_carried": carried_cos,
        "f_heading_cos_next": next_cos,
        "f_straightness": straightness[pos],
        "f_net_speed_mps": net_speed[pos],
        "f_context_dist_mean_m": context.mean(axis=0),
        "f_context_dist_max_m": context.max(axis=0),
        "f_anchor_match": arc == anchor[pos],
        "f_anchor_support": support,
        "f_connected_prev": connected(pos - 1, forward=False),
        "f_connected_next": connected(pos + 1, forward=True),
        "f_from_out_degree": out_degree[from_node[arc]],
        "f_to_out_degree": out_degree[to_node[arc]],
        "f_has_reverse": np.isin(reverse_keys, arc_keys),
        "f_speed_mps": (steps / dt_prev)[pos],
        "f_step_prev_m": steps[pos],
        "f_dt_prev_log": np.log1p(dt_prev)[pos],
        "f_dt_next_log": np.log1p(dt_next)[pos],
        "f_n_candidates": n_candidates[pos],
    }
    table = pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "obs_idx": track["obs_idx"].to_numpy()[pos].astype(np.int32),
            "t": times[pos],
            "gps_lon": track["lon"].to_numpy()[pos],
            "gps_lat": track["lat"].to_numpy()[pos],
            "arc_id": arc.astype(np.int32),
        }
    )
    for name in FEATURE_COLUMNS:
        table[name] = np.asarray(features[name], dtype=np.float32)
    return table
