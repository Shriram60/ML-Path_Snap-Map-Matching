"""Create per-GPS-point reference arcs from the ordered .route file.

The route says which arcs were driven and in what order, but not when. Each GPS
point is therefore aligned to a route position with dynamic programming, under
the constraint that the position can never move backwards.

Cost of a point is its distance to the route arc, capped so that stretches of
GPS away from the route do not distort the alignment, plus a cost for steady
movement against the arc's direction. Skipping whole route arcs
costs a little per metre, and less across a time gap, so loops are followed
instead of jumped over. Points that stay too far from their aligned arc, lie
beyond the ends of the route, or wander at the start or end of the trip for far
longer than the arc itself get no reference arc (-1). The same applies to
stopped points on a road the route drives out and straight back along, where
the direction cannot be known.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_processing.geometry import point_segment
from features.candidates import motion_summary


@dataclass(frozen=True)
class LabelConfig:
    capped_distance_m: float = 60.0
    skip_penalty_per_m: float = 0.1
    wrong_direction_cost_m: float = 20.0
    max_reference_distance_m: float = 50.0
    max_path_arc_ratio: float = 3.0
    path_allowance_m: float = 100.0


def _wandering_mask(trip, route_pos, config):
    """Flag the opening or closing run when its GPS path far exceeds its arc.

    Only the two end runs are tested. Mid-trip runs can legitimately build up a
    long path from GPS jitter while a vehicle waits on one arc.
    """
    steps = np.linalg.norm(np.diff(trip.track_xy, axis=0), axis=1)
    path = np.r_[0.0, steps]
    runs = np.cumsum(np.r_[0, np.diff(route_pos) != 0])
    run_path = np.bincount(runs, weights=path)
    route = trip.route[route_pos]
    arc_xy = trip.node_xy[trip.arcs[route]]
    arc_length = np.linalg.norm(arc_xy[:, 1] - arc_xy[:, 0], axis=1)
    limit = config.max_path_arc_ratio * arc_length + config.path_allowance_m
    end_run = (runs == runs[0]) | (runs == runs[-1])
    return end_run & (run_path[runs] > limit)


def _ambiguous_turnaround_mask(trip, route_pos):
    """Flag stopped points on a road the route drives out and straight back.

    The two directions share the same geometry, so without steady movement the
    GPS cannot show which leg the vehicle was on.
    """
    route_arcs = trip.arcs[trip.route]
    reverses = (route_arcs[:-1, 0] == route_arcs[1:, 1]) & (route_arcs[:-1, 1] == route_arcs[1:, 0])
    in_pair = np.r_[reverses, False] | np.r_[False, reverses]
    steady = motion_summary(trip.track_xy, trip.track["t"].to_numpy())[4]
    return in_pair[route_pos] & ~steady


def align_to_route(trip, config=LabelConfig()):
    """Return the route position, distance and along-arc position per GPS point."""
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"{trip.trip_id}: no reference route")
    points = trip.track_xy
    start_xy = trip.node_xy[trip.arcs[trip.route, 0]]
    end_xy = trip.node_xy[trip.arcs[trip.route, 1]]
    lengths = np.linalg.norm(end_xy - start_xy, axis=1)
    arc_end = np.cumsum(lengths)
    arc_start = arc_end - lengths

    n_points, n_arcs = len(points), len(trip.route)
    positions = np.arange(n_arcs)
    gaps = np.diff(trip.track["t"].to_numpy(), prepend=np.nan)
    penalties = config.skip_penalty_per_m / np.maximum(np.nan_to_num(gaps, nan=1.0), 1.0)
    previous = np.empty((n_points, n_arcs), dtype=np.int32)
    previous[0] = positions

    # Opposite directions of a road have identical geometry, so steady movement
    # against an arc's direction must cost something for the two to be separated.
    net_vector, net, _, _, steady = motion_summary(points, trip.track["t"].to_numpy())
    heading = np.zeros_like(points)
    heading[steady] = net_vector[steady] / net[steady, None]
    arc_direction = np.divide(
        end_xy - start_xy,
        lengths[:, None],
        out=np.zeros_like(end_xy),
        where=lengths[:, None] > 0,
    )

    def point_cost(index):
        distance = np.minimum(
            point_segment(points[index], start_xy, end_xy)[0], config.capped_distance_m
        )
        if not steady[index]:
            return distance
        opposition = 0.5 * (1.0 - arc_direction @ heading[index])
        return distance + config.wrong_direction_cost_m * opposition

    best = point_cost(0)
    for i in range(1, n_points):
        penalty = penalties[i]
        leave = best - penalty * arc_end
        running = np.minimum.accumulate(leave)
        source = np.maximum.accumulate(np.where(leave == running, positions, 0))
        arrive = np.full(n_arcs, np.inf)
        arrive[1:] = running[:-1] + penalty * arc_start[1:]
        move = arrive < best
        previous[i] = positions
        previous[i, 1:][move[1:]] = source[:-1][move[1:]]
        best = np.where(move, arrive, best) + point_cost(i)

    route_pos = np.empty(n_points, dtype=np.int32)
    route_pos[-1] = int(np.argmin(best))
    for i in range(n_points - 1, 0, -1):
        route_pos[i - 1] = previous[i, route_pos[i]]

    distance, along, _ = point_segment(points, start_xy[route_pos], end_xy[route_pos])
    return route_pos, distance, along


def reference_table(trip, config=LabelConfig()):
    """One row per GPS point, with ref_arc set to -1 when there is no reliable match."""
    route_pos, distance, along = align_to_route(trip, config)
    arc = trip.canonical_arc[trip.route[route_pos]]
    before_route = (route_pos == 0) & (along <= 0)
    after_route = (route_pos == len(trip.route) - 1) & (along >= 1)
    wandering = _wandering_mask(trip, route_pos, config)
    usable = (
        (distance <= config.max_reference_distance_m)
        & ~before_route
        & ~after_route
        & ~wandering
        & ~_ambiguous_turnaround_mask(trip, route_pos)
    )
    return pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "obs_idx": trip.track["obs_idx"].to_numpy(),
            "route_pos": route_pos,
            "ref_arc": np.where(usable, arc, -1).astype(np.int32),
            "ref_dist_m": distance.astype(np.float32),
        }
    )


def add_labels(candidates, trip, config=LabelConfig()):
    """Add label and ref_arc, keeping only GPS points with a reliable reference."""
    references = reference_table(trip, config)
    references = references.loc[references["ref_arc"] >= 0, ["obs_idx", "ref_arc"]]
    labelled = candidates.merge(references, on="obs_idx", how="inner")
    labelled["label"] = (labelled["arc_id"] == labelled["ref_arc"]).astype(np.int8)
    return labelled
