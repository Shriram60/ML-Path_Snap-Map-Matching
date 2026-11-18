"""Point-level and route-level scores for decoded trips."""
import numpy as np

from hmm.viterbi import RoadRouter

REACHABLE_SLACK_M = 100.0


def _arc_lengths(trip, arcs):
    start = trip.node_xy[trip.arcs[arcs, 0]]
    end = trip.node_xy[trip.arcs[arcs, 1]]
    return np.linalg.norm(end - start, axis=1)


def continuity(pred, trip):
    """Return strict and reachable continuity between consecutive predictions.

    Strict means the same arc or directly joined arcs. Reachable also accepts a
    short drivable path, since a moving vehicle can cross a short arc between
    two GPS points.
    """
    valid = pred["arc_id"].to_numpy() >= 0
    arcs = pred["arc_id"].to_numpy()[valid].astype(np.int64)
    if len(arcs) < 2:
        return 1.0, 1.0
    first, second = arcs[:-1], arcs[1:]
    strict = (first == second) | (trip.arcs[first, 1] == trip.arcs[second, 0])
    reachable = strict.copy()
    positions = np.flatnonzero(valid)
    router = RoadRouter(trip)
    for k in np.flatnonzero(~strict):
        gap = np.linalg.norm(trip.track_xy[positions[k + 1]] - trip.track_xy[positions[k]])
        radius = 2.0 * gap + REACHABLE_SLACK_M
        source = int(trip.arcs[first[k], 1])
        reachable[k] = int(trip.arcs[second[k], 0]) in router.reach(source, radius)
    return float(strict.mean()), float(reachable.mean())


def score_trip(pred, references, trip):
    """Compare predictions with references, on points that have a reference."""
    joined = references[["obs_idx", "route_pos", "ref_arc"]].merge(pred, on="obs_idx")
    labelled = joined[joined["ref_arc"] >= 0]
    predicted = labelled["arc_id"].to_numpy()
    reference = labelled["ref_arc"].to_numpy()
    strict, reachable = continuity(pred, trip)

    covered = trip.route[labelled["route_pos"].min():labelled["route_pos"].max() + 1]
    true_arcs = set(trip.canonical_arc[covered].tolist())
    pred_arcs = set(predicted[predicted >= 0].tolist())
    hit = np.array(sorted(true_arcs & pred_arcs), dtype=np.int64)
    true_list = np.array(sorted(true_arcs), dtype=np.int64)
    pred_list = np.array(sorted(pred_arcs), dtype=np.int64)
    hit_length = _arc_lengths(trip, hit).sum()
    precision = hit_length / max(_arc_lengths(trip, pred_list).sum(), 1e-9)
    recall = hit_length / max(_arc_lengths(trip, true_list).sum(), 1e-9)
    route_f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "labelled_points": len(labelled),
        "correct_points": int(np.sum(predicted == reference)),
        "top1_accuracy": float(np.mean(predicted == reference)),
        "route_continuity": strict,
        "reachable_continuity": reachable,
        "route_precision": float(precision),
        "route_recall": float(recall),
        "route_f1": float(route_f1),
    }
