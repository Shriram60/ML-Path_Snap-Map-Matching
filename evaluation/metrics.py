"""Evaluation metrics for PathSnap.

Two levels are measured:
- Row level: is each (GPS point, candidate arc) row a match or not.
  Used for Macro F1, precision, recall, confusion matrix and log loss.
- Point level: for each GPS point, is the chosen arc the reference arc.
  Used for Top-1 accuracy, candidate miss rate and route continuity.

Only points with a reliable reference (ref_arc >= 0) are scored.
Arc IDs are canonical in the candidate tables, so they can be compared directly.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

KEY = ["trip_id", "obs_idx"]
THRESHOLD = 0.5  # row is called a match when p_match >= 0.5


def top1_predictions(scored):
    """Classifier-only choice: the candidate with the highest p_match per GPS point."""
    best = scored.sort_values(KEY + ["p_match"], ascending=[True, True, False])
    best = best.drop_duplicates(KEY, keep="first")
    return best[KEY + ["arc_id"]].reset_index(drop=True)


def row_metrics(labelled):
    """Classification metrics on candidate rows. Needs `label` and `p_match`."""
    rows = labelled[labelled["ref_arc"] >= 0] if "ref_arc" in labelled else labelled
    y = rows["label"].to_numpy()
    p = np.clip(rows["p_match"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    yhat = (p >= THRESHOLD).astype(int)
    return {
        "accuracy": float((yhat == y).mean()),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def point_metrics(pred, labelled, references=None):
    """Point-level and row-level metrics.

    pred: one row per GPS point with trip_id, obs_idx, arc_id (-1 = no road).
    labelled: candidate table with label and ref_arc, plus p_match if available.
    references: optional per-point reference table. When given, points that got
    no candidates at all also count in the miss rate and as Top-1 errors.
    """
    if references is not None:
        points = references.loc[references["ref_arc"] >= 0, KEY + ["ref_arc"]]
    else:
        points = labelled.loc[labelled["ref_arc"] >= 0, KEY + ["ref_arc"]].drop_duplicates(KEY)

    merged = points.merge(pred[KEY + ["arc_id"]], on=KEY, how="left")
    merged["arc_id"] = merged["arc_id"].fillna(-1)
    top1 = float((merged["arc_id"] == merged["ref_arc"]).mean())

    has_true = labelled.loc[labelled["label"] == 1, KEY].drop_duplicates()
    found = points.merge(has_true, on=KEY, how="inner")
    miss_rate = 1.0 - len(found) / max(len(points), 1)

    out = {"top1_accuracy": top1, "candidate_miss_rate": float(miss_rate), "n_points": len(points)}
    if "p_match" in labelled:
        out.update(row_metrics(labelled))
    return out


def route_continuity(pred, trip):
    """Share of consecutive predicted points on the same arc or on connected arcs.

    Connected means the first arc ends at the node where the second arc starts.
    Points with arc_id -1 are skipped, so the pair is formed with the next assigned point.
    """
    seq = pred[pred["arc_id"] >= 0].sort_values("obs_idx")["arc_id"].to_numpy(dtype=np.int64)
    if len(seq) < 2:
        return float("nan")
    a, b = seq[:-1], seq[1:]
    canon = trip.canonical_arc
    same = canon[a] == canon[b]
    linked = trip.arcs[a, 1] == trip.arcs[b, 0]
    return float((same | linked).mean())
