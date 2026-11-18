"""
############################################################################
#  PLACEHOLDER PIPELINE - DEVELOPMENT ONLY - DELETE THIS FILE BEFORE THE DEMO
#
#  Temporary stand-ins for contract functions that are not written yet, so
#  the API and UI can be tested end to end. Nothing here is trained:
#    score            p_match = Gaussian of f_dist_m (no model)
#    decode           highest p_match per GPS point (no HMM)
#    point_metrics    simple Top-1 against ref_arc from add_labels
#    route_continuity same arc or to_node == next from_node
#  The API only uses a function from here if the real one cannot be
#  imported AND PATHSNAP_PLACEHOLDER=1 is set. Every response made with it
#  is flagged in the console and in the UI.
#  Never report any number produced with this file.
############################################################################
"""
import numpy as np
import pandas as pd

SIGMA_M = 10.0


def score(candidates):
    out = candidates.copy()
    out["p_match"] = np.exp(-0.5 * (out["f_dist_m"].to_numpy(dtype=float) / SIGMA_M) ** 2)
    return out


def decode(scored, trip, config):
    obs = trip.track["obs_idx"].to_numpy()
    pred = pd.DataFrame({"obs_idx": obs, "arc_id": -1})
    if len(scored):
        best = scored.loc[scored.groupby("obs_idx")["p_match"].idxmax(), ["obs_idx", "arc_id"]]
        pred = pred.set_index("obs_idx")
        pred.loc[best["obs_idx"].to_numpy(), "arc_id"] = best["arc_id"].to_numpy()
        pred = pred.reset_index()
    return pred


def point_metrics(pred, labelled):
    # add_labels keeps only points with a reliable reference, and gives each its ref_arc
    ref = labelled.groupby("obs_idx")["ref_arc"].first()
    predicted = pred.set_index("obs_idx")["arc_id"].reindex(ref.index, fill_value=-1)
    covered = labelled.groupby("obs_idx")["label"].max()
    return {"top1_accuracy": float((predicted == ref).mean()),
            "candidate_miss_rate": float(1 - covered.mean())}


def route_continuity(pred, trip):
    arcs = np.asarray(pred["arc_id"])
    arcs = arcs[arcs >= 0]
    if len(arcs) < 2:
        return 0.0
    prev, nxt = arcs[:-1], arcs[1:]
    ok = (prev == nxt) | (trip.arcs[prev, 1] == trip.arcs[nxt, 0])
    return float(ok.mean())
