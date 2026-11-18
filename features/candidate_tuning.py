"""Tune candidate generation with Bayesian optimisation on training trips only.

The objective balances two costs: missing the true road, and adding candidates.
One extra candidate per point must cut the miss rate by at least 0.05 percentage
points to be worthwhile. The radius search starts at 50 m, the largest distance
allowed between a labelled point and its reference arc, so every reference arc
stays within reach.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import trip_dirs
from features.candidates import find_nearby_arcs
from features.labels import reference_table

OUTPUT = Path(__file__).parent / "outputs" / "candidate_tuning.csv"
SEED = 42
TRIALS = 40
POINT_STRIDE = 5
MAX_RADIUS_M = 100
COST_PER_CANDIDATE = 0.0005
NOT_FOUND = 10**9


def collect(trip_dir):
    trip = load_trip(trip_dir)
    if trip.trip_id == "00000033":
        return None
    references = reference_table(trip)
    chosen = np.flatnonzero(references["ref_arc"].to_numpy() >= 0)[::POINT_STRIDE]
    ref_arc = references["ref_arc"].to_numpy()[chosen]
    pos, arc, dist, _, _ = find_nearby_arcs(trip, MAX_RADIUS_M, points=trip.track_xy[chosen])
    first = np.r_[True, pos[1:] != pos[:-1]]
    rank = np.arange(len(pos)) - np.flatnonzero(first)[np.cumsum(first) - 1]
    match = arc == ref_arc[pos]
    ref_rank = np.full(len(chosen), NOT_FOUND)
    ref_dist = np.full(len(chosen), np.inf)
    ref_rank[pos[match]] = rank[match]
    ref_dist[pos[match]] = dist[match]
    return pos, dist, ref_rank, ref_dist


def main():
    with ProcessPoolExecutor(max_workers=8) as pool:
        parts = [part for part in pool.map(collect, trip_dirs("train")) if part is not None]
    offsets = np.cumsum([0] + [len(part[2]) for part in parts[:-1]])
    pos = np.concatenate([part[0] + offset for part, offset in zip(parts, offsets)])
    dist = np.concatenate([part[1] for part in parts])
    ref_rank = np.concatenate([part[2] for part in parts])
    ref_dist = np.concatenate([part[3] for part in parts])
    n_points = len(ref_rank)

    def objective(trial):
        radius = trial.suggest_int("radius_m", 50, MAX_RADIUS_M)
        count = trial.suggest_int("max_candidates", 4, 40)
        miss_rate = float(np.mean((ref_dist > radius) | (ref_rank >= count)))
        per_point = np.bincount(pos[dist <= radius], minlength=n_points)
        mean_candidates = float(np.minimum(per_point, count).mean())
        trial.set_user_attr("miss_rate", miss_rate)
        trial.set_user_attr("mean_candidates", mean_candidates)
        return miss_rate + COST_PER_CANDIDATE * mean_candidates

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS)

    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    OUTPUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT, index=False, float_format="%.6f")
    print("Best", study.best_params, "objective", round(study.best_value, 5), study.best_trial.user_attrs)


if __name__ == "__main__":
    main()
