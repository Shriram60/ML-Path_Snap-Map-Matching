"""Temporary development scores, used until the team's final model is exported.

This baseline exists only to build and test the decoder. It is not one of the
reported models. Trips are split so the decoder is tuned on trips the baseline
did not train on.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from data_processing.paths import work_root

SEED = 42
TUNING_TRIPS = 19
MAX_TRAIN_ROWS = 500_000


def main():
    work = work_root()
    train = pd.read_parquet(work / "candidates" / "train")
    test = pd.read_parquet(work / "candidates" / "student_test")
    features = [c for c in train.columns if c.startswith("f_")]
    trips = np.array(sorted(train["trip_id"].unique().tolist()))
    np.random.default_rng(SEED).shuffle(trips)
    tuning = set(trips[:TUNING_TRIPS].tolist())
    is_tuning = train["trip_id"].isin(tuning)
    fit = train[~is_tuning]
    if len(fit) > MAX_TRAIN_ROWS:
        fit = fit.sample(MAX_TRAIN_ROWS, random_state=SEED)
    model = HistGradientBoostingClassifier(max_iter=150, random_state=SEED)
    model.fit(fit[features], fit["label"])

    output = work / "baseline_scores"
    for name, table in (("tuning", train[is_tuning]), ("student_test", test)):
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        scored = table[["trip_id", "obs_idx", "arc_id", "f_along_arc"]].copy()
        scored["p_match"] = model.predict_proba(table[features])[:, 1]
        for trip_id, rows in scored.groupby("trip_id"):
            rows.to_parquet(Path(folder) / f"{trip_id}.parquet", index=False)
        print(name, table["trip_id"].nunique(), "trips", len(table), "rows")


if __name__ == "__main__":
    main()
