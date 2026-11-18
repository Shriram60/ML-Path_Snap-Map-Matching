"""Create the classifier scores used to tune and test the decoder.

The decoder must be tuned on scores from trips the classifier did not train on.
One third of the training trips is held out, and a copy of the selected model is
fitted on a sample of the other trips with the same settings. Student-test trips
are scored with the exported final model.

Every GPS point is scored, including points with no reliable reference, because
the deployed decoder sees all of them. References are used only to compute metrics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_processing.loader import load_trip
from data_processing.paths import trip_dirs, work_root
from features.candidates import build_candidates
from models.common import SEED, build_model, load_partition, sample_points
from models.inference import score

CONFIG = Path("models") / "artifacts" / "model_config.json"
FOLDS = 3
SAMPLE_FRACTION = 0.2
KEEP = ["trip_id", "obs_idx", "arc_id", "f_along_arc", "p_match"]


def save(table, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.parquet"):
        old.unlink()
    for trip_id, rows in table[KEEP].groupby("trip_id"):
        rows.to_parquet(folder / f"{trip_id}.parquet", index=False)
    print(folder.name, table["trip_id"].nunique(), "trips", len(table), "rows", flush=True)


def all_point_candidates(partition, trip_ids=None):
    """Candidate rows for every GPS point, built without the route as at inference."""
    folders = [d for d in trip_dirs(partition) if trip_ids is None or d.name in trip_ids]
    return pd.concat(
        [build_candidates(load_trip(d).without_route()) for d in folders], ignore_index=True
    )


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    features = config["feature_columns"]
    train = load_partition("train")
    output = work_root() / "final_scores"

    splitter = GroupKFold(n_splits=FOLDS)
    fit_rows, held_rows = next(splitter.split(train, groups=train["trip_id"]))
    fit = sample_points(train.iloc[fit_rows], SAMPLE_FRACTION, SEED)
    held_trips = set(train["trip_id"].iloc[held_rows])
    assert not set(fit["trip_id"]) & held_trips
    model = build_model(config["model"], config["params"])
    model.fit(fit[features].to_numpy(), fit["label"].to_numpy())
    held = all_point_candidates("train", held_trips)
    held["p_match"] = model.predict_proba(held[features].to_numpy())[:, 1]
    save(held, output / "tuning")

    save(score(all_point_candidates("student_test")), output / "student_test")
    print("held-out trips:", int(np.unique(held["trip_id"]).size), "fit rows:", len(fit))


if __name__ == "__main__":
    main()
