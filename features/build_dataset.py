"""Build the candidate tables used for training and student testing.

Output, under PATHSNAP_WORK:
  candidates/<partition>/<trip>.parquet   labelled candidate rows
  references/<partition>/<trip>.parquet   one row per GPS point, for miss rates
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import PARTITIONS, trip_dirs, work_root
from features.candidates import build_candidates
from features.label_report import heading_agreement
from features.labels import add_labels, reference_table

SUMMARY = Path(__file__).parent / "outputs" / "build_summary.csv"
MIN_TRAIN_HEADING_AGREEMENT = 0.85


def build_trip(job):
    partition, trip_dir, work = job
    trip = load_trip(trip_dir)
    references = reference_table(trip)
    agreement = heading_agreement(trip, references)
    excluded = partition == "train" and agreement < MIN_TRAIN_HEADING_AGREEMENT

    # Features are built without the route, exactly as at inference.
    candidates = build_candidates(trip.without_route())
    labelled = add_labels(candidates, trip)

    reference_dir = work / "references" / partition
    candidate_dir = work / "candidates" / partition
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    references.to_parquet(reference_dir / f"{trip.trip_id}.parquet", index=False)
    target = candidate_dir / f"{trip.trip_id}.parquet"
    if excluded:
        target.unlink(missing_ok=True)
    else:
        labelled.to_parquet(target, index=False)

    has_reference = references["ref_arc"].to_numpy() >= 0
    found = labelled.loc[labelled["label"] == 1, "obs_idx"].nunique()
    return {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_points": len(references),
        "labelled_points": int(has_reference.sum()),
        "points_with_true_candidate": int(found),
        "candidate_miss_rate": 1 - found / max(int(has_reference.sum()), 1),
        "rows": len(labelled),
        "positive_share": float(labelled["label"].mean()) if len(labelled) else np.nan,
        "heading_agreement": agreement,
        "excluded_from_training": excluded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    work = work_root()
    jobs = [(p, d, work) for p in PARTITIONS for d in trip_dirs(p)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summary = pd.DataFrame(pool.map(build_trip, jobs))
    SUMMARY.parent.mkdir(exist_ok=True)
    summary.to_csv(SUMMARY, index=False, float_format="%.5f")
    kept = summary[~summary["excluded_from_training"]]
    totals = kept.groupby("partition").agg(
        trips=("trip_id", "size"),
        gps_points=("gps_points", "sum"),
        labelled_points=("labelled_points", "sum"),
        found=("points_with_true_candidate", "sum"),
        rows=("rows", "sum"),
    )
    totals["miss_rate"] = 1 - totals["found"] / totals["labelled_points"]
    totals["rows_per_point"] = totals["rows"] / totals["labelled_points"]
    totals["positive_share"] = totals["found"] / totals["rows"]
    print(totals.round(4).to_string())
    print("Excluded:", summary.loc[summary["excluded_from_training"], "trip_id"].tolist())


if __name__ == "__main__":
    main()
