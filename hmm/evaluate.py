"""Tune decoder settings and compare classifier-only with classifier plus HMM.

Expected input: <scores>/<tuning set>/<trip>.parquet and
<scores>/student_test/<trip>.parquet, with obs_idx, arc_id, f_along_arc, p_match.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import data_root, work_root
from features.labels import reference_table
from hmm.route_metrics import score_trip
from hmm.viterbi import DecoderConfig, classifier_only, decode, save_config

OUTPUTS = Path(__file__).parent / "outputs"
GRID = {
    "emission_weight": (0.5, 1.0, 2.0),
    "transition_scale_m": (5.0, 10.0, 20.0, 40.0),
    "unreachable_penalty": (10.0, 25.0, 50.0),
}


def trip_folder(trip_id, partition):
    source = "student_test" if partition == "student_test" else "train"
    return data_root() / source / trip_id


def _run(job):
    path, partition, config, include_classifier = job
    scored = pd.read_parquet(path)
    trip_id = path.stem
    trip = load_trip(trip_folder(trip_id, partition))
    references = reference_table(trip)
    rows = [{"trip_id": trip_id, "method": "hmm", **score_trip(decode(scored, trip, config), references, trip)}]
    if include_classifier:
        pred = classifier_only(scored, trip)
        rows.append({"trip_id": trip_id, "method": "classifier_only", **score_trip(pred, references, trip)})
    return rows


def _evaluate(pool, folder, partition, config, include_classifier=False):
    jobs = [(p, partition, config, include_classifier) for p in sorted(Path(folder).glob("*.parquet"))]
    return pd.DataFrame([row for rows in pool.map(_run, jobs) for row in rows])


def _summary(table):
    grouped = table.groupby("method")
    result = grouped[["route_continuity", "reachable_continuity", "route_f1"]].mean()
    totals = grouped[["correct_points", "labelled_points"]].sum()
    result.insert(0, "top1_accuracy", totals["correct_points"] / totals["labelled_points"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=None, help="folder of scored candidates")
    parser.add_argument("--tuning-set", default="tuning")
    parser.add_argument("--source", default="temporary baseline scores")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    scores = Path(args.scores) if args.scores else work_root() / "baseline_scores"
    OUTPUTS.mkdir(exist_ok=True)

    search = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for values in product(*GRID.values()):
            settings = dict(zip(GRID, values))
            config = replace(DecoderConfig(), **settings)
            table = _evaluate(pool, scores / args.tuning_set, "train", config)
            row = _summary(table).loc["hmm"].to_dict()
            search.append({**settings, **row})
            print(settings, {k: round(v, 4) for k, v in row.items()}, flush=True)
        search = pd.DataFrame(search).sort_values(
            ["top1_accuracy", "reachable_continuity"], ascending=False
        )
        search.to_csv(OUTPUTS / "decoder_search.csv", index=False, float_format="%.5f")
        best = search.iloc[0]
        config = replace(DecoderConfig(), **{name: float(best[name]) for name in GRID})
        save_config(config, args.source)

        tuning = _evaluate(pool, scores / args.tuning_set, "train", config, True)
        test = _evaluate(pool, scores / "student_test", "student_test", config, True)

    test.to_csv(OUTPUTS / "student_test_by_trip.csv", index=False, float_format="%.5f")
    comparison = pd.concat(
        {"tuning": _summary(tuning), "student_test": _summary(test)}, names=["dataset"]
    ).reset_index()
    comparison.insert(0, "score_source", args.source)
    comparison.to_csv(OUTPUTS / "hmm_comparison.csv", index=False, float_format="%.5f")
    print("Best", asdict(config))
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
