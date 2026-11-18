"""Measure label coverage, plausibility and sensitivity to the alignment settings."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import PARTITIONS, trip_dirs
from features.labels import LabelConfig, reference_table

OUTPUT = Path(__file__).parent / "outputs" / "label_quality.csv"
HEADING_WINDOW = 3
MIN_MOVEMENT_M = 10.0


def heading_agreement(trip, table):
    """Share of clearly moving labelled points travelling the same way as their arc."""
    points = trip.track_xy
    n = len(points)
    index = np.arange(n)
    motion = points[np.minimum(index + HEADING_WINDOW, n - 1)] - points[
        np.maximum(index - HEADING_WINDOW, 0)
    ]
    arcs = table["ref_arc"].to_numpy()
    mask = (arcs >= 0) & (np.linalg.norm(motion, axis=1) >= MIN_MOVEMENT_M)
    if not mask.any():
        return np.nan
    direction = trip.node_xy[trip.arcs[arcs[mask], 1]] - trip.node_xy[trip.arcs[arcs[mask], 0]]
    return float(np.mean(np.sum(motion[mask] * direction, axis=1) > 0))


def assess(job):
    partition, trip_dir = job
    trip = load_trip(trip_dir)
    base = reference_table(trip)
    labelled = base["ref_arc"].to_numpy() >= 0
    row = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_points": len(base),
        "route_arcs": len(trip.route),
        "labelled_share": labelled.mean(),
        "route_arcs_with_points": base.loc[labelled, "route_pos"].nunique() / len(trip.route),
        "median_ref_dist_m": base.loc[labelled, "ref_dist_m"].median(),
        "p95_ref_dist_m": base.loc[labelled, "ref_dist_m"].quantile(0.95),
        "heading_agreement": heading_agreement(trip, base),
    }
    for penalty in (0.05, 0.2):
        other = reference_table(trip, replace(LabelConfig(), skip_penalty_per_m=penalty))
        same = base["ref_arc"].to_numpy()[labelled] == other["ref_arc"].to_numpy()[labelled]
        row[f"agreement_penalty_{penalty}"] = same.mean()
    return row


def main():
    jobs = [(p, d) for p in PARTITIONS for d in trip_dirs(p)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        table = pd.DataFrame(pool.map(assess, jobs))
    OUTPUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT, index=False, float_format="%.5f")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    numeric = table.drop(columns=["trip_id"])
    print(numeric.groupby("partition").describe(percentiles=[0.05, 0.5]).T.round(4).to_string())
    print(table.sort_values("labelled_share").head(10).round(3).to_string(index=False))
    print(table.sort_values("heading_agreement").head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
