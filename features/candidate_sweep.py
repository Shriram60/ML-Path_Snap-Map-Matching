"""Choose the candidate radius and count from training trips only."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import trip_dirs
from features.candidates import find_nearby_arcs
from features.labels import reference_table

OUTPUT = Path(__file__).parent / "outputs" / "candidate_sweep.csv"
RADII = (20, 30, 40, 50, 60, 80, 100)
COUNTS = (2, 4, 6, 8, 12, 16, 24, 32)
POINT_STRIDE = 5


def measure(trip_dir):
    trip = load_trip(trip_dir)
    references = reference_table(trip)
    chosen = np.flatnonzero(references["ref_arc"].to_numpy() >= 0)[::POINT_STRIDE]
    ref_arc = references["ref_arc"].to_numpy()[chosen]
    pos, arc, dist, _, _ = find_nearby_arcs(trip, max(RADII), points=trip.track_xy[chosen])
    first = np.r_[True, pos[1:] != pos[:-1]]
    rank = np.arange(len(pos)) - np.flatnonzero(first)[np.cumsum(first) - 1]
    match = arc == ref_arc[pos]
    ref_rank = np.full(len(chosen), 10**9)
    ref_dist = np.full(len(chosen), np.inf)
    ref_rank[pos[match]] = rank[match]
    ref_dist[pos[match]] = dist[match]
    sizes = [np.bincount(pos[dist <= r], minlength=len(chosen)) for r in RADII]
    rows = []
    for r, per_point in zip(RADII, sizes):
        for k in COUNTS:
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "radius_m": r,
                    "max_candidates": k,
                    "points": len(chosen),
                    "misses": int(np.sum((ref_dist > r) | (ref_rank >= k))),
                    "candidates": int(np.minimum(per_point, k).sum()),
                }
            )
    return rows


def main():
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(measure, trip_dirs("train")))
    table = pd.DataFrame([row for rows in results for row in rows])
    grouped = table.groupby(["radius_m", "max_candidates"], as_index=False)[
        ["points", "misses", "candidates"]
    ].sum()
    grouped["miss_rate"] = grouped["misses"] / grouped["points"]
    grouped["mean_candidates"] = grouped["candidates"] / grouped["points"]
    OUTPUT.parent.mkdir(exist_ok=True)
    grouped.to_csv(OUTPUT, index=False, float_format="%.5f")
    print(grouped.pivot(index="radius_m", columns="max_candidates", values="miss_rate").round(4).to_string())
    print(grouped.pivot(index="radius_m", columns="max_candidates", values="mean_candidates").round(1).to_string())


if __name__ == "__main__":
    main()
