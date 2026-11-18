
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path(os.environ["PATHSNAP_DATA"])
OUTPUT_DIR = Path(os.environ["PATHSNAP_WORK"]) / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["longitude", "latitude", "timestamp"]

EARTH_RADIUS_M = 6_371_000


# --------------------------------------------------
# 2. HAVERSINE DISTANCE FUNCTION
# --------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between
    two GPS coordinates in metres.
    """

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# --------------------------------------------------
# 3. FIND ALL TRIP FOLDERS
# --------------------------------------------------

trip_folders = []

for partition in ["train", "student_test"]:

    partition_dir = DATA_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):

        if trip_dir.is_dir():
            trip_folders.append((partition, trip_dir))


# --------------------------------------------------
# 4. ANALYSE EACH TRIP
# --------------------------------------------------

results = []

all_time_gaps = []
all_speeds = []

for partition, trip_dir in trip_folders:

    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"

    if not track_file.exists():
        continue

    # Read GPS observations
    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    # Convert columns to numeric values
    track = track.apply(pd.to_numeric, errors="coerce")

    # Count invalid rows before removing them
    invalid_rows_removed = int(
        track.isna().any(axis=1).sum()
    )

    # Remove rows with missing or invalid values
    track = track.dropna().reset_index(drop=True)

    if len(track) < 2:
        print(f"Skipping {trip_id}: insufficient GPS points")
        continue

    # Extract GPS coordinates and timestamps
    lon = track["longitude"].to_numpy(dtype=float)
    lat = track["latitude"].to_numpy(dtype=float)
    timestamp = track["timestamp"].to_numpy(dtype=float)

    # Calculate time difference between consecutive points
    time_gaps = np.diff(timestamp)

    # Calculate distance between consecutive GPS points
    distances = haversine(
        lon[:-1],
        lat[:-1],
        lon[1:],
        lat[1:]
    )

    # Identify intervals with positive time differences
    valid_intervals = time_gaps > 0

    # Calculate speed in km/h
    speeds = (
        distances[valid_intervals]
        / time_gaps[valid_intervals]
    ) * 3.6

    # Store values for overall plots
    all_time_gaps.extend(time_gaps[valid_intervals])
    all_speeds.extend(speeds)

    # Calculate percentage of one-second GPS intervals
    one_second_count = np.sum(time_gaps == 1)

    one_second_share = (
        one_second_count / len(time_gaps)
    ) * 100

    # Calculate total GPS distance in kilometres
    total_distance_km = distances.sum() / 1000

    # Count extreme speed intervals
    extreme_speed_count = int(np.sum(speeds > 200))

    # Store trip-level statistics
    results.append({

        "trip_id": trip_id,

        "partition": partition,

        "gps_points": len(track),

        "invalid_rows_removed": invalid_rows_removed,

        "total_distance_km": total_distance_km,

        "median_gap_seconds": np.median(time_gaps),

        "max_gap_seconds": np.max(time_gaps),

        "one_second_steps_percent": one_second_share,

        "non_increasing_timestamps": int(
            np.sum(time_gaps <= 0)
        ),

        "median_speed_kmh": (
            np.median(speeds) if len(speeds) else np.nan
        ),

        "p95_speed_kmh": (
            np.percentile(speeds, 95) if len(speeds) else np.nan
        ),

        "max_speed_kmh": (
            np.max(speeds) if len(speeds) else np.nan
        ),

        "extreme_speed_steps_over_200_kmh": extreme_speed_count

    })

    print(
        f"Processed {trip_id}: "
        f"{len(track)} GPS points"
    )


# --------------------------------------------------
# 5. SAVE RESULTS TO CSV
# --------------------------------------------------

quality_df = pd.DataFrame(results)

quality_df = quality_df.sort_values(
    ["partition", "trip_id"]
)

csv_path = OUTPUT_DIR / "gps_quality.csv"

quality_df.to_csv(csv_path, index=False)

print("\nGPS QUALITY SUMMARY")
print("-------------------")

print(f"Trips analysed: {len(quality_df)}")

print(
    "Total invalid rows removed:",
    int(quality_df["invalid_rows_removed"].sum())
)

print(
    "Trips with non-increasing timestamps:",
    int(
        (quality_df["non_increasing_timestamps"] > 0).sum()
    )
)

print(
    "Total extreme speed intervals above 200 km/h:",
    int(quality_df["extreme_speed_steps_over_200_kmh"].sum())
)

print(
    "Largest time gap:",
    quality_df["max_gap_seconds"].max(),
    "seconds"
)

print(
    "Trip with largest time gap:",
    quality_df.loc[
        quality_df["max_gap_seconds"].idxmax(),
        "trip_id"
    ]
)

print(f"\nCSV saved to: {csv_path}")


# --------------------------------------------------
# 6. PLOT GPS TIME GAP DISTRIBUTION
# --------------------------------------------------

all_time_gaps = np.asarray(all_time_gaps)

plt.figure(figsize=(10, 6))

plt.hist(
    all_time_gaps,
    bins=50,
    edgecolor="black"
)

plt.xscale("log")

plt.xlabel("Time gap (seconds, logarithmic scale)")
plt.ylabel("Number of GPS steps")
plt.title("Distribution of GPS Time Gaps")

plt.grid(alpha=0.3)

gap_plot = OUTPUT_DIR / "gps_time_gaps.png"

plt.tight_layout()

plt.savefig(gap_plot, dpi=150)

plt.close()


# --------------------------------------------------
# 7. PLOT GPS SPEED DISTRIBUTION
# --------------------------------------------------

all_speeds = np.asarray(all_speeds)

plt.figure(figsize=(10, 6))

plt.hist(
    all_speeds,
    bins=50,
    edgecolor="black"
)

plt.xscale("log")

plt.xlabel("GPS speed (km/h, logarithmic scale)")
plt.ylabel("Number of GPS steps")
plt.title("Distribution of GPS Speeds")

plt.grid(alpha=0.3)

speed_plot = OUTPUT_DIR / "gps_speed_distribution.png"

plt.tight_layout()

plt.savefig(speed_plot, dpi=150)

plt.close()


# --------------------------------------------------
# 8. FINAL OUTPUT
# --------------------------------------------------

print("\nGENERATED FILES")
print("---------------")

print(f"CSV: {csv_path}")
print(f"Time gap plot: {gap_plot}")
print(f"Speed plot: {speed_plot}")

print("\nGPS QUALITY ANALYSIS COMPLETED SUCCESSFULLY.")
