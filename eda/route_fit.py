
"""
Task 4: GPS-to-Reference-Route Fit Analysis

For every GPS observation, calculate the distance to the
nearest segment of its reference route.

Distances are calculated in metres using a local
equirectangular projection.

Input: PATHSNAP_DATA
Output: PATHSNAP_WORK/eda/route_fit.csv
        PATHSNAP_WORK/eda/route_fit_analysis.png
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Configure input and output folders
# ---------------------------------------------------------

DATA_DIR = Path(os.environ["PATHSNAP_DATA"])
OUTPUT_DIR = Path(os.environ["PATHSNAP_WORK"]) / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6_371_000
OFF_ROUTE_THRESHOLD_M = 50


# ---------------------------------------------------------
# 2. Calculate GPS-to-segment distances
# ---------------------------------------------------------

def point_to_route_distances(points, segment_starts, segment_ends):
    """
    Find the minimum distance from every GPS point to
    any segment in the reference route.

    All coordinates must already be projected into metres.

    Points:          shape (number of GPS points, 2)
    Segment starts:  shape (number of route segments, 2)
    Segment ends:    shape (number of route segments, 2)
    """

    # Direction vector for each route segment.
    vectors = segment_ends - segment_starts

    # Squared segment lengths.
    segment_lengths_squared = np.sum(vectors ** 2, axis=1)

    # Avoid division by zero for zero-length segments.
    safe_lengths_squared = np.where(
        segment_lengths_squared == 0,
        1,
        segment_lengths_squared
    )

    minimum_distances = np.full(len(points), np.inf)

    # Process GPS points in chunks to control memory usage.
    chunk_size = 256

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))

        point_chunk = points[start:end]

        # Vector from each segment start to each GPS point.
        offsets = (
            point_chunk[:, np.newaxis, :]
            - segment_starts[np.newaxis, :, :]
        )

        # Project each point onto each segment.
        projection = np.sum(
            offsets * vectors[np.newaxis, :, :],
            axis=2
        ) / safe_lengths_squared[np.newaxis, :]

        # Keep projection within the segment.
        projection = np.clip(projection, 0, 1)

        # Find the closest point on each segment.
        closest_points = (
            segment_starts[np.newaxis, :, :]
            + projection[:, :, np.newaxis]
            * vectors[np.newaxis, :, :]
        )

        # Calculate Euclidean distances in metres.
        distances = np.linalg.norm(
            point_chunk[:, np.newaxis, :] - closest_points,
            axis=2
        )

        # Minimum distance to any route segment.
        minimum_distances[start:end] = np.min(
            distances,
            axis=1
        )

    return minimum_distances


# ---------------------------------------------------------
# 3. Analyse one trip
# ---------------------------------------------------------

def analyse_trip(trip_folder, partition):

    trip_id = trip_folder.name

    track_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    # Read GPS observations: longitude, latitude, timestamp.
    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"]
    )

    # Read road node coordinates.
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"]
    )

    # Read directed road arcs.
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    # Read reference route arc IDs.
    route = pd.read_csv(
        route_file,
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )["arc_id"].astype(int).to_numpy()

    # -----------------------------------------------------
    # 4. Validate route and coordinate references
    # -----------------------------------------------------

    if len(track) == 0:
        raise ValueError(f"Empty GPS track in trip {trip_id}")

    if len(route) == 0:
        raise ValueError(f"Empty reference route in trip {trip_id}")

    if not np.isfinite(
        track[["lon", "lat"]].to_numpy()
    ).all():
        raise ValueError(f"Invalid GPS coordinates in {trip_id}")

    if route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f"Invalid route arc ID in {trip_id}")

    arc_pairs = arcs.to_numpy(dtype=int)

    if (
        arc_pairs.min() < 0
        or arc_pairs.max() >= len(nodes)
    ):
        raise ValueError(f"Invalid node reference in {trip_id}")

    # -----------------------------------------------------
    # 5. Build reference route line segments
    # -----------------------------------------------------

    # Select the arcs in reference-route order.
    route_arcs = arc_pairs[route]

    # Each arc has a start node and an end node.
    start_node_ids = route_arcs[:, 0]
    end_node_ids = route_arcs[:, 1]

    route_starts_lonlat = nodes.iloc[
        start_node_ids
    ][["lon", "lat"]].to_numpy()

    route_ends_lonlat = nodes.iloc[
        end_node_ids
    ][["lon", "lat"]].to_numpy()

    gps_lonlat = track[["lon", "lat"]].to_numpy()

    # -----------------------------------------------------
    # 6. Convert longitude/latitude into local metres
    # -----------------------------------------------------

    # Use the mean GPS latitude as the projection reference.
    reference_latitude = np.deg2rad(
        np.mean(track["lat"])
    )

    longitude_scale = (
        EARTH_RADIUS_M * np.cos(reference_latitude)
    )

    latitude_scale = EARTH_RADIUS_M

    def project_coordinates(coordinates):
        longitude = np.deg2rad(coordinates[:, 0])
        latitude = np.deg2rad(coordinates[:, 1])

        x = longitude * longitude_scale
        y = latitude * latitude_scale

        return np.column_stack((x, y))

    gps_points = project_coordinates(gps_lonlat)

    segment_starts = project_coordinates(
        route_starts_lonlat
    )

    segment_ends = project_coordinates(
        route_ends_lonlat
    )

    # -----------------------------------------------------
    # 7. Calculate distance to nearest route segment
    # -----------------------------------------------------

    distances = point_to_route_distances(
        gps_points,
        segment_starts,
        segment_ends
    )

    # -----------------------------------------------------
    # 8. Calculate route fit statistics
    # -----------------------------------------------------

    median_distance = np.median(distances)
    p95_distance = np.percentile(distances, 95)
    maximum_distance = np.max(distances)

    points_over_50m = np.sum(
        distances > OFF_ROUTE_THRESHOLD_M
    )

    share_over_50m = (
        points_over_50m / len(distances)
    )

    first_point_distance = distances[0]
    last_point_distance = distances[-1]

    # -----------------------------------------------------
    # 9. Return trip-level results
    # -----------------------------------------------------

    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_point_count": len(track),
        "route_arc_count": len(route),
        "median_distance_m": median_distance,
        "p95_distance_m": p95_distance,
        "maximum_distance_m": maximum_distance,
        "points_over_50m": points_over_50m,
        "share_over_50m": share_over_50m,
        "first_point_distance_m": first_point_distance,
        "last_point_distance_m": last_point_distance,
        "first_point_over_50m": (
            first_point_distance > OFF_ROUTE_THRESHOLD_M
        ),
        "last_point_over_50m": (
            last_point_distance > OFF_ROUTE_THRESHOLD_M
        ),
    }


# ---------------------------------------------------------
# 10. Run analysis for all trips
# ---------------------------------------------------------

def main():

    results = []

    for partition in ["train", "student_test"]:

        partition_dir = DATA_DIR / partition

        trip_folders = sorted(
            folder
            for folder in partition_dir.iterdir()
            if folder.is_dir()
        )

        for trip_folder in trip_folders:

            result = analyse_trip(
                trip_folder,
                partition
            )

            results.append(result)

            print(
                f"Analysed {partition}/{trip_folder.name}"
            )

    df = pd.DataFrame(results)

    # -----------------------------------------------------
    # 11. Save results to CSV
    # -----------------------------------------------------

    csv_path = OUTPUT_DIR / "route_fit.csv"

    df.to_csv(csv_path, index=False)

    print("\nCSV saved:", csv_path)

    # -----------------------------------------------------
    # 12. Print overall summary
    # -----------------------------------------------------

    print("\n========== ROUTE FIT SUMMARY ==========")

    print("Total trips:", len(df))

    print(
        "Average median distance:",
        round(df["median_distance_m"].mean(), 2),
        "metres"
    )

    print(
        "Average p95 distance:",
        round(df["p95_distance_m"].mean(), 2),
        "metres"
    )

    print(
        "Maximum observed distance:",
        round(df["maximum_distance_m"].max(), 2),
        "metres"
    )

    print(
        "Trips with median distance above 50m:",
        (df["median_distance_m"] > 50).sum()
    )

    print(
        "Trips with GPS points beyond 50m:",
        (df["points_over_50m"] > 0).sum()
    )

    print(
        "Total GPS points beyond 50m:",
        df["points_over_50m"].sum()
    )

    print(
        "Trips with first GPS point beyond 50m:",
        df["first_point_over_50m"].sum()
    )

    print(
        "Trips with last GPS point beyond 50m:",
        df["last_point_over_50m"].sum()
    )

    # -----------------------------------------------------
    # 13. Show trips with the largest median distances
    # -----------------------------------------------------

    print("\nTrips with largest median distances:")

    print(
        df.sort_values(
            "median_distance_m",
            ascending=False
        )[
            [
                "trip_id",
                "partition",
                "median_distance_m",
                "p95_distance_m",
                "maximum_distance_m",
                "share_over_50m"
            ]
        ].head(15).to_string(index=False)
    )

    # -----------------------------------------------------
    # 14. Create plots
    # -----------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        df["median_distance_m"],
        bins=20,
        edgecolor="black"
    )

    axes[0].axvline(
        50,
        linestyle="--",
        label="50 metre threshold"
    )

    axes[0].set_title(
        "Median GPS-to-Route Distance"
    )

    axes[0].set_xlabel("Distance (metres)")
    axes[0].set_ylabel("Number of trips")
    axes[0].legend()

    axes[1].hist(
        df["share_over_50m"] * 100,
        bins=20,
        edgecolor="black"
    )

    axes[1].set_title(
        "GPS Points More Than 50 Metres from Route"
    )

    axes[1].set_xlabel("GPS points beyond 50m (%)")
    axes[1].set_ylabel("Number of trips")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "route_fit_analysis.png"

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPlot saved:", plot_path)


if __name__ == "__main__":
    main()
