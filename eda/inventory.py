
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Dataset and output locations
# --------------------------------------------------

DATA_DIR = Path(os.environ["PATHSNAP_DATA"])
WORK_DIR = Path(os.environ["PATHSNAP_WORK"]) / "eda"

WORK_DIR.mkdir(parents=True, exist_ok=True)

PARTITIONS = ["train", "student_test"]


# --------------------------------------------------
# 2. Function to read trip files
# --------------------------------------------------

def read_data_file(file_path, columns):
    """Read a whitespace-separated file without a header."""

    if not file_path.is_file():
        return None

    try:
        return pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=columns
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


# --------------------------------------------------
# 3. Analyse an individual trip
# --------------------------------------------------

def analyse_trip(partition, trip_dir):

    trip_id = trip_dir.name

    # Construct the four expected file paths
    files = {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "route": trip_dir / f"{trip_id}.route"
    }

    # Read each file using its correct column names
    track = read_data_file(
        files["track"],
        ["longitude", "latitude", "timestamp"]
    )

    nodes = read_data_file(
        files["nodes"],
        ["longitude", "latitude"]
    )

    arcs = read_data_file(
        files["arcs"],
        ["from_node", "to_node"]
    )

    route = read_data_file(
        files["route"],
        ["arc_id"]
    )

    data = {
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }

    # Start building the inventory row
    row = {
        "partition": partition,
        "trip_id": trip_id
    }

    # Record file existence and file sizes
    for name, file_path in files.items():

        row[f"{name}_exists"] = file_path.is_file()

        row[f"{name}_size_bytes"] = (
            file_path.stat().st_size
            if file_path.is_file()
            else 0
        )

    # Count records in each file
    for name, df in data.items():

        row[f"{name}_count"] = (
            len(df) if df is not None else None
        )

    # Count missing values
    missing_count = 0

    for df in data.values():

        if df is not None:
            missing_count += int(df.isna().sum().sum())

    row["missing_values"] = missing_count

    # Calculate GPS coordinate boundaries
    if track is not None and not track.empty:

        row["min_longitude"] = track["longitude"].min()
        row["max_longitude"] = track["longitude"].max()

        row["min_latitude"] = track["latitude"].min()
        row["max_latitude"] = track["latitude"].max()

    else:

        row["min_longitude"] = None
        row["max_longitude"] = None
        row["min_latitude"] = None
        row["max_latitude"] = None

    return row


# --------------------------------------------------
# 4. Analyse all trips in a partition
# --------------------------------------------------

def analyse_partition(partition):

    partition_dir = DATA_DIR / partition

    if not partition_dir.is_dir():
        raise FileNotFoundError(
            f"Partition not found: {partition_dir}"
        )

    trip_dirs = sorted(
        path for path in partition_dir.iterdir()
        if path.is_dir()
    )

    results = []

    for trip_dir in trip_dirs:

        print(
            f"Analysing {partition}: {trip_dir.name}"
        )

        result = analyse_trip(partition, trip_dir)

        results.append(result)

    return results


# --------------------------------------------------
# 5. Create GPS points histogram
# --------------------------------------------------

def create_gps_histogram(inventory):

    plt.figure(figsize=(10, 6))

    for partition in PARTITIONS:

        values = inventory.loc[
            inventory["partition"] == partition,
            "track_count"
        ].dropna()

        plt.hist(
            values,
            bins=15,
            alpha=0.6,
            label=partition
        )

    plt.title(
        "GPS Points per Trip - Train and Student Test"
    )

    plt.xlabel("Number of GPS points")
    plt.ylabel("Number of trips")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = WORK_DIR / "gps_points_histogram.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"\nHistogram saved: {output_file}")


# --------------------------------------------------
# 6. Main program
# --------------------------------------------------

def main():

    all_results = []

    for partition in PARTITIONS:

        results = analyse_partition(partition)

        all_results.extend(results)

    # Combine all trips into one DataFrame
    inventory = pd.DataFrame(all_results)

    # Save inventory CSV
    csv_file = WORK_DIR / "trip_inventory.csv"

    inventory.to_csv(csv_file, index=False)

    print(f"\nInventory saved: {csv_file}")

    # Generate histogram
    create_gps_histogram(inventory)

    # Print summary
    print("\n========== INVENTORY SUMMARY ==========")

    print("\nTrips per partition:")

    print(
        inventory.groupby("partition").size()
    )

    print("\nGPS point range:")

    print(
        inventory["track_count"].min(),
        "to",
        inventory["track_count"].max()
    )

    print("\nTotal missing values:")

    print(
        inventory["missing_values"].sum()
    )

    # Check missing files
    file_columns = [
        "track_exists",
        "nodes_exists",
        "arcs_exists",
        "route_exists"
    ]

    incomplete = inventory[
        ~inventory[file_columns].all(axis=1)
    ]

    print("\nTrips with missing files:", len(incomplete))

    if not incomplete.empty:

        print(
            incomplete[
                ["partition", "trip_id"] + file_columns
            ].to_string(index=False)
        )

    print("\n=======================================")


if __name__ == "__main__":
    main()
