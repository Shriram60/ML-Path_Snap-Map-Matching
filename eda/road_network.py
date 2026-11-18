
"""
Task 3: Road Network Analysis

Analyses the directed road network and reference routes for all trips.
Input data is read from PATHSNAP_DATA.
Generated CSV and plot are saved under PATHSNAP_WORK/eda.
"""

from pathlib import Path
from collections import Counter
import os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Set input and output folders
# ---------------------------------------------------------

DATA_DIR = Path(os.environ["PATHSNAP_DATA"])
OUTPUT_DIR = Path(os.environ["PATHSNAP_WORK"]) / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Read one trip's road network
# ---------------------------------------------------------

def analyse_trip(trip_folder, partition):
    trip_id = trip_folder.name

    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    # Each row in nodes represents one node.
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"]
    )

    # Each row in arcs represents a directed connection.
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    # Route contains arc IDs, one per row.
    route = pd.read_csv(
        route_file,
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )["arc_id"].astype(int).tolist()

    node_count = len(nodes)
    arc_count = len(arcs)

    # -----------------------------------------------------
    # 3. Validate node and arc references
    # -----------------------------------------------------

    if not arcs["from_node"].between(0, node_count - 1).all():
        raise ValueError(f"Invalid starting node in trip {trip_id}")

    if not arcs["to_node"].between(0, node_count - 1).all():
        raise ValueError(f"Invalid ending node in trip {trip_id}")

    if any(arc_id < 0 or arc_id >= arc_count for arc_id in route):
        raise ValueError(f"Invalid route arc ID in trip {trip_id}")

    # -----------------------------------------------------
    # 4. Build directed graph
    # -----------------------------------------------------

    graph = nx.DiGraph()

    # Add every node, including nodes with no connections.
    graph.add_nodes_from(range(node_count))

    # Convert arc rows into directed edges.
    arc_pairs = list(
        zip(arcs["from_node"], arcs["to_node"])
    )

    graph.add_edges_from(arc_pairs)

    # -----------------------------------------------------
    # 5. Basic network statistics
    # -----------------------------------------------------

    unique_arc_pairs = set(arc_pairs)

    duplicate_arc_count = arc_count - len(unique_arc_pairs)

    self_loop_count = sum(
        1 for start, end in arc_pairs if start == end
    )

    # Count unique directed arcs that have a reverse connection.
    reverse_arc_count = sum(
        1
        for start, end in unique_arc_pairs
        if start != end and (end, start) in unique_arc_pairs
    )

    reverse_arc_share = (
        reverse_arc_count / len(unique_arc_pairs)
        if unique_arc_pairs else 0
    )

    # -----------------------------------------------------
    # 6. Connected components
    # -----------------------------------------------------

    components = list(nx.weakly_connected_components(graph))

    component_count = len(components)

    largest_component_size = (
        max(map(len, components)) if components else 0
    )

    largest_component_share = (
        largest_component_size / node_count
        if node_count else 0
    )

    isolated_nodes = list(nx.isolates(graph))

    # -----------------------------------------------------
    # 7. Degree statistics
    # -----------------------------------------------------

    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    average_in_degree = (
        np.mean(list(in_degrees.values())) if node_count else 0
    )

    average_out_degree = (
        np.mean(list(out_degrees.values())) if node_count else 0
    )

    # -----------------------------------------------------
    # 8. Reference route connectivity
    # -----------------------------------------------------

    connected_steps = 0

    for i in range(len(route) - 1):
        current_arc = route[i]
        next_arc = route[i + 1]

        current_end = arc_pairs[current_arc][1]
        next_start = arc_pairs[next_arc][0]

        if current_end == next_start:
            connected_steps += 1

    total_route_steps = max(len(route) - 1, 0)

    route_connected_share = (
        connected_steps / total_route_steps
        if total_route_steps else 0
    )

    disconnected_steps = total_route_steps - connected_steps

    route_repeated_arcs = (
        len(route) - len(set(route))
    )

    # -----------------------------------------------------
    # 9. Return all results for this trip
    # -----------------------------------------------------

    return {
        "partition": partition,
        "trip_id": trip_id,
        "node_count": node_count,
        "arc_count": arc_count,
        "component_count": component_count,
        "largest_component_size": largest_component_size,
        "largest_component_share": largest_component_share,
        "isolated_node_count": len(isolated_nodes),
        "self_loop_count": self_loop_count,
        "duplicate_arc_count": duplicate_arc_count,
        "reverse_arc_share": reverse_arc_share,
        "average_in_degree": average_in_degree,
        "average_out_degree": average_out_degree,
        "route_arc_count": len(route),
        "route_connected_steps": connected_steps,
        "route_disconnected_steps": disconnected_steps,
        "route_connected_share": route_connected_share,
        "route_repeated_arc_count": route_repeated_arcs,
    }


# ---------------------------------------------------------
# 10. Analyse all training and student-test trips
# ---------------------------------------------------------

def main():
    results = []

    for partition in ["train", "student_test"]:
        partition_dir = DATA_DIR / partition

        trip_folders = sorted(
            folder for folder in partition_dir.iterdir()
            if folder.is_dir()
        )

        for trip_folder in trip_folders:
            result = analyse_trip(trip_folder, partition)
            results.append(result)

            print(
                f"Analysed {partition}/{trip_folder.name}"
            )

    df = pd.DataFrame(results)

    # -----------------------------------------------------
    # 11. Save CSV
    # -----------------------------------------------------

    csv_path = OUTPUT_DIR / "road_network.csv"
    df.to_csv(csv_path, index=False)

    print("\nCSV saved:", csv_path)

    # -----------------------------------------------------
    # 12. Print summary statistics
    # -----------------------------------------------------

    print("\n========== ROAD NETWORK SUMMARY ==========")

    print("Total trips:", len(df))
    print("Total nodes:", df["node_count"].sum())
    print("Total arcs:", df["arc_count"].sum())

    print(
        "Trips with disconnected route steps:",
        (df["route_disconnected_steps"] > 0).sum()
    )

    print(
        "Total disconnected route steps:",
        df["route_disconnected_steps"].sum()
    )

    print(
        "Average largest component share:",
        round(df["largest_component_share"].mean(), 4)
    )

    print(
        "Total isolated nodes:",
        df["isolated_node_count"].sum()
    )

    print(
        "Total duplicate arcs:",
        df["duplicate_arc_count"].sum()
    )

    print(
        "Total self-loops:",
        df["self_loop_count"].sum()
    )

    print(
        "Average route connected share:",
        round(df["route_connected_share"].mean(), 4)
    )

    # -----------------------------------------------------
    # 13. Create plots
    # -----------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(
        df["largest_component_share"],
        bins=20,
        edgecolor="black"
    )
    axes[0].set_title("Largest Connected Component Share")
    axes[0].set_xlabel("Share of nodes in largest component")
    axes[0].set_ylabel("Number of trips")

    axes[1].hist(
        df["route_connected_share"],
        bins=20,
        edgecolor="black"
    )
    axes[1].set_title("Reference Route Connectivity")
    axes[1].set_xlabel("Connected-step share")
    axes[1].set_ylabel("Number of trips")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "road_network_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nPlot saved:", plot_path)

    # -----------------------------------------------------
    # 14. Show trips with disconnected route steps
    # -----------------------------------------------------

    disconnected = df[
        df["route_disconnected_steps"] > 0
    ][
        [
            "trip_id",
            "partition",
            "route_disconnected_steps",
            "route_connected_share"
        ]
    ]

    print("\nTrips with disconnected route steps:")
    print(disconnected.to_string(index=False))


if __name__ == "__main__":
    main()
