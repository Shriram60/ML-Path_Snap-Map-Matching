
"""
Task 5: Interactive GPS and reference-route maps.

Usage:
    python eda/trip_map.py --trip 00000033
    python eda/trip_map.py --selected
"""

import argparse
import json
import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd


SELECTED_TRIPS = [
    "00000002",
    "00000033",
    "00000054",
    "00000043",
    "00000096",
]

# Limit how many nearby road segments are drawn on one map.
MAX_ROAD_SEGMENTS = 15000

# Extra map area around the GPS track and reference route.
ROAD_MARGIN_DEGREES = 0.003


def read_table(path):
    """Read a whitespace- or comma-separated dataset."""
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
    )


def find_trip_folder(trip_id):
    """Find a trip in either train or student_test."""
    dataset = Path(os.environ["PATHSNAP_DATA"])

    for partition in ["train", "student_test"]:
        folder = dataset / partition / trip_id

        if folder.exists():
            return folder, partition

    raise FileNotFoundError(
        f"Trip {trip_id} was not found in train or student_test."
    )


def load_trip_data(folder, trip_id):
    """Load GPS points, nodes, arcs and reference route."""

    track = read_table(folder / f"{trip_id}.track")
    nodes = read_table(folder / f"{trip_id}.nodes")
    arcs = read_table(folder / f"{trip_id}.arcs")
    route = read_table(folder / f"{trip_id}.route")

    # Track columns: longitude, latitude, timestamp.
    gps = track.iloc[:, :3].copy()
    gps.columns = ["lon", "lat", "timestamp"]

    # Node columns: longitude, latitude.
    nodes = nodes.iloc[:, :2].copy()
    nodes.columns = ["lon", "lat"]

    # Arc columns: start node ID, end node ID.
    arcs = arcs.iloc[:, :2].copy()
    arcs.columns = ["from_node", "to_node"]

    # Route contains arc IDs.
    route_ids = route.iloc[:, 0].astype(int).to_numpy()

    return gps, nodes, arcs, route_ids


def get_reference_segments(nodes, arcs, route_ids):
    """Convert reference arc IDs into individual map segments."""

    valid_ids = route_ids[
        (route_ids >= 0) & (route_ids < len(arcs))
    ]

    selected_arcs = arcs.iloc[valid_ids]

    from_ids = selected_arcs["from_node"].astype(int).to_numpy()
    to_ids = selected_arcs["to_node"].astype(int).to_numpy()

    valid_nodes = (
        (from_ids >= 0)
        & (to_ids >= 0)
        & (from_ids < len(nodes))
        & (to_ids < len(nodes))
    )

    from_ids = from_ids[valid_nodes]
    to_ids = to_ids[valid_nodes]

    start_coords = nodes.iloc[from_ids][["lon", "lat"]].to_numpy()
    end_coords = nodes.iloc[to_ids][["lon", "lat"]].to_numpy()

    return start_coords, end_coords


def get_nearby_roads(gps, route_start, route_end, nodes, arcs):
    """Select road segments inside a bounding box around the trip."""

    all_lons = np.concatenate([
        gps["lon"].to_numpy(),
        route_start[:, 0],
        route_end[:, 0],
    ])

    all_lats = np.concatenate([
        gps["lat"].to_numpy(),
        route_start[:, 1],
        route_end[:, 1],
    ])

    min_lon = all_lons.min() - ROAD_MARGIN_DEGREES
    max_lon = all_lons.max() + ROAD_MARGIN_DEGREES
    min_lat = all_lats.min() - ROAD_MARGIN_DEGREES
    max_lat = all_lats.max() + ROAD_MARGIN_DEGREES

    from_ids = arcs["from_node"].astype(int).to_numpy()
    to_ids = arcs["to_node"].astype(int).to_numpy()

    valid = (
        (from_ids >= 0)
        & (to_ids >= 0)
        & (from_ids < len(nodes))
        & (to_ids < len(nodes))
    )

    valid_positions = np.flatnonzero(valid)
    from_ids = from_ids[valid]
    to_ids = to_ids[valid]

    from_coords = nodes.iloc[from_ids][["lon", "lat"]].to_numpy()
    to_coords = nodes.iloc[to_ids][["lon", "lat"]].to_numpy()

    # Keep a road if either endpoint is inside the bounding box.
    inside = (
        (
            (from_coords[:, 0] >= min_lon)
            & (from_coords[:, 0] <= max_lon)
            & (from_coords[:, 1] >= min_lat)
            & (from_coords[:, 1] <= max_lat)
        )
        |
        (
            (to_coords[:, 0] >= min_lon)
            & (to_coords[:, 0] <= max_lon)
            & (to_coords[:, 1] >= min_lat)
            & (to_coords[:, 1] <= max_lat)
        )
    )

    from_coords = from_coords[inside]
    to_coords = to_coords[inside]

    if len(from_coords) > MAX_ROAD_SEGMENTS:
        sample_positions = np.linspace(
            0,
            len(from_coords) - 1,
            MAX_ROAD_SEGMENTS,
            dtype=int,
        )
        from_coords = from_coords[sample_positions]
        to_coords = to_coords[sample_positions]

    return from_coords, to_coords


def create_map(trip_id):
    """Create and save one interactive trip map."""

    folder, partition = find_trip_folder(trip_id)

    print(f"\nProcessing trip {trip_id} ({partition})...")

    gps, nodes, arcs, route_ids = load_trip_data(folder, trip_id)

    if gps.empty:
        raise ValueError(f"Trip {trip_id} has no GPS points.")

    # Convert reference route arc IDs into coordinate pairs.
    route_start, route_end = get_reference_segments(
        nodes, arcs, route_ids
    )

    # Select nearby roads only.
    road_start, road_end = get_nearby_roads(
        gps, route_start, route_end, nodes, arcs
    )

    # Folium requires coordinates in latitude, longitude order.
    gps_locations = gps[["lat", "lon"]].values.tolist()

    # Centre the map on the GPS track.
    center = [
        gps["lat"].median(),
        gps["lon"].median(),
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Layer 1: nearby road network.
    roads_layer = folium.FeatureGroup(
        name="Nearby road network",
        show=True,
    )

    road_features = []

    for start, end in zip(road_start, road_end):
        road_features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ],
            },
        })

    if road_features:
        road_geojson = {
            "type": "FeatureCollection",
            "features": road_features,
        }

        folium.GeoJson(
            road_geojson,
            name="Nearby roads",
            style_function=lambda feature: {
                "color": "gray",
                "weight": 1,
                "opacity": 0.55,
            },
        ).add_to(roads_layer)

    roads_layer.add_to(map_object)

    # Layer 2: reference route.
    reference_layer = folium.FeatureGroup(
        name="Reference route",
        show=True,
    )

    for start, end in zip(route_start, route_end):
        folium.PolyLine(
            locations=[
                [float(start[1]), float(start[0])],
                [float(end[1]), float(end[0])],
            ],
            color="red",
            weight=4,
            opacity=0.85,
        ).add_to(reference_layer)

    reference_layer.add_to(map_object)

    # Layer 3: observed GPS track.
    gps_layer = folium.FeatureGroup(
        name="GPS track",
        show=True,
    )

    folium.PolyLine(
        locations=gps_locations,
        color="blue",
        weight=3,
        opacity=0.9,
        tooltip=f"GPS track - Trip {trip_id}",
    ).add_to(gps_layer)

    gps_layer.add_to(map_object)

    # Start marker.
    folium.Marker(
        location=gps_locations[0],
        tooltip="GPS start",
        popup=f"Trip {trip_id} - Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(map_object)

    # End marker.
    folium.Marker(
        location=gps_locations[-1],
        tooltip="GPS end",
        popup=f"Trip {trip_id} - End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(map_object)

    # Add layer selection control.
    folium.LayerControl(collapsed=False).add_to(map_object)

    # Add a map legend.
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border: 1px solid grey;
        border-radius: 5px;
        font-size: 13px;
    ">
        <b>Map Legend</b><br>
        <span style="color:blue;">━━━</span> GPS track<br>
        <span style="color:red;">━━━</span> Reference route<br>
        <span style="color:gray;">━━━</span> Nearby roads<br>
        <span style="color:green;">●</span> Start<br>
        <span style="color:red;">●</span> End
    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # Fit the map to the GPS track and reference route.
    bounds_coords = np.vstack([
        gps[["lat", "lon"]].to_numpy(),
        route_start[:, [1, 0]],
        route_end[:, [1, 0]],
    ])

    map_object.fit_bounds([
        bounds_coords.min(axis=0).tolist(),
        bounds_coords.max(axis=0).tolist(),
    ])

    # Save HTML outside the Git repository.
    work_folder = Path(os.environ["PATHSNAP_WORK"])
    output_folder = work_folder / "eda" / "maps"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"{partition}_{trip_id}.html"

    map_object.save(str(output_file))

    print(f"GPS points: {len(gps):,}")
    print(f"Reference route segments: {len(route_start):,}")
    print(f"Nearby road segments displayed: {len(road_start):,}")
    print(f"Map saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive GPS and route maps."
    )

    parser.add_argument(
        "--trip",
        type=str,
        help="Create a map for one trip ID.",
    )

    parser.add_argument(
        "--selected",
        action="store_true",
        help="Create maps for all five selected trips.",
    )

    args = parser.parse_args()

    if args.trip:
        create_map(args.trip)

    elif args.selected:
        for trip_id in SELECTED_TRIPS:
            create_map(trip_id)

    else:
        parser.error("Use --trip TRIP_ID or --selected.")


if __name__ == "__main__":
    main()
