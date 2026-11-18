"""Read and validate one trip folder."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.geometry import project


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: pd.DataFrame  # obs_idx, lon, lat, t for valid GPS rows
    nodes: np.ndarray  # lon, lat per node
    arcs: np.ndarray  # from_node, to_node per arc
    route: np.ndarray | None  # reference arc IDs in driving order
    origin: tuple  # lon0, lat0 of the local projection
    track_xy: np.ndarray
    node_xy: np.ndarray
    canonical_arc: np.ndarray  # lowest arc ID sharing the same two nodes
    n_raw_points: int

    def without_route(self):
        """Copy used at inference, so reference data cannot reach the features."""
        return replace(self, route=None)


def _read(path, columns, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, dtype=dtype)


def _clean_track(raw):
    finite = np.isfinite(raw[["lon", "lat", "t"]].to_numpy()).all(axis=1)
    in_range = raw["lon"].between(-180, 180) & raw["lat"].between(-90, 90)
    not_null_island = ~((raw["lon"] == 0) & (raw["lat"] == 0))
    track = raw[finite & in_range & not_null_island].copy()
    increasing = track["t"] > track["t"].cummax().shift(fill_value=-np.inf)
    return track[increasing].reset_index(drop=True)


def _canonical_arcs(arcs, n_nodes):
    keys = arcs[:, 0].astype(np.int64) * n_nodes + arcs[:, 1]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return first[inverse]


def load_trip(trip_dir):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    raw = _read(trip_dir / f"{trip_id}.track", ["lon", "lat", "t"], float)
    raw.insert(0, "obs_idx", np.arange(len(raw)))
    track = _clean_track(raw)
    if track.empty:
        raise ValueError(f"{trip_id}: no valid GPS points")

    nodes = _read(trip_dir / f"{trip_id}.nodes", ["lon", "lat"], float).to_numpy()
    arcs = _read(trip_dir / f"{trip_id}.arcs", ["from_node", "to_node"], np.int64).to_numpy()
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"{trip_id}: arc refers to an unknown node")

    route_path = trip_dir / f"{trip_id}.route"
    route = None
    if route_path.exists():
        route = _read(route_path, ["arc_id"], np.int64)["arc_id"].to_numpy()
        if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
            raise ValueError(f"{trip_id}: route refers to an unknown arc")

    lon0 = float(track["lon"].mean())
    lat0 = float(track["lat"].mean())
    return Trip(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        origin=(lon0, lat0),
        track_xy=project(track["lon"], track["lat"], lon0, lat0),
        node_xy=project(nodes[:, 0], nodes[:, 1], lon0, lat0),
        canonical_arc=_canonical_arcs(arcs, len(nodes)),
        n_raw_points=len(raw),
    )
