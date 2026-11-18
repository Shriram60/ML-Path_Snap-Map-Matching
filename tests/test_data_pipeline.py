"""Checks for loading, labelling and candidates on a small synthetic trip."""
import numpy as np
import pandas as pd

from data_processing.geometry import point_segment, project
from data_processing.loader import load_trip
from features.candidates import FEATURE_COLUMNS, build_candidates
from features.labels import add_labels, reference_table

LAT = 50.0
METRE_LON = 1 / (111195 * np.cos(np.radians(LAT)))
METRE_LAT = 1 / 111195


def write_trip(folder, track_rows):
    """East-west road of three 200 m arcs, drivable in both directions."""
    trip_dir = folder / "00000001"
    trip_dir.mkdir()
    nodes = [(10 + i * 200 * METRE_LON, LAT) for i in range(4)]
    arcs = [(0, 1), (1, 2), (2, 3), (1, 0), (2, 1), (3, 2), (0, 1)]
    pd.DataFrame(track_rows).to_csv(trip_dir / "00000001.track", sep=" ", header=False, index=False)
    pd.DataFrame(nodes).to_csv(trip_dir / "00000001.nodes", sep=" ", header=False, index=False)
    pd.DataFrame(arcs).to_csv(trip_dir / "00000001.arcs", sep=" ", header=False, index=False)
    pd.DataFrame([0, 1, 2]).to_csv(trip_dir / "00000001.route", sep=" ", header=False, index=False)
    return trip_dir


def eastbound(count=60, step=10.0, north=3.0):
    return [
        (10 + (5 + i * step) * METRE_LON, LAT + north * METRE_LAT, float(i + 1))
        for i in range(count)
    ]


def test_projection_and_segment_distance():
    xy = project([10.0, 10.001], [50.0, 50.0], 10.0, 50.0)
    assert abs(xy[1, 0] - 71.47) < 0.1
    distance, along, _ = point_segment(
        np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert distance == 3.0 and along == 0.5


def test_loader_drops_bad_points_and_keeps_original_index(tmp_path):
    rows = eastbound(5)
    rows[1] = (float("nan"), LAT, 2.0)
    rows[3] = (rows[3][0], rows[3][1], 1.0)
    trip = load_trip(write_trip(tmp_path, rows))
    assert trip.track["obs_idx"].tolist() == [0, 2, 4]
    assert trip.n_raw_points == 5
    assert trip.canonical_arc[6] == 0


def test_labels_follow_route_in_order(tmp_path):
    trip = load_trip(write_trip(tmp_path, eastbound()))
    table = reference_table(trip)
    assert np.all(np.diff(table["route_pos"]) >= 0)
    assert table["ref_arc"].iloc[0] == 0
    assert table["ref_arc"].iloc[25] == 1
    assert table["ref_arc"].iloc[45] == 2


def test_points_far_from_route_have_no_reference(tmp_path):
    rows = eastbound(50)
    rows += [
        (rows[-1][0], LAT + (100 + i * 10) * METRE_LAT, 51.0 + i) for i in range(10)
    ]
    trip = load_trip(write_trip(tmp_path, rows))
    assert (reference_table(trip)["ref_arc"].to_numpy()[-10:] == -1).all()


def test_candidates_ignore_route_and_heading_picks_direction(tmp_path):
    trip = load_trip(write_trip(tmp_path, eastbound()))
    with_route = build_candidates(trip)
    without_route = build_candidates(trip.without_route())
    pd.testing.assert_frame_equal(with_route, without_route)
    assert not np.isin(6, with_route["arc_id"])
    assert not with_route[FEATURE_COLUMNS].isna().any().any()

    labelled = add_labels(with_route, trip)
    middle = labelled[labelled["obs_idx"] == 25].set_index("arc_id")
    assert middle.loc[1, "label"] == 1 and middle.loc[4, "label"] == 0
    assert middle.loc[1, "f_heading_cos"] > 0.9 and middle.loc[4, "f_heading_cos"] < -0.9
