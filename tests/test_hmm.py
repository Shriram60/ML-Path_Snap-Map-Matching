"""Checks for Viterbi decoding on a small synthetic trip."""
import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from features.candidates import build_candidates
from features.labels import add_labels
from hmm.viterbi import DecoderConfig, classifier_only, decode
from test_data_pipeline import eastbound, write_trip


def scored_trip(tmp_path, wrong_obs=()):
    trip = load_trip(write_trip(tmp_path, eastbound()))
    table = add_labels(build_candidates(trip), trip)
    table["p_match"] = np.where(table["label"] == 1, 0.8, 0.1)
    wrong = table["obs_idx"].isin(wrong_obs)
    table.loc[wrong, "p_match"] = np.where(table.loc[wrong, "label"] == 1, 0.3, 0.1)
    reverse = wrong & (table["arc_id"] == 4)
    table.loc[reverse, "p_match"] = 0.6
    return trip, table


def test_decoder_fixes_isolated_wrong_prediction(tmp_path):
    trip, table = scored_trip(tmp_path, wrong_obs=[25])
    independent = classifier_only(table, trip).set_index("obs_idx")["arc_id"]
    decoded = decode(table, trip, DecoderConfig()).set_index("obs_idx")["arc_id"]
    assert independent[25] == 4
    assert decoded[25] == 1
    assert decoded[0] == 0 and decoded[59] == 2


def test_points_without_candidates_return_minus_one(tmp_path):
    trip, table = scored_trip(tmp_path)
    decoded = decode(table[table["obs_idx"] != 30], trip, DecoderConfig())
    assert decoded.loc[decoded["obs_idx"] == 30, "arc_id"].item() == -1
    assert len(decoded) == len(trip.track)


def test_empty_scores_are_handled(tmp_path):
    trip, table = scored_trip(tmp_path)
    decoded = decode(pd.DataFrame(columns=table.columns), trip, DecoderConfig())
    assert (decoded["arc_id"] == -1).all()
