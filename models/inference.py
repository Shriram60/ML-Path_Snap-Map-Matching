"""Score candidate rows with the saved final model.

    from models.inference import score
    scored = score(candidates)   # adds p_match

Works on tables with no label or ref_arc column. Only the saved f_ columns are used.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACTS = Path(__file__).parent / "artifacts"


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(ARTIFACTS / "final_model.joblib")
    config = json.loads((ARTIFACTS / "model_config.json").read_text())
    return model, config["feature_columns"]


def score(candidates):
    model, features = _load()
    missing = [c for c in features if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidate table is missing feature columns: {missing}")
    out = candidates.copy()
    if len(out):
        out["p_match"] = model.predict_proba(out[features].to_numpy())[:, 1]
    else:
        out["p_match"] = []
    return out
