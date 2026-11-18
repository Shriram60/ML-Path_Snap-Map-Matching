"""Server settings, read once from environment variables.

PATHSNAP_DATA      folder that holds train/ and student_test/ (required)
PATHSNAP_EVAL_DIR  folder of trips the API serves. Defaults to PATHSNAP_DATA/student_test
                   so we can rehearse. Point it at the evaluator folder on demo day.
PATHSNAP_PLACEHOLDER=1  allow the temporary placeholder pipeline (development only)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = REPO_ROOT / "models" / "artifacts" / "final_model.joblib"
MODEL_CONFIG_PATH = REPO_ROOT / "models" / "artifacts" / "model_config.json"
HMM_CONFIG_PATH = REPO_ROOT / "hmm" / "hmm_config.json"


def data_dir() -> Path | None:
    value = os.environ.get("PATHSNAP_DATA")
    return Path(value) if value else None


def eval_dir() -> Path:
    value = os.environ.get("PATHSNAP_EVAL_DIR")
    if value:
        return Path(value)
    base = data_dir()
    if base is None:
        raise RuntimeError("Set PATHSNAP_DATA (or PATHSNAP_EVAL_DIR) before starting the API")
    return base / "student_test"


def placeholder_allowed() -> bool:
    return os.environ.get("PATHSNAP_PLACEHOLDER", "0") == "1"
