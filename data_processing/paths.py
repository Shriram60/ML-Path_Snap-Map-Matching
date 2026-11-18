"""Locations of the dataset and of regenerated files."""
import os
from pathlib import Path

PARTITIONS = ("train", "student_test")


def _env_dir(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable")
    return Path(value)


def data_root():
    return _env_dir("PATHSNAP_DATA")


def work_root():
    path = _env_dir("PATHSNAP_WORK")
    path.mkdir(parents=True, exist_ok=True)
    return path


def trip_dirs(partition, root=None):
    base = Path(root) if root else data_root()
    folder = base / partition
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.is_dir())
