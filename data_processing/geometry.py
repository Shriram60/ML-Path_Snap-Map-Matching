"""Planar geometry helpers.

Each trip is projected to metres with an equirectangular projection centred on
that trip. Trips span tens of kilometres at most, so the scale error is well
under 1 percent, far below GPS noise for the short distances used here.
"""
import numpy as np

EARTH_RADIUS_M = 6371008.8


def project(lon, lat, lon0, lat0):
    """Convert degrees to local x, y metres around (lon0, lat0)."""
    dlon = (np.asarray(lon, float) - lon0 + 180.0) % 360.0 - 180.0
    dlat = np.asarray(lat, float) - lat0
    x = EARTH_RADIUS_M * np.radians(dlon) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(dlat)
    return np.column_stack((x, y))


def point_segment(points, start, end):
    """Return distance, position along the segment (0 to 1) and closest point."""
    segment = end - start
    offset = points - start
    length2 = np.sum(segment * segment, axis=-1)
    dot = np.sum(offset * segment, axis=-1)
    along = np.divide(dot, length2, out=np.zeros_like(length2), where=length2 > 0)
    along = np.clip(along, 0.0, 1.0)
    closest = start + along[..., None] * segment
    distance = np.linalg.norm(points - closest, axis=-1)
    return distance, along, closest
