"""Stereo math for the two-camera system.

Builds on court_model.CameraModel (pinhole in court FEET; origin front-left
floor seam, x right 0-21, y front 0 -> back 32, z up). Public functions take
RAW pixels and undistort internally via court_model.undistort_point; the
CameraModel primitives themselves stay in undistorted pixel space.
"""
import numpy as np

import court_model
from court_model import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    OUT_LINE_HEIGHT_FT,
    TIN_TOP_HEIGHT_FT,
)

PARALLEL_EPS = 1e-9


def _undistorted_ray(model, pixel):
    pixel = court_model.undistort_point(pixel, model.distortion)
    return model.ray(pixel)


def triangulate(model_a, model_b, px_a, px_b):
    """Closest-approach midpoint of the two viewing rays.

    Returns (point_ft, gap_ft); (None, inf) for near-parallel rays or when
    the closest approach lies behind either camera (s or t <= 0).
    """
    o1, d1 = _undistorted_ray(model_a, px_a)
    o2, d2 = _undistorted_ray(model_b, px_b)
    w0 = o1 - o2
    b = float(np.dot(d1, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = 1.0 - b * b          # a = c = 1 for unit directions
    if denom < PARALLEL_EPS:
        return None, np.inf
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    if s <= 0.0 or t <= 0.0:
        return None, np.inf
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2))
