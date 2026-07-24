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


BACK_WALL_OUT_HEIGHT_FT = 7.0

_SURFACE_PLANES = {
    "floor": (np.zeros(3), np.array([0.0, 0.0, 1.0])),
    "front_wall": (np.zeros(3), np.array([0.0, 1.0, 0.0])),
    "back_wall": (np.array([0.0, COURT_LENGTH_FT, 0.0]), np.array([0.0, -1.0, 0.0])),
    "left_wall": (np.zeros(3), np.array([1.0, 0.0, 0.0])),
    "right_wall": (np.array([COURT_WIDTH_FT, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
}
SURFACES = set(_SURFACE_PLANES)
_BOUNDS_SLACK_FT = 0.5


def surface_plane(name):
    point, normal = _SURFACE_PLANES[name]
    return point.copy(), normal.copy()


def side_wall_out_height_ft(y_ft):
    """WSF side-wall out line: 15 ft at the front wall, 7 ft at the back."""
    return OUT_LINE_HEIGHT_FT + (
        BACK_WALL_OUT_HEIGHT_FT - OUT_LINE_HEIGHT_FT) * (y_ft / COURT_LENGTH_FT)


def call_for_impact(surface, point_ft):
    """(call, margin_ft) for an impact point known to lie on `surface`.

    margin_ft is the distance to the deciding line — how clear the call is.
    """
    x, y, z = (float(v) for v in point_ft)
    if surface == "floor":
        return "bounce", 0.0
    if surface == "front_wall":
        if z >= OUT_LINE_HEIGHT_FT:
            return "out", z - OUT_LINE_HEIGHT_FT
        if z <= TIN_TOP_HEIGHT_FT:
            return "down", TIN_TOP_HEIGHT_FT - z
        return "in", min(OUT_LINE_HEIGHT_FT - z, z - TIN_TOP_HEIGHT_FT)
    if surface in ("left_wall", "right_wall"):
        line = side_wall_out_height_ft(y)
        return ("out", z - line) if z >= line else ("in", line - z)
    if surface == "back_wall":
        line = BACK_WALL_OUT_HEIGHT_FT
        return ("out", z - line) if z >= line else ("in", line - z)
    raise ValueError(f"Unknown surface: {surface}")


def _in_surface_bounds(surface, point_ft):
    x, y, z = (float(v) for v in point_ft)
    lo = -_BOUNDS_SLACK_FT
    if surface == "floor":
        return lo <= x <= COURT_WIDTH_FT + _BOUNDS_SLACK_FT and lo <= y <= COURT_LENGTH_FT + _BOUNDS_SLACK_FT
    if surface in ("front_wall", "back_wall"):
        return lo <= x <= COURT_WIDTH_FT + _BOUNDS_SLACK_FT and lo <= z
    return lo <= y <= COURT_LENGTH_FT + _BOUNDS_SLACK_FT and lo <= z


def snap_to_plane(model, px, surface):
    """Intersect the (undistorted) viewing ray with a court surface plane."""
    plane_point, normal = _SURFACE_PLANES[surface]
    origin, direction = _undistorted_ray(model, px)
    denom = float(np.dot(direction, normal))
    if abs(denom) < PARALLEL_EPS:
        return None
    t = float(np.dot(plane_point - origin, normal)) / denom
    if t <= 0.0:
        return None
    point = origin + t * direction
    return point if _in_surface_bounds(surface, point) else None


def fuse_snaps(p_a, p_b):
    if p_a is None:
        return p_b
    if p_b is None:
        return p_a
    return (p_a + p_b) / 2.0
