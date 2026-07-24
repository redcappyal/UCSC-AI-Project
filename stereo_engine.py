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


from dataclasses import dataclass


@dataclass(frozen=True)
class TrackSample:
    t_s: float
    px: tuple


@dataclass(frozen=True)
class TrackPoint3D:
    t_s: float
    point_ft: np.ndarray
    gap_ft: float


@dataclass(frozen=True)
class Impact:
    t_s: float
    surface: str
    point_ft: np.ndarray
    call: str
    margin_ft: float
    confidence: str
    snap_disagreement_ft: float | None


FIT_WINDOW_SAMPLES = 7
MIN_FIT_SAMPLES = 4
SNAP_DISAGREEMENT_MAX_FT = 0.3
IMPACT_PROXIMITY_FT = 1.5
_IMPACT_MERGE_S = 0.060
_PRE_IMPACT_WINDOW_S = 0.25
_PRE_IMPACT_GUARD_S = 1.0 / 240.0


_WINDOW_GAP_RATIO_MAX = 3.0


def eval_pixel_track(samples, t_s, window=FIT_WINDOW_SAMPLES):
    """Quadratic least-squares over the `window` samples nearest t_s."""
    if not samples:
        return None
    nearest = sorted(samples, key=lambda s: abs(s.t_s - t_s))[:window]
    if len(nearest) < MIN_FIT_SAMPLES:
        return None
    nearest_ts_sorted = sorted(s.t_s for s in nearest)
    gaps = np.diff(nearest_ts_sorted)
    # A dropped-frame occlusion can still fill the window with enough
    # samples by reaching across the hole, splicing two disjoint time
    # clusters together (e.g. samples from well before the gap and well
    # after it). A quadratic fit across that kind of internal gap models
    # nothing physical, so require the window to actually be local: no
    # internal gap far larger than the window's own typical spacing.
    if gaps.size:
        typical_gap = float(np.median(gaps))
        if typical_gap > 0.0 and float(gaps.max()) > _WINDOW_GAP_RATIO_MAX * typical_gap:
            return None
    ts = np.array([s.t_s for s in nearest]) - t_s   # center for conditioning
    us = np.array([s.px[0] for s in nearest])
    vs = np.array([s.px[1] for s in nearest])
    coeff_u = np.polyfit(ts, us, 2)
    coeff_v = np.polyfit(ts, vs, 2)
    return np.array([np.polyval(coeff_u, 0.0), np.polyval(coeff_v, 0.0)])


def build_track3d(model_a, samples_a, model_b, samples_b, hz=120.0):
    if not samples_a or not samples_b:
        return []
    t_lo = max(samples_a[0].t_s, samples_b[0].t_s)
    t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s)
    track = []
    for t_s in np.arange(t_lo, t_hi, 1.0 / hz):
        px_a = eval_pixel_track(samples_a, t_s)
        px_b = eval_pixel_track(samples_b, t_s)
        if px_a is None or px_b is None:
            continue
        point_ft, gap_ft = triangulate(model_a, model_b, px_a, px_b)
        if point_ft is None:
            continue
        track.append(TrackPoint3D(t_s=float(t_s), point_ft=point_ft, gap_ft=gap_ft))
    return track


def _plane_distance(surface, point_ft):
    plane_point, normal = _SURFACE_PLANES[surface]
    return float(np.dot(point_ft - plane_point, normal))


def _pre_impact_eval(samples, t_impact):
    window = [s for s in samples
              if t_impact - _PRE_IMPACT_WINDOW_S <= s.t_s <= t_impact - _PRE_IMPACT_GUARD_S]
    if len(window) < MIN_FIT_SAMPLES:
        return None
    return eval_pixel_track(window, t_impact, window=len(window))


def detect_impacts(model_a, samples_a, model_b, samples_b, track=None):
    if track is None:
        track = build_track3d(model_a, samples_a, model_b, samples_b)
    if len(track) < 3:
        return []
    impacts = []
    for surface in SURFACES:
        dists = np.array([_plane_distance(surface, p.point_ft) for p in track])
        for i in range(1, len(track) - 1):
            if dists[i] > IMPACT_PROXIMITY_FT:
                continue
            if not (dists[i] <= dists[i - 1] and dists[i] <= dists[i + 1]):
                continue
            approaching = dists[i] - dists[i - 1]
            leaving = dists[i + 1] - dists[i]
            if not (approaching < 0.0 and leaving > 0.0):
                continue
            t_impact = track[i].t_s
            snap_a = snap_b = None
            px_a = _pre_impact_eval(samples_a, t_impact)
            px_b = _pre_impact_eval(samples_b, t_impact)
            if px_a is not None:
                snap_a = snap_to_plane(model_a, px_a, surface)
            if px_b is not None:
                snap_b = snap_to_plane(model_b, px_b, surface)
            disagreement = None
            if snap_a is not None and snap_b is not None:
                disagreement = float(np.linalg.norm(snap_a - snap_b))
                confidence = "high" if disagreement <= SNAP_DISAGREEMENT_MAX_FT else "one_view"
                point = fuse_snaps(snap_a, snap_b)
            elif snap_a is not None or snap_b is not None:
                confidence = "one_view"
                point = fuse_snaps(snap_a, snap_b)
            else:
                confidence = "no_call"
                point = track[i].point_ft
            call, margin = call_for_impact(surface, point)
            impacts.append(Impact(
                t_s=t_impact, surface=surface, point_ft=point, call=call,
                margin_ft=margin, confidence=confidence,
                snap_disagreement_ft=disagreement))
    impacts.sort(key=lambda imp: imp.t_s)
    merged = []
    for imp in impacts:
        if merged and merged[-1].surface == imp.surface and \
                imp.t_s - merged[-1].t_s < _IMPACT_MERGE_S:
            prev_d = abs(_plane_distance(imp.surface, merged[-1].point_ft))
            cur_d = abs(_plane_distance(imp.surface, imp.point_ft))
            if cur_d < prev_d:
                merged[-1] = imp
            continue
        merged.append(imp)
    return merged
