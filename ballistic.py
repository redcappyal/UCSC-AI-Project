"""Gravity-constrained 3D trajectory fitting for ball tracks.

A ball in free flight follows X(t) = X0 + V0*tau + 0.5*g*tau^2 in court
coordinates (feet, z up). With a calibrated camera each detection pixel
contributes two linear constraints on X(t) (the cross-product form of the
projection equation), and X(t) is linear in (X0, V0) - so fitting an arc to
any frame range is one linear least-squares solve in 6 unknowns. Contacts
are where consecutive arcs meet (segmentation lives here too; see
segment_track / arc_boundary_events).
"""

from dataclasses import dataclass

import numpy as np

from court_model import G_FT_PER_S2, undistort_point

GRAVITY_VEC = np.array([0.0, 0.0, -G_FT_PER_S2])


@dataclass(frozen=True)
class BallisticArc:
    t_ref: float
    x0: np.ndarray
    v0: np.ndarray
    rms_px: float
    start: int
    end: int

    def position(self, t):
        tau = t - self.t_ref
        return self.x0 + self.v0 * tau + 0.5 * GRAVITY_VEC * tau * tau

    def velocity(self, t):
        return self.v0 + GRAVITY_VEC * (t - self.t_ref)


def fit_arc(times, pixels_und, camera, start=0, end=None):
    """Fit one ballistic arc to samples [start:end); None if degenerate.

    times: (N,) seconds. pixels_und: (N,2) undistorted pixels. The rows are
    normalized so the algebraic residual approximates pixel error; rms_px is
    then computed exactly by reprojection.
    """
    end = len(times) if end is None else end
    if end - start < 3:
        return None
    projection = camera.projection_matrix()
    t_ref = float(times[start])
    rows, rhs = [], []
    for index in range(start, end):
        tau = float(times[index]) - t_ref
        drop = 0.5 * GRAVITY_VEC * tau * tau
        for matrix_row, pixel_coord in (
            (projection[0], float(pixels_und[index][0])),
            (projection[1], float(pixels_und[index][1])),
        ):
            constraint = pixel_coord * projection[2] - matrix_row  # 4-vector
            spatial = constraint[:3]
            norm = np.linalg.norm(spatial)
            if norm < 1e-12:
                continue
            rows.append(np.concatenate([spatial, tau * spatial]) / norm)
            rhs.append(-(spatial @ drop + constraint[3]) / norm)
    if len(rows) < 6:
        return None
    system = np.asarray(rows)
    try:
        solution, _, rank, _ = np.linalg.lstsq(system, np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 6:
        return None
    x0, v0 = solution[:3], solution[3:]

    errors = []
    for index in range(start, end):
        tau = float(times[index]) - t_ref
        point = x0 + v0 * tau + 0.5 * GRAVITY_VEC * tau * tau
        try:
            u, v = camera.project(point)
        except ValueError:
            return None
        du = u - float(pixels_und[index][0])
        dv = v - float(pixels_und[index][1])
        errors.append(du * du + dv * dv)
    return BallisticArc(
        t_ref=t_ref, x0=x0, v0=v0,
        rms_px=float(np.sqrt(np.mean(errors))),
        start=int(start), end=int(end),
    )


def segment_track(times, pixels_und, camera, rms_px, min_points):
    """Greedy maximal ballistic arcs, mirroring event_engine.segment_into_arcs:
    grow each arc until adding the next sample pushes reprojection rms past
    rms_px. Unfittable ranges come back as (start, end) tuples."""
    segments = []
    start, count = 0, len(times)
    while start < count:
        end = min(start + min_points, count)
        best = fit_arc(times, pixels_und, camera, start, end)
        if best is None:
            segments.append((start, end))
            start = end
            continue
        while end < count:
            # Aggregate rms dilutes a single contaminated sample as the arc
            # grows (a boundary sample from the NEXT flight segment can hide
            # inside a long arc's average) — so the newest sample must also
            # pass the gate individually, or the arc overruns the contact.
            # Checked against `best` (the fit *before* this sample is folded
            # in): checking against the refit `grown` arc instead lets the
            # least-squares solve redistribute the contaminated sample's
            # error across every point, diluting its own residual below
            # rms_px too and defeating the gate.
            try:
                u, v = camera.project(best.position(float(times[end])))
            except ValueError:
                break
            du = u - float(pixels_und[end][0])
            dv = v - float(pixels_und[end][1])
            if (du * du + dv * dv) ** 0.5 > rms_px:
                break
            grown = fit_arc(times, pixels_und, camera, start, end + 1)
            if grown is None or grown.rms_px > rms_px:
                break
            best, end = grown, end + 1
        segments.append(best)
        start = end
    return segments


def refine_impact(arc_a, arc_b, t_lo, t_hi):
    """Closest approach of two arcs. Both share the gravity term, so their
    difference is linear in t and the minimizer is closed-form."""
    t_mid = 0.5 * (t_lo + t_hi)
    offset = arc_a.position(t_mid) - arc_b.position(t_mid)
    relative = arc_a.velocity(t_mid) - arc_b.velocity(t_mid)
    denom = float(relative @ relative)
    t_star = t_mid if denom < 1e-9 else t_mid - float(offset @ relative) / denom
    t_star = min(max(t_star, t_lo), t_hi)
    point = 0.5 * (arc_a.position(t_star) + arc_b.position(t_star))
    return t_star, point, arc_a.velocity(t_star), arc_b.velocity(t_star)


def _projected_velocity(camera, arc, t, dt=1.0 / 120.0):
    """Image-plane velocity (px/s) of the arc at time t, for 2D-compatible
    event fields."""
    u1, v1 = camera.project(arc.position(t - dt))
    u2, v2 = camera.project(arc.position(t + dt))
    return np.array([(u2 - u1) / (2 * dt), (v2 - v1) / (2 * dt)])


def arc_boundary_events(frames, timestamps, positions, tracks, camera, cfg):
    """Contact events at every boundary between adjacent fitted 3D arcs.

    Event dicts match event_engine._make_event plus a "contact_3d" payload.
    positions are raw pixels; undistortion happens here.
    """
    from event_engine import _make_event  # shared event shape, no cycle at import time

    rms_px = cfg["arc3d_rms_px"]
    min_points = cfg["arc_min_points"]
    events = []
    for track_start, track_end in tracks:
        if track_end - track_start < 2 * min_points:
            continue
        times = timestamps[track_start:track_end]
        pixels = np.asarray(
            [undistort_point(p, camera.distortion)
             for p in positions[track_start:track_end]]
        )
        segments = segment_track(times, pixels, camera, rms_px, min_points)
        for k in range(1, len(segments)):
            previous, current = segments[k - 1], segments[k]
            if not isinstance(previous, BallisticArc) or not isinstance(
                current, BallisticArc
            ):
                continue  # unfittable gap: derivative/audio methods still cover it
            if (previous.end - previous.start < min_points
                    or current.end - current.start < min_points):
                continue
            t_lo = float(times[previous.end - 1])
            t_hi = float(times[current.start])
            t_star, point, v_in_3d, v_out_3d = refine_impact(
                previous, current, t_lo, t_hi)
            try:
                v_in_px = _projected_velocity(camera, previous, t_lo)
                v_out_px = _projected_velocity(camera, current, t_hi)
            except ValueError:
                continue
            event = _make_event(
                track_start + current.start, frames, timestamps, positions,
                v_in_px, v_out_px, "ballistic",
            )
            event["contact_3d"] = {
                "time": float(t_star),
                "point_ft": [float(c) for c in point],
                "v_in_ft_s": [float(c) for c in v_in_3d],
                "v_out_ft_s": [float(c) for c in v_out_3d],
                "arc_rms_px": [float(previous.rms_px), float(current.rms_px)],
            }
            events.append(event)
    return events
