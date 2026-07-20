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

from court_model import G_FT_PER_S2

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
        solution, *_ = np.linalg.lstsq(system, np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:
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
