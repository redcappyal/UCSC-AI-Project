"""Metric squash-court model and floor-plane homography.

Court coordinate convention (the contract shared with the browser wizard):
origin is the front-wall/floor seam at the LEFT corner as seen from the
back-wall camera; x runs rightward 0 -> 21 ft across the court; y runs from
the front wall toward the back wall 0 -> 32 ft. Units are feet.

The floor homography maps image pixels (undistorted) to this court plane.
Calibration schema v2 stores the landmark correspondences; this module is
the authoritative parser/fitter — the client-side JS fit is cosmetic only.
"""

from dataclasses import dataclass

import numpy as np


# WSF datum convention (Specifications for Squash Courts 2013/17; constants
# table in docs/mount-spec.md §2.2). Court lines are 50 mm wide and are OUT;
# every WSF dimension is datumed to a specific EDGE of a line, not its middle:
# the short line is 4260 mm from the back wall to its NEAREST (back-wall-side)
# edge, and service boxes are 1600 mm square INTERNAL (clear floor between the
# inside faces of their lines, WSF 5.05.03). The ft constants below land on
# those datum edges to within ~4 mm — inside WSF's ±10 mm construction
# tolerance, so they are deliberately left as round feet:
#
#   SHORT_LINE_FROM_FRONT_FT  18.0 ft = 5486 mm ~ 9750-4260 = 5490 mm
#       -> y=18.0 is the short line's BACK edge (the edge nearer the back wall)
#   SERVICE_BOX_FT            5.25 ft = 1600.2 mm ~ 1600 mm internal side
#       -> x=5.25 (left box) is the inner side line's INTERIOR (wall-side) edge
#   SERVICE_BOX_BACK_FT       23.25 ft = 7086.6 mm ~ 5490+1600 = 7090 mm
#       -> y=23.25 is the box back line's INTERIOR (front-facing) edge
#   HALF_COURT_X_FT           10.5 ft = 3200.4 mm ~ 6400/2 = 3200 mm
#       -> x=10.5 is the half-court line's CENTERLINE (the one landmark datum
#          that is a line middle, not an edge — WSF centers it between walls)
#
# Landmark labels below must name the exact edge/corner to tap: a 50 mm line
# spans ~0.164 ft, so "the short line" without an edge is a 5 cm ambiguity —
# far above the fit's px-level residual targets. Do not change these constants
# without checking every consumer of court coordinates (eval_line_calls.py,
# judge_call.py, bounce detectors).
COURT_WIDTH_FT = 21.0
COURT_LENGTH_FT = 32.0
SHORT_LINE_FROM_FRONT_FT = 18.0
SERVICE_BOX_FT = 5.25
HALF_COURT_X_FT = COURT_WIDTH_FT / 2.0
SERVICE_BOX_BACK_FT = SHORT_LINE_FROM_FRONT_FT + SERVICE_BOX_FT

FLOOR_ZONE_COLUMNS = 3
FLOOR_ZONE_ROWS = 4

# Wizard tap order: large, sharp back-half intersections first; the far
# front-seam corners last (by then the running fit can hint where to look).
FLOOR_LANDMARKS = [
    {
        "id": "short_line_left",
        "court_ft": [0.0, SHORT_LINE_FROM_FRONT_FT],
        "label": (
            "Where the BACK edge of the short line (the edge nearer the back "
            "wall) meets the LEFT side wall"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "short_line_right",
        "court_ft": [COURT_WIDTH_FT, SHORT_LINE_FROM_FRONT_FT],
        "label": (
            "Where the BACK edge of the short line (the edge nearer the back "
            "wall) meets the RIGHT side wall"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "t_point",
        "court_ft": [HALF_COURT_X_FT, SHORT_LINE_FROM_FRONT_FT],
        "label": (
            "The T — where the MIDDLE of the half-court line's width meets "
            "the BACK edge of the short line"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "left_box_inner_back",
        "court_ft": [SERVICE_BOX_FT, SERVICE_BOX_BACK_FT],
        "label": (
            "Back-inside corner of the LEFT service box — the corner of the "
            "unpainted floor INSIDE the box, where the inner edges of its "
            "back and side lines meet"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "right_box_inner_back",
        "court_ft": [COURT_WIDTH_FT - SERVICE_BOX_FT, SERVICE_BOX_BACK_FT],
        "label": (
            "Back-inside corner of the RIGHT service box — the corner of the "
            "unpainted floor INSIDE the box, where the inner edges of its "
            "back and side lines meet"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "front_seam_left",
        "court_ft": [0.0, 0.0],
        "label": (
            "Front-LEFT corner — the floor seam where front wall and LEFT "
            "side wall meet (a wall junction, not a painted line)"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "front_seam_right",
        "court_ft": [COURT_WIDTH_FT, 0.0],
        "label": (
            "Front-RIGHT corner — the floor seam where front wall and RIGHT "
            "side wall meet (a wall junction, not a painted line)"
        ),
        "optional": False,
        "snap_lines": ["h", "v"],
    },
    {
        "id": "half_court_back",
        "court_ft": [HALF_COURT_X_FT, COURT_LENGTH_FT],
        "label": (
            "Where the MIDDLE of the half-court line's width meets the back "
            "wall (skip if hidden)"
        ),
        "optional": True,
        "snap_lines": ["v", "h"],
    },
]

FLOOR_LANDMARKS_BY_ID = {mark["id"]: mark for mark in FLOOR_LANDMARKS}

# Segments in court feet; the single render source for the wizard overlay,
# the floor bounce map, and any future diagrams.
FLOOR_WIREFRAME = [
    # Court outline
    [[0.0, 0.0], [COURT_WIDTH_FT, 0.0]],
    [[COURT_WIDTH_FT, 0.0], [COURT_WIDTH_FT, COURT_LENGTH_FT]],
    [[COURT_WIDTH_FT, COURT_LENGTH_FT], [0.0, COURT_LENGTH_FT]],
    [[0.0, COURT_LENGTH_FT], [0.0, 0.0]],
    # Short line
    [[0.0, SHORT_LINE_FROM_FRONT_FT], [COURT_WIDTH_FT, SHORT_LINE_FROM_FRONT_FT]],
    # Half-court line (short line to back wall)
    [[HALF_COURT_X_FT, SHORT_LINE_FROM_FRONT_FT], [HALF_COURT_X_FT, COURT_LENGTH_FT]],
    # Left service box (inner side + back edge)
    [[SERVICE_BOX_FT, SHORT_LINE_FROM_FRONT_FT], [SERVICE_BOX_FT, SERVICE_BOX_BACK_FT]],
    [[0.0, SERVICE_BOX_BACK_FT], [SERVICE_BOX_FT, SERVICE_BOX_BACK_FT]],
    # Right service box
    [
        [COURT_WIDTH_FT - SERVICE_BOX_FT, SHORT_LINE_FROM_FRONT_FT],
        [COURT_WIDTH_FT - SERVICE_BOX_FT, SERVICE_BOX_BACK_FT],
    ],
    [
        [COURT_WIDTH_FT - SERVICE_BOX_FT, SERVICE_BOX_BACK_FT],
        [COURT_WIDTH_FT, SERVICE_BOX_BACK_FT],
    ],
]


def court_model_public():
    """JSON-safe court model served to the browser wizard (one source of truth)."""
    return {
        "units": "feet",
        "convention": (
            "origin front-left floor corner viewed from back-wall camera; "
            "x right 0-21; y toward back wall 0-32"
        ),
        "width_ft": COURT_WIDTH_FT,
        "length_ft": COURT_LENGTH_FT,
        "short_line_from_front_ft": SHORT_LINE_FROM_FRONT_FT,
        "service_box_ft": SERVICE_BOX_FT,
        "landmarks": FLOOR_LANDMARKS,
        "wireframe": FLOOR_WIREFRAME,
        "zone_rows": FLOOR_ZONE_ROWS,
        "zone_columns": FLOOR_ZONE_COLUMNS,
    }


def _as_points(points):
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Points must be an Nx2 array.")
    return array


def _collinear(points, tol=1e-9):
    centered = points - points.mean(axis=0)
    # Smallest singular value ~ 0 means all points lie on one line.
    singular = np.linalg.svd(centered, compute_uv=False)
    return singular[-1] <= tol * max(1.0, singular[0])


def _normalization(points):
    """Hartley normalization: translate centroid to origin, mean radius sqrt(2)."""
    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    mean_radius = radii.mean()
    scale = (2.0 ** 0.5) / mean_radius if mean_radius > 0 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return transform


def fit_homography(src_points, dst_points):
    """Fit H so that H @ [src, 1] ~ [dst, 1] (normalized DLT, least squares).

    Returns (H, residuals) where residuals are per-point errors in dst units.
    Raises ValueError for <4 correspondences or degenerate (collinear) input.
    """
    src = _as_points(src_points)
    dst = _as_points(dst_points)
    if src.shape != dst.shape:
        raise ValueError("Source and destination point counts must match.")
    if len(src) < 4:
        raise ValueError("Homography requires at least 4 point correspondences.")
    if _collinear(src) or _collinear(dst):
        raise ValueError("Homography points are collinear (degenerate).")

    src_norm_t = _normalization(src)
    dst_norm_t = _normalization(dst)
    src_h = np.hstack([src, np.ones((len(src), 1))]) @ src_norm_t.T
    dst_h = np.hstack([dst, np.ones((len(dst), 1))]) @ dst_norm_t.T

    rows = []
    for (sx, sy, _), (dx, dy, _) in zip(src_h, dst_h):
        rows.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        rows.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
    system = np.asarray(rows)

    _, singular, vt = np.linalg.svd(system)
    # A well-posed fit has exactly one (near-)null direction; a second tiny
    # singular value means the correspondences do not pin down a homography.
    if singular[-2] <= 1e-9 * max(1.0, singular[0]):
        raise ValueError("Homography system is rank-deficient (degenerate points).")
    h_norm = vt[-1].reshape(3, 3)

    homography = np.linalg.inv(dst_norm_t) @ h_norm @ src_norm_t
    if abs(homography[2, 2]) <= 1e-12:
        raise ValueError("Degenerate homography (zero scale term).")
    homography = homography / homography[2, 2]

    projected = np.array([apply_homography(homography, point) for point in src])
    residuals = np.linalg.norm(projected - dst, axis=1)
    return homography, residuals


def apply_homography(homography, point):
    vector = np.asarray(homography, dtype=float) @ np.array(
        [float(point[0]), float(point[1]), 1.0]
    )
    if abs(vector[2]) <= 1e-12:
        raise ValueError("Point maps to infinity under this homography.")
    return (vector[0] / vector[2], vector[1] / vector[2])


def invert_homography(homography):
    inverse = np.linalg.inv(np.asarray(homography, dtype=float))
    return inverse / inverse[2, 2]


def _distortion_params(distortion):
    if not distortion:
        return None
    if distortion.get("model") != "division_k1":
        raise ValueError(f"Unsupported distortion model: {distortion.get('model')!r}")
    k1 = float(distortion["k1"])
    cx, cy = (float(value) for value in distortion["center_px"])
    norm = float(distortion.get("norm_px") or 1000.0)
    return k1, cx, cy, norm


def undistort_point(point, distortion=None):
    """Division model: p_u = c + (p_d - c) / (1 + k1 * r^2), r = |p_d - c| / norm_px.

    Identity when distortion is None. This formula is the JS<->Python contract.
    """
    params = _distortion_params(distortion)
    if params is None:
        return (float(point[0]), float(point[1]))
    k1, cx, cy, norm = params
    dx = float(point[0]) - cx
    dy = float(point[1]) - cy
    r2 = (dx * dx + dy * dy) / (norm * norm)
    factor = 1.0 + k1 * r2
    if abs(factor) <= 1e-9:
        raise ValueError("Distortion factor collapsed to zero.")
    return (cx + dx / factor, cy + dy / factor)


def distort_point(point, distortion=None):
    """Inverse of undistort_point (closed form for the division model)."""
    params = _distortion_params(distortion)
    if params is None:
        return (float(point[0]), float(point[1]))
    k1, cx, cy, norm = params
    dx = float(point[0]) - cx
    dy = float(point[1]) - cy
    ru = (dx * dx + dy * dy) ** 0.5 / norm
    if ru <= 1e-12 or abs(k1) <= 1e-12:
        return (float(point[0]), float(point[1]))
    # Solve ru = rd / (1 + k1 * rd^2) for rd, taking the root -> ru as k1 -> 0.
    discriminant = 1.0 - 4.0 * k1 * ru * ru
    if discriminant < 0:
        raise ValueError("Point is outside the invertible range of the distortion.")
    rd = (1.0 - discriminant ** 0.5) / (2.0 * k1 * ru)
    scale = rd / ru
    return (cx + dx * scale, cy + dy * scale)


@dataclass(frozen=True)
class FloorMap:
    """Image -> court-plane mapping resolved from a v2 calibration."""

    homography_court_from_image: np.ndarray
    distortion: dict | None
    fit_rms_px: float | None
    max_residual_px: float | None
    landmark_count: int
    source: str  # "refit" or "stored_matrix"

    def image_to_court(self, x, y):
        """Map an image pixel to court feet (x: 0-21 across, y: 0-32 front->back)."""
        undistorted = undistort_point((x, y), self.distortion)
        return apply_homography(self.homography_court_from_image, undistorted)


def _floor_landmark_points(floor_plane):
    image_points = []
    court_points = []
    for landmark in floor_plane.get("landmarks", []):
        if landmark.get("skipped"):
            continue
        pixel = landmark.get("refined_px") or landmark.get("tap_px")
        court = landmark.get("court_ft")
        if pixel is None or court is None:
            continue
        image_points.append([float(pixel[0]), float(pixel[1])])
        court_points.append([float(court[0]), float(court[1])])
    return image_points, court_points


def load_floor_calibration(calibration):
    """Parse a calibration dict; return a FloorMap or None.

    v2 parser gate mirroring judge_call.load_calibration_lines. Returns None
    for v1 calibrations, missing floor planes, or structurally bad data —
    floor mapping is additive and must never break front-wall judging.
    Re-fits the homography from stored landmark points (the client JS fit is
    advisory); falls back to the stored matrix only if the refit fails.
    """
    if not isinstance(calibration, dict):
        return None
    planes = calibration.get("planes")
    if not isinstance(planes, dict):
        return None
    floor_plane = planes.get("floor")
    if not isinstance(floor_plane, dict):
        return None

    try:
        distortion = calibration.get("distortion") or None
        if distortion is not None:
            _distortion_params(distortion)  # validate early

        image_points, court_points = _floor_landmark_points(floor_plane)
        if len(image_points) >= 4:
            try:
                undistorted = [
                    undistort_point(point, distortion) for point in image_points
                ]
                homography, residuals = fit_homography(court_points, [
                    list(point) for point in undistorted
                ])
                return FloorMap(
                    homography_court_from_image=invert_homography(homography),
                    distortion=distortion,
                    fit_rms_px=float(np.sqrt(np.mean(residuals**2))),
                    max_residual_px=float(residuals.max()),
                    landmark_count=len(image_points),
                    source="refit",
                )
            except (ValueError, np.linalg.LinAlgError):
                pass

        stored = floor_plane.get("homography_image_from_court")
        if stored is not None:
            matrix = np.asarray(stored, dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                return None
            return FloorMap(
                homography_court_from_image=invert_homography(matrix),
                distortion=distortion,
                fit_rms_px=floor_plane.get("fit_rms_px"),
                max_residual_px=floor_plane.get("max_residual_px"),
                landmark_count=len(floor_plane.get("landmarks", [])),
                source="stored_matrix",
            )
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
        return None

    return None


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def floor_zone_for_point(x_ft, y_ft, columns=FLOOR_ZONE_COLUMNS, rows=FLOOR_ZONE_ROWS):
    """Classify a court-plane point into the analytics grid plus court flags.

    Zone numbering matches the front-wall target zones: row-major, 1-based,
    row 0 nearest the front wall.
    """
    x = _clamp(x_ft, 0.0, COURT_WIDTH_FT)
    y = _clamp(y_ft, 0.0, COURT_LENGTH_FT)
    column = min(columns - 1, int(x / COURT_WIDTH_FT * columns))
    row = min(rows - 1, int(y / COURT_LENGTH_FT * rows))
    behind_short_line = y >= SHORT_LINE_FROM_FRONT_FT
    in_box_depth = SHORT_LINE_FROM_FRONT_FT <= y <= SERVICE_BOX_BACK_FT
    return {
        "zone": row * columns + column + 1,
        "row": row,
        "column": column,
        "x_ft": round(x, 2),
        "y_ft": round(y, 2),
        "behind_short_line": bool(behind_short_line),
        "in_left_service_box": bool(in_box_depth and x <= SERVICE_BOX_FT),
        "in_right_service_box": bool(
            in_box_depth and x >= COURT_WIDTH_FT - SERVICE_BOX_FT
        ),
        "side": "left" if x < HALF_COURT_X_FT else "right",
    }


def build_floor_zone_summary(hits, columns=FLOOR_ZONE_COLUMNS, rows=FLOOR_ZONE_ROWS):
    """Floor-bounce analogue of job_runner.build_target_zone_summary."""
    zones = [
        {
            "zone": row * columns + column + 1,
            "row": row,
            "column": column,
            "count": 0,
            "percentage": 0.0,
        }
        for row in range(rows)
        for column in range(columns)
    ]
    by_zone = {zone["zone"]: zone for zone in zones}
    floor_hits = [
        hit
        for hit in hits
        if hit.get("event_type") == "floor" and hit.get("floor_zone") is not None
    ]
    for hit in floor_hits:
        zone = by_zone.get(int(hit["floor_zone"]["zone"]))
        if zone is not None:
            zone["count"] += 1

    total = len(floor_hits)
    if total:
        for zone in zones:
            zone["percentage"] = zone["count"] / total * 100.0

    common = [
        dict(zone)
        for zone in sorted(zones, key=lambda item: (-item["count"], item["zone"]))
        if zone["count"] > 0
    ][:3]
    missing = [dict(zone) for zone in zones if zone["count"] == 0]
    return {
        "rows": rows,
        "columns": columns,
        "total_floor_bounces": total,
        "zones": zones,
        "common_zones": common,
        "missing_zones": missing,
    }


# --- Full camera model (pose + focal) --------------------------------------

G_FT_PER_S2 = 32.174

OUT_LINE_HEIGHT_FT = 15.0
TIN_TOP_HEIGHT_FT = 19.0 / 12.0

_WALL_LINE_HEIGHTS_FT = {
    "out_line_lower_edge": OUT_LINE_HEIGHT_FT,
    "tin_top_edge": TIN_TOP_HEIGHT_FT,
}


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole camera in court coordinates (feet).

    Operates in undistorted pixel space: `project` returns undistorted
    pixels and `ray` expects them; callers undistort observations with
    `undistort_point(p, self.distortion)` first.
    """

    focal_px: float
    center_px: tuple
    rotation: np.ndarray        # 3x3, world -> camera
    camera_center_ft: np.ndarray  # (3,)
    distortion: dict | None
    fit_rms_px: float | None
    point_count: int

    def project(self, court_xyz):
        camera_point = self.rotation @ (
            np.asarray(court_xyz, dtype=float) - self.camera_center_ft
        )
        if camera_point[2] <= 1e-9:
            raise ValueError("Point is at or behind the camera.")
        cx, cy = self.center_px
        return (
            self.focal_px * camera_point[0] / camera_point[2] + cx,
            self.focal_px * camera_point[1] / camera_point[2] + cy,
        )

    def ray(self, pixel):
        cx, cy = self.center_px
        direction_camera = np.array(
            [(float(pixel[0]) - cx) / self.focal_px,
             (float(pixel[1]) - cy) / self.focal_px,
             1.0]
        )
        direction = self.rotation.T @ direction_camera
        return (
            self.camera_center_ft.copy(),
            direction / np.linalg.norm(direction),
        )

    def depth_ft(self, court_xyz):
        camera_point = self.rotation @ (
            np.asarray(court_xyz, dtype=float) - self.camera_center_ft
        )
        return float(camera_point[2])

    def projection_matrix(self):
        cx, cy = self.center_px
        intrinsics = np.array(
            [[self.focal_px, 0.0, cx], [0.0, self.focal_px, cy], [0.0, 0.0, 1.0]]
        )
        translation = (-self.rotation @ self.camera_center_ft).reshape(3, 1)
        return intrinsics @ np.hstack([self.rotation, translation])

    def to_dict(self):
        return {
            "focal_px": float(self.focal_px),
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
            "rotation": self.rotation.tolist(),
            "camera_center_ft": self.camera_center_ft.tolist(),
            "distortion": self.distortion,
            "fit_rms_px": self.fit_rms_px,
            "point_count": self.point_count,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            focal_px=float(data["focal_px"]),
            center_px=(float(data["center_px"][0]), float(data["center_px"][1])),
            rotation=np.asarray(data["rotation"], dtype=float),
            camera_center_ft=np.asarray(data["camera_center_ft"], dtype=float),
            distortion=data.get("distortion"),
            fit_rms_px=data.get("fit_rms_px"),
            point_count=int(data.get("point_count") or 0),
        )


def _camera_correspondences(calibration):
    """Image px (raw/distorted) <-> court 3D points from a v2 calibration.

    Floor landmarks sit at z=0. Front-wall line endpoints sit at
    (0, 0, h) and (21, 0, h): line_from_calibration's contract stores them
    left-then-right, and the wizard spans the wall, so the endpoints are the
    side-wall junctions. If a calibration violates that, the pose fit's
    residual gate (solve_camera_model) rejects it.
    """
    image_points, court_points = [], []
    planes = calibration.get("planes") or {}
    floor_plane = planes.get("floor") or {}
    for landmark in floor_plane.get("landmarks", []):
        if landmark.get("skipped"):
            continue
        pixel = landmark.get("refined_px") or landmark.get("tap_px")
        court = landmark.get("court_ft")
        if pixel is None or court is None:
            continue
        image_points.append([float(pixel[0]), float(pixel[1])])
        court_points.append([float(court[0]), float(court[1]), 0.0])
    for line in calibration.get("lines", []):
        height = _WALL_LINE_HEIGHTS_FT.get(line.get("name"))
        endpoints = line.get("endpoints") or []
        if height is None or len(endpoints) != 2:
            continue
        for x_ft, endpoint in zip((0.0, COURT_WIDTH_FT), endpoints):
            image_points.append([float(endpoint[0]), float(endpoint[1])])
            court_points.append([x_ft, 0.0, height])
    return (
        np.asarray(image_points, dtype=float).reshape(-1, 2),
        np.asarray(court_points, dtype=float).reshape(-1, 3),
    )


def _init_camera_from_floor(calibration, frame_size):
    """Approximate (focal, R, C) from the floor homography, Zhang-style.

    H maps court floor (x, y) -> undistorted image px, H ~ K [r1 r2 t].
    With the principal point pinned at the image center, the orthonormality
    of r1, r2 yields two closed-form estimates of the focal length.
    """
    floor_map = load_floor_calibration(calibration)
    if floor_map is None:
        return None
    homography = invert_homography(floor_map.homography_court_from_image)
    cx, cy = frame_size[0] / 2.0, frame_size[1] / 2.0

    def reduced(column):
        return (
            column[0] - cx * column[2],
            column[1] - cy * column[2],
            column[2],
        )

    a1, b1, c1 = reduced(homography[:, 0])
    a2, b2, c2 = reduced(homography[:, 1])
    focal_sq = []
    if abs(c1 * c2) > 1e-12:
        value = -(a1 * a2 + b1 * b2) / (c1 * c2)
        if value > 0:
            focal_sq.append(value)
    if abs(c2 * c2 - c1 * c1) > 1e-12:
        value = ((a1 * a1 + b1 * b1) - (a2 * a2 + b2 * b2)) / (c2 * c2 - c1 * c1)
        if value > 0:
            focal_sq.append(value)
    if not focal_sq:
        return None
    focal = float(np.sqrt(np.mean(focal_sq)))

    intrinsics_inv = np.array(
        [[1.0 / focal, 0.0, -cx / focal],
         [0.0, 1.0 / focal, -cy / focal],
         [0.0, 0.0, 1.0]]
    )
    r1 = intrinsics_inv @ homography[:, 0]
    r2 = intrinsics_inv @ homography[:, 1]
    translation = intrinsics_inv @ homography[:, 2]
    scale = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    r1, r2, translation = r1 * scale, r2 * scale, translation * scale
    if translation[2] < 0:
        # Court origin must be in front of the camera.
        r1, r2, translation = -r1, -r2, -translation
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    camera_center = -rotation.T @ translation
    return focal, rotation, camera_center
