"""Automatic squash-court detection from a handful of frames.

Finds the court's lines in an image and derives every calibration landmark as
an intersection of two of them, producing exactly the structures the manual
wizard's buildJson() produces (schema squash-calibration-v2). Design spec:
docs/superpowers/specs/2026-07-27-auto-court-detection-design.md

Two invariants this module is built around:

  * Never key on hue. Court lines are navy at Bay Club and red at
    SquashAnalytics; the property that holds across both is "a thin stripe
    darker and/or more chromatic than its local surround" (spec §4.1).
  * Never solve 3-D pose. Both fits are plane homographies, which keeps this
    clear of the solve_camera_model chirality defect that rejects every real
    stored calibration.

Pure functions over numpy arrays: no Flask, no disk access, no globals.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

import court_model

# --- temporal median -------------------------------------------------------
MOTION_DELTA = 18        # grey levels; below this a pixel counts as unchanged
MOTION_FRACTION = 0.25   # fraction of changed pixels that means "the camera moved"


def median_frame(frames):
    """Temporal median of same-size BGR frames, plus a camera-motion verdict.

    Two players moving through a static court change a few percent of the
    pixels and vanish into the median. A pan changes most of them and medians
    into a ghost, so the caller must fall back to a single frame — which is
    exactly what SquashAnalytics.mp4 does and the product's fin mount does not.
    """
    stack = np.stack([np.asarray(frame) for frame in frames])
    if stack.shape[0] < 2:
        return stack[0].copy(), False

    median = np.median(stack, axis=0).astype(np.uint8)
    reference = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.int16)
    changed = [
        float(np.mean(
            np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
                   - reference) > MOTION_DELTA))
        for frame in stack
    ]
    return median, bool(float(np.mean(changed)) > MOTION_FRACTION)


# --- response maps ---------------------------------------------------------
BLACKHAT_K = 21          # px; wider than a court line, narrower than a panel
CHROMA_K = 31            # px; odd, as cv2.medianBlur requires
HOUGH_VOTES = 80
HOUGH_GAP = 14
MERGE_ANGLE_DEG = 2.0
MERGE_OFFSET_PX = 6.0


def line_response(bgr):
    """Painted-line saliency: thin structures darker and/or more chromatic
    than their local surround.

    Black-hat (closing minus image) responds to anything darker than its
    surround and thinner than the kernel; the chroma term catches paint that
    is coloured but not much darker. Neither names a hue, which is the point:
    Bay Club's lines are navy and SquashAnalytics' are red.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    dark = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_BLACKHAT, kernel)
    background = np.stack(
        [cv2.medianBlur(lab[:, :, channel], CHROMA_K) for channel in (1, 2)],
        axis=2).astype(np.int16)
    chroma = np.clip(
        np.linalg.norm(lab[:, :, 1:].astype(np.int16) - background, axis=2),
        0, 255).astype(np.uint8)
    return cv2.normalize(cv2.max(dark, chroma), None, 0, 255, cv2.NORM_MINMAX)


def edge_response(bgr):
    """Surface-boundary saliency.

    The wall/floor seams are junctions, not paint: they can be a pure colour
    step with no dark ridge, so line_response can miss them entirely. Scharr
    over all three LAB channels catches the step whichever way it goes.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    strongest = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        gradient_x = cv2.Scharr(lab[:, :, channel], cv2.CV_32F, 1, 0)
        gradient_y = cv2.Scharr(lab[:, :, channel], cv2.CV_32F, 0, 1)
        strongest = np.maximum(strongest, cv2.magnitude(gradient_x, gradient_y))
    return cv2.normalize(strongest, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def paint_mask(bgr):
    """Binary mask of painted pixels; the datum refit in Task 5 reads this."""
    _, mask = cv2.threshold(line_response(bgr), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# --- line extraction -------------------------------------------------------
@dataclass(frozen=True)
class DetectedLine:
    """A straight image feature, as a segment between its extreme support."""

    x1: float
    y1: float
    x2: float
    y2: float
    support: int

    @property
    def length_px(self):
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self):
        """Orientation in (-90, 90]; 0 is horizontal."""
        angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        if angle > 90:
            angle -= 180
        if angle <= -90:
            angle += 180
        return angle

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def y_at(self, x):
        """Image y where this line crosses column x; None if it is vertical."""
        if abs(self.x2 - self.x1) < 1e-9:
            return None
        slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        return self.y1 + slope * (x - self.x1)

    def intersect(self, other):
        """Crossing point of the two infinite lines, or None if near-parallel."""
        dx1, dy1 = self.x2 - self.x1, self.y2 - self.y1
        dx2, dy2 = other.x2 - other.x1, other.y2 - other.y1
        denominator = dx1 * dy2 - dy1 * dx2
        if abs(denominator) < 1e-9:
            return None
        along = (((other.x1 - self.x1) * dy2 - (other.y1 - self.y1) * dx2)
                 / denominator)
        return (self.x1 + dx1 * along, self.y1 + dy1 * along)


def _segment_direction(segment):
    """Unit direction of a segment, or None if it is degenerate."""
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    return dx / length, dy / length


def _offset_along(direction, point):
    """Signed perpendicular offset of `point` from the origin in a direction frame.

    Only comparable between two segments expressed in the SAME frame: negating
    a direction negates the offset. That is why fragments are aligned to their
    group's direction before this is measured, rather than each canonicalising
    itself -- any per-fragment rule (wrap the angle, force uy >= 0) merely moves
    the sign flip to whatever orientation its tie-break sits on, and a flip
    turns a 0 px offset difference into ~2*y.
    """
    ux, uy = direction
    return -uy * point[0] + ux * point[1]


def _aligned_to(direction, reference):
    """`direction`, flipped if needed to point the same way as `reference`."""
    ux, uy = direction
    rx, ry = reference
    return (-ux, -uy) if ux * rx + uy * ry < 0 else (ux, uy)


def _direction_delta_deg(first, second):
    """Unsigned angle between two unit directions, in degrees."""
    ax, ay = first
    bx, by = second
    return abs(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))


def _fit_group(segments):
    """Total-least-squares fit through every segment in a collinear group.

    Sampling each segment every ~4 px weights the fit by segment length, so a
    long clean run of the out line outvotes a short glare artefact that
    happened to land on the same infinite line.
    """
    points = []
    for x1, y1, x2, y2 in segments:
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1) / 4))
        for index in range(steps + 1):
            fraction = index / steps
            points.append((x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction))
    array = np.asarray(points, dtype=float)
    centre = array.mean(axis=0)
    _, _, right = np.linalg.svd(array - centre)
    direction = right[0]
    along = (array - centre) @ direction
    start = centre + direction * float(along.min())
    end = centre + direction * float(along.max())
    return DetectedLine(float(start[0]), float(start[1]),
                        float(end[0]), float(end[1]), len(segments))


def find_lines(response, min_length_px):
    """Hough segments from a response map, merged into whole lines.

    Hough returns a court line as several broken pieces wherever a player or a
    glare patch interrupted it, so collinear pieces are grouped and refitted
    as one — otherwise a line's usable span is whatever survived occlusion.
    """
    _, binary = cv2.threshold(response, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    found = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=HOUGH_VOTES,
                            minLineLength=int(min_length_px),
                            maxLineGap=HOUGH_GAP)
    if found is None:
        return []

    groups = []
    # OpenCV 4 returns HoughLinesP segments as (N, 1, 4), while OpenCV 5
    # returns the same x1/y1/x2/y2 values as (N, 4). Flatten only the wrapper
    # dimensions so the detector remains compatible with both versions.
    for raw in np.asarray(found).reshape(-1, 4):
        segment = [float(value) for value in raw]
        direction = _segment_direction(segment)
        if direction is None:
            continue
        for group in groups:
            aligned = _aligned_to(direction, group["direction"])
            if _direction_delta_deg(aligned, group["direction"]) > MERGE_ANGLE_DEG:
                continue
            if abs(_offset_along(aligned, segment[:2])
                   - group["offset"]) > MERGE_OFFSET_PX:
                continue
            group["segments"].append(segment)
            break
        else:
            groups.append({"direction": direction,
                           "offset": _offset_along(direction, segment[:2]),
                           "segments": [segment]})

    lines = [_fit_group(group["segments"]) for group in groups]
    return sorted(lines, key=lambda line: line.length_px, reverse=True)


# --- entity assignment -----------------------------------------------------
HORIZONTAL_MAX_DEG = 25.0   # steeper than this and a line is not a court-x line


def _horizontals(lines):
    return [line for line in lines if abs(line.angle_deg) <= HORIZONTAL_MAX_DEG]


def _diagonals(lines):
    return [line for line in lines if abs(line.angle_deg) > HORIZONTAL_MAX_DEG]


def _upper_end(line):
    """The endpoint nearer the top of the image."""
    return (line.x1, line.y1) if line.y1 <= line.y2 else (line.x2, line.y2)


def assign_lines(paint_lines, edge_lines, frame_shape):
    """Name detected lines as court entities (spec §7.4).

    From a back-wall mount the court presents a fixed vertical ordering — out
    line, service line, tin, wall/floor seam, then floor lines — and two long
    diagonals running away from the front corners. That ordering plus the side
    a diagonal falls on is enough to name everything, with no hue and no pose.

    Returns a dict whose values are DetectedLine or None. A None means that
    entity was not found; the caller decides whether that is fatal.
    """
    height, width = frame_shape[:2]
    centre_x = width / 2.0
    assigned = {name: None for name in
                ("out", "service", "tin", "front_seam",
                 "left_seam", "right_seam", "short_line")}

    # Side-wall floor seams: the longest steep line on each half of the frame,
    # sloping the way that half's seam must slope. Looking down-court, the
    # left seam runs down-and-left (negative angle) and the right seam runs
    # down-and-right (positive angle).
    diagonals = _diagonals(edge_lines) + _diagonals(paint_lines)
    left = [line for line in diagonals
            if line.midpoint[0] < centre_x and line.angle_deg < 0]
    right = [line for line in diagonals
             if line.midpoint[0] > centre_x and line.angle_deg > 0]
    if left:
        assigned["left_seam"] = max(left, key=lambda line: line.length_px)
    if right:
        assigned["right_seam"] = max(right, key=lambda line: line.length_px)

    # Front wall/floor seam: the horizontal nearest to where both diagonals
    # begin. It is a junction, so look in the boundary map first.
    anchors = [_upper_end(assigned[name]) for name in ("left_seam", "right_seam")
               if assigned[name] is not None]
    horizontals = _horizontals(edge_lines) + _horizontals(paint_lines)
    if anchors and horizontals:
        def seam_error(line):
            errors = []
            for anchor_x, anchor_y in anchors:
                at = line.y_at(anchor_x)
                errors.append(abs(at - anchor_y) if at is not None else 1e9)
            return sum(errors) / len(errors)
        best = min(horizontals, key=seam_error)
        if seam_error(best) < height * 0.05:
            assigned["front_seam"] = best

    seam_y = (assigned["front_seam"].midpoint[1]
              if assigned["front_seam"] is not None else height * 0.65)

    # Front-wall paint: horizontals above the seam, widest span wins the out
    # line, then the remaining two are service (upper) and tin (lower).
    above = sorted((line for line in _horizontals(paint_lines)
                    if line.midpoint[1] < seam_y - 2),
                   key=lambda line: line.midpoint[1])
    if above:
        assigned["out"] = max(above, key=lambda line: line.length_px)
        rest = [line for line in above if line is not assigned["out"]
                and line.midpoint[1] > assigned["out"].midpoint[1]]
        if len(rest) >= 2:
            assigned["service"], assigned["tin"] = rest[0], rest[-1]
        elif len(rest) == 1:
            assigned["service"] = rest[0]

    # Short line: the widest horizontal paint line below the seam.
    below = [line for line in _horizontals(paint_lines)
             if line.midpoint[1] > seam_y + 2]
    if below:
        assigned["short_line"] = max(below, key=lambda line: line.length_px)

    return assigned


# --- datum refit -------------------------------------------------------------
DATUM_BAND_PX = 6          # how far either side of the fitted line to look
MIN_DATUM_COLUMNS = 40     # mirrors index.html's MIN_COLS
CHECK_OK_PX = 4.0          # a self-verification prediction this close is "ok"


@dataclass(frozen=True)
class LineFit:
    """A line in the calibration's stored form: y = slope * x + intercept."""

    slope: float
    intercept: float
    rms_px: float
    x_span: tuple
    pixel_count: int


def _robust_line_fit(xs, ys):
    """Least squares, then one pass dropping points beyond 2.5 sigma."""
    xs_array = np.asarray(xs, dtype=float)
    ys_array = np.asarray(ys, dtype=float)
    for _ in range(2):
        slope, intercept = np.polyfit(xs_array, ys_array, 1)
        residuals = ys_array - (slope * xs_array + intercept)
        spread = float(np.std(residuals))
        if spread < 1e-6:
            break
        keep = np.abs(residuals) <= 2.5 * spread
        if keep.sum() < MIN_DATUM_COLUMNS or keep.all():
            break
        xs_array, ys_array = xs_array[keep], ys_array[keep]
    residuals = ys_array - (slope * xs_array + intercept)
    return (float(slope), float(intercept),
            float(np.sqrt(np.mean(residuals ** 2))),
            (float(xs_array.min()), float(xs_array.max())),
            int(len(xs_array)))


def refit_to_datum(mask, line, mode):
    """Refit `line` onto the named boundary of the painted stripe under it.

    `mode` is 'min' (topmost mask row per column) or 'max' (lowest) — the same
    contract as index.html's extractEdge, because front-wall lines are stored
    by EDGE, not centreline (spec §5). Returning the stripe's middle instead
    would put a systematic half-line-width bias into the out-line call.
    """
    return _refit(mask, line, mode)


def refit_to_centreline(mask, line):
    """Refit `line` onto the middle of the stripe under it.

    Floor landmarks are stored by paint CENTRELINE (court_model.py:49-62), the
    opposite convention to the front-wall lines above.
    """
    return _refit(mask, line, "mid")


def _refit(mask, line, mode):
    height, width = mask.shape[:2]
    first = int(max(0, math.floor(min(line.x1, line.x2))))
    last = int(min(width - 1, math.ceil(max(line.x1, line.x2))))
    xs, ys = [], []
    for x in range(first, last + 1):
        centre = line.y_at(x)
        if centre is None:
            continue
        low = int(max(0, round(centre - DATUM_BAND_PX)))
        high = int(min(height - 1, round(centre + DATUM_BAND_PX)))
        rows = [y for y in range(low, high + 1) if mask[y, x]]
        if not rows:
            continue
        if mode == "max":
            chosen = max(rows)
        elif mode == "min":
            chosen = min(rows)
        else:
            chosen = (min(rows) + max(rows)) / 2.0
        xs.append(float(x))
        ys.append(float(chosen))
    if len(xs) < MIN_DATUM_COLUMNS:
        return None
    slope, intercept, rms, span, count = _robust_line_fit(xs, ys)
    return LineFit(slope, intercept, rms, span, count)


def _line_payload(name, fit, source):
    """A LineFit in the exact shape buildJson() writes (index.html:2593)."""
    x0, x1 = fit.x_span
    return {
        "name": name,
        "endpoints": [[round(x0, 2), round(fit.slope * x0 + fit.intercept, 2)],
                      [round(x1, 2), round(fit.slope * x1 + fit.intercept, 2)]],
        "slope": fit.slope,
        "intercept": fit.intercept,
        "fit_rms_px": round(fit.rms_px, 2),
        "x_span_px": [round(x0, 2), round(x1, 2)],
        "x_span_source": "fit",
        "edge_pixels_used": fit.pixel_count,
        "source": source,
    }


def _fit_as_line(fit):
    """A LineFit back as a DetectedLine, so it can be intersected."""
    x0, x1 = fit.x_span
    return DetectedLine(x0, fit.slope * x0 + fit.intercept,
                        x1, fit.slope * x1 + fit.intercept, support=1)


# --- the whole detection -----------------------------------------------------
# Only the corner IDS live here. Their court placements are datum values and
# belong to court_model (CLAUDE.md: "Court datum values live in court_model.py
# and must never be redefined") -- court_model._camera_correspondences owns the
# id -> court_ft mapping, and nothing in this module needs the feet, only the
# names and their order in the emitted wall plane.
_WALL_CORNER_IDS = ("top_left", "top_right", "bottom_left", "bottom_right")

_FLOOR_ANCHORS = (
    ("front_seam_left", (0.0, 0.0)),
    ("front_seam_right", (court_model.COURT_WIDTH_FT, 0.0)),
    ("short_line_left", (0.0, court_model.SHORT_LINE_CENTER_Y_FT)),
    ("short_line_right", (court_model.COURT_WIDTH_FT,
                          court_model.SHORT_LINE_CENTER_Y_FT)),
)

# A plane homography has 8 degrees of freedom, so the 4 correspondences above
# determine it EXACTLY: the fit reproduces its own input points to ~1e-13 px
# whatever those points are. Measured 2026-07-27 by corrupting the anchors --
# swapping left for right, i.e. a fully mirrored court, still fitted to ~2e-13
# px RMS. A residual number here would therefore be a fabricated precision
# signal rather than a measurement, so the payload reports null instead. If a
# 5th anchor is ever added the fit becomes over-determined and the residuals
# start carrying information again, which is what this constant tracks.
_FLOOR_FIT_IS_EXACT = len(_FLOOR_ANCHORS) <= 4

# Court points that no anchor was fitted to. The third element says whether the
# point is INDEPENDENT evidence, i.e. whether its position is pinned by the
# anchors regardless of whether the fit is right.
#
# `t_point` is not: it lies on the segment joining short_line_left and
# short_line_right, which were themselves fitted onto real short-line paint, so
# a projective map sends it somewhere on that same painted line no matter how
# wrong everything else is. distanceTransform duly reads ~0 px for it even on a
# mirrored court. It stays in the payload -- a gross failure can still move it,
# and the UI lists it when it goes off -- but it can never be the thing that
# earns a confident verdict, so it does not count toward checks_verified.
_FLOOR_CHECKS = (
    ("t_point", "the T", (court_model.HALF_COURT_X_FT,
                          court_model.SHORT_LINE_CENTER_Y_FT), False),
    ("left_box_inner_back", "the left service box's back inner corner",
     (court_model.LEFT_BOX_INNER_CENTER_X_FT,
      court_model.BOX_BACK_CENTER_Y_FT), True),
    ("right_box_inner_back", "the right service box's back inner corner",
     (court_model.RIGHT_BOX_INNER_CENTER_X_FT,
      court_model.BOX_BACK_CENTER_Y_FT), True),
)

# Human words for the entities assign_lines names. The failure reason reaches
# the player verbatim, and "Could not find: front_seam." is an internal
# identifier, not user copy (DESIGN.md §0.7: sentence case, plain language).
_ENTITY_LABELS = {
    "out": "the out line",
    "service": "the service line",
    "tin": "the tin",
    "front_seam": "the front wall's floor line",
    "left_seam": "the left wall's floor line",
    "right_seam": "the right wall's floor line",
    "short_line": "the short line",
}


def _readable_list(labels):
    """'a', 'a and b', or 'a, b, and c'."""
    labels = list(labels)
    if len(labels) <= 1:
        return "".join(labels)
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _failure(status, frame_shape, reason, warnings=()):
    """A non-'ok' reply whose FIRST warning is why detection actually failed.

    The client shows warnings[0]. Appending the reason after the incidental
    ones (e.g. "Camera appears to be moving") sent the player off to re-mount a
    phone that was mounted fine, so the reason leads and the rest follow.
    """
    height, width = (frame_shape[:2] if frame_shape is not None else (0, 0))
    return {"ok": True, "status": status,
            "frame_width": int(width), "frame_height": int(height),
            "lines": [], "planes": {}, "checks": [], "checks_verified": 0,
            "confidence": "low", "warnings": [reason] + list(warnings)}


def detect_court(frames):
    """Detect the court from one or more frames of the same fixed viewpoint.

    Returns squash-calibration-v2 structures (spec §6) and never raises: the
    wizard falls back to manual taps on any non-'ok' status.
    """
    if not len(frames):
        return _failure("no_frames", None, "No frames were supplied.")

    warnings = []
    image, moved = median_frame(frames)
    if moved:
        image = np.asarray(frames[len(frames) // 2])
        warnings.append("Camera appears to be moving; used a single frame.")

    height, width = image.shape[:2]
    minimum = width * 0.10
    mask = paint_mask(image)
    assigned = assign_lines(find_lines(line_response(image), minimum),
                            find_lines(edge_response(image), minimum),
                            image.shape)

    # All three front-wall lines are REQUIRED, not optional. A calibration
    # without them is unusable by construction: judge_call.load_calibration_
    # lines rejects any calibration missing tin_top_edge, and index.html's
    # buildJson() dereferences S.lines[k].fit for all three -- so a "successful"
    # detection missing one presents as green on the confirm screen and then
    # throws a TypeError two screens later at TRACK BALL. Failing here instead
    # drops the player into the manual wizard, which is the working fallback.
    required = ("out", "service", "tin", "front_seam",
                "left_seam", "right_seam", "short_line")
    missing = [name for name in required if assigned.get(name) is None]
    if missing:
        return _failure(
            "insufficient_lines", image.shape,
            "Could not find "
            f"{_readable_list(_ENTITY_LABELS[name] for name in missing)}.",
            warnings)

    # Front-wall lines, each pulled onto its own stored datum (spec §5).
    lines = []
    for name, key, mode in (
            ("out_line_lower_edge", "out", "max"),
            ("service_line_top_edge", "service", "min"),
            ("tin_top_edge", "tin", "min")):
        fit = refit_to_datum(mask, assigned[key], mode)
        if fit is None:
            return _failure(
                "insufficient_lines", image.shape,
                f"There was too little clean paint on {_ENTITY_LABELS[key]} "
                "to fit it.", warnings)
        lines.append(_line_payload(name, fit, "detected"))

    out_fit = lines[0]

    # Anchors as intersections of long lines.
    out_line = DetectedLine(*out_fit["endpoints"][0], *out_fit["endpoints"][1],
                            support=1)
    seam = assigned["front_seam"]
    short_fit = refit_to_centreline(mask, assigned["short_line"])
    short_line = (_fit_as_line(short_fit) if short_fit is not None
                  else assigned["short_line"])

    # The wall-top corners are NOT out_line.intersect(seam): out_line sits on
    # the 3-D line {y=0, z=OUT_LINE_HEIGHT_FT}, the seam lines sit on
    # {x=0 or COURT_WIDTH_FT, z=0} -- two lines at different heights that
    # never actually meet in 3-D, so their projections only cross at some
    # arbitrary point far from the true corner (measured 35-500+ px off,
    # growing *worse* at longer focal lengths, at court_camera(700)/(1600)/
    # (3000)/(9000) respectively -- not a noise artifact). The corner IS,
    # however, exactly where out_line's own fitted extent ends, because
    # render_court draws the out-line band across the FULL front-wall width
    # (x in [0, COURT_WIDTH_FT]), so a clean detection's x_span already
    # reaches both corners (measured ~0.13 px off truth). x1 < x2 always
    # (DetectedLine built from fit.x_span's (min, max)), so x1 is the
    # screen-left end -- the same side `assigned["left_seam"]` was chosen
    # from (assign_lines: midpoint[0] < centre_x) -- matching bottom_left's
    # convention below.
    anchors = {
        "top_left": (out_line.x1, out_line.y1),
        "top_right": (out_line.x2, out_line.y2),
        "bottom_left": seam.intersect(assigned["left_seam"]),
        "bottom_right": seam.intersect(assigned["right_seam"]),
        "short_line_left": short_line.intersect(assigned["left_seam"]),
        "short_line_right": short_line.intersect(assigned["right_seam"]),
    }
    if any(point is None for point in anchors.values()):
        return _failure("insufficient_lines", image.shape,
                        "The court lines found did not cross where a court's "
                        "corners should be.", warnings)

    floor_pixels = {
        "front_seam_left": anchors["bottom_left"],
        "front_seam_right": anchors["bottom_right"],
        "short_line_left": anchors["short_line_left"],
        "short_line_right": anchors["short_line_right"],
    }
    court_points = [court_ft for _, court_ft in _FLOOR_ANCHORS]
    image_points = [floor_pixels[name] for name, _ in _FLOOR_ANCHORS]
    try:
        homography, residuals = court_model.fit_homography(court_points,
                                                           image_points)
    except (ValueError, np.linalg.LinAlgError):
        return _failure("insufficient_lines", image.shape,
                        "The court lines found do not describe a flat court "
                        "floor.", warnings)

    # Self-verification: how far predictions of UNUSED court points fall from
    # real paint. distanceTransform turns that into a lookup.
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    checks = []
    for check_id, check_label, court_ft, independent in _FLOOR_CHECKS:
        # apply_homography raises when a point maps to infinity -- a check
        # point on the horizon of a shallow fit does exactly that. This
        # function's contract is that it never raises (the wizard's whole
        # fallback design rests on a status field, not an exception), and the
        # endpoint would otherwise answer 500 instead of falling back, so a
        # check we cannot place is simply a check we cannot verify.
        try:
            px, py = court_model.apply_homography(homography, court_ft)
            if not (math.isfinite(px) and math.isfinite(py)):
                raise ValueError("Check point is not finite.")
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            checks.append({"id": check_id, "label": check_label,
                           "independent": independent, "predicted_px": None,
                           "residual_px": None, "status": "unverified"})
            continue
        inside = 0 <= int(py) < height and 0 <= int(px) < width
        residual = float(distance[int(py), int(px)]) if inside else None
        checks.append({
            "id": check_id,
            "label": check_label,
            "independent": independent,
            "predicted_px": [round(px, 2), round(py, 2)],
            "residual_px": None if residual is None else round(residual, 2),
            "status": ("unverified" if residual is None
                       else "ok" if residual <= CHECK_OK_PX else "off"),
        })

    # What is actually knowable at this point:
    #   * all three front-wall lines are present -- guaranteed above, since a
    #     detection missing one now fails rather than reaching here;
    #   * all six anchors were derived -- guaranteed by the intersection check;
    #   * whether any check landed off real paint;
    #   * how many checks were INDEPENDENTLY verifiable at all.
    # The floor fit's own residuals are deliberately not an input: they are ~0
    # by construction (see _FLOOR_FIT_IS_EXACT) and would only launder an exact
    # fit into a quality claim.
    off_checks = [check for check in checks if check["status"] == "off"]
    checks_verified = sum(1 for check in checks
                          if check["independent"] and check["status"] != "unverified")

    reasons = []
    if off_checks:
        reasons.append(
            "Predicted court markings did not land on real paint: "
            f"{_readable_list(check['label'] for check in off_checks)}.")
    if not checks_verified:
        # No independent evidence at all: the fit reproduces the four points it
        # was built from and nothing else was in frame to contradict it. That
        # is not a reason to call the detection wrong, but it is a reason not
        # to call it confident.
        reasons.append("No independent court marking was in frame to check "
                       "this fit against.")
    confidence = "low" if reasons else "high"
    warnings.extend(reasons)

    floor_rms = (None if _FLOOR_FIT_IS_EXACT
                 else round(float(np.sqrt(np.mean(residuals ** 2))), 2))
    floor_worst = (None if _FLOOR_FIT_IS_EXACT
                   else round(float(residuals.max()), 2))

    return {
        "ok": True,
        "status": "ok",
        "frame_width": int(width),
        "frame_height": int(height),
        "lines": lines,
        "planes": {
            "wall": {
                "corners": [
                    {"id": corner_id,
                     "tap_px": [round(anchors[corner_id][0], 2),
                                round(anchors[corner_id][1], 2)],
                     # top_left/top_right come from out_line's own fitted
                     # extent (see the comment above `anchors`), not a line
                     # intersection like the other two.
                     "source": ("line_extent" if corner_id.startswith("top_")
                                else "intersection")}
                    for corner_id in _WALL_CORNER_IDS
                ],
                "fitted_by": "auto_detect",
            },
            "floor": {
                "landmarks": [
                    {"id": name,
                     "court_ft": list(court_ft),
                     "tap_px": [round(floor_pixels[name][0], 2),
                                round(floor_pixels[name][1], 2)],
                     "refined_px": [round(floor_pixels[name][0], 2),
                                    round(floor_pixels[name][1], 2)],
                     "method": "intersection",
                     # null, not 0.0: an exact fit has no residual to report
                     # (see _FLOOR_FIT_IS_EXACT). court_model.load_floor_
                     # calibration re-fits from these landmark points and never
                     # reads the stored numbers on that path, and index.html's
                     # applyDetection reads only tap_px/refined_px/method, so
                     # both consumers are unaffected by the null.
                     "residual_px": (None if _FLOOR_FIT_IS_EXACT
                                     else round(float(residuals[index]), 2)),
                     "skipped": False,
                     "source": "intersection"}
                    for index, (name, court_ft) in enumerate(_FLOOR_ANCHORS)
                ],
                "homography_image_from_court": [
                    [float(value) for value in row] for row in homography],
                "fit_rms_px": floor_rms,
                "max_residual_px": floor_worst,
                "fitted_by": "auto_detect",
            },
        },
        "checks": checks,
        # How many of `checks` were independent AND actually placeable in this
        # frame -- i.e. how much real evidence stands behind `confidence`. The
        # confirm screen says so out loud rather than implying the fit was
        # checked when nothing could be.
        "checks_verified": checks_verified,
        "confidence": confidence,
        "warnings": warnings,
    }
