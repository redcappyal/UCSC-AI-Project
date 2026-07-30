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

import itertools
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


def _perpendicular_distance(anchor, direction, point):
    """Distance from `point` to the line through `anchor` along `direction`.

    Measured between the two points themselves, never extrapolated back to the
    image origin. An origin-referenced offset (-uy*x + ux*y, compared between
    fragments) levers each fragment's angular error by its distance from the
    origin: two pieces of one 4 px stripe, one starting at x=715 and one at
    x=776 with a 0.4 deg tilt, measure 555 and 564 -- a 9 px gap that splits
    one court line into two groups. Anchoring the comparison where the
    fragments actually are removes the lever arm, and being unsigned it is
    also immune to the direction flip that offsets had to be aligned against.
    """
    ux, uy = direction
    return abs(-uy * (point[0] - anchor[0]) + ux * (point[1] - anchor[1]))


def _midpoint(segment):
    """Centre of a segment -- the point a group is anchored and compared at,
    because it is the fragment's least-extrapolated position."""
    x1, y1, x2, y2 = segment
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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


def _merge_hough_segments(found):
    """Group collinear Hough segments and refit each group as one line.

    Hough returns a court line as several broken pieces wherever a player or a
    glare patch interrupted it, so collinear pieces are grouped and refitted
    as one — otherwise a line's usable span is whatever survived occlusion.
    """
    groups = []
    # OpenCV 4 returns HoughLinesP segments as (N, 1, 4), while OpenCV 5
    # returns the same x1/y1/x2/y2 values as (N, 4). Flatten only the wrapper
    # dimensions so the detector remains compatible with both versions.
    for raw in np.asarray(found).reshape(-1, 4):
        segment = [float(value) for value in raw]
        direction = _segment_direction(segment)
        if direction is None:
            continue
        midpoint = _midpoint(segment)
        for group in groups:
            aligned = _aligned_to(direction, group["direction"])
            if _direction_delta_deg(aligned, group["direction"]) > MERGE_ANGLE_DEG:
                continue
            if _perpendicular_distance(group["anchor"], group["direction"],
                                       midpoint) > MERGE_OFFSET_PX:
                continue
            group["segments"].append(segment)
            break
        else:
            groups.append({"direction": direction,
                           "anchor": midpoint,
                           "segments": [segment]})

    lines = [_fit_group(group["segments"]) for group in groups]
    return sorted(lines, key=lambda line: line.length_px, reverse=True)


def find_lines(response, min_length_px):
    """Hough segments from a response map, merged into whole lines."""
    _, binary = cv2.threshold(response, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    found = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=HOUGH_VOTES,
                            minLineLength=int(min_length_px),
                            maxLineGap=HOUGH_GAP)
    if found is None:
        return []
    return _merge_hough_segments(found)


# --- entity assignment -----------------------------------------------------
HORIZONTAL_MAX_DEG = 25.0   # steeper than this and a line is not a court-x line


def _horizontals(lines):
    return [line for line in lines if abs(line.angle_deg) <= HORIZONTAL_MAX_DEG]


def _diagonals(lines):
    return [line for line in lines if abs(line.angle_deg) > HORIZONTAL_MAX_DEG]


def assign_lines(paint_lines, edge_lines, frame_shape, mask=None, image=None):
    """Name detected lines as court entities (spec 2026-07-29 §3).

    Viewpoint-general: the front-wall stack is selected by its cross-ratio (a
    projective invariant of the four known line heights), and the floor is
    resolved by hypothesis — candidate short lines and steep floor lines take
    on court identities, each combination fits a floor homography from points
    AND infinite lines, and paint evidence picks the winner. No vertical
    frame ordering, no hue, no pose.

    `mask`/`image` sharpen the search (clutter rejection, faint-floor-paint
    recovery, paint-based hypothesis scoring); passing just `image` derives
    the mask. detect_court always passes both. Without them the choice falls
    back to geometric gates alone — enough for clean synthetic frames, not
    for real glass-court footage.

    Returns a dict whose values are DetectedLine or None. A None means that
    entity was not found; the caller decides whether that is fatal. Seams the
    hypothesis fit implies but no detected line supports are returned as
    constructed lines (their `support` is 1).
    """
    if mask is None and image is not None:
        mask = paint_mask(image)
    result = _analyse(paint_lines, edge_lines, frame_shape, mask, image)
    assigned = {name: None for name in
                ("out", "service", "tin", "front_seam",
                 "left_seam", "right_seam", "short_line", "half_court")}
    if result["status"] == "ok":
        for name in assigned:
            assigned[name] = result.get(name)
    return assigned


# --- viewpoint-general assignment (spec 2026-07-29) --------------------------
# The front wall's out line, service line, tin top, and the wall/floor seam
# are four parallel lines of one plane, so the cross-ratio of the four points
# they cut on ANY image transversal is this viewpoint-free constant.
_STACK_HEIGHTS_FT = (court_model.OUT_LINE_HEIGHT_FT,
                     court_model.SERVICE_LINE_HEIGHT_FT,
                     court_model.TIN_TOP_HEIGHT_FT,
                     0.0)


def _cross_ratio(a, b, c, d):
    return ((a - c) * (b - d)) / ((b - c) * (a - d))


_STACK_CROSS_RATIO = _cross_ratio(*_STACK_HEIGHTS_FT)
STACK_CR_TOL = 0.05          # |cross-ratio error| a quadruple may carry
# Longest horizontals entering the C(n,4) search. This must stay generous: an
# occlusion-fragmented out line can rank far down the length order (228 px on
# the Squash Zone frame, ~rank 36 of 46) and a cap that drops it silently
# forfeits the whole stack. The mutual-overlap gate prunes most combinations
# before any cross-ratio work, so n=48 stays well under a second.
STACK_MAX_CANDIDATES = 48
GAP_CONTAMINATION_MAX = 0.15  # paint fraction tolerated between stack lines
FLOOR_CHROMA_MIN = 3.5       # LAB a/b deviation; hazy-glass floor paint sits
                             # at 4-9 against a 0-3 background (measured)
FLOOR_HOUGH_VOTES = 40       # gentler than HOUGH_VOTES: floor paint is faint
FLOOR_HOUGH_GAP = 25
VPX_TOL_DEG = 2.5            # short-line candidates must aim at the stack VP
RANK_GAP_MIN = 5e-3          # below this the floor system is rank-deficient
SEAM_CORNER_TOL_PX = 25.0    # a side seam must pass this close to its corner
CORNER_REPRODUCTION_TOL_PX = 10.0  # the fit must give back its own corners

# Identities a steep floor line can take, in court-x order. The court values
# come from court_model (CLAUDE.md: datum values are never redefined here).
_Y_LINE_IDENTITIES = (
    ("left_seam", 0.0),
    ("left_box_inner", court_model.LEFT_BOX_INNER_CENTER_X_FT),
    ("half_court", court_model.HALF_COURT_X_FT),
    ("right_box_inner", court_model.RIGHT_BOX_INNER_CENTER_X_FT),
    ("right_seam", court_model.COURT_WIDTH_FT),
)

# Floor paint a correct homography must land on wherever it is visible, used
# to score competing floor hypotheses. Segments, in court feet.
_FLOOR_COVERAGE_SEGMENTS = (
    ((0.0, court_model.SHORT_LINE_CENTER_Y_FT),
     (court_model.COURT_WIDTH_FT, court_model.SHORT_LINE_CENTER_Y_FT)),
    ((court_model.HALF_COURT_X_FT, court_model.SHORT_LINE_CENTER_Y_FT),
     (court_model.HALF_COURT_X_FT, court_model.SHORT_LINE_CENTER_Y_FT + 9.0)),
    ((court_model.LEFT_BOX_INNER_CENTER_X_FT, court_model.SHORT_LINE_CENTER_Y_FT),
     (court_model.LEFT_BOX_INNER_CENTER_X_FT, court_model.BOX_BACK_CENTER_Y_FT)),
    ((court_model.RIGHT_BOX_INNER_CENTER_X_FT, court_model.SHORT_LINE_CENTER_Y_FT),
     (court_model.RIGHT_BOX_INNER_CENTER_X_FT, court_model.BOX_BACK_CENTER_Y_FT)),
    ((0.0, court_model.BOX_BACK_CENTER_Y_FT),
     (court_model.SERVICE_BOX_FT, court_model.BOX_BACK_CENTER_Y_FT)),
    ((court_model.COURT_WIDTH_FT - court_model.SERVICE_BOX_FT,
      court_model.BOX_BACK_CENTER_Y_FT),
     (court_model.COURT_WIDTH_FT, court_model.BOX_BACK_CENTER_Y_FT)),
)


def _hline(line):
    """Homogeneous (a, b, c) with a^2 + b^2 = 1 for a DetectedLine."""
    coeffs = np.cross([line.x1, line.y1, 1.0], [line.x2, line.y2, 1.0])
    return coeffs / math.hypot(coeffs[0], coeffs[1])


def _vanishing_point(lines):
    """Least-squares common point of the lines, homogeneous (may be at
    infinity — exactly parallel image lines are a legal pencil here)."""
    _, _, vt = np.linalg.svd(np.asarray([_hline(l) for l in lines]))
    return vt[-1]


def _passes_through(line, vp, tol_deg):
    """Does `line` pass through the (possibly infinite) point vp?

    Measured as the angle at the line's midpoint between its own direction
    and the direction toward vp, so the test stays meaningful for vanishing
    points at any distance including infinity.
    """
    mx, my = line.midpoint
    dx = vp[0] - mx * vp[2]
    dy = vp[1] - my * vp[2]
    reach = math.hypot(dx, dy)
    if reach < 1e-12:
        return True
    ux = (line.x2 - line.x1) / line.length_px
    uy = (line.y2 - line.y1) / line.length_px
    cross = abs(ux * dy / reach - uy * dx / reach)
    return math.degrees(math.asin(min(1.0, cross))) <= tol_deg


def _span(line):
    return (min(line.x1, line.x2), max(line.x1, line.x2))


def _gap_contamination(mask, lines, lo, hi):
    """Mean paint-mask density in the bands BETWEEN consecutive stack lines.

    A squash front wall is plain between its painted lines, so the true
    stack's gaps are nearly empty. Clutter clusters that pass the cross-ratio
    by chance (glass frit combs, fog-region fragments — both observed on the
    Squash Zone footage) have more of the same clutter inside their gaps.
    A 4 px margin off each line keeps the lines' own paint out of the count.
    """
    height, width = mask.shape[:2]
    total, hit = 0, 0
    for x in np.linspace(lo, hi, 40):
        ys = [line.y_at(x) for line in lines]
        if any(y is None for y in ys):
            continue
        for upper, lower in zip(ys, ys[1:]):
            for fraction in np.linspace(0.0, 1.0, 12):
                y = upper + 4 + (lower - upper - 8) * fraction
                if not (upper + 4 < y < lower - 4):
                    continue
                col, row = int(round(x)), int(round(y))
                if 0 <= row < height and 0 <= col < width:
                    total += 1
                    if mask[row, col]:
                        hit += 1
    return hit / total if total else 1.0


def _find_stacks(horizontals, mask=None, limit=12):
    """Candidate (out, service, tin, seam) quadruples, best first.

    Quadruples passing the cross-ratio test are checked cheapest-error first
    for clean gaps, and the `limit` best clean ones are all returned rather
    than only the winner: a chance cross-ratio match can outrank the real
    stack (measured: an occluder rectangle's edge posing as the service
    line), and the caller holds the evidence that settles it — the impostor
    has no consistent datum paint or floor beneath it, so trying candidates
    in order until the full pipeline succeeds picks the true stack without
    another scoring heuristic. Neither total length nor vertical extent
    discriminates on real clutter — the Squash Zone frit comb produces four
    ~1000 px collinear merges that beat the real stack on both — but its
    gaps are full of its own bars.
    """
    cands = sorted(horizontals, key=lambda l: l.length_px,
                   reverse=True)[:STACK_MAX_CANDIDATES]
    cands.sort(key=lambda l: l.midpoint[1])
    passers = []
    for combo in itertools.combinations(range(len(cands)), 4):
        lines = [cands[index] for index in combo]
        spans = [_span(l) for l in lines]
        lo = max(span[0] for span in spans)
        hi = min(span[1] for span in spans)
        if hi - lo < 60:
            continue
        errors = []
        ok = True
        for x in (lo + (hi - lo) * f for f in (0.15, 0.5, 0.85)):
            ys = [line.y_at(x) for line in lines]
            if (any(y is None for y in ys)
                    or any(ys[i] >= ys[i + 1] for i in range(3))):
                ok = False
                break
            errors.append(abs(_cross_ratio(*ys) - _STACK_CROSS_RATIO))
        if not ok:
            continue
        error = sum(errors) / len(errors)
        if error < STACK_CR_TOL:
            passers.append((error, lines, lo, hi))
    passers.sort(key=lambda passer: passer[0])
    # Contamination costs ~2 ms per quadruple, and a noisy frame can produce
    # thousands of chance cross-ratio passers (measured: 22 s on a
    # noise_sigma=40 render). If none of the 40 best-fitting quadruples has
    # clean gaps, a real stack is not in this frame; on real footage the true
    # stack ranks in the top handful (Squash Zone: top ten).
    stacks = []

    def duplicates_accepted(lines, lo, hi):
        """A stripe's two edges (or a re-merge of the same paint) produce
        near-identical quadruples; keeping variants of one stack would spend
        every retry slot on the same wrong answer."""
        columns = (lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75)
        for _, accepted in stacks:
            same = True
            for mine, theirs in zip(lines, accepted):
                for x in columns:
                    my_y, their_y = mine.y_at(x), theirs.y_at(x)
                    if my_y is None or their_y is None \
                            or abs(my_y - their_y) > 12.0:
                        same = False
                        break
                if not same:
                    break
            if same:
                return True
        return False

    for error, lines, lo, hi in passers[:40]:
        if duplicates_accepted(lines, lo, hi):
            continue
        if mask is None or (_gap_contamination(mask, lines, lo, hi)
                            <= GAP_CONTAMINATION_MAX):
            stacks.append((error, lines))
            if len(stacks) >= limit:
                break
    return stacks


def _floor_chroma_mask(bgr, seam, margin_px=10):
    """Binary mask of chromatic floor paint below the front seam.

    Floor paint seen through hazy glass keeps almost no darkness contrast but
    keeps a consistent chromatic offset from the wood (measured 4-9 LAB units
    on Squash Zone against 0-3 background noise). The measure is hue-agnostic —
    |a*-dev| + |b*-dev| fires on red and navy alike — and the white glass
    frit is achromatic, so it vanishes here even though it dominates the
    luminance map. A fixed low threshold on the smoothed deviation is used
    instead of Otsu: the map is almost entirely zeros, which makes Otsu
    collapse to whatever clutter is brightest (measured: it kept only a blue
    logo sticker and lost every court line). Everything above the seam is
    zeroed so wall paint cannot dominate.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    deviation = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in (1, 2):
        plane = lab[:, :, channel]
        deviation += np.abs(plane.astype(np.float32)
                            - cv2.medianBlur(plane, 31).astype(np.float32))
    deviation = cv2.GaussianBlur(deviation, (5, 5), 1.5)
    height, width = deviation.shape
    rows = np.arange(height, dtype=float)[:, None]
    seam_rows = np.array([[seam.y_at(x) if seam.y_at(x) is not None else 0.0
                           for x in range(width)]])
    deviation[rows < seam_rows + margin_px] = 0.0
    return np.where(deviation >= FLOOR_CHROMA_MIN, np.uint8(255), np.uint8(0))


def _find_floor_lines(binary, min_length_px):
    """find_lines' Hough+merge, tuned for faint fragmented floor paint.

    Merged lines with three or more near-parallel neighbours within a court
    line's width are dropped: real floor paint is isolated (no two court
    lines run parallel closer than ~1.4 ft), while decorative combs and
    textured bands merge into dense parallel stacks that would otherwise
    flood the short-line candidate pool — and, worse, offer a wrong
    hypothesis a bed of false paint to score against.
    """
    found = cv2.HoughLinesP(binary, 1, np.pi / 360,
                            threshold=FLOOR_HOUGH_VOTES,
                            minLineLength=int(min_length_px),
                            maxLineGap=FLOOR_HOUGH_GAP)
    if found is None:
        return []
    lines = _merge_hough_segments(found)
    kept = []
    for line in lines:
        direction = _segment_direction((line.x1, line.y1, line.x2, line.y2))
        neighbours = 0
        for other in lines:
            if other is line:
                continue
            if _direction_delta_deg(
                    _aligned_to(_segment_direction(
                        (other.x1, other.y1, other.x2, other.y2)), direction),
                    direction) > 3.0:
                continue
            if _perpendicular_distance((line.x1, line.y1), direction,
                                       other.midpoint) <= 16.0:
                neighbours += 1
        if neighbours < 3:
            kept.append(line)
    return kept


def _fit_floor(corners, seam, short, named_lines):
    """One floor-homography hypothesis via the mixed point/line DLT.

    Line correspondences carry no endpoint information, so a court line whose
    middle (or either end) is occluded constrains the fit exactly as a fully
    visible one does — the straight-line fill the product promises.
    """
    points = [((0.0, 0.0), corners["bottom_left"]),
              ((court_model.COURT_WIDTH_FT, 0.0), corners["bottom_right"])]
    lines = [((0.0, 1.0, 0.0), _hline(seam)),
             ((0.0, 1.0, -court_model.SHORT_LINE_CENTER_Y_FT), _hline(short))]
    for court_x, line in named_lines:
        lines.append(((1.0, 0.0, -court_x), _hline(line)))
    return court_model.fit_homography_mixed(points, lines)


def _analyse(paint_lines, edge_lines, frame_shape, mask=None, image=None):
    """Name the court's structures from merged lines (spec 2026-07-29 §3).

    Returns a dict. On failure it carries only {"status": "no_stack" |
    "no_floor"}; on success {"status": "ok"} plus the named DetectedLines
    (out/service/tin/front_seam/short_line, left_seam/right_seam/half_court
    where known), wall corners, edge columns, the floor homography, and the
    chroma floor mask when an image was supplied.
    """
    horizontals = _horizontals(edge_lines) + _horizontals(paint_lines)
    steeps = _diagonals(edge_lines) + _diagonals(paint_lines)

    # Candidate stacks are tried best-first until one carries a full court:
    # a chance cross-ratio impostor has no datum paint or floor beneath it,
    # so the attempt fails fast and the next candidate gets its turn.
    best_failure = {"status": "no_stack"}
    for _, (out, service, tin, seam) in _find_stacks(horizontals, mask):
        result = _resolve_with_stack(out, service, tin, seam, horizontals,
                                     steeps, frame_shape, mask, image)
        if result["status"] == "ok":
            return result
        if result["status"] == "no_floor":
            best_failure = result
    return best_failure


def _resolve_with_stack(out, service, tin, seam, horizontals, steeps,
                        frame_shape, mask, image):
    """One full pipeline attempt for one candidate stack (see _find_stacks)."""
    height, width = frame_shape[:2]

    # Wall edge columns — where the paint physically stops. The coarse
    # estimate is the raw reach of the near-full stack members: a fragmented
    # member (an occlusion-shortened out line) must not drag the edge inward,
    # and an EXTENDED search must not run past the corner — on all-glass
    # courts the panes reflect the wall paint at its own image heights, so a
    # refit allowed to roam leftward happily follows the mirror image
    # (measured: 60 px of bleed on Squash Zone). The datum refits, bounded by
    # the coarse window, can then only pull the edge inward onto real paint;
    # their median start/end tolerates one occlusion-truncated member.
    longest = max(l.length_px for l in (out, service, tin, seam))
    full = [l for l in (out, service, tin, seam)
            if l.length_px >= 0.7 * longest]
    x_left = min(_span(l)[0] for l in full)
    x_right = max(_span(l)[1] for l in full)
    refits = None
    if mask is not None:
        refits = {}
        spans = []
        for key, line, mode in (("out", out, "max"),
                                ("service", service, "min"),
                                ("tin", tin, "min")):
            fit = refit_to_datum(mask, line, mode, x_range=(x_left, x_right))
            if fit is None:
                return {"status": "no_stack"}
            refits[key] = fit
            spans.append(fit.x_span)
        x_left = sorted(span[0] for span in spans)[1]
        x_right = sorted(span[1] for span in spans)[1]

    # Corner verticals (the wall's own side edges) override an extent vote.
    wall_mid_y = (out.midpoint[1] + seam.midpoint[1]) / 2.0
    for line in steeps:
        if abs(abs(line.angle_deg) - 90.0) > 20.0:
            continue
        if not (min(line.y1, line.y2) < wall_mid_y < max(line.y1, line.y2)):
            continue
        x_at = line.midpoint[0]
        if abs(x_at - x_left) < 25.0:
            x_left = x_at
        elif abs(x_at - x_right) < 25.0:
            x_right = x_at

    # Final datum refits over the settled wall span — these are the payload's
    # line fits, so their spans must state the wall's reach, not whatever the
    # coarse search window was.
    if refits is not None:
        for key, line, mode in (("out", out, "max"),
                                ("service", service, "min"),
                                ("tin", tin, "min")):
            fit = refit_to_datum(mask, line, mode, x_range=(x_left, x_right))
            if fit is None:
                return {"status": "no_stack"}
            refits[key] = fit

    out_for_corners = out
    if refits is not None:
        out_for_corners = _fit_as_line(refits["out"])
    corner_ys = [out_for_corners.y_at(x_left), out_for_corners.y_at(x_right),
                 seam.y_at(x_left), seam.y_at(x_right)]
    if any(y is None for y in corner_ys):
        return {"status": "no_stack"}
    corners = {
        "top_left": (x_left, corner_ys[0]),
        "top_right": (x_right, corner_ys[1]),
        "bottom_left": (x_left, corner_ys[2]),
        "bottom_right": (x_right, corner_ys[3]),
    }

    # Faint chroma floor paint below the seam (hazy-glass viewpoints).
    floor_mask = None
    floor_lines = []
    if image is not None:
        floor_mask = _floor_chroma_mask(image, seam)
        floor_lines = _find_floor_lines(floor_mask, width * 0.05)
        horizontals = horizontals + _horizontals(floor_lines)
        steeps = steeps + _diagonals(floor_lines)

    vpx = _vanishing_point([out, service, tin, seam])
    stack_ids = {id(l) for l in (out, service, tin, seam)}
    centre_column = (x_left + x_right) / 2.0
    seam_y_mid = seam.y_at(centre_column)

    shorts = []
    for line in horizontals:
        if id(line) in stack_ids:
            continue
        seam_here = seam.y_at(line.midpoint[0])
        if seam_here is None or line.midpoint[1] < seam_here + 5:
            continue
        if not _passes_through(line, vpx, VPX_TOL_DEG):
            continue
        shorts.append(line)

    y_cands = []
    for line in steeps:
        # Court-y floor lines exist only below the seam; anything whose upper
        # end reaches meaningfully above it (glass pane edges, wall corner
        # verticals — all full-height) is not floor paint.
        top_x, top_y = ((line.x1, line.y1) if line.y1 <= line.y2
                        else (line.x2, line.y2))
        seam_at_top = seam.y_at(top_x)
        if seam_at_top is not None and top_y < seam_at_top - 15:
            continue
        if seam_y_mid is not None and line.midpoint[1] < seam_y_mid - 5:
            continue
        y_cands.append(line)

    distance = None
    if floor_mask is not None:
        distance = cv2.distanceTransform(255 - floor_mask, cv2.DIST_L2, 3)

    def hypothesis_score(homography):
        """(covered_fraction, verified_checks, -mean_check_residual) against
        the CHROMA mask only: floor court paint is chromatic, while the
        dominant floor clutter (white frit, reflections) is not. Scoring
        against the luminance mask let a wrong short-line hypothesis "cover"
        itself with frit bars (measured)."""
        if distance is None:
            return (0.0, 0, 0.0)
        inside, hit = 0, 0
        for (ax, ay), (bx, by) in _FLOOR_COVERAGE_SEGMENTS:
            for fraction in np.linspace(0.0, 1.0, 24):
                try:
                    px, py = court_model.apply_homography(
                        homography,
                        (ax + (bx - ax) * fraction, ay + (by - ay) * fraction))
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                if not (0 <= int(py) < height and 0 <= int(px) < width):
                    continue
                inside += 1
                if float(distance[int(py), int(px)]) <= 3.0:
                    hit += 1
        covered = hit / inside if inside >= 24 else 0.0
        verified, residuals = 0, []
        for _, _, court_ft, independent in _FLOOR_CHECKS:
            if not independent:
                continue
            try:
                px, py = court_model.apply_homography(homography, court_ft)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not (0 <= int(py) < height and 0 <= int(px) < width):
                continue
            residual = float(distance[int(py), int(px)])
            residuals.append(residual)
            if residual <= CHECK_OK_PX:
                verified += 1
        mean_residual = (sum(residuals) / len(residuals)) if residuals else 0.0
        return (covered, verified, -mean_residual)

    def near_corner(line, corner):
        coeffs = _hline(line)
        return abs(coeffs[0] * corner[0] + coeffs[1] * corner[1]
                   + coeffs[2]) <= SEAM_CORNER_TOL_PX

    hypotheses = []
    for short in shorts:
        crossings = []
        for line in y_cands:
            crossing = short.intersect(line)
            if crossing is None:
                continue
            if not (-width * 0.5 <= crossing[0] <= width * 1.5):
                continue
            crossings.append((crossing[0], line))
        crossings.sort(key=lambda item: item[0])
        crossings = crossings[:8]

        # Identity assignments must be in increasing court-x order, seams must
        # pass near their corner, and non-seams must not.
        options = [[]]
        for _, line in crossings:
            grown = []
            near_left = near_corner(line, corners["bottom_left"])
            near_right = near_corner(line, corners["bottom_right"])
            for option in options:
                grown.append(option)
                last = max((index for index, _ in option), default=-1)
                for index in range(last + 1, len(_Y_LINE_IDENTITIES)):
                    name = _Y_LINE_IDENTITIES[index][0]
                    if name == "left_seam" and not near_left:
                        continue
                    if name == "right_seam" and not near_right:
                        continue
                    if name not in ("left_seam", "right_seam") and (
                            near_left or near_right):
                        continue
                    grown.append(option + [(index, line)])
            options = grown

        for option in options:
            if not option:
                continue
            named = [(_Y_LINE_IDENTITIES[index][1], line)
                     for index, line in option]
            homography, rank_gap = _fit_floor(corners, seam, short, named)
            if rank_gap < RANK_GAP_MIN:
                continue
            try:
                front_left = court_model.apply_homography(homography, (0.0, 0.0))
                front_right = court_model.apply_homography(
                    homography, (court_model.COURT_WIDTH_FT, 0.0))
                short_left = court_model.apply_homography(
                    homography, (0.0, court_model.SHORT_LINE_CENTER_Y_FT))
                short_right = court_model.apply_homography(
                    homography, (court_model.COURT_WIDTH_FT,
                                 court_model.SHORT_LINE_CENTER_Y_FT))
            except (ValueError, np.linalg.LinAlgError):
                continue
            anchors = (front_left, front_right, short_left, short_right)
            if not all(math.isfinite(v) for point in anchors for v in point):
                continue
            # Floor orientation sanity for a camera behind the court: the
            # short line sits below the seam and at least as wide.
            if not (short_left[1] > front_left[1]
                    and short_right[1] > front_right[1]):
                continue
            if not (short_left[0] < short_right[0]
                    and front_left[0] < front_right[0]):
                continue
            if not (short_left[0] < front_left[0] + 5
                    and short_right[0] > front_right[0] - 5):
                continue
            if math.hypot(short_right[0] - short_left[0],
                          short_right[1] - short_left[1]) < 0.7 * math.hypot(
                    front_right[0] - front_left[0],
                    front_right[1] - front_left[1]):
                continue
            # The least-squares fit must reproduce its own corner inputs — a
            # compromise solution that drags them is a wrong hypothesis.
            reproduction = max(
                math.hypot(front_left[0] - corners["bottom_left"][0],
                           front_left[1] - corners["bottom_left"][1]),
                math.hypot(front_right[0] - corners["bottom_right"][0],
                           front_right[1] - corners["bottom_right"][1]))
            if reproduction > CORNER_REPRODUCTION_TOL_PX:
                continue
            seams_used = sum(1 for index, _ in option
                             if _Y_LINE_IDENTITIES[index][0]
                             in ("left_seam", "right_seam"))
            hypotheses.append((hypothesis_score(homography), seams_used,
                               len(option), homography, short, option))

    if not hypotheses:
        return {"status": "no_floor"}
    hypotheses.sort(key=lambda h: (h[0], h[1], h[2]), reverse=True)
    score, _, _, homography, short, option = hypotheses[0]

    named = {_Y_LINE_IDENTITIES[index][0]: line for index, line in option}
    for name, court_x in (("left_seam", 0.0),
                          ("right_seam", court_model.COURT_WIDTH_FT)):
        if name not in named:
            near = court_model.apply_homography(homography, (court_x, 1.0))
            far = court_model.apply_homography(
                homography, (court_x, court_model.SHORT_LINE_CENTER_Y_FT))
            named[name] = DetectedLine(near[0], near[1], far[0], far[1],
                                       support=1)

    return {
        "status": "ok",
        "out": out, "service": service, "tin": tin, "front_seam": seam,
        "short_line": short,
        "left_seam": named.get("left_seam"),
        "right_seam": named.get("right_seam"),
        "half_court": named.get("half_court"),
        "corners": corners,
        "x_left": x_left, "x_right": x_right,
        "refits": refits,
        "floor_homography": homography,
        "floor_mask": floor_mask,
        "floor_score": score,
    }


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


def refit_to_datum(mask, line, mode, x_range=None):
    """Refit `line` onto the named boundary of the painted stripe under it.

    `mode` is 'min' (topmost mask row per column) or 'max' (lowest) — the same
    contract as index.html's extractEdge, because front-wall lines are stored
    by EDGE, not centreline (spec §5). Returning the stripe's middle instead
    would put a systematic half-line-width bias into the out-line call.

    `x_range` widens (or narrows) the columns searched beyond the line's own
    detected span. Court lines are straight, so a line whose middle or far end
    was occluded still has a valid extension — the refit follows it and simply
    collects whatever mask support exists along the way, which is how a
    detection recovers a line's full reach from a fragment.
    """
    return _refit(mask, line, mode, x_range)


def _refit(mask, line, mode, x_range=None):
    height, width = mask.shape[:2]
    lo, hi = (min(line.x1, line.x2), max(line.x1, line.x2)) \
        if x_range is None else x_range
    first = int(max(0, math.floor(lo)))
    last = int(min(width - 1, math.ceil(hi)))
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
    analysed = _analyse(find_lines(line_response(image), minimum),
                        find_lines(edge_response(image), minimum),
                        image.shape, mask, image)

    # Everything below is REQUIRED, not optional. A calibration missing a
    # front-wall line is unusable by construction: judge_call.load_calibration_
    # lines rejects any calibration missing tin_top_edge, and index.html's
    # buildJson() dereferences S.lines[k].fit for all three -- so a "successful"
    # detection missing one presents as green on the confirm screen and then
    # throws a TypeError two screens later at TRACK BALL. Failing here instead
    # drops the player into the manual wizard, which is the working fallback.
    if analysed["status"] == "no_stack":
        return _failure(
            "insufficient_lines", image.shape,
            "Could not find the front wall's out line, service line, tin, "
            "and floor line together.", warnings)
    if analysed["status"] == "no_floor":
        return _failure(
            "insufficient_lines", image.shape,
            "Could not find enough of the floor's lines to place the court.",
            warnings)

    # Front-wall lines, each pulled onto its own stored datum (spec §5).
    # _analyse already refit them across the wall's full width: the lines are
    # straight, so a fragment left by occlusion or glare still fixes the line
    # and the refit collected whatever paint existed along its extension (the
    # occlusion contract, spec 2026-07-29 §4).
    x_left, x_right = analysed["x_left"], analysed["x_right"]
    lines = [_line_payload(name, analysed["refits"][key], "detected")
             for name, key in (("out_line_lower_edge", "out"),
                               ("service_line_top_edge", "service"),
                               ("tin_top_edge", "tin"))]
    out_fit = lines[0]

    # The wall-top corners are NOT out_line.intersect(side seams): the out
    # line sits on the 3-D line {y=0, z=OUT_LINE_HEIGHT_FT}, the seams on
    # {x=0 or COURT_WIDTH_FT, z=0} -- two lines at different heights whose
    # projections only cross far from the true corner (measured 35-500+ px
    # off). The corner IS where the wall's paint stops: the datum-refit out
    # line evaluated at the wall's edge columns, which _analyse votes from
    # every stack member's paint reach so one occluded line cannot drag them.
    corners = analysed["corners"]
    anchors = {
        "top_left": (x_left, out_fit["slope"] * x_left + out_fit["intercept"]),
        "top_right": (x_right,
                      out_fit["slope"] * x_right + out_fit["intercept"]),
        "bottom_left": corners["bottom_left"],
        "bottom_right": corners["bottom_right"],
    }

    # The floor homography comes from _analyse's mixed point/line fit (seam
    # corners + seam, short line, and whichever steep floor lines were
    # identified -- side seams when visible, the half-court and service-box
    # lines when not). The four anchor landmarks the payload stores are its
    # projections, so an anchor is defined even where fog or occlusion hid
    # the paint its manual-wizard equivalent would have needed.
    homography = analysed["floor_homography"]
    try:
        floor_pixels = {
            name: court_model.apply_homography(homography, court_ft)
            for name, court_ft in _FLOOR_ANCHORS
        }
    except (ValueError, np.linalg.LinAlgError):
        return _failure("insufficient_lines", image.shape,
                        "The court lines found do not describe a flat court "
                        "floor.", warnings)

    # Self-verification: how far predictions of UNUSED court points fall from
    # real paint. distanceTransform turns that into a lookup. Faint floor
    # paint (hazy glass) often misses the luminance mask, so the chroma floor
    # mask is OR-ed in before measuring.
    check_mask = mask
    if analysed.get("floor_mask") is not None:
        check_mask = np.maximum(mask, analysed["floor_mask"])
    distance = cv2.distanceTransform(255 - check_mask, cv2.DIST_L2, 3)
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
