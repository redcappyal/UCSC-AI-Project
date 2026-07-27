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


def _angle_delta(first, second):
    """Smallest angle between two orientations, accounting for the ±90 wrap."""
    delta = abs(first - second) % 180.0
    return min(delta, 180.0 - delta)


def _normal_form(segment):
    """(orientation, signed perpendicular offset from the origin).

    The direction is canonicalised into the upper half-plane before the normal
    is taken. Wrapping the ANGLE alone is not enough: two fragments of one
    near-vertical line can be measured at +89.97 and -89.97 degrees, whose
    normals point in opposite directions, so their offsets come out negated
    (-500 vs +500) and the merge rejects them as 1000 px apart. Canonicalising
    the direction keeps one line's fragments sharing one normal.
    """
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, float(y1)
    ux, uy = dx / length, dy / length
    if uy < 0 or (uy == 0.0 and ux < 0):
        ux, uy = -ux, -uy
    return math.degrees(math.atan2(uy, ux)), -uy * x1 + ux * y1


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
    for segment in found[:, 0, :]:
        angle, offset = _normal_form([float(value) for value in segment])
        for group in groups:
            if (_angle_delta(angle, group["angle"]) <= MERGE_ANGLE_DEG
                    and abs(offset - group["offset"]) <= MERGE_OFFSET_PX):
                group["segments"].append([float(value) for value in segment])
                break
        else:
            groups.append({"angle": angle, "offset": offset,
                           "segments": [[float(value) for value in segment]]})

    lines = [_fit_group(group["segments"]) for group in groups]
    return sorted(lines, key=lambda line: line.length_px, reverse=True)
