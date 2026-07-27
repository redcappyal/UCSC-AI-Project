"""Detection tests for court_detect, scored against a synthetic court.

The synthetic court is rendered from a known CameraModel, so every assertion
compares the detector's answer to the camera that produced the image. Paint
bands carry real WSF width (50 mm), which is what makes the datum rules in
docs/superpowers/specs/2026-07-27-auto-court-detection-design.md §5 testable:
returning a stripe's centre instead of its named edge is a half-line-width
bias, and it must fail here rather than in a match.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import court_detect
from court_model import LEFT_BOX_INNER_CENTER_X_FT, LINE_WIDTH_FT
from synthetic_court import FLOOR_BGR, LINE_BGR, WALL_BGR, court_camera, render_court

REPO_ROOT = Path(__file__).resolve().parents[1]
SQUASH_ANALYTICS_MP4 = REPO_ROOT / "SquashAnalytics.mp4"
BAY_CLUB_MOV = Path("/Users/Ian2/Desktop/Training Data/Bay Club Squash 1 Rally+audio.mov")


def test_render_court_paints_lines_where_the_camera_projects_them():
    camera = court_camera()
    image, truth = render_court(camera)

    assert image.shape == (1080, 1920, 3)

    # The out line's stored datum is its LOWER edge, so the pixel just ABOVE
    # that datum is paint and the pixel well above it is bare wall.
    (x1, y1), (x2, y2) = truth["out_line_lower_edge"]
    mid_x, mid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    assert np.allclose(image[mid_y - 1, mid_x], LINE_BGR, atol=2)
    assert np.allclose(image[mid_y - 40, mid_x], WALL_BGR, atol=2)

    # Below the front-wall/floor seam is floor, above it is wall.
    (sx1, sy1), (sx2, sy2) = truth["front_seam"]
    seam_x, seam_y = int(round((sx1 + sx2) / 2)), int(round((sy1 + sy2) / 2))
    assert np.allclose(image[seam_y + 25, seam_x], FLOOR_BGR, atol=2)
    assert np.allclose(image[seam_y - 25, seam_x], WALL_BGR, atol=2)


def test_service_box_lines_straddle_their_centreline_not_their_edge():
    """The left service box's inner side-line band must straddle
    LEFT_BOX_INNER_CENTER_X_FT, not the raw SERVICE_BOX_FT edge datum.

    SERVICE_BOX_FT is a WSF EDGE datum (the paint's interior-facing
    boundary, court_model.py:27-35), not a centreline. An earlier version of
    the renderer straddled that raw datum, which draws the whole band a half
    line-width (~25 mm) toward the box interior -- silently disagreeing with
    the *_CENTER_* constants that Task 5 scores its self-verification
    against. Neither existing render test would have caught this, since both
    only check `out_line_lower_edge` and `front_seam`.
    """
    # court_camera()'s default framing (focal_px=1600) is narrow enough that
    # y >= ~10.5 ft projects below row 1080 -- it was tuned in Task 1 only to
    # keep the out line and front seam in frame. The service box (y in
    # [18.0, 23.25]) needs a wider FOV to be visible at all, so this test
    # overrides focal_px; every other test in the suite still uses the
    # canonical framing.
    camera = court_camera(focal_px=500.0)
    image, _truth = render_court(camera)

    # A y comfortably inside [SHORT_LINE_FROM_FRONT_FT, SERVICE_BOX_BACK_FT]
    # so the sample point sits on the box's inner side line, away from its
    # front/back ends.
    y_test = 20.0
    quarter_width = LINE_WIDTH_FT / 4.0

    def pixel_at(x_ft):
        px, py = camera.project((x_ft, y_test, 0.0))
        return image[int(round(py)), int(round(px))]

    # Correct band: [CENTER - half, CENTER + half]. Paint must appear both
    # at the centreline and on either side of it, within the band.
    assert np.allclose(pixel_at(LEFT_BOX_INNER_CENTER_X_FT), LINE_BGR, atol=2)
    assert np.allclose(
        pixel_at(LEFT_BOX_INNER_CENTER_X_FT - quarter_width), LINE_BGR, atol=2)

    # The buggy version straddled SERVICE_BOX_FT instead, whose band ends
    # exactly at LEFT_BOX_INNER_CENTER_X_FT (SERVICE_BOX_FT + half). A point
    # a quarter line-width further out is inside the correct band but past
    # the end of the buggy one, so it is bare floor under the bug and paint
    # once fixed.
    assert np.allclose(
        pixel_at(LEFT_BOX_INNER_CENTER_X_FT + quarter_width), LINE_BGR, atol=2)


def _frames_with_moving_player(base, count=5):
    """Static court, one dark rectangle that moves — the real occlusion case."""
    frames = []
    for index in range(count):
        frame = base.copy()
        left = 300 + index * 180
        frame[500:900, left:left + 150] = (30, 30, 35)
        frames.append(frame)
    return frames


def test_median_frame_erases_a_moving_player():
    camera = court_camera()
    base, _ = render_court(camera)
    frames = _frames_with_moving_player(base)
    median, moved = court_detect.median_frame(frames)

    assert moved is False

    # A bare "brightness floor" check over this crop is not specific enough:
    # LINE_BGR = (90, 45, 30) paint sits in this region and its minimum
    # channel (30) is identical to the player patch's minimum channel (30 in
    # (30, 30, 35)), so `.min() > threshold` would fail on a clean render for
    # a reason that has nothing to do with the player. Instead compare the
    # swept crop directly to the same crop of the untouched, player-free
    # render (deterministic: render_court has no randomness at noise_sigma=0).
    # Each column is covered by the player in exactly one of the five
    # non-overlapping sweeps, so the per-pixel temporal median always picks
    # one of the four untouched samples and should reproduce the clean crop
    # exactly.
    crop_median = median[500:900, 300:1350]
    crop_clean = base[500:900, 300:1350]
    assert np.array_equal(crop_median, crop_clean)


def test_median_frame_flags_a_panning_camera():
    camera = court_camera()

    # A bare noise-free render will not do here: it is two huge flat-colour
    # slabs (wall, floor) plus a few thin lines, and most full-width rows of
    # the wall slab are one uniform colour end to end. np.roll along either
    # axis maps those uniform runs onto themselves, so no amount of shift
    # ever moves more than ~15% of pixels past MOTION_DELTA -- the fixture
    # would fail to prove a real pan regardless of the implementation under
    # test. noise_sigma is NOT a stand-in for real sensor/lens noise -- it is
    # fixed per-pixel texture painted onto the render, which np.roll then
    # displaces, so the fraction of changed pixels reflects the roll instead
    # of the render's artificial flatness. This only proves the fixture can
    # cross MOTION_FRACTION at a noise level (40.0) 20x this suite's normal
    # 2.0; it says nothing about whether 0.25 is the right threshold for real
    # footage. That validation lives in
    # test_median_frame_flags_a_real_pan_on_squashanalytics and
    # test_median_frame_does_not_flag_the_bay_club_fixed_mount below, which
    # measure actual video and bracket 0.25 with wide margins on both sides.
    base, _ = render_court(camera, noise_sigma=40.0, seed=0)
    frames = [np.roll(base, shift * 90, axis=1) for shift in range(5)]

    _, moved = court_detect.median_frame(frames)

    assert moved is True


def _sample_frames(path, count=5):
    """Seek `count` frames spread evenly across a video, cv2.VideoCapture-only.

    Mirrors the real caller (design spec §7: "the client posts 5 JPEG
    frames, sampled across the clip using the seek machinery"). Only the
    `count` target frames are decoded -- never the whole video.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fracs = [i / (count - 1) for i in range(count)]
        frames = []
        for frac in fracs:
            target = min(int(round(frac * (total - 1))), total - 1)
            # Some containers (.mov) fail to decode the exact last frame via
            # POS_FRAMES seeking; step back a few frames until one decodes.
            ok = False
            for backoff in range(10):
                probe = max(target - backoff, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
                ok, frame = cap.read()
                if ok:
                    break
            if not ok:
                raise RuntimeError(f"failed to read a frame near {target} in {path}")
            frames.append(frame)
        return frames
    finally:
        cap.release()


def test_median_frame_flags_a_real_pan_on_squashanalytics():
    """SquashAnalytics.mp4 (in repo) genuinely pans -- this is the real-frame
    smoke test promised by the auto-court-detection design spec §11.

    Measured 2026-07-27 with `.venv/bin/python`, sampling 5 frames spread
    evenly across the clip's 18713 frames (indices 0, 4678, 9356, 14034,
    18712): mean changed-pixel fraction 0.63, comfortably above
    MOTION_FRACTION (0.25). A second sampling at fractions
    [0.1, 0.3, 0.5, 0.7, 0.9] measured 0.31 -- still clear of the threshold,
    so the result isn't an artifact of which frames get picked.
    """
    frames = _sample_frames(SQUASH_ANALYTICS_MP4)

    _, moved = court_detect.median_frame(frames)

    assert moved is True


@pytest.mark.skipif(
    not BAY_CLUB_MOV.exists(),
    reason="Bay Club footage lives outside the repo (real fixed-mount test asset)",
)
def test_median_frame_does_not_flag_the_bay_club_fixed_mount():
    """The false-positive side of the guard: real footage from the product's
    actual fixed fin mount must not be flagged as a pan.

    Measured 2026-07-27, same 5-frames-evenly-spread sampling as the
    SquashAnalytics test above (indices 0, 127, 254, 381, 502 of 509 total
    frames): mean changed-pixel fraction 0.12, comfortably below
    MOTION_FRACTION (0.25). A second sampling at fractions
    [0.1, 0.3, 0.5, 0.7, 0.9] measured 0.10 -- consistently well clear of the
    threshold. Together with the SquashAnalytics measurement above, 0.25
    separates real panning from a real fixed mount by a wide margin on both
    sides (~0.13-0.15 below across samplings, ~0.06-0.38 above).
    """
    frames = _sample_frames(BAY_CLUB_MOV)

    _, moved = court_detect.median_frame(frames)

    assert moved is False


def _nearest_line_to(lines, point_a, point_b):
    """The detected line whose ends best match a truth segment's ends."""
    def error(line):
        forward = (np.hypot(line.x1 - point_a[0], line.y1 - point_a[1])
                   + np.hypot(line.x2 - point_b[0], line.y2 - point_b[1]))
        backward = (np.hypot(line.x1 - point_b[0], line.y1 - point_b[1])
                    + np.hypot(line.x2 - point_a[0], line.y2 - point_a[1]))
        return min(forward, backward)
    return min(lines, key=error)


def test_find_lines_recovers_the_front_wall_paint():
    camera = court_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.line_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    for name in ("out_line_lower_edge", "service_line_top_edge", "tin_top_edge"):
        start, end = truth[name]
        found = _nearest_line_to(lines, start, end)
        # The stripe is ~LINE_WIDTH_FT wide, so a line fitted to the whole
        # stripe may sit up to half a width off the named datum. Half
        # LINE_WIDTH_FT at this camera scale measures ~4.4 px, and the three
        # named lines measure 3.1-4.1 px here -- Task 5 pulls it onto the
        # exact edge; here we only require the right stripe.
        assert abs(found.y_at(start[0]) - start[1]) < 6, name


def test_edge_response_finds_the_wall_floor_seams():
    camera = court_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.edge_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    start, end = truth["front_seam"]
    found = _nearest_line_to(lines, start, end)
    assert abs(found.y_at(start[0]) - start[1]) < 8


def test_normal_form_agrees_for_a_near_vertical_line_regardless_of_traversal():
    """Regression test for a wrap-discontinuity bug in `_normal_form`: it used
    to wrap the ANGLE into (-90, 90] and then derive the perpendicular offset
    from that wrapped angle via (-sin, cos) -- a direction that flips sign
    right at the wrap. Two fragments of one near-vertical line, traversed in
    opposite directions, land on opposite sides of it: one measures at
    +89.97 degrees, the other at -89.97 degrees, and their offsets come out
    negated (-499.94 vs +500.46, |diff| ~= 1000.4) even though both lie on
    the same real line. The fix canonicalises the direction before taking
    the normal, so both fragments share one normal and their offsets agree.
    """
    seg_a = (500.0, 100.0, 500.4, 800.0)  # tiny +0.4px tilt
    seg_b = (500.4, 100.0, 500.0, 800.0)  # same real line, opposite traversal

    angle_a, offset_a = court_detect._normal_form(seg_a)
    angle_b, offset_b = court_detect._normal_form(seg_b)

    assert court_detect._angle_delta(angle_a, angle_b) <= court_detect.MERGE_ANGLE_DEG
    assert abs(offset_a - offset_b) <= court_detect.MERGE_OFFSET_PX


def test_find_lines_merges_a_broken_near_vertical_line_into_one():
    """find_lines-level regression test for the same wrap bug: the half-court
    line and service-box side lines are near-vertical court features that get
    interrupted by players and glare like any other line. Real HoughLinesP
    output for a near-vertical line already exhibits the +90/-90 sign flip
    between different fragments -- verified empirically here with nothing
    more than a 4 px total tilt over a 700 px run -- so under the old
    `_normal_form` this broken line under-merges into two DetectedLines
    instead of one.
    """
    height, width = 900, 400
    y_top, y_bottom = 100, 800
    x_top, x_bottom = 200.0, 196.0
    gaps = ((300, 340), (500, 540))

    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(y_top, y_bottom):
        if any(lo <= y < hi for lo, hi in gaps):
            continue
        fraction = (y - y_top) / (y_bottom - y_top)
        x = int(round(x_top + (x_bottom - x_top) * fraction))
        cv2.line(image, (x, y), (x, y), 255, thickness=1)

    lines = court_detect.find_lines(image, min_length_px=width * 0.10)

    assert len(lines) == 1
    (line,) = lines
    # Spans most of the drawn extent (measured ~678 px of the 700 px run)
    # despite the two gaps, not just one fragment's worth (~160 px).
    assert abs(line.y1 - line.y2) > 600


def test_detected_line_intersect_returns_none_for_parallels():
    first = court_detect.DetectedLine(0.0, 0.0, 100.0, 0.0, support=1)
    second = court_detect.DetectedLine(0.0, 50.0, 100.0, 50.0, support=1)
    assert first.intersect(second) is None

    crossing = court_detect.DetectedLine(50.0, -50.0, 50.0, 50.0, support=1)
    point = first.intersect(crossing)
    assert point is not None and abs(point[0] - 50.0) < 1e-6
