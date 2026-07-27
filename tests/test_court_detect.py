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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import court_detect
from court_model import LEFT_BOX_INNER_CENTER_X_FT, LINE_WIDTH_FT
from synthetic_court import FLOOR_BGR, LINE_BGR, WALL_BGR, court_camera, render_court


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
    # test. noise_sigma gives the render per-pixel texture (a stand-in for
    # real sensor/lens noise), which a shift genuinely misaligns, so the
    # fraction of changed pixels reflects the roll instead of the render's
    # artificial flatness. Confirmed stable across noise seeds (~0.33 vs the
    # 0.25 threshold, independent of which seed generates the pattern).
    base, _ = render_court(camera, noise_sigma=40.0, seed=0)
    frames = [np.roll(base, shift * 90, axis=1) for shift in range(5)]

    _, moved = court_detect.median_frame(frames)

    assert moved is True
