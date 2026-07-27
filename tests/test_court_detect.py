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
