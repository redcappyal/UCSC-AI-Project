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

from synthetic3d import make_camera
from synthetic_court import FLOOR_BGR, LINE_BGR, WALL_BGR, render_court


def test_render_court_paints_lines_where_the_camera_projects_them():
    # make_camera()'s default look_at=(10.5, 0.0, 5.0) tilts the camera down
    # just enough that the out line (z=15 ft) projects to y < 0 -- off the
    # top of the frame -- so this test aims a bit higher to keep the whole
    # out-line band, plus the 40px of wall above it the assertions check,
    # inside the visible image. Per the brief: adjust look_at only here,
    # never the renderer's court constants.
    camera = make_camera(look_at=(10.5, 0.0, 6.5))
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
