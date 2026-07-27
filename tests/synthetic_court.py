"""Synthetic squash-court renderer for court_detect tests.

Draws a court as seen by a `court_model.CameraModel`, so a detector's
recovered geometry can be scored against the camera that produced the image.
Paint bands have real WSF width (LINE_WIDTH_FT), which is what makes the
datum rules in the auto-court-detection spec (§5) testable.

Colours are deliberately Bay Club's (navy paint on maple), not
SquashAnalytics' (red paint): the detector must not key on hue, and a test
that passes only in the palette it was written for would hide that.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic3d import make_camera

from court_model import (
    COURT_WIDTH_FT,
    HALF_COURT_X_FT,
    LINE_WIDTH_FT,
    OUT_LINE_HEIGHT_FT,
    SERVICE_BOX_BACK_FT,
    SERVICE_BOX_FT,
    SERVICE_LINE_HEIGHT_FT,
    SHORT_LINE_CENTER_Y_FT,
    SHORT_LINE_FROM_FRONT_FT,
    TIN_TOP_HEIGHT_FT,
)

FLOOR_BGR = (150, 190, 215)   # warm maple
WALL_BGR = (228, 230, 231)    # off-white plaster
LINE_BGR = (90, 45, 30)       # navy court paint
SEAM_BGR = (120, 120, 122)    # the shadowed wall/floor junction

SEAM_WIDTH_FT = 0.08          # ~1 inch of shadow at the junction


def court_camera(**overrides):
    """The canonical camera for court-detection tests.

    synthetic3d.make_camera()'s default look_at=(10.5, 0, 5) tilts far enough
    down that the out line (15 ft up the front wall) projects above the top of
    the frame, where negative numpy indices silently wrap to the bottom rows
    and assertions pass against the wrong pixels. Raising the aim point puts
    the whole court in view. Every test in this suite uses this camera, so
    detector results stay comparable across tasks.
    """
    settings = {"look_at": (10.5, 0.0, 6.5)}
    settings.update(overrides)
    return make_camera(**settings)


def _polygon(camera, corners_ft):
    """Project 3-D corners to an int pixel polygon, or None if any is behind."""
    pixels = []
    for corner in corners_ft:
        try:
            pixels.append(camera.project(corner))
        except ValueError:
            return None
    return np.round(np.asarray(pixels, dtype=float)).astype(np.int32)


def _fill(image, camera, corners_ft, colour):
    polygon = _polygon(camera, corners_ft)
    if polygon is not None:
        cv2.fillPoly(image, [polygon], colour)   # fillPoly clips to the image


def _wall_band(z_low, z_high):
    """A band across the full front wall (y = 0) between two heights."""
    return [
        (0.0, 0.0, z_low), (COURT_WIDTH_FT, 0.0, z_low),
        (COURT_WIDTH_FT, 0.0, z_high), (0.0, 0.0, z_high),
    ]


def _floor_band(x_low, x_high, y_low, y_high):
    """A rectangle on the floor (z = 0)."""
    return [
        (x_low, y_low, 0.0), (x_high, y_low, 0.0),
        (x_high, y_high, 0.0), (x_low, y_high, 0.0),
    ]


def render_court(camera, visible_depth_ft=26.0, noise_sigma=0.0, seed=0):
    """Render the court and return (image_bgr, truth).

    `visible_depth_ft` bounds how far down-court the floor is drawn; it must
    stay in front of the camera (make_camera sits at y = 30 ft), because
    CameraModel.project raises for points at or behind it.
    """
    width = int(camera.frame_width)
    height = int(camera.frame_height)
    image = np.full((height, width, 3), WALL_BGR, dtype=np.uint8)

    # Floor slab, then the shadowed junction lines that bound it.
    _fill(image, camera, _floor_band(0.0, COURT_WIDTH_FT, 0.0, visible_depth_ft),
          FLOOR_BGR)
    _fill(image, camera, _floor_band(0.0, COURT_WIDTH_FT, 0.0, SEAM_WIDTH_FT),
          SEAM_BGR)
    _fill(image, camera, _floor_band(0.0, SEAM_WIDTH_FT, 0.0, visible_depth_ft),
          SEAM_BGR)
    _fill(image, camera,
          _floor_band(COURT_WIDTH_FT - SEAM_WIDTH_FT, COURT_WIDTH_FT,
                      0.0, visible_depth_ft),
          SEAM_BGR)

    # Front-wall paint. Each band sits on the side of its datum where the real
    # paint lies, so the stored edge is a real boundary in the image:
    #   out line   -> datum is the LOWER edge, paint runs upward from it
    #   service    -> datum is the TOP edge, paint runs downward from it
    #   tin        -> datum is the TOP edge, paint runs downward from it
    _fill(image, camera,
          _wall_band(OUT_LINE_HEIGHT_FT, OUT_LINE_HEIGHT_FT + LINE_WIDTH_FT),
          LINE_BGR)
    _fill(image, camera,
          _wall_band(SERVICE_LINE_HEIGHT_FT - LINE_WIDTH_FT,
                     SERVICE_LINE_HEIGHT_FT),
          LINE_BGR)
    _fill(image, camera,
          _wall_band(TIN_TOP_HEIGHT_FT - LINE_WIDTH_FT, TIN_TOP_HEIGHT_FT),
          LINE_BGR)

    # Floor paint. These are stored by CENTRELINE, so each band straddles its
    # datum (court_model.py:49-62).
    half = LINE_WIDTH_FT / 2.0
    _fill(image, camera,
          _floor_band(0.0, COURT_WIDTH_FT,
                      SHORT_LINE_CENTER_Y_FT - half,
                      SHORT_LINE_CENTER_Y_FT + half),
          LINE_BGR)
    _fill(image, camera,
          _floor_band(HALF_COURT_X_FT - half, HALF_COURT_X_FT + half,
                      SHORT_LINE_FROM_FRONT_FT, visible_depth_ft),
          LINE_BGR)
    for x_inner in (SERVICE_BOX_FT, COURT_WIDTH_FT - SERVICE_BOX_FT):
        _fill(image, camera,
              _floor_band(x_inner - half, x_inner + half,
                          SHORT_LINE_FROM_FRONT_FT, SERVICE_BOX_BACK_FT),
              LINE_BGR)
    for x_low, x_high in ((0.0, SERVICE_BOX_FT),
                          (COURT_WIDTH_FT - SERVICE_BOX_FT, COURT_WIDTH_FT)):
        _fill(image, camera,
              _floor_band(x_low, x_high,
                          SERVICE_BOX_BACK_FT - half,
                          SERVICE_BOX_BACK_FT + half),
              LINE_BGR)

    if noise_sigma > 0:
        generator = np.random.default_rng(seed)
        noisy = image.astype(np.int16) + generator.normal(
            0.0, noise_sigma, image.shape).astype(np.int16)
        image = np.clip(noisy, 0, 255).astype(np.uint8)

    project = camera.project
    truth = {
        "out_line_lower_edge": (project((0.0, 0.0, OUT_LINE_HEIGHT_FT)),
                                project((COURT_WIDTH_FT, 0.0, OUT_LINE_HEIGHT_FT))),
        "service_line_top_edge": (project((0.0, 0.0, SERVICE_LINE_HEIGHT_FT)),
                                  project((COURT_WIDTH_FT, 0.0,
                                           SERVICE_LINE_HEIGHT_FT))),
        "tin_top_edge": (project((0.0, 0.0, TIN_TOP_HEIGHT_FT)),
                         project((COURT_WIDTH_FT, 0.0, TIN_TOP_HEIGHT_FT))),
        "front_seam": (project((0.0, 0.0, 0.0)),
                       project((COURT_WIDTH_FT, 0.0, 0.0))),
        "short_line": (project((0.0, SHORT_LINE_CENTER_Y_FT, 0.0)),
                       project((COURT_WIDTH_FT, SHORT_LINE_CENTER_Y_FT, 0.0))),
        "wall_top_left": project((0.0, 0.0, OUT_LINE_HEIGHT_FT)),
        "wall_top_right": project((COURT_WIDTH_FT, 0.0, OUT_LINE_HEIGHT_FT)),
        "wall_bottom_left": project((0.0, 0.0, 0.0)),
        "wall_bottom_right": project((COURT_WIDTH_FT, 0.0, 0.0)),
        "short_line_left": project((0.0, SHORT_LINE_CENTER_Y_FT, 0.0)),
        "short_line_right": project((COURT_WIDTH_FT, SHORT_LINE_CENTER_Y_FT, 0.0)),
        "t_point": project((HALF_COURT_X_FT, SHORT_LINE_CENTER_Y_FT, 0.0)),
    }
    return image, truth
