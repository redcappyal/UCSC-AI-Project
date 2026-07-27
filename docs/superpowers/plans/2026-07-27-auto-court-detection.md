# Auto Court Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the judge-a-clip calibration wizard's ~12–19 taps across 7 screens with one confirm screen and at most one tap, by detecting the court's lines in a frame and deriving every calibration landmark as a line intersection.

**Architecture:** A new pure-Python module `court_detect.py` (cv2 + numpy, no Flask) takes a handful of frames, takes their temporal median, finds painted lines and surface boundaries with two colour-agnostic response maps, names them as court entities by geometry, and intersects them into the exact structures `buildJson()` already produces. A new `POST /api/detect-court` wraps it. `index.html` gains one new `confirm` phase that overlays the result and lets the user accept it or drag an anchor. Everything stays 2D — no 3D pose is solved anywhere.

**Tech Stack:** Python 3 / OpenCV 4.10 / NumPy / Flask / pytest; vanilla ES2020 + Canvas 2D in a single-file HTML app.

**Spec:** `docs/superpowers/specs/2026-07-27-auto-court-detection-design.md`. Read it before Task 1 — §5 (datum rules) is the single most likely way to get this subtly and silently wrong.

## Global Constraints

- **The venv is `.venv`. System `python3` has no flask or cv2.** Every command below uses `.venv/bin/python`.
- **Full suite must stay green:** `.venv/bin/python -m pytest tests/ -q` → `318 passed, 1 deselected`. That deselection is expected, not a skip to chase. Your new tests raise the passed count.
- **A PostToolUse hook auto-runs the paired test file** whenever you edit a `*.py` that has a `tests/test_*.py`. Failures come back as a *blocked edit*, not a warning. So `tests/test_court_detect.py` must stay fast (< ~10 s).
- **Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite`.** This plan never does — the endpoint uses `cv2.imdecode` on an in-memory buffer, which is unaffected.
- **The manual wizard must remain wired and behaviourally unchanged.** No task deletes or edits `tap_out` / `tap_tin` / `tap_service` / `review` / `tap_wall` / `tap_floor` logic.
- **Schema stays `squash-calibration-v2`.** The only addition is an additive `source` string per element; no existing key changes meaning.
- **All UI work follows `DESIGN.md`.** Binding rules that bite here: §0.2 one accent (yellow `#ffd60a`); §0.3 colour = meaning — **red is reserved for OUT verdicts, never for a bad anchor**, and colour is never the sole carrier of meaning; §0.6 44 px minimum touch targets, 48 px for primaries; §0.7 uppercase controls, sentence-case guidance; §0.9 no layout shift; §0.11 respect the shell; §0.12 verify both themes at 390 × 844.
- **Keep every import at the top of `court_detect.py`.** Tasks 3 and 5 introduce `math`, `dataclass`, and `court_model`; the code blocks show them where they are first needed for readability, but when you add them, hoist them into the module's single import block at the top.
- **Court datum values are already defined in `court_model.py`.** Never redefine them: `COURT_WIDTH_FT` 21.0, `COURT_LENGTH_FT` 32.0, `OUT_LINE_HEIGHT_FT` 15.0, `SERVICE_LINE_HEIGHT_FT` ≈ 5.8399, `TIN_TOP_HEIGHT_FT` = 19/12 ≈ 1.5833, `SHORT_LINE_CENTER_Y_FT` ≈ 17.918, `HALF_COURT_X_FT` 10.5, `SERVICE_BOX_FT` 5.25, `SERVICE_BOX_BACK_FT` 23.25, `LINE_WIDTH_FT` = 50/304.8 ≈ 0.164.

---

## File Structure

| File | Responsibility |
|---|---|
| `court_detect.py` | **new.** All detection. Pure functions over numpy arrays; imports `court_model` for geometry and `cv2`/`numpy` only. Never imports Flask, never touches disk. |
| `tests/synthetic_court.py` | **new.** Renders a synthetic court image from a `CameraModel`, plus the ground-truth pixel positions of every datum. Test helper, mirrors the existing `tests/synthetic3d.py` pattern. Not collected as a test (no `test_` prefix). |
| `tests/test_court_detect.py` | **new.** Paired test for `court_detect.py`; auto-run by the hook. |
| `app.py` | **modify.** Add `POST /api/detect-court` beside the existing `/api/camera-check` (`app.py:1010`). |
| `tests/test_camera_endpoints.py` | **modify.** Add the endpoint test; this file already owns the single-camera calibration endpoints. |
| `index.html` | **modify.** Detection trigger on the frame step, new `confirm` phase, anchor dragging. |
| `DESIGN.md` | **modify.** Document the confirm screen as a new §8 component. |

Why detection lives in Python and not the browser: cv2 supplies the morphology and Hough tooling the feasibility spike proved out, `court_model.py` is already the authoritative fitter, it is unit-testable under pytest, and one endpoint serves both the web app and the iOS app.

---

### Task 1: Synthetic court renderer (test foundation)

Everything else is scored against this, so it comes first. The renderer draws paint bands with **real WSF width**, which is what makes the spec's §5 datum rules testable at all: a detector that returns a stripe's centre instead of its named edge must fail a test here.

**Files:**
- Create: `tests/synthetic_court.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: `court_model.CameraModel` (has `.project((x, y, z)) -> (px, py)`, raises `ValueError` behind the camera); `tests/synthetic3d.make_camera(...)`.
- Produces:
  - `render_court(camera, visible_depth_ft=26.0, noise_sigma=0.0, seed=0) -> (image_bgr, truth)`
  - `truth` is a `dict[str, tuple]`: line datums map to a 2-tuple of endpoint pixels; point datums map to one pixel. Keys: `out_line_lower_edge`, `service_line_top_edge`, `tin_top_edge`, `front_seam`, `short_line`, `wall_top_left`, `wall_top_right`, `wall_bottom_left`, `wall_bottom_right`, `short_line_left`, `short_line_right`, `t_point`.
  - Module constants `FLOOR_BGR`, `WALL_BGR`, `LINE_BGR`, `SEAM_BGR`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_court_detect.py`:

```python
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
    camera = make_camera()
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'synthetic_court'`

- [ ] **Step 3: Write the renderer**

Create `tests/synthetic_court.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: PASS — `1 passed`

If the wall/floor assertions fail, print `truth["front_seam"]` and check the seam is inside the frame; adjust `make_camera`'s `look_at` only inside the test, never the renderer's court constants.

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic_court.py tests/test_court_detect.py
git commit -m "test(detect): synthetic court renderer with real paint-band widths"
```

---

### Task 2: Temporal median and camera-motion guard

**Files:**
- Create: `court_detect.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `median_frame(frames) -> (median_bgr, moved: bool)`. `frames` is a non-empty sequence of same-shape uint8 BGR arrays. Module constants `MOTION_DELTA = 18`, `MOTION_FRACTION = 0.25`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_court_detect.py`:

```python
import court_detect
from synthetic_court import render_court   # already imported above; keep one import


def _frames_with_moving_player(camera, count=5):
    """Static court, one dark rectangle that moves — the real occlusion case."""
    base, _ = render_court(camera)
    frames = []
    for index in range(count):
        frame = base.copy()
        left = 300 + index * 180
        frame[500:900, left:left + 150] = (30, 30, 35)
        frames.append(frame)
    return frames


def test_median_frame_erases_a_moving_player():
    camera = make_camera()
    frames = _frames_with_moving_player(camera)
    median, moved = court_detect.median_frame(frames)

    assert moved is False
    # No column keeps the player's near-black patch after the median.
    assert median[500:900, 300:1350].min() > 60


def test_median_frame_flags_a_panning_camera():
    camera = make_camera()
    base, _ = render_court(camera)
    frames = [np.roll(base, shift * 90, axis=1) for shift in range(5)]

    _, moved = court_detect.median_frame(frames)

    assert moved is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'court_detect'`

- [ ] **Step 3: Write the implementation**

Create `court_detect.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: PASS — `3 passed`

- [ ] **Step 5: Commit**

```bash
git add court_detect.py tests/test_court_detect.py
git commit -m "feat(detect): temporal median with a camera-motion guard"
```

---

### Task 3: Response maps and line extraction

Two response maps, because the court presents two physically different structures. **Painted lines** (out, service, tin, short line) are thin dark chromatic stripes. **Surface boundaries** (the wall/floor seams) are junctions, which can be a pure colour step with no dark ridge at all. One map cannot find both reliably.

**Files:**
- Modify: `court_detect.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: `median_frame` (Task 2).
- Produces:
  - `line_response(bgr) -> uint8 HxW`, `edge_response(bgr) -> uint8 HxW`
  - `paint_mask(bgr) -> uint8 HxW` (0/255), the binarised `line_response`
  - `find_lines(response, min_length_px) -> list[DetectedLine]`
  - `class DetectedLine` — frozen dataclass with fields `x1, y1, x2, y2, support` and members `length_px`, `angle_deg`, `midpoint`, `y_at(x)`, `intersect(other) -> (x, y) | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_court_detect.py`:

```python
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
    camera = make_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.line_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    for name in ("out_line_lower_edge", "service_line_top_edge", "tin_top_edge"):
        start, end = truth[name]
        found = _nearest_line_to(lines, start, end)
        # The stripe is ~LINE_WIDTH_FT wide, so a line fitted to the whole
        # stripe may sit up to half a width off the named datum. Task 5 pulls
        # it onto the exact edge; here we only require the right stripe.
        assert abs(found.y_at(start[0]) - start[1]) < 12, name


def test_edge_response_finds_the_wall_floor_seams():
    camera = make_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.edge_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    start, end = truth["front_seam"]
    found = _nearest_line_to(lines, start, end)
    assert abs(found.y_at(start[0]) - start[1]) < 8


def test_detected_line_intersect_returns_none_for_parallels():
    first = court_detect.DetectedLine(0.0, 0.0, 100.0, 0.0, support=1)
    second = court_detect.DetectedLine(0.0, 50.0, 100.0, 50.0, support=1)
    assert first.intersect(second) is None

    crossing = court_detect.DetectedLine(50.0, -50.0, 50.0, 50.0, support=1)
    point = first.intersect(crossing)
    assert point is not None and abs(point[0] - 50.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: FAIL — `AttributeError: module 'court_detect' has no attribute 'line_response'`

- [ ] **Step 3: Write the implementation**

Append to `court_detect.py`:

```python
import math
from dataclasses import dataclass

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
    """(orientation, signed perpendicular offset from the origin)."""
    x1, y1, x2, y2 = segment
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle > 90:
        angle -= 180
    if angle <= -90:
        angle += 180
    radians = math.radians(angle)
    return angle, -math.sin(radians) * x1 + math.cos(radians) * y1


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: PASS — `6 passed`

If `test_find_lines_recovers_the_front_wall_paint` fails on the tin only, the tin band may be merging with the seam band: raise `MERGE_OFFSET_PX` scrutiny by *lowering* it to `4.0` rather than widening any tolerance in the test.

- [ ] **Step 5: Commit**

```bash
git add court_detect.py tests/test_court_detect.py
git commit -m "feat(detect): colour-agnostic paint and boundary line extraction"
```

---

### Task 4: Name the lines as court entities

**Files:**
- Modify: `court_detect.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: `DetectedLine`, `find_lines`, `line_response`, `edge_response` (Task 3).
- Produces: `assign_lines(paint_lines, edge_lines, frame_shape) -> dict[str, DetectedLine | None]` with keys `out`, `service`, `tin`, `front_seam`, `left_seam`, `right_seam`, `short_line`. Module constant `HORIZONTAL_MAX_DEG = 25.0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_court_detect.py`:

```python
def test_assign_lines_names_every_required_entity():
    camera = make_camera()
    image, truth = render_court(camera, noise_sigma=2.0)
    minimum = image.shape[1] * 0.10

    assigned = court_detect.assign_lines(
        court_detect.find_lines(court_detect.line_response(image), minimum),
        court_detect.find_lines(court_detect.edge_response(image), minimum),
        image.shape,
    )

    for name in ("out", "service", "tin", "front_seam",
                 "left_seam", "right_seam", "short_line"):
        assert assigned.get(name) is not None, f"missing {name}"

    # The out line sits above the service line, which sits above the tin,
    # which sits above the seam, which sits above the short line.
    order = [assigned[name].midpoint[1]
             for name in ("out", "service", "tin", "front_seam", "short_line")]
    assert order == sorted(order), order

    # The seams are on the sides they are named for.
    assert assigned["left_seam"].midpoint[0] < image.shape[1] / 2
    assert assigned["right_seam"].midpoint[0] > image.shape[1] / 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: FAIL — `AttributeError: module 'court_detect' has no attribute 'assign_lines'`

- [ ] **Step 3: Write the implementation**

Append to `court_detect.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: PASS — `7 passed`

- [ ] **Step 5: Commit**

```bash
git add court_detect.py tests/test_court_detect.py
git commit -m "feat(detect): name detected lines as court entities by geometry"
```

---

### Task 5: Datum refit, intersections, homographies — `detect_court()`

**This is the task the spec's §5 warning is about.** Front-wall lines are stored by painted-band **edge** (out = lower, service and tin = top); floor landmarks are stored by **centreline**. Getting it backwards is a systematic half-line-width bias landed on the IN/OUT call.

**Files:**
- Modify: `court_detect.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: everything from Tasks 2–4; `court_model.fit_homography(src, dst) -> (H, residuals)` where `H @ [src,1] ~ [dst,1]`.
- Produces: `detect_court(frames) -> dict`. Shape:
  ```python
  {"ok": True, "status": "ok" | "insufficient_lines" | "no_frames",
   "frame_width": int, "frame_height": int,
   "lines": [{"name", "endpoints", "slope", "intercept", "fit_rms_px",
              "x_span_px", "x_span_source", "edge_pixels_used", "source"}],
   "planes": {"wall": {"corners": [{"id", "tap_px", "source"}],
                       "fitted_by": "auto_detect"},
              "floor": {"landmarks": [{"id", "court_ft", "tap_px",
                                       "refined_px", "method", "residual_px",
                                       "skipped", "source"}],
                        "homography_image_from_court": [[float] * 3] * 3,
                        "fit_rms_px": float, "max_residual_px": float,
                        "fitted_by": "auto_detect"}},
   "checks": [{"id", "predicted_px", "residual_px", "status"}],
   "confidence": "high" | "low",
   "warnings": [str]}
  ```
  Also `LineFit` (frozen dataclass: `slope`, `intercept`, `rms_px`, `x_span`, `pixel_count`) and `refit_to_datum(mask, line, mode) -> LineFit | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_court_detect.py`:

```python
import court_model
from judge_call import load_calibration_lines


def test_refit_to_datum_lands_on_the_named_edge_not_the_centre():
    camera = make_camera()
    image, truth = render_court(camera)
    mask = court_detect.paint_mask(image)
    lines = court_detect.find_lines(court_detect.line_response(image),
                                    image.shape[1] * 0.10)

    start, end = truth["out_line_lower_edge"]
    stripe = _nearest_line_to(lines, start, end)
    fit = court_detect.refit_to_datum(mask, stripe, "max")

    assert fit is not None
    for point in (start, end):
        assert abs(fit.slope * point[0] + fit.intercept - point[1]) < 2.0

    # And the 'min' datum is a genuinely different answer — proof the mode is
    # doing work rather than both modes landing on the stripe's centre.
    top = court_detect.refit_to_datum(mask, stripe, "min")
    assert abs(top.intercept - fit.intercept) > 1.0


def test_detect_court_recovers_the_camera_that_drew_the_court():
    camera = make_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    result = court_detect.detect_court([image])

    assert result["status"] == "ok"
    assert result["confidence"] == "high", result["warnings"]

    # Front-wall lines land on their true datums.
    by_name = {line["name"]: line for line in result["lines"]}
    for name in ("out_line_lower_edge", "service_line_top_edge", "tin_top_edge"):
        start, end = truth[name]
        fitted = by_name[name]
        for point in (start, end):
            predicted = fitted["slope"] * point[0] + fitted["intercept"]
            assert abs(predicted - point[1]) < 3.0, name

    # Wall corners land on their true intersections.
    corners = {corner["id"]: corner["tap_px"]
               for corner in result["planes"]["wall"]["corners"]}
    for corner_id, truth_key in (("top_left", "wall_top_left"),
                                 ("top_right", "wall_top_right"),
                                 ("bottom_left", "wall_bottom_left"),
                                 ("bottom_right", "wall_bottom_right")):
        assert np.linalg.norm(
            np.asarray(corners[corner_id]) - np.asarray(truth[truth_key])) < 4.0


def test_detect_court_output_parses_with_the_existing_consumers():
    camera = make_camera()
    image, _ = render_court(camera, noise_sigma=2.0)

    result = court_detect.detect_court([image])
    calibration = {
        "schema": "squash-calibration-v2",
        "frame_width": result["frame_width"],
        "frame_height": result["frame_height"],
        "lines": result["lines"],
        "planes": result["planes"],
        "distortion": None,
    }

    top, bottom = load_calibration_lines(calibration)
    assert top.left.x < top.right.x

    floor_map = court_model.load_floor_calibration(calibration)
    assert floor_map is not None
    x_ft, y_ft = floor_map.image_to_court(*result["checks"][0]["predicted_px"])
    assert -2.0 < x_ft < court_model.COURT_WIDTH_FT + 2.0


def test_detect_court_reports_failure_rather_than_guessing():
    blank = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    result = court_detect.detect_court([blank])
    assert result["status"] == "insufficient_lines"
    assert result["confidence"] == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: FAIL — `AttributeError: module 'court_detect' has no attribute 'refit_to_datum'`

- [ ] **Step 3: Write the implementation**

Append to `court_detect.py`:

```python
# --- datum refit -----------------------------------------------------------
import court_model

DATUM_BAND_PX = 6          # how far either side of the fitted line to look
MIN_DATUM_COLUMNS = 40     # mirrors index.html's MIN_COLS
CHECK_OK_PX = 4.0          # a self-verification prediction this close is "ok"
HOMOGRAPHY_RMS_OK_PX = 3.0


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


# --- the whole detection ---------------------------------------------------
_WALL_CORNER_COURT_FT = {
    "top_left": (0.0, court_model.OUT_LINE_HEIGHT_FT),
    "top_right": (court_model.COURT_WIDTH_FT, court_model.OUT_LINE_HEIGHT_FT),
    "bottom_left": (0.0, 0.0),
    "bottom_right": (court_model.COURT_WIDTH_FT, 0.0),
}

_FLOOR_ANCHORS = (
    ("front_seam_left", (0.0, 0.0)),
    ("front_seam_right", (court_model.COURT_WIDTH_FT, 0.0)),
    ("short_line_left", (0.0, court_model.SHORT_LINE_CENTER_Y_FT)),
    ("short_line_right", (court_model.COURT_WIDTH_FT,
                          court_model.SHORT_LINE_CENTER_Y_FT)),
)

# Court points that no anchor was fitted to. Their distance from real paint is
# free evidence the two homographies are right, at zero human cost.
_FLOOR_CHECKS = (
    ("t_point", (court_model.HALF_COURT_X_FT,
                 court_model.SHORT_LINE_CENTER_Y_FT)),
    ("left_box_inner_back", (court_model.LEFT_BOX_INNER_CENTER_X_FT,
                             court_model.BOX_BACK_CENTER_Y_FT)),
    ("right_box_inner_back", (court_model.RIGHT_BOX_INNER_CENTER_X_FT,
                              court_model.BOX_BACK_CENTER_Y_FT)),
)


def _failure(status, frame_shape, warnings):
    height, width = (frame_shape[:2] if frame_shape is not None else (0, 0))
    return {"ok": True, "status": status,
            "frame_width": int(width), "frame_height": int(height),
            "lines": [], "planes": {}, "checks": [],
            "confidence": "low", "warnings": warnings}


def detect_court(frames):
    """Detect the court from one or more frames of the same fixed viewpoint.

    Returns squash-calibration-v2 structures (spec §6) and never raises: the
    wizard falls back to manual taps on any non-'ok' status.
    """
    if not len(frames):
        return _failure("no_frames", None, ["No frames were supplied."])

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

    required = ("out", "front_seam", "left_seam", "right_seam", "short_line")
    missing = [name for name in required if assigned.get(name) is None]
    if missing:
        return _failure("insufficient_lines", image.shape,
                        warnings + [f"Could not find: {', '.join(missing)}."])

    # Front-wall lines, each pulled onto its own stored datum (spec §5).
    lines = []
    for name, key, mode in (
            ("out_line_lower_edge", "out", "max"),
            ("service_line_top_edge", "service", "min"),
            ("tin_top_edge", "tin", "min")):
        if assigned.get(key) is None:
            warnings.append(f"{name} was not detected.")
            continue
        fit = refit_to_datum(mask, assigned[key], mode)
        if fit is None:
            warnings.append(f"{name} had too little clean paint to fit.")
            continue
        lines.append(_line_payload(name, fit, "detected"))

    out_fit = next((line for line in lines
                    if line["name"] == "out_line_lower_edge"), None)
    if out_fit is None:
        return _failure("insufficient_lines", image.shape,
                        warnings + ["The out line could not be fitted."])

    # Anchors as intersections of long lines.
    out_line = DetectedLine(*out_fit["endpoints"][0], *out_fit["endpoints"][1],
                            support=1)
    seam = assigned["front_seam"]
    short_fit = refit_to_centreline(mask, assigned["short_line"])
    short_line = (_fit_as_line(short_fit) if short_fit is not None
                  else assigned["short_line"])

    anchors = {
        "top_left": out_line.intersect(assigned["left_seam"]),
        "top_right": out_line.intersect(assigned["right_seam"]),
        "bottom_left": seam.intersect(assigned["left_seam"]),
        "bottom_right": seam.intersect(assigned["right_seam"]),
        "short_line_left": short_line.intersect(assigned["left_seam"]),
        "short_line_right": short_line.intersect(assigned["right_seam"]),
    }
    if any(point is None for point in anchors.values()):
        return _failure("insufficient_lines", image.shape,
                        warnings + ["Court lines did not intersect."])

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
                        warnings + ["The floor homography was degenerate."])

    # Self-verification: how far predictions of UNUSED court points fall from
    # real paint. distanceTransform turns that into a lookup.
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    checks = []
    for check_id, court_ft in _FLOOR_CHECKS:
        px, py = court_model.apply_homography(homography, court_ft)
        inside = 0 <= int(py) < height and 0 <= int(px) < width
        residual = float(distance[int(py), int(px)]) if inside else None
        checks.append({
            "id": check_id,
            "predicted_px": [round(px, 2), round(py, 2)],
            "residual_px": None if residual is None else round(residual, 2),
            "status": ("unverified" if residual is None
                       else "ok" if residual <= CHECK_OK_PX else "off"),
        })

    floor_rms = float(np.sqrt(np.mean(residuals ** 2)))
    worst_check = max((check["residual_px"] for check in checks
                       if check["residual_px"] is not None), default=None)
    confidence = "high"
    if len(lines) < 3:
        confidence = "low"
    elif floor_rms > HOMOGRAPHY_RMS_OK_PX:
        confidence = "low"
        warnings.append(f"Floor fit RMS {floor_rms:.1f} px.")
    elif worst_check is None or worst_check > CHECK_OK_PX:
        confidence = "low"
        warnings.append("Predicted court markings did not land on real paint.")

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
                     "source": "intersection"}
                    for corner_id in _WALL_CORNER_COURT_FT
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
                     "residual_px": round(float(residuals[index]), 2),
                     "skipped": False,
                     "source": "intersection"}
                    for index, (name, court_ft) in enumerate(_FLOOR_ANCHORS)
                ],
                "homography_image_from_court": [
                    [float(value) for value in row] for row in homography],
                "fit_rms_px": round(floor_rms, 2),
                "max_residual_px": round(float(residuals.max()), 2),
                "fitted_by": "auto_detect",
            },
        },
        "checks": checks,
        "confidence": confidence,
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_court_detect.py -q`
Expected: PASS — `11 passed`

Then the full suite, because this task imports `court_model` at module scope:
Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `329 passed, 1 deselected`

- [ ] **Step 5: Commit**

```bash
git add court_detect.py tests/test_court_detect.py
git commit -m "feat(detect): datum-correct line fits, anchor intersections, floor homography"
```

---

### Task 6: `POST /api/detect-court`

**Files:**
- Modify: `app.py` (add after the `/api/camera-model` route, which ends at `app.py:1053`)
- Test: `tests/test_camera_endpoints.py`

**Interfaces:**
- Consumes: `court_detect.detect_court(frames) -> dict` (Task 5).
- Produces: `POST /api/detect-court`, multipart form field `frames` (repeated, JPEG). Always HTTP 200 with a `status` field, matching the never-raise convention `/api/camera-check` already uses.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_camera_endpoints.py`:

```python
import cv2

import court_model
from judge_call import load_calibration_lines
from synthetic_court import render_court


def _jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buffer.tobytes()


def test_detect_court_endpoint_returns_usable_calibration_structures():
    import io

    client = _client()
    image, _ = render_court(make_camera(), noise_sigma=2.0)
    payload = {"frames": [(io.BytesIO(_jpeg(image)), f"frame{index}.jpg")
                          for index in range(3)]}

    body = client.post("/api/detect-court", data=payload,
                       content_type="multipart/form-data").get_json()

    assert body["ok"] is True and body["status"] == "ok"
    calibration = {
        "schema": "squash-calibration-v2",
        "frame_width": body["frame_width"],
        "frame_height": body["frame_height"],
        "lines": body["lines"],
        "planes": body["planes"],
        "distortion": None,
    }
    assert load_calibration_lines(calibration) is not None
    assert court_model.load_floor_calibration(calibration) is not None


def test_detect_court_endpoint_is_200_with_a_status_when_given_nothing():
    client = _client()
    body = client.post("/api/detect-court", data={},
                       content_type="multipart/form-data").get_json()
    assert body["ok"] is True and body["status"] == "no_frames"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_camera_endpoints.py -q`
Expected: FAIL — 404, so `body["status"]` raises `TypeError: 'NoneType' object is not subscriptable`

- [ ] **Step 3: Write the implementation**

Add to `app.py` immediately after the `/api/camera-model` route:

```python
MAX_DETECT_FRAMES = 8


@app.post("/api/detect-court")
def detect_court_endpoint():
    """Detect the court from a few frames of one fixed viewpoint.

    Read-only wizard feedback like /api/camera-check: nothing is stored, and
    the reply is always 200 with a status field so the client can fall back to
    the manual tap wizard on any failure.

    Frames arrive as JPEG bytes and are decoded with cv2.imdecode, which reads
    from memory -- so the CLAUDE.md warning about cv2.imread and non-ASCII
    paths does not apply here.
    """
    uploads = request.files.getlist("frames")
    if not uploads:
        return jsonify({"ok": True, "status": "no_frames",
                        "warnings": ["No frames were supplied."]})

    frames = []
    for upload in uploads[:MAX_DETECT_FRAMES]:
        buffer = np.frombuffer(upload.read(), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is not None and (not frames or image.shape == frames[0].shape):
            frames.append(image)

    if not frames:
        return jsonify({"ok": True, "status": "no_frames",
                        "warnings": ["No frame could be decoded."]})

    return jsonify(court_detect.detect_court(frames))
```

Then add the imports at the top of `app.py`. `import cv2` is already there at `app.py:11`;
`numpy` is **not** imported and must be added. The result should read:

```python
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import court_detect
import court_model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera_endpoints.py -q`
Expected: PASS

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `331 passed, 1 deselected`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_camera_endpoints.py
git commit -m "feat(api): POST /api/detect-court for automatic calibration"
```

---

### Task 7: Client — detect on USE FRAME, and the confirm screen

**Files:**
- Modify: `index.html`
- Modify: `DESIGN.md`

**Interfaces:**
- Consumes: `POST /api/detect-court` (Task 6).
- Produces (all in `index.html`): `grabDetectionFrames(count)` → `Promise<Blob[]>`; `runCourtDetection()` → `Promise<void>`; `applyDetection(result)`; `drawConfirmOverlay()`; new phase key `'confirm'`; new section `#p-confirm`.

Read `DESIGN.md` §0 and §8 before writing any markup.

- [ ] **Step 1: Add the phase section markup**

In `index.html`, immediately after the `<!-- PHASE: floor calibration -->` section's closing `</section>` (before `<!-- PHASE: clip -->`), add:

```html
  <!-- PHASE: confirm (auto-detected court, review before use) -->
  <section class="hidden" id="p-confirm">
    <p class="status" id="confirmSummary">&nbsp;</p>
    <div class="targetZones">
      <div class="targetHead"><strong>Court anchors</strong><span id="confirmCount"></span></div>
      <div class="wallCornerList" id="confirmAnchorList"></div>
    </div>
    <button class="primary proxied" id="confirmUseBtn">Use this calibration</button>
    <div class="row">
      <button class="small" id="confirmManualBtn" type="button">Tap it manually</button>
    </div>
    <p class="status" id="confirmWarn">&nbsp;</p>
  </section>
```

The `&nbsp;` placeholders reserve the status lines' height so nothing shifts when a message appears (DESIGN.md §0.9). `.wallCornerList` is reused rather than a new class — it already styles a `done`/`active`/`warned` list.

- [ ] **Step 2: Register the phase**

In `STEP_META` (`index.html:1229`), add after the `tap_floor` entry:

```javascript
  confirm:{label:'Confirm court', instr:'Check the overlay hugs the real court. Drag any anchor that sits off the line.',
           action:{btn:'confirmUseBtn', label:'Use calibration'}},
```

In `setPhase` (`index.html:1301`), add `'p-confirm'` to the id list that gets hidden, and add these two lines beside the existing per-phase blocks:

```javascript
  if(p === 'confirm'){ $('p-confirm').classList.remove('hidden'); buildConfirmView(); }
```

and extend the zoom-control condition so the confirm phase keeps pinch-zoom:

```javascript
  } else if(p === 'tap_wall' || p === 'tap_floor' || p === 'confirm'){
    $('zoomCtrl').classList.remove('hidden');
```

In the `hdrBack` handler (`index.html:1885`), add a branch beside the `tap_out` one:

```javascript
  } else if(S.phase === 'confirm'){
    S.detect = null; setPhase('frame'); ctx.drawImage(vid, 0, 0, S.W, S.H);
```

Add `detect:null,` to the `S` state object beside `floor:null`.

- [ ] **Step 3: Grab frames and call the endpoint**

Add near the other calibration helpers, after `finishCalWizard()`:

```javascript
/* ---------- automatic court detection ----------
   Posts a handful of frames so the server can take a temporal median (which
   deletes the players) and detect the court's lines. The reply arrives in the
   same shapes buildJson() writes, so it drops straight into wizard state and
   every downstream consumer is unchanged. */
const DETECT_FRAME_COUNT = 5;

async function grabDetectionFrames(count = DETECT_FRAME_COUNT){
  const scratch = document.createElement('canvas');
  scratch.width = S.W; scratch.height = S.H;
  const scratchCtx = scratch.getContext('2d');
  const wasAt = vid.currentTime;
  const duration = clipDuration();
  const blobs = [];
  for(let i = 0; i < count; i++){
    // seekVideo, not a raw currentTime assignment: WebKit fires 'seeked'
    // before the new frame is presented, so a hand-rolled seek captures a
    // stale or black frame and the median would be built from nothing.
    await seekVideo(duration * (i + 0.5) / count);
    scratchCtx.drawImage(vid, 0, 0, S.W, S.H);
    const blob = await new Promise(r => scratch.toBlob(r, 'image/jpeg', 0.92));
    if(blob) blobs.push(blob);
  }
  await seekVideo(wasAt);
  return blobs;
}

async function runCourtDetection(){
  $('analyzeOverlay').classList.remove('hidden');
  let result = null;
  try{
    const form = new FormData();
    for(const [i, blob] of (await grabDetectionFrames()).entries())
      form.append('frames', blob, `frame${i}.jpg`);
    const response = await fetch('/api/detect-court', {method:'POST', body:form});
    if(response.ok) result = await response.json();
  } catch(_){ /* offline or decode failure: fall through to the manual wizard */ }
  $('analyzeOverlay').classList.add('hidden');

  ctx.drawImage(vid, 0, 0, S.W, S.H);
  S.base = ctx.getImageData(0, 0, S.W, S.H);
  S.frameTime = vid.currentTime;

  if(!result || result.status !== 'ok'){
    setPhase('tap_out');
    $('tapStatus').textContent = result && result.warnings && result.warnings.length
      ? `Could not detect the court (${result.warnings[0]}) — tap the lines instead.`
      : 'Could not detect the court — tap the lines instead.';
    $('tapStatus').className = 'status warn';
    return;
  }
  applyDetection(result);
  setPhase('confirm');
}
```

- [ ] **Step 4: Apply the result into wizard state**

Add after `runCourtDetection`:

```javascript
/* The detector's reply uses buildJson()'s own shapes, so applying it is a
   direct fill of the same state the tap wizard produces — no translation
   layer, and nothing downstream can tell the two apart. */
function applyDetection(result){
  S.detect = result;
  S.profile = null;
  const byName = {};
  for(const line of result.lines) byName[line.name] = line;
  for(const key of ORDER){
    const stored = byName[CFG[key].edgeName];
    S.lines[key] = stored
      ? { seeds:[], maskCanvas:null, fit:fitFromStoredLine(stored) }
      : null;
  }
  S.wall = wallFromCalibration(result);
  S.floor = null;
  const floorPlane = result.planes && result.planes.floor;
  if(COURT && floorPlane && Array.isArray(floorPlane.landmarks)){
    S.floor = newFloorState();
    const byId = {};
    for(const landmark of floorPlane.landmarks) byId[landmark.id] = landmark;
    for(const landmark of S.floor.landmarks){
      const stored = byId[landmark.id];
      if(!stored) continue;
      landmark.tap_px = stored.tap_px || null;
      landmark.refined_px = stored.refined_px || null;
      landmark.method = stored.method || 'intersection';
      landmark.skipped = false;
    }
    S.floor.activeIdx = -1;
    refitFloor();
    if(!S.floor.H) S.floor = null;
  }
}
```

`fitFromStoredLine` (`index.html:3258`) and `wallFromCalibration` (`index.html:3268`) already exist and take exactly these shapes — that is why the endpoint mirrors `buildJson()`.

- [ ] **Step 5: Draw the overlay and build the list**

Add after `applyDetection`:

```javascript
/* Anchor colours follow the floor wizard's sanctioned palette (DESIGN.md
   §8.10): dim -> green done -> amber warned. Never red — §0.3 reserves red
   for OUT verdicts — and never colour alone, so the list below names every
   anchor the checks flagged. */
const CONFIRM_OK = '#3ddc84', CONFIRM_WARN = '#f5c518';

function confirmAnchorPoints(){
  const points = [];
  if(S.wall) WALL_CORNERS.forEach((corner, i) => {
    if(S.wall.points[i]) points.push({key:`wall:${i}`, label:corner.label,
                                      xy:S.wall.points[i]});
  });
  if(S.floor) for(const landmark of S.floor.landmarks){
    if((landmark.id === 'short_line_left' || landmark.id === 'short_line_right')
        && landmark.refined_px)
      points.push({key:`floor:${landmark.id}`,
                   label:landmark.id.replace(/_/g, ' '), xy:landmark.refined_px});
  }
  return points;
}

function confirmOffChecks(){
  const checks = (S.detect && S.detect.checks) || [];
  return checks.filter(check => check.status === 'off');
}

function drawConfirmOverlay(){
  if(S.phase !== 'confirm') return;
  const warned = confirmOffChecks().length > 0;
  ctx.save();
  const radius = Math.max(9, S.W / 110);
  for(const anchor of confirmAnchorPoints()){
    ctx.beginPath();
    ctx.arc(anchor.xy[0], anchor.xy[1], radius, 0, Math.PI * 2);
    ctx.fillStyle = warned ? CONFIRM_WARN : CONFIRM_OK;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,.6)';
    ctx.lineWidth = Math.max(1.5, S.W / 1400);
    ctx.stroke();
  }
  ctx.restore();
}

function buildConfirmView(){
  const off = confirmOffChecks();
  const anchors = confirmAnchorPoints();
  $('confirmCount').textContent = `${anchors.length} placed`;
  $('confirmAnchorList').innerHTML = anchors.map(anchor =>
    `<span class="done">${anchor.label} set</span>`).join('');
  $('confirmSummary').textContent = off.length
    ? 'Detected, but some court markings did not line up.'
    : 'Court detected — the overlay should sit on the real lines.';
  $('confirmSummary').className = 'status' + (off.length ? ' warn' : ' ok');
  $('confirmWarn').innerHTML = off.length
    ? `Off: ${off.map(check => check.id.replace(/_/g, ' ')).join(', ')}. Drag the nearest anchor, or tap it manually.`
    : '&nbsp;';
  $('confirmWarn').className = 'status' + (off.length ? ' warn' : '');
  render();
}
```

In `render()` (`index.html:2534`), add `drawConfirmOverlay();` immediately after the existing `drawFloorOverlay();` call.

- [ ] **Step 6: Wire the buttons**

The existing `$('useFrame')` handler is at `index.html:1468`. Keep every line of it —
`transportStop()`, the `S.frameView.gen++` abort, and the `seekVideo` call are all load-
bearing — and change only its last line from `setPhase('tap_out')` to detection. The frame
capture inside it is left in place so `S.base` is populated even if detection later fails:

```javascript
$('useFrame').onclick = async () => {
  transportStop();
  S.frameView.gen++;               // abort any in-flight strip render
  await seekVideo(S.frameView.cursor);   // strips may have the decoder elsewhere
  ctx.drawImage(vid, 0, 0, S.W, S.H);
  S.base = ctx.getImageData(0, 0, S.W, S.H);
  S.frameTime = S.frameView.cursor;
  await runCourtDetection();
};
```

Add beside the other calibration button handlers:

```javascript
$('confirmUseBtn').onclick = () => { finishCalWizard(); };
$('confirmManualBtn').onclick = () => {
  S.detect = null;
  S.lines = {out:null, tin:null, service:null};
  S.wall = null; S.floor = null; S.work = null;
  setPhase('tap_out');
};
```

- [ ] **Step 7: Document the component in DESIGN.md**

Add a new subsection after §8.15 (Hero action cards):

```markdown
### 8.16 Auto-detect confirm screen (`#p-confirm`)

The single review surface for an automatically detected court. Overlays the
fitted calibration edges (their existing reserved hues — out `#35e0ff`,
service `#ff9f43`, tin `#b4ff3a`) and the floor wireframe on the frame, with
draggable anchor pucks at the four wall corners and the two short-line ends.

- Anchor pucks use the floor-wizard residual palette (§8.10): `#3ddc84` when
  the self-verification checks agree, `#f5c518` when one is off. **Never red**
  — §0.3 reserves red for OUT verdicts. Colour is never the only carrier: the
  status line names every anchor the checks flagged.
- Both status lines reserve their height with `&nbsp;` (§0.9) so a warning
  appearing after a drag never pushes the buttons around.
- Detection runs behind the sanctioned analyzing scrim (§8.12).
- Primary is the proxied `USE THIS CALIBRATION` (§3.4); the manual wizard is
  always one tap away via the secondary `TAP IT MANUALLY`.
```

- [ ] **Step 8: Verify in the browser**

Start the server and drive it with the `/verify` skill:

Run: `.venv/bin/python app.py` then load a test video, pick a frame, tap **USE FRAME**.
Expected: the analyzing scrim appears, then the confirm screen with the wireframe sitting on the court.

Check both themes at 390 × 844 (DESIGN.md §0.12). Confirm the buttons are ≥ 44 px and the primary ≥ 48 px.

- [ ] **Step 9: Commit**

```bash
git add index.html DESIGN.md
git commit -m "feat(ui): auto-detect the court and confirm it on one screen"
```

---

### Task 8: Client — draggable anchors, the divergence rule, and final verification

Spec §8.1 is the contract here: **anchors are authoritative for `planes`; detected line fits stay authoritative for `lines[]`.** A drag must never re-derive a line from two dragged corners — that would regress a hundreds-of-pixels fit to a 2-point fingertip fit, which is the precision loss this whole feature exists to avoid.

**Files:**
- Modify: `index.html`

**Interfaces:**
- Consumes: `confirmAnchorPoints()`, `buildConfirmView()`, `drawConfirmOverlay()` (Task 7).
- Produces: `S.confirmDrag` state; `confirmDragHitTest(x, y)`; `moveConfirmAnchor(key, x, y)`; `confirmDivergence()`.

- [ ] **Step 1: Add the drag state and hit test**

Add after `buildConfirmView()`:

```javascript
/* Dragging an anchor moves that anchor and refits the homographies. It never
   refits S.lines: those come from hundreds of detected edge pixels, and
   rebuilding a line from two dragged corners would be strictly less accurate
   than what detection already found (spec §8.1). Where a dragged corner ends
   up far from its line, we say so rather than silently picking a winner. */
const CONFIRM_DIVERGE_PX = 6;

function confirmDragHitTest(x, y){
  const radius = Math.max(18, S.W / 60);
  let best = null, bestDist = radius * radius;
  for(const anchor of confirmAnchorPoints()){
    const dx = anchor.xy[0] - x, dy = anchor.xy[1] - y;
    const dist = dx * dx + dy * dy;
    if(dist <= bestDist){ bestDist = dist; best = anchor.key; }
  }
  return best;
}

function moveConfirmAnchor(key, x, y){
  const [kind, id] = key.split(':');
  if(kind === 'wall' && S.wall) S.wall.points[+id] = [x, y];
  if(kind === 'floor' && S.floor){
    const landmark = S.floor.landmarks.find(l => l.id === id);
    if(landmark){ landmark.tap_px = [x, y]; landmark.refined_px = [x, y]; }
  }
  if(S.wall && S.floor) seedFrontSeamFromWall(S.floor);   // the seam is shared
  refitFloor();
}

/* Which fitted lines a dragged corner has walked away from. */
function confirmDivergence(){
  if(!S.wall || S.wall.points.length !== WALL_CORNERS.length) return [];
  const out = S.lines.out && S.lines.out.fit;
  if(!out) return [];
  const names = [];
  for(const index of [0, 1]){          // top_left, top_right sit on the out line
    const [x, y] = S.wall.points[index];
    if(Math.abs(out.m * x + out.b - y) > CONFIRM_DIVERGE_PX){
      names.push(WALL_CORNERS[index].label);
    }
  }
  return names;
}
```

- [ ] **Step 2: Route pointer events**

In the canvas `pointerdown` handler (`index.html:2003`), change the guard line to include the confirm phase and add the hit test:

```javascript
canvas.addEventListener('pointerdown', e => {
  if(S.phase !== 'tap_out' && S.phase !== 'tap_tin' && S.phase !== 'tap_service' && S.phase !== 'tap_wall' && S.phase !== 'tap_floor' && S.phase !== 'confirm') return;
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  if(S.phase === 'confirm' && pointers.size === 0){
    const r = canvas.getBoundingClientRect();
    const x = Math.round((e.clientX - r.left) * S.W / r.width);
    const y = Math.round((e.clientY - r.top) * S.H / r.height);
    S.confirmDrag = confirmDragHitTest(x, y);
  }
  pointers.set(e.pointerId, { x:e.clientX, y:e.clientY, startX:e.clientX, startY:e.clientY, moved:false, multi:false });
  // ... rest of the handler unchanged (the pinch block)
```

In the `pointermove` handler, insert this **immediately before** the existing
`if(pointers.size === 2 && pinch){` line — after `p.x`/`p.y` are updated, but before
either the pinch or the pan branch, so a drag wins over panning at zoom > 1:

```javascript
  if(S.confirmDrag && pointers.size === 1){
    const r = canvas.getBoundingClientRect();
    moveConfirmAnchor(S.confirmDrag,
      Math.min(S.W - 1, Math.max(0, Math.round((e.clientX - r.left) * S.W / r.width))),
      Math.min(S.H - 1, Math.max(0, Math.round((e.clientY - r.top) * S.H / r.height))));
    buildConfirmView();
    return;
  }
```

In `endPointer`, insert **after** the existing `if(pointers.size < 2) pinch = null;` line —
not before it. Returning early ahead of that line would leave a stale `pinch` object behind
and the next two-finger gesture would zoom from the wrong anchor:

```javascript
  if(S.confirmDrag && pointers.size === 0){
    S.confirmDrag = null;
    buildConfirmView();
    return;
  }
```

Add `confirmDrag:null,` to the `S` state object beside `detect:null`.

- [ ] **Step 3: Surface divergence in the confirm view**

In `buildConfirmView()`, replace the `$('confirmWarn')` block with:

```javascript
  const diverged = confirmDivergence();
  const messages = [];
  if(off.length)
    messages.push(`Off: ${off.map(check => check.id.replace(/_/g, ' ')).join(', ')}.`);
  if(diverged.length)
    messages.push(`${diverged.join(' and ')} no longer sits on the fitted out line — tap that line manually if the fit is wrong.`);
  $('confirmWarn').innerHTML = messages.length ? messages.join(' ') : '&nbsp;';
  $('confirmWarn').className = 'status' + (messages.length ? ' warn' : '');
```

- [ ] **Step 4: Verify the whole flow in the browser**

Use the `/verify` skill. Load a test video, pick a frame, tap **USE FRAME**, then:

1. Confirm the wireframe sits on the real court.
2. Drag a wall corner ~20 px off the out line → the divergence message appears, the floor wireframe moves, and the fitted cyan out line **does not move**.
3. Tap **USE THIS CALIBRATION** → lands on the clip editor.
4. Run a track end to end and confirm the run completes and produces calls.
5. Go back and tap **TAP IT MANUALLY** → the old wizard starts at the out line, unchanged.
6. Repeat 1–3 in the light theme at 390 × 844.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `331 passed, 1 deselected`

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "feat(ui): draggable confirm anchors with an explicit divergence rule"
```

---

## Done criteria (spec §13)

1. On both Bay Club clips, **USE FRAME** produces a confirm screen whose wireframe visibly hugs the real court with no dragging required.
2. Accepting produces a `calibration.json` that `judge_call` and `court_model.load_floor_calibration` parse with no code changes, and a run completes end to end.
3. The manual wizard still works, reached from the confirm screen, unchanged.
4. `.venv/bin/python -m pytest tests/ -q` green; UI verified in both themes at 390 × 844.

## Deferred (spec §9) — do not build these here

Vanishing-point line grouping; the tin+service → out-line extrapolation fallback; the scoring harness against all 46 stored calibrations; recording-time detection; lens distortion.

**One correction to spec §8, found while writing this plan.** The spec says the record
phase's on-site calibration "inherits this for free". That is true of the *accept* half —
`finishCalWizard()` already branches on `S.rec.calibrating`, so Task 7's confirm screen
works from either entry point — but **not** of the trigger. `recCalibrateBtn` freezes a
live camera frame and calls `setPhase('tap_out')` directly, and `grabDetectionFrames()`
seeks `vid`, which the record path is not using. Giving the record phase automatic
detection needs a second frame source (repeated grabs off the live `camVid` stream) and is
**out of scope here**: the record phase keeps the manual wizard, unchanged.
