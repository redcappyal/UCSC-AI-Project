# Stereo Core — Python Authority (Plan B1: spec Phase 3, part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Python stereo authority for the two-camera system: two-ray triangulation,
court-surface plane-snap with line calls, asynchronous-track interpolation + impact
detection, deterministic golden vectors (consumed later by Swift parity tests), and three
additive server endpoints (`/api/camera-model`, `camera_id` filtering on
`/api/calibration/latest`, `/api/camera-pair-check` agreement gate).

**Architecture:** All stereo math lives in a new `stereo_engine.py` (numpy only),
building on the existing `court_model.CameraModel` (pinhole in court feet, `ray()` /
`project()` primitives, undistorted-pixel contract). The server solves camera models
(Python stays the calibration authority); phones will exchange solved
`CameraModel.to_dict()` JSON in Plan B2. Golden vectors are generated once,
deterministically, into both `tests/` (Python) and `ios/Tests/Fixtures/` (Swift, Plan B2).

**Tech Stack:** Python 3 + numpy (no new dependencies), Flask (existing app.py), pytest.

## Global Constraints

- Python: numpy only — no new pip dependencies. No `np.random` without a fixed seed; the golden generator uses `np.random.default_rng(7)`.
- Court frame (court_model.py module docstring): origin = front-wall/floor seam at the LEFT corner seen from the back-wall camera; x rightward 0→21 ft; y front wall 0 → back wall 32 ft; z up; units FEET everywhere.
- `CameraModel` contract: operates in UNDISTORTED pixel space — `project` returns undistorted pixels, `ray` expects them; observations are undistorted with `court_model.undistort_point(p, model.distortion)` first. `stereo_engine` public functions take RAW pixels + the model and undistort internally (documented per function).
- Constants come from court_model.py — never re-derive: `COURT_WIDTH_FT = 21.0`, `COURT_LENGTH_FT = 32.0`, `OUT_LINE_HEIGHT_FT = 15.0`, `TIN_TOP_HEIGHT_FT = 19.0/12.0`. New: back-wall out line 7.0 ft; side-wall out line slopes linearly 15.0 ft (y=0) → 7.0 ft (y=32).
- Server changes are ADDITIVE: existing routes' behavior with existing payloads is byte-identical; new keys/params are optional. `/api/camera-check`'s always-200 convention applies to the new endpoints too.
- Test conventions (match exactly): pytest, files `tests/test_<module>.py`, each file starts with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` before repo imports; Flask routes via `app_module.app.test_client()`; synthetic cameras via `tests/synthetic3d.make_camera`.
- Full-suite gate: `pytest tests/ -q` green before every commit (baseline: the suite that CI runs; do not break existing tests).
- Commit after every task with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Branch: `claude/stereo-core` (already created off main @ 1c0eb57).

## File Structure

- `stereo_engine.py` (new, repo root — sibling of court_model.py): all stereo math. Sections: triangulation → surfaces/calls → snap → interpolation/track → impacts → pair agreement.
- `tests/test_stereo_engine.py` (new): unit tests for triangulation/snap/calls.
- `tests/test_stereo_track.py` (new): interpolation + impact-detection tests (synthetic trajectory).
- `tests/generate_stereo_goldens.py` (new, not a test): deterministic golden generator writing `tests/stereo_goldens.json` and `ios/Tests/Fixtures/stereo_goldens.json` (identical content).
- `tests/test_stereo_goldens.py` (new): asserts stereo_engine reproduces the checked-in goldens.
- `app.py` (modify): three additive endpoint changes.
- `tests/test_stereo_endpoints.py` (new): endpoint tests.

---

### Task 1: Two-ray triangulation

**Files:**
- Create: `stereo_engine.py`
- Test: `tests/test_stereo_engine.py`

**Interfaces:**
- Consumes: `court_model.CameraModel` (`.ray(pixel) -> (origin, unit_dir)`, `.project(court_xyz) -> (u, v)`), `court_model.undistort_point(pixel, distortion)`, `tests/synthetic3d.make_camera`.
- Produces (later tasks + Plan B2 Swift parity depend on these exact names):
  ```python
  def triangulate(model_a, model_b, px_a, px_b):
      """RAW pixels in; undistorts internally. Returns (point_ft: np.ndarray(3),
      gap_ft: float) or (None, inf) when rays are near-parallel or the
      closest-approach parameters put a point behind either camera."""
  PARALLEL_EPS = 1e-9
  ```
  Method: closest-approach midpoint (s,t solved from the 2x2 normal equations), reject `s <= 0 or t <= 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stereo_engine.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import stereo_engine
from synthetic3d import make_camera


def make_fin_pair():
    """Two phones on the back-wall fins, 3 ft apart, 7 ft up, aimed at the
    front wall — the production mounting from the spec."""
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    return left, right


def test_triangulate_recovers_known_point():
    left, right = make_fin_pair()
    point = np.array([8.0, 12.0, 3.5])
    point_ft, gap_ft = stereo_engine.triangulate(
        left, right, left.project(point), right.project(point))
    assert gap_ft < 1e-9
    assert np.allclose(point_ft, point, atol=1e-9)


def test_triangulate_pixel_noise_bounded_error():
    left, right = make_fin_pair()
    point = np.array([10.5, 16.0, 2.0])   # mid-court
    px_a = np.asarray(left.project(point)) + np.array([2.0, -2.0])
    px_b = np.asarray(right.project(point)) + np.array([-2.0, 2.0])
    point_ft, gap_ft = stereo_engine.triangulate(left, right, px_a, px_b)
    # Narrow-baseline depth error dominates; lateral/height stay tight.
    assert abs(point_ft[0] - point[0]) < 0.2
    assert abs(point_ft[2] - point[2]) < 0.2
    assert abs(point_ft[1] - point[1]) < 1.5
    assert gap_ft < 0.5


def test_triangulate_parallel_rays_returns_none():
    left, _ = make_fin_pair()
    # Same camera twice with the same pixel: identical rays are degenerate.
    point_ft, gap_ft = stereo_engine.triangulate(
        left, left, (960.0, 540.0), (960.0, 540.0))
    assert point_ft is None
    assert gap_ft == np.inf


def test_triangulate_behind_camera_rejected():
    left, right = make_fin_pair()
    # A point BEHIND the cameras (y > camera y): project() would raise, so
    # build pixels from a valid point but flip one ray by picking pixels
    # whose closest approach lands behind: use crossing rays aimed away.
    # Construct directly: pixel far left on one camera, far right on the
    # other, so rays diverge and closest approach is at negative s/t.
    point_ft, gap_ft = stereo_engine.triangulate(
        left, right, (-4000.0, 540.0), (5900.0, 540.0))
    assert point_ft is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_engine.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'stereo_engine'`.

- [ ] **Step 3: Implement**

```python
# stereo_engine.py
"""Stereo math for the two-camera system.

Builds on court_model.CameraModel (pinhole in court FEET; origin front-left
floor seam, x right 0-21, y front 0 -> back 32, z up). Public functions take
RAW pixels and undistort internally via court_model.undistort_point; the
CameraModel primitives themselves stay in undistorted pixel space.
"""
import numpy as np

import court_model
from court_model import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    OUT_LINE_HEIGHT_FT,
    TIN_TOP_HEIGHT_FT,
)

PARALLEL_EPS = 1e-9


def _undistorted_ray(model, pixel):
    pixel = court_model.undistort_point(pixel, model.distortion)
    return model.ray(pixel)


def triangulate(model_a, model_b, px_a, px_b):
    """Closest-approach midpoint of the two viewing rays.

    Returns (point_ft, gap_ft); (None, inf) for near-parallel rays or when
    the closest approach lies behind either camera (s or t <= 0).
    """
    o1, d1 = _undistorted_ray(model_a, px_a)
    o2, d2 = _undistorted_ray(model_b, px_b)
    w0 = o1 - o2
    b = float(np.dot(d1, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = 1.0 - b * b          # a = c = 1 for unit directions
    if denom < PARALLEL_EPS:
        return None, np.inf
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    if s <= 0.0 or t <= 0.0:
        return None, np.inf
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_engine.py -q` — expected: 4 passed.
Then the full suite: `pytest tests/ -q` — expected: green, no regressions.

- [ ] **Step 5: Commit**

```bash
git add stereo_engine.py tests/test_stereo_engine.py
git commit -m "feat(stereo): two-ray closest-approach triangulation"
```

---

### Task 2: Court surfaces, plane snap, and line calls

**Files:**
- Modify: `stereo_engine.py` (append)
- Test: `tests/test_stereo_engine.py` (append)

**Interfaces:**
- Consumes: Task 1's `_undistorted_ray`; court_model constants.
- Produces (exact names — Plan B2 mirrors them):
  ```python
  BACK_WALL_OUT_HEIGHT_FT = 7.0
  SURFACES = {"floor", "front_wall", "left_wall", "right_wall", "back_wall"}
  def surface_plane(name) -> (point_ft: np.ndarray(3), normal: np.ndarray(3))  # inward normal
  def side_wall_out_height_ft(y_ft) -> float       # 15.0 at y=0 -> 7.0 at y=32, linear
  def call_for_impact(surface, point_ft) -> (call: str, margin_ft: float)
      # front_wall: "out" (z >= 15), "down" (z <= tin), else "in"
      # left/right_wall: "out" above the sloped line, else "in"
      # back_wall: "out" above 7.0, else "in"
      # floor: ("bounce", 0.0)
      # margin_ft >= 0 always: distance to the deciding line (for "in": distance
      # to the NEAREST deciding line on that surface)
  def snap_to_plane(model, px, surface) -> np.ndarray(3) | None
      # RAW pixel; ray-plane intersection; None if ray parallel to plane,
      # intersection behind camera, or landing outside the court's bounds for
      # that surface (0.5 ft slack)
  def fuse_snaps(p_a, p_b) -> np.ndarray(3) | None   # mean; passes through a lone point; None if both None
  ```

- [ ] **Step 1: Write the failing tests (append to tests/test_stereo_engine.py)**

```python
def test_surface_planes_and_side_out_slope():
    point, normal = stereo_engine.surface_plane("front_wall")
    assert point[1] == 0.0 and np.allclose(normal, [0.0, 1.0, 0.0])
    point, normal = stereo_engine.surface_plane("floor")
    assert point[2] == 0.0 and np.allclose(normal, [0.0, 0.0, 1.0])
    assert stereo_engine.side_wall_out_height_ft(0.0) == 15.0
    assert stereo_engine.side_wall_out_height_ft(32.0) == 7.0
    assert stereo_engine.side_wall_out_height_ft(16.0) == 11.0


def test_calls_front_wall():
    out_call = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 15.4]))
    assert out_call == ("out", 0.3999999999999986) or (
        out_call[0] == "out" and abs(out_call[1] - 0.4) < 1e-9)
    call, margin = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 1.0]))
    assert call == "down" and abs(margin - (19.0 / 12.0 - 1.0)) < 1e-9
    call, margin = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 8.0]))
    assert call == "in" and abs(margin - 7.0) < 1e-9   # nearer to out line (15-8) than tin


def test_calls_side_and_back_walls_and_floor():
    call, margin = stereo_engine.call_for_impact("left_wall", np.array([0.0, 16.0, 11.5]))
    assert call == "out" and abs(margin - 0.5) < 1e-9
    call, margin = stereo_engine.call_for_impact("right_wall", np.array([21.0, 16.0, 10.0]))
    assert call == "in" and abs(margin - 1.0) < 1e-9
    call, margin = stereo_engine.call_for_impact("back_wall", np.array([5.0, 32.0, 7.5]))
    assert call == "out" and abs(margin - 0.5) < 1e-9
    assert stereo_engine.call_for_impact("floor", np.array([5.0, 20.0, 0.0])) == ("bounce", 0.0)


def test_snap_to_plane_recovers_wall_point():
    left, right = make_fin_pair()
    impact = np.array([13.0, 0.0, 12.0])   # on the front wall
    snap_a = stereo_engine.snap_to_plane(left, left.project(impact), "front_wall")
    snap_b = stereo_engine.snap_to_plane(right, right.project(impact), "front_wall")
    assert np.allclose(snap_a, impact, atol=1e-9)
    assert np.allclose(snap_b, impact, atol=1e-9)
    fused = stereo_engine.fuse_snaps(snap_a, snap_b)
    assert np.allclose(fused, impact, atol=1e-9)
    assert stereo_engine.fuse_snaps(None, snap_b) is snap_b
    assert stereo_engine.fuse_snaps(None, None) is None


def test_snap_rejects_out_of_bounds_and_parallel():
    left, _ = make_fin_pair()
    # A pixel whose floor intersection lies far outside the court.
    assert stereo_engine.snap_to_plane(left, (100000.0, 540.0), "floor") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_engine.py -q`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'surface_plane'`.

- [ ] **Step 3: Implement (append to stereo_engine.py)**

```python
BACK_WALL_OUT_HEIGHT_FT = 7.0

_SURFACE_PLANES = {
    "floor": (np.zeros(3), np.array([0.0, 0.0, 1.0])),
    "front_wall": (np.zeros(3), np.array([0.0, 1.0, 0.0])),
    "back_wall": (np.array([0.0, COURT_LENGTH_FT, 0.0]), np.array([0.0, -1.0, 0.0])),
    "left_wall": (np.zeros(3), np.array([1.0, 0.0, 0.0])),
    "right_wall": (np.array([COURT_WIDTH_FT, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
}
SURFACES = set(_SURFACE_PLANES)
_BOUNDS_SLACK_FT = 0.5


def surface_plane(name):
    point, normal = _SURFACE_PLANES[name]
    return point.copy(), normal.copy()


def side_wall_out_height_ft(y_ft):
    """WSF side-wall out line: 15 ft at the front wall, 7 ft at the back."""
    return OUT_LINE_HEIGHT_FT + (
        BACK_WALL_OUT_HEIGHT_FT - OUT_LINE_HEIGHT_FT) * (y_ft / COURT_LENGTH_FT)


def call_for_impact(surface, point_ft):
    """(call, margin_ft) for an impact point known to lie on `surface`.

    margin_ft is the distance to the deciding line — how clear the call is.
    """
    x, y, z = (float(v) for v in point_ft)
    if surface == "floor":
        return "bounce", 0.0
    if surface == "front_wall":
        if z >= OUT_LINE_HEIGHT_FT:
            return "out", z - OUT_LINE_HEIGHT_FT
        if z <= TIN_TOP_HEIGHT_FT:
            return "down", TIN_TOP_HEIGHT_FT - z
        return "in", min(OUT_LINE_HEIGHT_FT - z, z - TIN_TOP_HEIGHT_FT)
    if surface in ("left_wall", "right_wall"):
        line = side_wall_out_height_ft(y)
        return ("out", z - line) if z >= line else ("in", line - z)
    if surface == "back_wall":
        line = BACK_WALL_OUT_HEIGHT_FT
        return ("out", z - line) if z >= line else ("in", line - z)
    raise ValueError(f"Unknown surface: {surface}")


def _in_surface_bounds(surface, point_ft):
    x, y, z = (float(v) for v in point_ft)
    lo = -_BOUNDS_SLACK_FT
    if surface == "floor":
        return lo <= x <= COURT_WIDTH_FT + _BOUNDS_SLACK_FT and lo <= y <= COURT_LENGTH_FT + _BOUNDS_SLACK_FT
    if surface in ("front_wall", "back_wall"):
        return lo <= x <= COURT_WIDTH_FT + _BOUNDS_SLACK_FT and lo <= z
    return lo <= y <= COURT_LENGTH_FT + _BOUNDS_SLACK_FT and lo <= z


def snap_to_plane(model, px, surface):
    """Intersect the (undistorted) viewing ray with a court surface plane."""
    plane_point, normal = _SURFACE_PLANES[surface]
    origin, direction = _undistorted_ray(model, px)
    denom = float(np.dot(direction, normal))
    if abs(denom) < PARALLEL_EPS:
        return None
    t = float(np.dot(plane_point - origin, normal)) / denom
    if t <= 0.0:
        return None
    point = origin + t * direction
    return point if _in_surface_bounds(surface, point) else None


def fuse_snaps(p_a, p_b):
    if p_a is None:
        return p_b
    if p_b is None:
        return p_a
    return (p_a + p_b) / 2.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_engine.py -q` — expected: 9 passed. Then `pytest tests/ -q` green.

- [ ] **Step 5: Commit**

```bash
git add stereo_engine.py tests/test_stereo_engine.py
git commit -m "feat(stereo): court surfaces, plane snap, and line calls"
```

---

### Task 3: Asynchronous-track interpolation and impact detection

**Files:**
- Modify: `stereo_engine.py` (append)
- Create: `tests/test_stereo_track.py`

**Interfaces:**
- Consumes: Tasks 1–2 (`triangulate`, `snap_to_plane`, `fuse_snaps`, `call_for_impact`, `SURFACES`).
- Produces (exact names):
  ```python
  @dataclass(frozen=True) class TrackSample:  t_s: float; px: tuple  # RAW pixel
  @dataclass(frozen=True) class TrackPoint3D: t_s: float; point_ft: np.ndarray; gap_ft: float
  @dataclass(frozen=True) class Impact:
      t_s: float; surface: str; point_ft: np.ndarray; call: str
      margin_ft: float; confidence: str        # "high" | "one_view" | "no_call"
      snap_disagreement_ft: float | None
  FIT_WINDOW_SAMPLES = 7
  MIN_FIT_SAMPLES = 4
  SNAP_DISAGREEMENT_MAX_FT = 0.3
  IMPACT_PROXIMITY_FT = 1.5
  def eval_pixel_track(samples, t_s, window=FIT_WINDOW_SAMPLES) -> np.ndarray(2) | None
      # quadratic least-squares over the `window` samples nearest t_s (per
      # u and v); None if fewer than MIN_FIT_SAMPLES in the window
  def build_track3d(model_a, samples_a, model_b, samples_b, hz=120.0) -> list[TrackPoint3D]
      # common timeline over the overlap of both cameras' time ranges;
      # skips timeline points where either eval is None or triangulate fails
  def detect_impacts(model_a, samples_a, model_b, samples_b, track=None) -> list[Impact]
  ```
  Impact algorithm: on the 3D track, for each surface, find local minima of
  distance-to-plane below `IMPACT_PROXIMITY_FT` where the velocity component
  along the plane normal changes sign across the minimum; `t_impact` = the
  minimum's time. Per camera, refit the pixel track using only samples in
  `[t_impact - 0.25, t_impact - 1/240]` (pre-impact side; the 2D path kinks at
  the impact), evaluate at `t_impact`, snap to the surface. Confidence: both
  snaps present and within `SNAP_DISAGREEMENT_MAX_FT` → `"high"`; exactly one
  snap → `"one_view"`; both missing → `"no_call"` (point falls back to the
  track minimum's 3D point). Consecutive impacts on the same surface within
  60 ms merge (keep the deeper minimum).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stereo_track.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from synthetic3d import make_camera

GRAVITY_FT_S2 = -32.174


def make_fin_pair():
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    return left, right


def simulate_front_wall_shot(dt=1.0 / 480.0):
    """Ballistic flight into the front wall; returns (states, impact) where
    states is [(t, xyz)] and impact is the exact (t, xyz) of the wall hit."""
    pos = np.array([16.0, 26.0, 4.0])
    vel = np.array([-6.0, -55.0, 9.0])
    t = 0.0
    states, impact = [], None
    while t < 0.9:
        states.append((t, pos.copy()))
        nxt = pos + vel * dt + 0.5 * np.array([0, 0, GRAVITY_FT_S2]) * dt * dt
        vel = vel + np.array([0, 0, GRAVITY_FT_S2]) * dt
        if impact is None and nxt[1] <= 0.0:
            frac = pos[1] / (pos[1] - nxt[1])
            impact = (t + frac * dt, pos + (nxt - pos) * frac)
            vel[1] = -vel[1] * 0.7
            nxt = pos + vel * dt
        pos = nxt
        t += dt
    return states, impact


def sample_camera(states, model, fps=60.0, phase_s=0.0):
    samples, next_t = [], phase_s
    for t, xyz in states:
        if t >= next_t:
            try:
                samples.append(TrackSample(t_s=t, px=tuple(model.project(xyz))))
            except ValueError:
                pass
            next_t += 1.0 / fps
    return samples


def test_eval_pixel_track_matches_projection():
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    t_query = samples[6].t_s + 0.004        # off-sample time, pre-impact
    px = stereo_engine.eval_pixel_track(samples, t_query)
    true_xyz = min(states, key=lambda s: abs(s[0] - t_query))[1]
    assert np.linalg.norm(px - np.asarray(left.project(true_xyz))) < 3.0


def test_build_track3d_tracks_flight():
    left, right = make_fin_pair()
    states, impact = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    track = stereo_engine.build_track3d(left, samples_a, right, samples_b)
    assert len(track) > 40
    mid = track[len(track) // 4]
    true_xyz = min(states, key=lambda s: abs(s[0] - mid.t_s))[1]
    # Depth (y) is the weak axis at this baseline; lateral/height are tight.
    assert abs(mid.point_ft[0] - true_xyz[0]) < 0.3
    assert abs(mid.point_ft[2] - true_xyz[2]) < 0.3


def test_detect_impacts_front_wall_call_and_position():
    left, right = make_fin_pair()
    states, (t_true, p_true) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    front = [i for i in impacts if i.surface == "front_wall"]
    assert len(front) == 1
    hit = front[0]
    assert abs(hit.t_s - t_true) < 0.012
    assert hit.confidence == "high"
    assert np.linalg.norm(hit.point_ft - p_true) < 0.15
    call, margin = stereo_engine.call_for_impact("front_wall", p_true)
    assert hit.call == call


def test_detect_impacts_one_view_when_occluded():
    left, right = make_fin_pair()
    states, (t_true, _) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = [s for s in sample_camera(states, right, fps=60.0, phase_s=0.007)
                 if not (t_true - 0.3 <= s.t_s <= t_true + 0.1)]
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    front = [i for i in impacts if i.surface == "front_wall"]
    assert len(front) == 1
    assert front[0].confidence == "one_view"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_track.py -q`
Expected: FAIL with `ImportError: cannot import name 'TrackSample'`.

- [ ] **Step 3: Implement (append to stereo_engine.py)**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackSample:
    t_s: float
    px: tuple


@dataclass(frozen=True)
class TrackPoint3D:
    t_s: float
    point_ft: np.ndarray
    gap_ft: float


@dataclass(frozen=True)
class Impact:
    t_s: float
    surface: str
    point_ft: np.ndarray
    call: str
    margin_ft: float
    confidence: str
    snap_disagreement_ft: float | None


FIT_WINDOW_SAMPLES = 7
MIN_FIT_SAMPLES = 4
SNAP_DISAGREEMENT_MAX_FT = 0.3
IMPACT_PROXIMITY_FT = 1.5
_IMPACT_MERGE_S = 0.060
_PRE_IMPACT_WINDOW_S = 0.25
_PRE_IMPACT_GUARD_S = 1.0 / 240.0


def eval_pixel_track(samples, t_s, window=FIT_WINDOW_SAMPLES):
    """Quadratic least-squares over the `window` samples nearest t_s."""
    if not samples:
        return None
    nearest = sorted(samples, key=lambda s: abs(s.t_s - t_s))[:window]
    if len(nearest) < MIN_FIT_SAMPLES:
        return None
    ts = np.array([s.t_s for s in nearest]) - t_s   # center for conditioning
    us = np.array([s.px[0] for s in nearest])
    vs = np.array([s.px[1] for s in nearest])
    coeff_u = np.polyfit(ts, us, 2)
    coeff_v = np.polyfit(ts, vs, 2)
    return np.array([np.polyval(coeff_u, 0.0), np.polyval(coeff_v, 0.0)])


def build_track3d(model_a, samples_a, model_b, samples_b, hz=120.0):
    if not samples_a or not samples_b:
        return []
    t_lo = max(samples_a[0].t_s, samples_b[0].t_s)
    t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s)
    track = []
    for t_s in np.arange(t_lo, t_hi, 1.0 / hz):
        px_a = eval_pixel_track(samples_a, t_s)
        px_b = eval_pixel_track(samples_b, t_s)
        if px_a is None or px_b is None:
            continue
        point_ft, gap_ft = triangulate(model_a, model_b, px_a, px_b)
        if point_ft is None:
            continue
        track.append(TrackPoint3D(t_s=float(t_s), point_ft=point_ft, gap_ft=gap_ft))
    return track


def _plane_distance(surface, point_ft):
    plane_point, normal = _SURFACE_PLANES[surface]
    return float(np.dot(point_ft - plane_point, normal))


def _pre_impact_eval(samples, t_impact):
    window = [s for s in samples
              if t_impact - _PRE_IMPACT_WINDOW_S <= s.t_s <= t_impact - _PRE_IMPACT_GUARD_S]
    if len(window) < MIN_FIT_SAMPLES:
        return None
    return eval_pixel_track(window, t_impact, window=len(window))


def detect_impacts(model_a, samples_a, model_b, samples_b, track=None):
    if track is None:
        track = build_track3d(model_a, samples_a, model_b, samples_b)
    if len(track) < 3:
        return []
    impacts = []
    for surface in SURFACES:
        dists = np.array([_plane_distance(surface, p.point_ft) for p in track])
        for i in range(1, len(track) - 1):
            if dists[i] > IMPACT_PROXIMITY_FT:
                continue
            if not (dists[i] <= dists[i - 1] and dists[i] <= dists[i + 1]):
                continue
            approaching = dists[i] - dists[i - 1]
            leaving = dists[i + 1] - dists[i]
            if not (approaching < 0.0 and leaving > 0.0):
                continue
            t_impact = track[i].t_s
            snap_a = snap_b = None
            px_a = _pre_impact_eval(samples_a, t_impact)
            px_b = _pre_impact_eval(samples_b, t_impact)
            if px_a is not None:
                snap_a = snap_to_plane(model_a, px_a, surface)
            if px_b is not None:
                snap_b = snap_to_plane(model_b, px_b, surface)
            disagreement = None
            if snap_a is not None and snap_b is not None:
                disagreement = float(np.linalg.norm(snap_a - snap_b))
                confidence = "high" if disagreement <= SNAP_DISAGREEMENT_MAX_FT else "one_view"
                point = fuse_snaps(snap_a, snap_b)
            elif snap_a is not None or snap_b is not None:
                confidence = "one_view"
                point = fuse_snaps(snap_a, snap_b)
            else:
                confidence = "no_call"
                point = track[i].point_ft
            call, margin = call_for_impact(surface, point)
            impacts.append(Impact(
                t_s=t_impact, surface=surface, point_ft=point, call=call,
                margin_ft=margin, confidence=confidence,
                snap_disagreement_ft=disagreement))
    impacts.sort(key=lambda imp: imp.t_s)
    merged = []
    for imp in impacts:
        if merged and merged[-1].surface == imp.surface and \
                imp.t_s - merged[-1].t_s < _IMPACT_MERGE_S:
            prev_d = abs(_plane_distance(imp.surface, merged[-1].point_ft))
            cur_d = abs(_plane_distance(imp.surface, imp.point_ft))
            if cur_d < prev_d:
                merged[-1] = imp
            continue
        merged.append(imp)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_track.py tests/test_stereo_engine.py -q` — expected: all pass.
If `test_detect_impacts_front_wall_call_and_position` misses tolerance, debug the
pre-impact window first (the 2D path kinks at the wall; the fit must not straddle it) —
do not widen tolerances without understanding why. Then `pytest tests/ -q` green.

- [ ] **Step 5: Commit**

```bash
git add stereo_engine.py tests/test_stereo_track.py
git commit -m "feat(stereo): async track interpolation and impact detection"
```

---

### Task 4: Deterministic golden vectors (Python + Swift fixture)

**Files:**
- Create: `tests/generate_stereo_goldens.py`
- Create: `tests/stereo_goldens.json` (generated, committed)
- Create: `ios/Tests/Fixtures/stereo_goldens.json` (identical copy, committed — Plan B2's Swift parity tests read it)
- Test: `tests/test_stereo_goldens.py`

**Interfaces:**
- Consumes: Tasks 1–3 (all public functions), `CameraModel.to_dict()`.
- Produces: the golden JSON schema (Plan B2 depends on these exact keys):
  ```json
  {
    "schema": "stereo-goldens-v1",
    "cameras": {"left": {<CameraModel.to_dict()>}, "right": {...}},
    "triangulation_cases": [{"px_a": [u,v], "px_b": [u,v], "point_ft": [x,y,z], "gap_ft": g}],
    "snap_cases": [{"camera": "left"|"right", "px": [u,v], "surface": s, "point_ft": [x,y,z]}],
    "call_cases": [{"surface": s, "point_ft": [x,y,z], "call": c, "margin_ft": m}],
    "trajectory": {
      "samples_a": [{"t_s": t, "px": [u,v]}], "samples_b": [...],
      "impacts": [{"t_s": t, "surface": s, "point_ft": [x,y,z], "call": c,
                    "margin_ft": m, "confidence": conf}]
    }
  }
  ```

- [ ] **Step 1: Write the generator**

```python
# tests/generate_stereo_goldens.py
"""Regenerate the stereo golden vectors (deterministic — rng seed 7).

Run from the repo root:  python tests/generate_stereo_goldens.py
Writes tests/stereo_goldens.json and ios/Tests/Fixtures/stereo_goldens.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from synthetic3d import make_camera
from test_stereo_track import sample_camera, simulate_front_wall_shot


def main():
    rng = np.random.default_rng(7)
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))

    triangulation_cases = []
    for _ in range(12):
        point = np.array([rng.uniform(2, 19), rng.uniform(2, 28), rng.uniform(0.5, 12)])
        px_a, px_b = left.project(point), right.project(point)
        got, gap = stereo_engine.triangulate(left, right, px_a, px_b)
        triangulation_cases.append({
            "px_a": list(map(float, px_a)), "px_b": list(map(float, px_b)),
            "point_ft": [float(v) for v in got], "gap_ft": float(gap)})

    snap_cases, call_cases = [], []
    wall_points = [("front_wall", np.array([13.0, 0.0, 12.0])),
                   ("front_wall", np.array([6.0, 0.0, 15.8])),
                   ("front_wall", np.array([10.5, 0.0, 1.2])),
                   ("left_wall", np.array([0.0, 10.0, 12.9])),
                   ("right_wall", np.array([21.0, 24.0, 8.0])),
                   ("floor", np.array([8.0, 20.0, 0.0]))]
    for surface, point in wall_points:
        for name, model in (("left", left), ("right", right)):
            snap = stereo_engine.snap_to_plane(model, model.project(point), surface)
            snap_cases.append({"camera": name, "px": list(map(float, model.project(point))),
                               "surface": surface, "point_ft": [float(v) for v in snap]})
        call, margin = stereo_engine.call_for_impact(surface, point)
        call_cases.append({"surface": surface, "point_ft": [float(v) for v in point],
                           "call": call, "margin_ft": float(margin)})

    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)

    goldens = {
        "schema": "stereo-goldens-v1",
        "cameras": {"left": left.to_dict(), "right": right.to_dict()},
        "triangulation_cases": triangulation_cases,
        "snap_cases": snap_cases,
        "call_cases": call_cases,
        "trajectory": {
            "samples_a": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_a],
            "samples_b": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_b],
            "impacts": [{"t_s": i.t_s, "surface": i.surface,
                          "point_ft": [float(v) for v in i.point_ft], "call": i.call,
                          "margin_ft": i.margin_ft, "confidence": i.confidence}
                         for i in impacts],
        },
    }
    payload = json.dumps(goldens, indent=2, sort_keys=True)
    (REPO / "tests" / "stereo_goldens.json").write_text(payload)
    fixture_dir = REPO / "ios" / "Tests" / "Fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "stereo_goldens.json").write_text(payload)
    print(f"wrote {len(payload)} bytes to tests/ and ios/Tests/Fixtures/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing golden test**

```python
# tests/test_stereo_goldens.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from court_model import CameraModel

GOLDENS = Path(__file__).resolve().parent / "stereo_goldens.json"


def load():
    data = json.loads(GOLDENS.read_text())
    left = CameraModel.from_dict(data["cameras"]["left"])
    right = CameraModel.from_dict(data["cameras"]["right"])
    return data, left, right


def test_goldens_file_matches_ios_fixture():
    ios_copy = GOLDENS.parents[1] / "ios" / "Tests" / "Fixtures" / "stereo_goldens.json"
    assert GOLDENS.read_text() == ios_copy.read_text()


def test_triangulation_goldens():
    data, left, right = load()
    for case in data["triangulation_cases"]:
        point, gap = stereo_engine.triangulate(left, right, case["px_a"], case["px_b"])
        assert np.allclose(point, case["point_ft"], atol=1e-9)
        assert abs(gap - case["gap_ft"]) < 1e-9


def test_snap_and_call_goldens():
    data, left, right = load()
    models = {"left": left, "right": right}
    for case in data["snap_cases"]:
        snap = stereo_engine.snap_to_plane(models[case["camera"]], case["px"], case["surface"])
        assert np.allclose(snap, case["point_ft"], atol=1e-9)
    for case in data["call_cases"]:
        call, margin = stereo_engine.call_for_impact(case["surface"], np.array(case["point_ft"]))
        assert call == case["call"] and abs(margin - case["margin_ft"]) < 1e-9


def test_trajectory_impact_goldens():
    data, left, right = load()
    samples_a = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in data["trajectory"]["samples_a"]]
    samples_b = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in data["trajectory"]["samples_b"]]
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    expected = data["trajectory"]["impacts"]
    assert len(impacts) == len(expected)
    for got, want in zip(impacts, expected):
        assert got.surface == want["surface"] and got.call == want["call"]
        assert abs(got.t_s - want["t_s"]) < 1e-9
        assert np.allclose(got.point_ft, want["point_ft"], atol=1e-9)
```

- [ ] **Step 3: Run to verify failure, then generate**

Run: `pytest tests/test_stereo_goldens.py -q` — expected FAIL (missing `tests/stereo_goldens.json`).
Then: `python tests/generate_stereo_goldens.py` — expected: "wrote N bytes".

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_goldens.py -q` — expected: 4 passed. Then `pytest tests/ -q` green.
Also confirm determinism: run the generator a second time; `git status --short` must show no changes.

- [ ] **Step 5: Commit**

```bash
git add tests/generate_stereo_goldens.py tests/test_stereo_goldens.py tests/stereo_goldens.json ios/Tests/Fixtures/stereo_goldens.json
git commit -m "feat(stereo): deterministic golden vectors for python and swift parity"
```

---

### Task 5: `/api/camera-model` endpoint

**Files:**
- Modify: `app.py` (add route next to `/api/camera-check`, app.py:508)
- Test: `tests/test_stereo_endpoints.py` (create)

**Interfaces:**
- Consumes: `court_model.solve_camera_model(calibration) -> (CameraModel | None, info)`, `CameraModel.to_dict()`.
- Produces: `POST /api/camera-model` — request `{"calibration": {...}}` (or `{"calibration_json": "..."}` like camera-check); response always 200: on solve success `{"ok": true, "status": "ok", "camera_model": {<to_dict>}, ...info}`; on failure the same shape as camera-check (`ok: true`, failing `status`, no `camera_model` key). Plan B2's phones call this to obtain their solved model.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stereo_endpoints.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from court_model import CameraModel
from synthetic3d import make_camera
from test_camera_model import _synthetic_calibration


def _client():
    import app as app_module
    return app_module.app.test_client()


def test_camera_model_endpoint_returns_solved_model():
    client = _client()
    camera = make_camera()
    body = client.post("/api/camera-model",
                       json={"calibration": _synthetic_calibration(camera)}).get_json()
    assert body["ok"] is True and body["status"] == "ok"
    model = CameraModel.from_dict(body["camera_model"])
    # The solved model must project a known court point close to the
    # synthetic camera's own projection.
    point = np.array([10.5, 16.0, 3.0])
    assert np.linalg.norm(
        np.asarray(model.project(point)) - np.asarray(camera.project(point))) < 2.0


def test_camera_model_endpoint_failure_has_no_model_key():
    client = _client()
    body = client.post("/api/camera-model", json={}).get_json()
    assert body["ok"] is True
    assert body["status"] in ("no_frame_size", "invalid_json")
    assert "camera_model" not in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_endpoints.py -q`
Expected: FAIL with 404 (`get_json()` returns None → AttributeError/assertion on None).

- [ ] **Step 3: Implement (add to app.py, directly below the `/api/camera-check` route)**

```python
@app.route("/api/camera-model", methods=["POST"])
def api_camera_model():
    """Solve the full camera model for a calibration and return it as JSON.

    Same input contract and always-200 convention as /api/camera-check; adds
    the solved model (court_model.CameraModel.to_dict) under "camera_model"
    when the solve succeeds. Phones exchange these solved models at pairing.
    """
    payload = request.get_json(silent=True) or {}
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raw = payload.get("calibration_json", "")
        try:
            calibration = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return jsonify({"ok": True, "status": "invalid_json"})
    model, info = court_model.solve_camera_model(calibration)
    response = {"ok": True, **info}
    if model is not None and info.get("status") == "ok":
        response["camera_model"] = model.to_dict()
    return jsonify(response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_endpoints.py -q` — expected: 2 passed. Then `pytest tests/ -q` green.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_stereo_endpoints.py
git commit -m "feat(server): camera-model endpoint returning the solved model"
```

---

### Task 6: Calibration camera identity + latest filter

**Files:**
- Modify: `app.py:473-505` (`/api/calibration/latest`)
- Test: `tests/test_stereo_endpoints.py` (append)

**Interfaces:**
- Consumes: existing storage layout (`RUNS_DIR.glob("*/calibration.json")`, mtime-latest).
- Produces: optional `?camera_id=<id>` query param on `GET /api/calibration/latest`. A calibration whose JSON contains a top-level `"camera_id"` key matching the param is eligible; when the param is present, non-matching (or id-less) calibrations are skipped. Without the param, behavior is byte-identical to today (all calibrations eligible). `camera_id` itself is pass-through data — `/api/track` already stores calibration JSON verbatim, so no write-path change is needed.

- [ ] **Step 1: Write the failing tests (append to tests/test_stereo_endpoints.py)**

First read `tests/test_calibration_latest.py` and reuse its RUNS_DIR fixture pattern
verbatim (it patches `app_module.RUNS_DIR` to a tmp_path). The test below assumes that
pattern; adapt the two helper lines marked `# per test_calibration_latest.py` to match
the existing file's actual helper names if they differ.

```python
import json as _json
import time


def _write_run(runs_dir, run_id, calibration):
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "calibration.json").write_text(_json.dumps(calibration))
    time.sleep(0.01)   # distinct mtimes


def test_calibration_latest_camera_id_filter(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)   # per test_calibration_latest.py
    _write_run(tmp_path, "run-a", {"schema": "squash-calibration-v2", "camera_id": "ucsc-left-fin"})
    _write_run(tmp_path, "run-b", {"schema": "squash-calibration-v2", "camera_id": "ucsc-right-fin"})
    _write_run(tmp_path, "run-c", {"schema": "squash-calibration-v2"})   # no id (legacy)

    client = app_module.app.test_client()
    # Unfiltered: newest overall (run-c), exactly today's behavior.
    body = client.get("/api/calibration/latest").get_json()
    assert body["ok"] is True and body["run_id"] == "run-c"
    # Filtered: newest with the matching id.
    body = client.get("/api/calibration/latest?camera_id=ucsc-left-fin").get_json()
    assert body["ok"] is True and body["run_id"] == "run-a"
    assert body["calibration"]["camera_id"] == "ucsc-left-fin"
    # Filtered with no match: 404.
    response = client.get("/api/calibration/latest?camera_id=nope")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_endpoints.py::test_calibration_latest_camera_id_filter -q`
Expected: FAIL — the filtered request returns run-c (filter not implemented).

- [ ] **Step 3: Implement**

In `/api/calibration/latest`, the current code selects `max` over the glob then reads the
file. Change the selection to iterate candidates newest-first and pick the first eligible
one, reading each candidate at most once:

```python
    camera_id = request.args.get("camera_id") or None
    candidates = sorted(
        RUNS_DIR.glob("*/calibration.json"),
        key=lambda p: (p.stat().st_mtime_ns, p.parent.name),
        reverse=True,
    )
    for path in candidates:
        try:
            calibration = json.loads(path.read_text())
        except (OSError, ValueError):
            if camera_id is None:
                return jsonify({"ok": False,
                                "error": "Latest calibration could not be read."}), 500
            continue
        if camera_id is not None and calibration.get("camera_id") != camera_id:
            continue
        saved_at = ...  # keep the existing mtime->ISO8601 formatting line unchanged
        return jsonify({"ok": True, "run_id": path.parent.name,
                        "saved_at": saved_at, "calibration": calibration})
    return jsonify({"ok": False, "error": "No saved calibration found. "
                    "Run a calibrated analysis first."}), 404
```

Preserve the existing unfiltered semantics exactly: with no `camera_id`, the newest file
is chosen and an unreadable newest file is still a 500 (the `if camera_id is None`
branch above keeps that). Reuse the route's existing ISO-8601 `saved_at` expression
verbatim where the `...` placeholder-comment sits — it is existing code being moved, not
new code (read the current implementation at app.py:473-505 and carry its exact line).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_endpoints.py -q` and the existing
`pytest tests/test_calibration_latest.py -q` (must stay green — that file pins today's
unfiltered behavior). Then `pytest tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_stereo_endpoints.py
git commit -m "feat(server): camera_id filter for latest calibration"
```

---

### Task 7: Cross-camera agreement gate

**Files:**
- Modify: `stereo_engine.py` (append `pair_agreement`)
- Modify: `app.py` (add `/api/camera-pair-check` below `/api/camera-model`)
- Test: `tests/test_stereo_endpoints.py` (append) and `tests/test_stereo_engine.py` (append)

**Interfaces:**
- Consumes: `triangulate` (Task 1), `court_model.solve_camera_model`.
- Produces:
  ```python
  PAIR_GATE_MAX_MEDIAN_FT = 0.1   # ~3 cm, per the spec's agreement gate
  def pair_agreement(model_a, model_b, grid=None) -> dict
      # {"median_err_ft", "max_err_ft", "baseline_ft", "point_count", "ok_pair"}
      # grid defaults to a 3x4x2 lattice over the court volume
      # (x in {5.25, 10.5, 15.75}, y in {4, 12, 20, 28}, z in {1.0, 8.0});
      # for each point visible to both cameras (project() succeeds), project
      # through both models, triangulate back, and measure 3D error.
  POST /api/camera-pair-check  {"calibration_a": {...}, "calibration_b": {...}}
      # always 200; solves both (statuses reported as "status_a"/"status_b");
      # when both solve ok: {"ok": true, "status": "ok", **pair_agreement}
      # else {"ok": true, "status": "solve_failed", "status_a": ..., "status_b": ...}
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stereo_engine.py`:

```python
def test_pair_agreement_synthetic_pair_passes_gate():
    left, right = make_fin_pair()
    report = stereo_engine.pair_agreement(left, right)
    assert report["ok_pair"] is True
    assert report["median_err_ft"] < 1e-6
    assert abs(report["baseline_ft"] - 3.0) < 1e-9
    assert report["point_count"] >= 12


def test_pair_agreement_disagreeing_pair_fails_gate():
    import dataclasses
    left, right = make_fin_pair()
    shifted = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + np.array([0.0, 0.0, 0.5]))
    report = stereo_engine.pair_agreement(left, shifted)
    assert report["ok_pair"] is False
    assert report["median_err_ft"] > stereo_engine.PAIR_GATE_MAX_MEDIAN_FT
```

Append to `tests/test_stereo_endpoints.py`:

```python
def test_camera_pair_check_endpoint():
    client = _client()
    cam_a = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    cam_b = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    body = client.post("/api/camera-pair-check", json={
        "calibration_a": _synthetic_calibration(cam_a),
        "calibration_b": _synthetic_calibration(cam_b),
    }).get_json()
    assert body["ok"] is True and body["status"] == "ok"
    assert body["ok_pair"] is True
    assert body["median_err_ft"] < 0.05
    assert 2.0 < body["baseline_ft"] < 4.0

    bad = client.post("/api/camera-pair-check", json={
        "calibration_a": _synthetic_calibration(cam_a),
        "calibration_b": {},
    }).get_json()
    assert bad["ok"] is True and bad["status"] == "solve_failed"
    assert bad["status_a"] == "ok" and bad["status_b"] == "no_frame_size"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stereo_engine.py -q tests/test_stereo_endpoints.py -q`
Expected: `AttributeError: ... no attribute 'pair_agreement'` and a 404 body for the endpoint test.

- [ ] **Step 3: Implement**

Append to `stereo_engine.py`:

```python
PAIR_GATE_MAX_MEDIAN_FT = 0.1   # ~3 cm agreement gate (spec Component 3)


def _default_agreement_grid():
    grid = []
    for x in (5.25, 10.5, 15.75):
        for y in (4.0, 12.0, 20.0, 28.0):
            for z in (1.0, 8.0):
                grid.append(np.array([x, y, z]))
    return grid


def pair_agreement(model_a, model_b, grid=None):
    """Triangulation self-consistency of two independently solved cameras.

    Projects known court points through both models, triangulates back, and
    reports the 3D disagreement. Catches two individually-plausible solves
    that disagree about where the court is.
    """
    errors = []
    for point in (grid if grid is not None else _default_agreement_grid()):
        try:
            px_a, px_b = model_a.project(point), model_b.project(point)
        except ValueError:
            continue
        recovered, _gap = triangulate(model_a, model_b, px_a, px_b)
        if recovered is None:
            continue
        errors.append(float(np.linalg.norm(recovered - point)))
    baseline = float(np.linalg.norm(model_a.camera_center_ft - model_b.camera_center_ft))
    if not errors:
        return {"median_err_ft": np.inf, "max_err_ft": np.inf,
                "baseline_ft": baseline, "point_count": 0, "ok_pair": False}
    median = float(np.median(errors))
    return {"median_err_ft": median, "max_err_ft": float(max(errors)),
            "baseline_ft": baseline, "point_count": len(errors),
            "ok_pair": median <= PAIR_GATE_MAX_MEDIAN_FT}
```

Add to `app.py` (below `/api/camera-model`; add `import stereo_engine` next to the
existing `import court_model`):

```python
@app.route("/api/camera-pair-check", methods=["POST"])
def api_camera_pair_check():
    """Cross-camera agreement gate: do two solved cameras agree on the court?"""
    payload = request.get_json(silent=True) or {}
    results = {}
    models = {}
    for key in ("a", "b"):
        calibration = payload.get(f"calibration_{key}")
        if not isinstance(calibration, dict):
            results[f"status_{key}"] = "invalid_json"
            continue
        model, info = court_model.solve_camera_model(calibration)
        results[f"status_{key}"] = info.get("status")
        if model is not None and info.get("status") == "ok":
            models[key] = model
    if len(models) < 2:
        return jsonify({"ok": True, "status": "solve_failed", **results})
    report = stereo_engine.pair_agreement(models["a"], models["b"])
    return jsonify({"ok": True, "status": "ok", **results, **report})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stereo_engine.py tests/test_stereo_endpoints.py -q` — all pass.
Then `pytest tests/ -q` green.

- [ ] **Step 5: Commit**

```bash
git add stereo_engine.py app.py tests/test_stereo_engine.py tests/test_stereo_endpoints.py
git commit -m "feat(stereo): cross-camera agreement gate and pair-check endpoint"
```

---

## After Plan B1

Plan B2 (authored after this plan executes, against the generated goldens): Swift
`CameraModel` + `StereoMath` with golden-parity tests reading
`ios/Tests/Fixtures/stereo_goldens.json`, the live `StereoEngine` consuming
`BallTracker` + `RemoteDetectionStore` + `ClockSync.remoteToLocal`, solved-model
exchange over the existing `.calibration` control message, and the DEBUG bench stereo
status line.

## Self-review notes

- **Spec coverage (Phase 3, Python half):** triangulation (T1), plane-snap + calls
  (T2), interpolation + impact + confidence tiers incl. one_view/no_call (T3), golden
  vectors for cross-language parity (T4), solved-model endpoint (T5), calibration
  identity + filtered retrieval (T6), agreement gate ≤ 3 cm with baseline reporting
  (T7). The Swift/live half is explicitly Plan B2.
- **Known judgment calls:** `stereo_engine` functions accept RAW pixels and undistort
  internally (documented) — synthetic tests use distortion-free cameras so
  undistort is identity there; the golden cameras are distortion-free too, keeping
  Swift parity simple in B2. `detect_impacts` scans all five surfaces each pass —
  O(5·track) is negligible at rally scale. Task 6's step-3 code carries one
  deliberate `...` placeholder-comment marking an EXISTING line to move (the
  `saved_at` expression), with explicit instructions — not new unwritten code.
- **Type consistency check:** `TrackSample.px` is a tuple; `sample_camera` produces
  tuples; goldens serialize as lists and tests rebuild tuples. `make_fin_pair`
  positions/baseline consistent (3.0 ft) across tests and generator. Endpoint tests
  reuse `_synthetic_calibration` from `tests/test_camera_model.py` via the tests-dir
  sys.path insert — both files verified to exist by the scout.
