# 3D Contact Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect every ball contact and classify it (floor / racket / front-wall / side-wall) by solving the full camera pose from existing calibration and fitting piecewise 3D ballistic arcs to the ball track, feeding 3D evidence into the existing fusion grammar.

**Architecture:** `court_model.py` gains a `CameraModel` (pose + focal solved from the floor landmarks + front-wall lines already in calibration v2). New `ballistic.py` fits gravity-constrained 3D arcs to ball detections via linear least squares; arc boundaries are contacts with sub-frame 3D impact estimates. `event_engine.py` swaps its trajectory event source and emission scores to the 3D versions when a camera solves, keeping audio evidence, the skip-state Viterbi, and the squash grammar unchanged. Every phase degrades to today's 2D behavior.

**Tech Stack:** Python 3, numpy only (no scipy/cv2), pytest, existing Flask/job pipeline.

**Spec:** `docs/superpowers/specs/2026-07-20-3d-contact-detection-design.md`

## Global Constraints

- Pure numpy for all new math — no scipy, no OpenCV.
- Units: feet; gravity `G = 32.174 ft/s²`.
- Court 3D frame: origin front-left floor corner seen from the back-wall camera; x → right 0–21 ft; y → front wall to back wall 0–32 ft; z → up. Floor plane z=0, front wall y=0, left wall x=0, right wall x=21.
- Camera model: pinhole, fx=fy=`focal_px`, principal point fixed at image center, image y grows downward. `CameraModel` works in **undistorted pixel space**; callers undistort observed pixels with `court_model.undistort_point(p, distortion)`.
- Heights on the front wall: out-line lower edge 15.0 ft; tin top edge 19/12 ft.
- Never fail a run: any camera/3D failure degrades to the current 2D path with a status payload, exactly like `load_floor_calibration`.
- The 2D path must stay behaviorally identical: all 95 existing tests keep passing untouched.
- Test runner: `.venv/bin/python -m pytest tests -q` from repo root.
- Capture norm 4K@60fps, but nothing may hard-code 60 fps — use timestamps.
- Commit after every task; message style follows repo history (imperative, no prefix enforced).

## File Structure

- `build_eval_set.py` (modify): skip ground-truth runs that never produced `detected_hits.json`.
- `court_model.py` (modify): `CameraModel` dataclass, correspondence gathering, homography-based init, LM refinement, `solve_camera_model(calibration)`.
- `ballistic.py` (create): ballistic arc fit, greedy 3D segmentation, impact refinement, event generation. Depends only on numpy + `CameraModel`.
- `event_engine.py` (modify): `camera=` parameter, ballistic trajectory source, 3D emission scores, 3D wall/floor impact handoff.
- `job_runner.py` (modify): solve camera when `fusion_3d` enabled, pass to engine, serialize camera + warnings into payload; `judge_hits` prefers engine-supplied court positions.
- `rerun_detection.py` (create): offline A/B harness — re-runs fusion over stored run CSVs into a mirror dir for eval rebuilds.
- `tune_fusion_weights.py` (create): grid search of emission weights against the eval set.
- `tests/synthetic3d.py` (create): shared synthetic camera + trajectory generators.
- `tests/test_camera_model.py` (create), `tests/test_ballistic.py` (create); `tests/test_event_engine.py`, `tests/test_eval_set.py` (extend).

---

### Task 1: Eval baseline — stop counting label-only runs as detector misses

The current 13/17 missed-bounce figure counts ground-truth runs that have **no** `detected_hits.json` (label-only runs where detection never ran). Those aren't detector misses. Fix the builder, rebuild, and snapshot the trustworthy baseline.

**Files:**
- Modify: `build_eval_set.py:143` (in `collect_ground_truth_cases`)
- Test: `tests/test_eval_set.py`
- Create: `eval_set/BASELINE-2026-07-20.md`

**Interfaces:**
- Produces: `collect_ground_truth_cases` returns `[]` for run dirs lacking `detected_hits.json`; manifest gains `"ground_truth_runs_skipped_no_detections"` count.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_set.py` (follow the file's existing fixture style for creating a temp run dir with `ground_truth.json`; the key assertion):

```python
def test_ground_truth_skipped_without_detected_hits(tmp_path):
    run_dir = tmp_path / "runs" / "labelonly"
    run_dir.mkdir(parents=True)
    (run_dir / "ground_truth.json").write_text(json.dumps({
        "events": [{"frame": 10, "type": "floor"}],
    }))
    # NOTE: no detected_hits.json on purpose
    cases, manifest = build_eval_set.build_eval_set(tmp_path / "runs")
    gt_cases = [c for c in cases if c["kind"] == "ground_truth_event"]
    assert gt_cases == []
    assert manifest["ground_truth_runs_skipped_no_detections"] == 1
```

Match the actual return signature of `build_eval_set.build_eval_set` (check `build_eval_set.py:205`) — if it returns `(cases, manifest)` differently, adapt the unpacking, not the assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_set.py -q`
Expected: FAIL (ground-truth case is currently produced with `matched_detected: None`, and the manifest key doesn't exist).

- [ ] **Step 3: Implement**

In `collect_ground_truth_cases` (`build_eval_set.py:143`), replace the silent default:

```python
    detected_path = run_dir / "detected_hits.json"
    if not detected_path.exists():
        # Label-only run: detection never ran, so an unmatched event is not
        # a detector miss. Skip the run and count it in the manifest.
        return None
    detected = load_json(detected_path) or {}
```

In `build_eval_set()` (`build_eval_set.py:205`), handle the `None` sentinel where `collect_ground_truth_cases` is called (around line 225): increment a new counter `ground_truth_runs_skipped_no_detections` and continue; include the counter in the manifest dict (around line 239). Print the skip count in `main()` so the cap is never silent.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Rebuild the eval set and snapshot the baseline**

```bash
.venv/bin/python build_eval_set.py --runs-dir ui_runs --out eval_set
.venv/bin/python eval_line_calls.py --eval-set eval_set/cases.jsonl --verbose
```

(Confirm `--out` semantics with `--help` first — pass the value the CLI expects for the tracked `eval_set/` location.) Copy the full printed report into a new `eval_set/BASELINE-2026-07-20.md` with a one-paragraph header: date, commit hash, and the note that label-only runs are now excluded from the missed-bounce axis. This file is the number Task 11 must beat.

- [ ] **Step 6: Commit**

```bash
git add build_eval_set.py tests/test_eval_set.py eval_set/
git commit -m "Eval: exclude label-only runs from missed-bounce axis; snapshot baseline"
```

---

### Task 2: CameraModel dataclass — projection, rays, serialization

**Files:**
- Modify: `court_model.py` (append after `FloorMap` section)
- Create: `tests/synthetic3d.py`
- Create: `tests/test_camera_model.py`

**Interfaces:**
- Produces:
  - `court_model.G_FT_PER_S2 = 32.174`
  - `court_model.CameraModel` frozen dataclass: fields `focal_px: float`, `center_px: tuple[float, float]`, `rotation: np.ndarray (3,3 world→camera)`, `camera_center_ft: np.ndarray (3,)`, `distortion: dict | None`, `fit_rms_px: float | None`, `point_count: int`; methods `project(court_xyz) -> (u, v)` (undistorted px; raises `ValueError` if the point is at/behind the camera), `ray(pixel) -> (origin_ft, unit_dir)` (undistorted px in), `projection_matrix() -> np.ndarray (3,4)`, `depth_ft(court_xyz) -> float`, `to_dict() -> dict`, `CameraModel.from_dict(d) -> CameraModel`.
  - `tests/synthetic3d.py`: `make_camera(focal_px=1600.0, center=(960.0, 540.0), position=(10.5, 30.0, 7.0), look_at=(10.5, 0.0, 5.0)) -> CameraModel`.

- [ ] **Step 1: Write the synthetic helper**

Create `tests/synthetic3d.py`:

```python
"""Synthetic camera + trajectory generators shared by the 3D test files."""
import numpy as np

from court_model import CameraModel


def make_camera(focal_px=1600.0, center=(960.0, 540.0),
                position=(10.5, 30.0, 7.0), look_at=(10.5, 0.0, 5.0)):
    """Back-wall-mounted synthetic camera looking toward the front wall."""
    position = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # image y grows downward
    rotation = np.vstack([right, down, forward])  # world -> camera rows
    return CameraModel(
        focal_px=float(focal_px),
        center_px=(float(center[0]), float(center[1])),
        rotation=rotation,
        camera_center_ft=position,
        distortion=None,
        fit_rms_px=0.0,
        point_count=0,
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_camera_model.py`:

```python
import numpy as np
import pytest

import court_model
from court_model import CameraModel
from synthetic3d import make_camera


def test_project_ray_roundtrip():
    camera = make_camera()
    point = np.array([6.0, 12.0, 3.0])
    u, v = camera.project(point)
    origin, direction = camera.ray((u, v))
    # The ray from the camera through that pixel must pass through the point.
    to_point = point - origin
    to_point /= np.linalg.norm(to_point)
    assert np.allclose(direction, to_point, atol=1e-9)


def test_project_center_pixel():
    camera = make_camera(center=(960.0, 540.0), look_at=(10.5, 0.0, 7.0))
    # A point straight along the optical axis lands on the principal point.
    u, v = camera.project((10.5, 0.0, 7.0))
    assert u == pytest.approx(960.0, abs=1e-6)
    assert v == pytest.approx(540.0, abs=1e-6)


def test_project_behind_camera_raises():
    camera = make_camera(position=(10.5, 30.0, 7.0))
    with pytest.raises(ValueError):
        camera.project((10.5, 33.0, 7.0))  # behind the back-wall camera


def test_serialization_roundtrip():
    camera = make_camera()
    restored = CameraModel.from_dict(camera.to_dict())
    assert np.allclose(restored.rotation, camera.rotation)
    assert np.allclose(restored.camera_center_ft, camera.camera_center_ft)
    assert restored.focal_px == camera.focal_px
    u1, v1 = camera.project((5.0, 10.0, 2.0))
    u2, v2 = restored.project((5.0, 10.0, 2.0))
    assert (u1, v1) == pytest.approx((u2, v2))


def test_projection_matrix_agrees_with_project():
    camera = make_camera()
    point = np.array([3.0, 20.0, 1.0, 1.0])
    projected = camera.projection_matrix() @ point
    u, v = projected[0] / projected[2], projected[1] / projected[2]
    assert (u, v) == pytest.approx(camera.project(point[:3]))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py -q`
Expected: FAIL with `ImportError: cannot import name 'CameraModel'`.

- [ ] **Step 4: Implement CameraModel**

Append to `court_model.py` (after the `FloorMap`/floor section):

```python
# --- Full camera model (pose + focal) --------------------------------------

G_FT_PER_S2 = 32.174

OUT_LINE_HEIGHT_FT = 15.0
TIN_TOP_HEIGHT_FT = 19.0 / 12.0


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole camera in court coordinates (feet).

    Operates in undistorted pixel space: `project` returns undistorted
    pixels and `ray` expects them; callers undistort observations with
    `undistort_point(p, self.distortion)` first.
    """

    focal_px: float
    center_px: tuple
    rotation: np.ndarray        # 3x3, world -> camera
    camera_center_ft: np.ndarray  # (3,)
    distortion: dict | None
    fit_rms_px: float | None
    point_count: int

    def project(self, court_xyz):
        camera_point = self.rotation @ (
            np.asarray(court_xyz, dtype=float) - self.camera_center_ft
        )
        if camera_point[2] <= 1e-9:
            raise ValueError("Point is at or behind the camera.")
        cx, cy = self.center_px
        return (
            self.focal_px * camera_point[0] / camera_point[2] + cx,
            self.focal_px * camera_point[1] / camera_point[2] + cy,
        )

    def ray(self, pixel):
        cx, cy = self.center_px
        direction_camera = np.array(
            [(float(pixel[0]) - cx) / self.focal_px,
             (float(pixel[1]) - cy) / self.focal_px,
             1.0]
        )
        direction = self.rotation.T @ direction_camera
        return (
            self.camera_center_ft.copy(),
            direction / np.linalg.norm(direction),
        )

    def depth_ft(self, court_xyz):
        camera_point = self.rotation @ (
            np.asarray(court_xyz, dtype=float) - self.camera_center_ft
        )
        return float(camera_point[2])

    def projection_matrix(self):
        cx, cy = self.center_px
        intrinsics = np.array(
            [[self.focal_px, 0.0, cx], [0.0, self.focal_px, cy], [0.0, 0.0, 1.0]]
        )
        translation = (-self.rotation @ self.camera_center_ft).reshape(3, 1)
        return intrinsics @ np.hstack([self.rotation, translation])

    def to_dict(self):
        return {
            "focal_px": float(self.focal_px),
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
            "rotation": self.rotation.tolist(),
            "camera_center_ft": self.camera_center_ft.tolist(),
            "distortion": self.distortion,
            "fit_rms_px": self.fit_rms_px,
            "point_count": self.point_count,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            focal_px=float(data["focal_px"]),
            center_px=(float(data["center_px"][0]), float(data["center_px"][1])),
            rotation=np.asarray(data["rotation"], dtype=float),
            camera_center_ft=np.asarray(data["camera_center_ft"], dtype=float),
            distortion=data.get("distortion"),
            fit_rms_px=data.get("fit_rms_px"),
            point_count=int(data.get("point_count") or 0),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py tests/test_court_model.py -q`
Expected: PASS (including the untouched court-model tests).

- [ ] **Step 6: Commit**

```bash
git add court_model.py tests/synthetic3d.py tests/test_camera_model.py
git commit -m "Add CameraModel: calibrated pinhole projection/rays in court feet"
```

---

### Task 3: Correspondence gathering + pose/focal initialization from the floor homography

**Files:**
- Modify: `court_model.py`
- Test: `tests/test_camera_model.py`

**Interfaces:**
- Consumes: `load_floor_calibration`, `invert_homography`, `undistort_point`, `CameraModel` (Task 2).
- Produces:
  - `court_model._camera_correspondences(calibration) -> (image_px Nx2 raw, court_xyz Nx3)` — floor landmarks at z=0 plus front-wall line endpoints at `(0|21, 0, height)`. Line endpoints are treated as the side-wall junctions (`line_from_calibration` stores them left-then-right); a violated assumption surfaces as high solver residual, which Task 4 gates on.
  - `court_model._init_camera_from_floor(calibration, frame_size) -> (focal_px, rotation, camera_center) | None` — Zhang-style single-homography init with known principal point.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_camera_model.py`:

```python
def _synthetic_calibration(camera, frame_size=(1920, 1080)):
    """Project the real landmark/line geometry through a known camera."""
    landmarks = []
    for mark in court_model.FLOOR_LANDMARKS:
        if mark["optional"]:
            continue
        x_ft, y_ft = mark["court_ft"]
        u, v = camera.project((x_ft, y_ft, 0.0))
        landmarks.append({"id": mark["id"], "court_ft": [x_ft, y_ft],
                          "refined_px": [u, v]})
    lines = []
    for name, height in (("out_line_lower_edge", court_model.OUT_LINE_HEIGHT_FT),
                         ("tin_top_edge", court_model.TIN_TOP_HEIGHT_FT)):
        left = camera.project((0.0, 0.0, height))
        right = camera.project((court_model.COURT_WIDTH_FT, 0.0, height))
        lines.append({"name": name, "endpoints": [list(left), list(right)]})
    return {
        "schema": "squash-calibration-v2",
        "frame_width": frame_size[0], "frame_height": frame_size[1],
        "lines": lines,
        "planes": {"floor": {"landmarks": landmarks}},
    }


def test_camera_correspondences_extracts_floor_and_wall():
    camera = make_camera()
    calibration = _synthetic_calibration(camera)
    image_px, court_xyz = court_model._camera_correspondences(calibration)
    assert len(image_px) == len(court_xyz) == 7 + 4  # 7 required landmarks + 4 line ends
    heights = sorted(set(round(z, 4) for z in court_xyz[:, 2]))
    assert heights == [0.0, round(court_model.TIN_TOP_HEIGHT_FT, 4),
                       court_model.OUT_LINE_HEIGHT_FT]


def test_init_camera_from_floor_recovers_pose():
    camera = make_camera(focal_px=1500.0, position=(10.5, 29.0, 7.0),
                         look_at=(10.5, 0.0, 4.0))
    calibration = _synthetic_calibration(camera)
    result = court_model._init_camera_from_floor(calibration, (1920, 1080))
    assert result is not None
    focal, rotation, center = result
    assert focal == pytest.approx(1500.0, rel=0.05)
    assert np.allclose(center, camera.camera_center_ft, atol=0.75)
    assert np.allclose(rotation, camera.rotation, atol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py -q`
Expected: FAIL with `AttributeError` on `_camera_correspondences`.

- [ ] **Step 3: Implement**

Append to `court_model.py`:

```python
_WALL_LINE_HEIGHTS_FT = {
    "out_line_lower_edge": OUT_LINE_HEIGHT_FT,
    "tin_top_edge": TIN_TOP_HEIGHT_FT,
}


def _camera_correspondences(calibration):
    """Image px (raw/distorted) <-> court 3D points from a v2 calibration.

    Floor landmarks sit at z=0. Front-wall line endpoints sit at
    (0, 0, h) and (21, 0, h): line_from_calibration's contract stores them
    left-then-right, and the wizard spans the wall, so the endpoints are the
    side-wall junctions. If a calibration violates that, the pose fit's
    residual gate (solve_camera_model) rejects it.
    """
    image_points, court_points = [], []
    planes = calibration.get("planes") or {}
    floor_plane = planes.get("floor") or {}
    for landmark in floor_plane.get("landmarks", []):
        if landmark.get("skipped"):
            continue
        pixel = landmark.get("refined_px") or landmark.get("tap_px")
        court = landmark.get("court_ft")
        if pixel is None or court is None:
            continue
        image_points.append([float(pixel[0]), float(pixel[1])])
        court_points.append([float(court[0]), float(court[1]), 0.0])
    for line in calibration.get("lines", []):
        height = _WALL_LINE_HEIGHTS_FT.get(line.get("name"))
        endpoints = line.get("endpoints") or []
        if height is None or len(endpoints) != 2:
            continue
        for x_ft, endpoint in zip((0.0, COURT_WIDTH_FT), endpoints):
            image_points.append([float(endpoint[0]), float(endpoint[1])])
            court_points.append([x_ft, 0.0, height])
    return (
        np.asarray(image_points, dtype=float).reshape(-1, 2),
        np.asarray(court_points, dtype=float).reshape(-1, 3),
    )


def _init_camera_from_floor(calibration, frame_size):
    """Approximate (focal, R, C) from the floor homography, Zhang-style.

    H maps court floor (x, y) -> undistorted image px, H ~ K [r1 r2 t].
    With the principal point pinned at the image center, the orthonormality
    of r1, r2 yields two closed-form estimates of the focal length.
    """
    floor_map = load_floor_calibration(calibration)
    if floor_map is None:
        return None
    homography = invert_homography(floor_map.homography_court_from_image)
    cx, cy = frame_size[0] / 2.0, frame_size[1] / 2.0

    def reduced(column):
        return (
            column[0] - cx * column[2],
            column[1] - cy * column[2],
            column[2],
        )

    a1, b1, c1 = reduced(homography[:, 0])
    a2, b2, c2 = reduced(homography[:, 1])
    focal_sq = []
    if abs(c1 * c2) > 1e-12:
        value = -(a1 * a2 + b1 * b2) / (c1 * c2)
        if value > 0:
            focal_sq.append(value)
    if abs(c2 * c2 - c1 * c1) > 1e-12:
        value = ((a1 * a1 + b1 * b1) - (a2 * a2 + b2 * b2)) / (c2 * c2 - c1 * c1)
        if value > 0:
            focal_sq.append(value)
    if not focal_sq:
        return None
    focal = float(np.sqrt(np.mean(focal_sq)))

    intrinsics_inv = np.array(
        [[1.0 / focal, 0.0, -cx / focal],
         [0.0, 1.0 / focal, -cy / focal],
         [0.0, 0.0, 1.0]]
    )
    r1 = intrinsics_inv @ homography[:, 0]
    r2 = intrinsics_inv @ homography[:, 1]
    translation = intrinsics_inv @ homography[:, 2]
    scale = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    r1, r2, translation = r1 * scale, r2 * scale, translation * scale
    if translation[2] < 0:
        # Court origin must be in front of the camera.
        r1, r2, translation = -r1, -r2, -translation
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    camera_center = -rotation.T @ translation
    return focal, rotation, camera_center
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add court_model.py tests/test_camera_model.py
git commit -m "Camera pose init: correspondences + focal/pose from floor homography"
```

---

### Task 4: Levenberg–Marquardt refinement + `solve_camera_model` with residual gate

**Files:**
- Modify: `court_model.py`
- Test: `tests/test_camera_model.py`

**Interfaces:**
- Consumes: Task 2–3 symbols.
- Produces:
  - `court_model.solve_camera_model(calibration) -> (CameraModel | None, info: dict)`. `info["status"]` ∈ `"ok" | "no_frame_size" | "insufficient_points" | "init_failed" | "refine_failed" | "high_residual"`; on `"ok"` and `"high_residual"` it also has `"rms_px"`, `"max_px"`, `"point_count"`.
  - `court_model.CAMERA_MAX_RMS_PX = 4.0` (threshold at 1920-wide frames, scaled linearly by `frame_width / 1920`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_camera_model.py`:

```python
def test_solve_camera_model_recovers_noisy_pose():
    rng = np.random.default_rng(7)
    camera = make_camera(focal_px=1550.0, position=(10.5, 30.5, 7.0),
                         look_at=(10.5, 0.0, 4.5))
    calibration = _synthetic_calibration(camera)
    for landmark in calibration["planes"]["floor"]["landmarks"]:
        landmark["refined_px"] = list(
            np.asarray(landmark["refined_px"]) + rng.normal(0, 0.5, 2))
    for line in calibration["lines"]:
        line["endpoints"] = [list(np.asarray(p) + rng.normal(0, 0.5, 2))
                             for p in line["endpoints"]]
    solved, info = court_model.solve_camera_model(calibration)
    assert info["status"] == "ok"
    assert solved is not None
    assert solved.focal_px == pytest.approx(1550.0, rel=0.03)
    assert np.allclose(solved.camera_center_ft, camera.camera_center_ft, atol=0.5)
    assert solved.fit_rms_px < 2.0


def test_solve_camera_model_rejects_bad_geometry():
    camera = make_camera()
    calibration = _synthetic_calibration(camera)
    # Corrupt one wall line badly: endpoints not at the side walls.
    calibration["lines"][0]["endpoints"][0][0] += 300.0
    solved, info = court_model.solve_camera_model(calibration)
    assert solved is None
    assert info["status"] == "high_residual"


def test_solve_camera_model_requires_wall_points():
    camera = make_camera()
    calibration = _synthetic_calibration(camera)
    calibration["lines"] = []
    solved, info = court_model.solve_camera_model(calibration)
    assert solved is None
    assert info["status"] == "insufficient_points"


def test_solve_camera_model_requires_frame_size():
    solved, info = court_model.solve_camera_model({"lines": [], "planes": {}})
    assert solved is None
    assert info["status"] == "no_frame_size"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py -q`
Expected: FAIL with `AttributeError` on `solve_camera_model`.

- [ ] **Step 3: Implement**

Append to `court_model.py`:

```python
CAMERA_MAX_RMS_PX = 4.0  # gate at 1920-wide frames; scaled by frame_width/1920


def _rotation_from_axis_angle(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    axis = omega / theta
    skew = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _camera_residuals(focal, rotation, center, center_px, court_xyz, image_und):
    """Flat residual vector (2N,) of undistorted-pixel reprojection errors,
    or None if any point falls at/behind the camera."""
    cx, cy = center_px
    residuals = np.empty(2 * len(court_xyz))
    for index, (point, observed) in enumerate(zip(court_xyz, image_und)):
        camera_point = rotation @ (point - center)
        if camera_point[2] <= 1e-6:
            return None
        residuals[2 * index] = focal * camera_point[0] / camera_point[2] + cx - observed[0]
        residuals[2 * index + 1] = focal * camera_point[1] / camera_point[2] + cy - observed[1]
    return residuals


def _refine_camera(focal, rotation, center, center_px, court_xyz, image_und,
                   iterations=60):
    """Levenberg-Marquardt over [focal, axis-angle(3), center(3)] with a
    numeric Jacobian. Local rotation parameterization: each accepted step
    right-multiplies the current rotation and resets omega to zero."""
    damping = 1e-3
    residuals = _camera_residuals(focal, rotation, center, center_px,
                                  court_xyz, image_und)
    if residuals is None:
        return None
    cost = float(residuals @ residuals)
    for _ in range(iterations):
        params = np.concatenate([[focal], np.zeros(3), center])
        jacobian = np.empty((len(residuals), 7))
        for column in range(7):
            step = np.zeros(7)
            step[column] = 1e-4 if column == 0 else 1e-6
            plus = params + step
            trial = _camera_residuals(
                plus[0], rotation @ _rotation_from_axis_angle(plus[1:4]),
                plus[4:7], center_px, court_xyz, image_und)
            if trial is None:
                return None
            jacobian[:, column] = (trial - residuals) / step[column]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        improved = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(
                    normal + damping * np.diag(np.diag(normal)), -gradient)
            except np.linalg.LinAlgError:
                return None
            trial_focal = focal + delta[0]
            trial_rotation = rotation @ _rotation_from_axis_angle(delta[1:4])
            trial_center = center + delta[4:7]
            trial = _camera_residuals(trial_focal, trial_rotation, trial_center,
                                      center_px, court_xyz, image_und)
            if trial is not None and float(trial @ trial) < cost:
                focal, rotation, center = trial_focal, trial_rotation, trial_center
                residuals, cost = trial, float(trial @ trial)
                damping = max(damping / 3.0, 1e-9)
                improved = True
                break
            damping *= 4.0
        if not improved:
            break
    return focal, rotation, center, residuals


def solve_camera_model(calibration):
    """Full pose + focal from a v2 calibration. Returns (CameraModel|None, info).

    Mirrors load_floor_calibration's philosophy: never raises on bad input,
    always explains itself through info["status"].
    """
    if not isinstance(calibration, dict):
        return None, {"status": "no_frame_size"}
    frame_width = calibration.get("frame_width")
    frame_height = calibration.get("frame_height")
    if not frame_width or not frame_height:
        return None, {"status": "no_frame_size"}
    center_px = (float(frame_width) / 2.0, float(frame_height) / 2.0)

    try:
        image_px, court_xyz = _camera_correspondences(calibration)
        distortion = calibration.get("distortion") or None
        floor_count = int(np.sum(court_xyz[:, 2] == 0.0)) if len(court_xyz) else 0
        wall_count = len(court_xyz) - floor_count
        if floor_count < 4 or wall_count < 2:
            return None, {"status": "insufficient_points",
                          "floor_points": floor_count, "wall_points": wall_count}
        image_und = np.asarray(
            [undistort_point(pixel, distortion) for pixel in image_px])
        init = _init_camera_from_floor(calibration, (frame_width, frame_height))
        if init is None:
            return None, {"status": "init_failed"}
        refined = _refine_camera(*init, center_px, court_xyz, image_und)
        if refined is None:
            return None, {"status": "refine_failed"}
        focal, rotation, center, residuals = refined
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
        return None, {"status": "refine_failed"}

    per_point = np.sqrt(residuals[0::2] ** 2 + residuals[1::2] ** 2)
    rms = float(np.sqrt(np.mean(per_point ** 2)))
    info = {"rms_px": rms, "max_px": float(per_point.max()),
            "point_count": len(court_xyz)}
    threshold = CAMERA_MAX_RMS_PX * float(frame_width) / 1920.0
    if rms > threshold:
        info["status"] = "high_residual"
        return None, info
    info["status"] = "ok"
    camera = CameraModel(
        focal_px=float(focal), center_px=center_px, rotation=rotation,
        camera_center_ft=np.asarray(center, dtype=float),
        distortion=distortion, fit_rms_px=rms, point_count=len(court_xyz),
    )
    return camera, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera_model.py tests/test_court_model.py -q`
Expected: PASS. If `test_solve_camera_model_recovers_noisy_pose` fails on tolerance rather than error, inspect `info["rms_px"]` first — a healthy fit is < 1 px at 0.5 px noise; loosen only the pose tolerances, never the residual assertion.

- [ ] **Step 5: Sanity-check against a real calibration**

```bash
.venv/bin/python -c "
import json, court_model
cal = json.load(open('ui_runs/1784571843421/calibration.json'))
camera, info = court_model.solve_camera_model(cal)
print(info)
print(None if camera is None else camera.camera_center_ft)"
```

Expected: either `status: ok` with a camera center near the back wall (y in roughly 24–34 ft, z in 3–9 ft), or an explained failure status. Record the output in the commit message body — this is the first real-footage evidence.

- [ ] **Step 6: Commit**

```bash
git add court_model.py tests/test_camera_model.py
git commit -m "solve_camera_model: LM-refined pose+focal with residual gate"
```

---

### Task 5: `ballistic.py` — gravity-constrained arc fit (linear least squares)

**Files:**
- Create: `ballistic.py`
- Create: `tests/test_ballistic.py`

**Interfaces:**
- Consumes: `CameraModel.projection_matrix()`, `CameraModel.project()`, `court_model.G_FT_PER_S2`.
- Produces:
  - `ballistic.GRAVITY_VEC` — `np.array([0, 0, -G_FT_PER_S2])`.
  - `ballistic.BallisticArc` frozen dataclass: `t_ref: float`, `x0: np.ndarray (3,)`, `v0: np.ndarray (3,)`, `rms_px: float`, `start: int`, `end: int` (sample index range, end-exclusive); methods `position(t) -> np.ndarray (3,)`, `velocity(t) -> np.ndarray (3,)`.
  - `ballistic.fit_arc(times, pixels_und, camera, start=0, end=None) -> BallisticArc | None` — None when < 3 samples, the system is degenerate, or the fitted arc projects behind the camera.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ballistic.py`:

```python
import numpy as np
import pytest

import ballistic
from ballistic import BallisticArc, GRAVITY_VEC
from synthetic3d import make_camera


def _project_trajectory(camera, x0, v0, times, t_ref=None):
    t_ref = times[0] if t_ref is None else t_ref
    pixels = []
    for t in times:
        tau = t - t_ref
        point = np.asarray(x0) + np.asarray(v0) * tau + 0.5 * GRAVITY_VEC * tau * tau
        pixels.append(camera.project(point))
    return np.asarray(pixels)


def test_fit_arc_recovers_state():
    camera = make_camera()
    times = np.arange(12) / 60.0
    x0 = np.array([4.0, 25.0, 3.0])
    v0 = np.array([10.0, -55.0, 12.0])   # driven toward the front wall
    pixels = _project_trajectory(camera, x0, v0, times)
    arc = ballistic.fit_arc(times, pixels, camera)
    assert arc is not None
    assert np.allclose(arc.x0, x0, atol=0.15)
    assert np.allclose(arc.v0, v0, atol=1.5)
    assert arc.rms_px < 0.1


def test_fit_arc_tolerates_pixel_noise():
    rng = np.random.default_rng(3)
    camera = make_camera()
    times = np.arange(15) / 60.0
    pixels = _project_trajectory(
        camera, [16.0, 22.0, 2.0], [-8.0, -60.0, 15.0], times)
    noisy = pixels + rng.normal(0, 1.0, pixels.shape)
    arc = ballistic.fit_arc(times, noisy, camera)
    assert arc is not None
    assert arc.rms_px < 3.0


def test_fit_arc_too_short_returns_none():
    camera = make_camera()
    times = np.arange(2) / 60.0
    pixels = _project_trajectory(camera, [10.0, 16.0, 3.0], [0.0, -40.0, 5.0], times)
    assert ballistic.fit_arc(times, pixels, camera) is None


def test_position_velocity_evaluation():
    arc = BallisticArc(t_ref=1.0, x0=np.zeros(3), v0=np.array([1.0, 2.0, 3.0]),
                       rms_px=0.0, start=0, end=5)
    assert np.allclose(arc.position(1.0), [0.0, 0.0, 0.0])
    expected_z = 3.0 * 0.5 + 0.5 * GRAVITY_VEC[2] * 0.25
    assert np.allclose(arc.position(1.5), [0.5, 1.0, expected_z])
    assert np.allclose(arc.velocity(1.5), [1.0, 2.0, 3.0 + GRAVITY_VEC[2] * 0.5])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ballistic.py -q`
Expected: FAIL with `ModuleNotFoundError: ballistic`.

- [ ] **Step 3: Implement**

Create `ballistic.py`:

```python
"""Gravity-constrained 3D trajectory fitting for ball tracks.

A ball in free flight follows X(t) = X0 + V0*tau + 0.5*g*tau^2 in court
coordinates (feet, z up). With a calibrated camera each detection pixel
contributes two linear constraints on X(t) (the cross-product form of the
projection equation), and X(t) is linear in (X0, V0) - so fitting an arc to
any frame range is one linear least-squares solve in 6 unknowns. Contacts
are where consecutive arcs meet (segmentation lives here too; see
segment_track / arc_boundary_events).
"""

from dataclasses import dataclass

import numpy as np

from court_model import G_FT_PER_S2

GRAVITY_VEC = np.array([0.0, 0.0, -G_FT_PER_S2])


@dataclass(frozen=True)
class BallisticArc:
    t_ref: float
    x0: np.ndarray
    v0: np.ndarray
    rms_px: float
    start: int
    end: int

    def position(self, t):
        tau = t - self.t_ref
        return self.x0 + self.v0 * tau + 0.5 * GRAVITY_VEC * tau * tau

    def velocity(self, t):
        return self.v0 + GRAVITY_VEC * (t - self.t_ref)


def fit_arc(times, pixels_und, camera, start=0, end=None):
    """Fit one ballistic arc to samples [start:end); None if degenerate.

    times: (N,) seconds. pixels_und: (N,2) undistorted pixels. The rows are
    normalized so the algebraic residual approximates pixel error; rms_px is
    then computed exactly by reprojection.
    """
    end = len(times) if end is None else end
    if end - start < 3:
        return None
    projection = camera.projection_matrix()
    t_ref = float(times[start])
    rows, rhs = [], []
    for index in range(start, end):
        tau = float(times[index]) - t_ref
        drop = 0.5 * GRAVITY_VEC * tau * tau
        for matrix_row, pixel_coord in (
            (projection[0], float(pixels_und[index][0])),
            (projection[1], float(pixels_und[index][1])),
        ):
            constraint = pixel_coord * projection[2] - matrix_row  # 4-vector
            spatial = constraint[:3]
            norm = np.linalg.norm(spatial)
            if norm < 1e-12:
                continue
            rows.append(np.concatenate([spatial, tau * spatial]) / norm)
            rhs.append(-(spatial @ drop + constraint[3]) / norm)
    if len(rows) < 6:
        return None
    system = np.asarray(rows)
    try:
        solution, *_ = np.linalg.lstsq(system, np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:
        return None
    x0, v0 = solution[:3], solution[3:]

    errors = []
    for index in range(start, end):
        tau = float(times[index]) - t_ref
        point = x0 + v0 * tau + 0.5 * GRAVITY_VEC * tau * tau
        try:
            u, v = camera.project(point)
        except ValueError:
            return None
        du = u - float(pixels_und[index][0])
        dv = v - float(pixels_und[index][1])
        errors.append(du * du + dv * dv)
    return BallisticArc(
        t_ref=t_ref, x0=x0, v0=v0,
        rms_px=float(np.sqrt(np.mean(errors))),
        start=int(start), end=int(end),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ballistic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ballistic.py tests/test_ballistic.py
git commit -m "ballistic.py: linear least-squares gravity-constrained arc fit"
```

---

### Task 6: Greedy 3D segmentation + sub-frame impact refinement + event records

**Files:**
- Modify: `ballistic.py`
- Test: `tests/test_ballistic.py`

**Interfaces:**
- Consumes: `fit_arc`, `BallisticArc` (Task 5).
- Produces:
  - `ballistic.segment_track(times, pixels_und, camera, rms_px, min_points) -> list[BallisticArc | tuple]` — greedy maximal arcs mirroring `event_engine.segment_into_arcs`; ranges too short/degenerate to fit appear as plain `(start, end)` tuples so no samples are silently dropped.
  - `ballistic.refine_impact(arc_a, arc_b, t_lo, t_hi) -> (t_star, point_xyz, v_in_3d, v_out_3d)` — closest approach of two arcs (linear in t because both share gravity), clamped to `[t_lo, t_hi]`.
  - `ballistic.arc_boundary_events(frames, timestamps, positions, tracks, camera, cfg) -> list[dict]` — event dicts shaped exactly like `event_engine._make_event` output (`index/frame/time/x/y/v_in/v_out/speed_before/speed_after/dv_magnitude/turn_degrees/methods={"ballistic"}`), plus a `"contact_3d"` dict: `{"time", "point_ft": (3,) list, "v_in_ft_s": list, "v_out_ft_s": list, "arc_rms_px": [a, b]}`. Positions are raw pixels; this function undistorts internally with `camera.distortion`. `cfg` keys used: `"arc3d_rms_px"`, `"arc_min_points"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ballistic.py`:

```python
def _bounce_trajectory(camera, fps=60.0):
    """Drive at the front wall, bounce off it, 3D ground truth throughout.
    Returns (times, raw_pixels, bounce_time, bounce_point)."""
    x0 = np.array([10.0, 24.0, 4.0])
    v0 = np.array([2.0, -70.0, 6.0])
    t_hit = None
    times, pixels, points = [], [], []
    t, dt = 0.0, 1.0 / fps
    position, velocity = x0.copy(), v0.copy()
    for _ in range(40):
        times.append(t)
        points.append(position.copy())
        pixels.append(camera.project(position))
        # integrate one step; reflect off the front wall plane y=0
        velocity_next = velocity + GRAVITY_VEC * dt
        position_next = position + velocity * dt + 0.5 * GRAVITY_VEC * dt * dt
        if position_next[1] <= 0.0 and t_hit is None:
            fraction = position[1] / (position[1] - position_next[1])
            t_hit = t + fraction * dt
            hit_point = position + (position_next - position) * fraction
            position_next[1] = -position_next[1]
            velocity_next[1] = -0.7 * velocity_next[1]  # restitution
        position, velocity = position_next, velocity_next
        t += dt
    return (np.asarray(times), np.asarray(pixels), t_hit, hit_point)


def test_segment_track_finds_wall_bounce():
    camera = make_camera()
    times, pixels, t_hit, _ = _bounce_trajectory(camera)
    arcs = ballistic.segment_track(times, pixels, camera,
                                   rms_px=3.0, min_points=5)
    fitted = [a for a in arcs if isinstance(a, BallisticArc)]
    assert len(fitted) == 2
    boundary_time = times[fitted[0].end]
    assert boundary_time == pytest.approx(t_hit, abs=3.0 / 60.0)


def test_refine_impact_locates_wall_contact():
    camera = make_camera()
    times, pixels, t_hit, hit_point = _bounce_trajectory(camera)
    arcs = [a for a in ballistic.segment_track(times, pixels, camera, 3.0, 5)
            if isinstance(a, BallisticArc)]
    arc_a, arc_b = arcs
    t_star, point, v_in, v_out = ballistic.refine_impact(
        arc_a, arc_b, times[arc_a.end - 1], times[arc_b.start])
    assert t_star == pytest.approx(t_hit, abs=1.5 / 60.0)
    assert point[1] == pytest.approx(0.0, abs=1.0)       # on the front wall
    assert np.allclose(point, hit_point, atol=1.5)
    assert v_in[1] < 0 < v_out[1]                        # depth reversal


def test_arc_boundary_events_shape():
    camera = make_camera()
    times, pixels, _, _ = _bounce_trajectory(camera)
    frames = np.arange(len(times))
    cfg = {"arc3d_rms_px": 3.0, "arc_min_points": 5}
    events = ballistic.arc_boundary_events(
        frames, times, pixels, [(0, len(times))], camera, cfg)
    assert len(events) == 1
    event = events[0]
    assert event["methods"] == {"ballistic"}
    assert "contact_3d" in event
    assert len(event["contact_3d"]["point_ft"]) == 3
    for key in ("v_in", "v_out", "speed_before", "dv_magnitude", "turn_degrees"):
        assert event[key] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ballistic.py -q`
Expected: FAIL with `AttributeError` on `segment_track`.

- [ ] **Step 3: Implement**

Append to `ballistic.py`:

```python
from court_model import undistort_point


def segment_track(times, pixels_und, camera, rms_px, min_points):
    """Greedy maximal ballistic arcs, mirroring event_engine.segment_into_arcs:
    grow each arc until adding the next sample pushes reprojection rms past
    rms_px. Unfittable ranges come back as (start, end) tuples."""
    segments = []
    start, count = 0, len(times)
    while start < count:
        end = min(start + min_points, count)
        best = fit_arc(times, pixels_und, camera, start, end)
        if best is None:
            segments.append((start, end))
            start = end
            continue
        while end < count:
            grown = fit_arc(times, pixels_und, camera, start, end + 1)
            if grown is None or grown.rms_px > rms_px:
                break
            best, end = grown, end + 1
        segments.append(best)
        start = end
    return segments


def refine_impact(arc_a, arc_b, t_lo, t_hi):
    """Closest approach of two arcs. Both share the gravity term, so their
    difference is linear in t and the minimizer is closed-form."""
    t_mid = 0.5 * (t_lo + t_hi)
    offset = arc_a.position(t_mid) - arc_b.position(t_mid)
    relative = arc_a.velocity(t_mid) - arc_b.velocity(t_mid)
    denom = float(relative @ relative)
    t_star = t_mid if denom < 1e-9 else t_mid - float(offset @ relative) / denom
    t_star = min(max(t_star, t_lo), t_hi)
    point = 0.5 * (arc_a.position(t_star) + arc_b.position(t_star))
    return t_star, point, arc_a.velocity(t_star), arc_b.velocity(t_star)


def _projected_velocity(camera, arc, t, dt=1.0 / 120.0):
    """Image-plane velocity (px/s) of the arc at time t, for 2D-compatible
    event fields."""
    u1, v1 = camera.project(arc.position(t - dt))
    u2, v2 = camera.project(arc.position(t + dt))
    return np.array([(u2 - u1) / (2 * dt), (v2 - v1) / (2 * dt)])


def arc_boundary_events(frames, timestamps, positions, tracks, camera, cfg):
    """Contact events at every boundary between adjacent fitted 3D arcs.

    Event dicts match event_engine._make_event plus a "contact_3d" payload.
    positions are raw pixels; undistortion happens here.
    """
    from event_engine import _make_event  # shared event shape, no cycle at import time

    rms_px = cfg["arc3d_rms_px"]
    min_points = cfg["arc_min_points"]
    events = []
    for track_start, track_end in tracks:
        if track_end - track_start < 2 * min_points:
            continue
        times = timestamps[track_start:track_end]
        pixels = np.asarray(
            [undistort_point(p, camera.distortion)
             for p in positions[track_start:track_end]]
        )
        segments = segment_track(times, pixels, camera, rms_px, min_points)
        for k in range(1, len(segments)):
            previous, current = segments[k - 1], segments[k]
            if not isinstance(previous, BallisticArc) or not isinstance(
                current, BallisticArc
            ):
                continue  # unfittable gap: derivative/audio methods still cover it
            if (previous.end - previous.start < min_points
                    or current.end - current.start < min_points):
                continue
            t_lo = float(times[previous.end - 1])
            t_hi = float(times[current.start])
            t_star, point, v_in_3d, v_out_3d = refine_impact(
                previous, current, t_lo, t_hi)
            try:
                v_in_px = _projected_velocity(camera, previous, t_lo)
                v_out_px = _projected_velocity(camera, current, t_hi)
            except ValueError:
                continue
            event = _make_event(
                track_start + current.start, frames, timestamps, positions,
                v_in_px, v_out_px, "ballistic",
            )
            event["contact_3d"] = {
                "time": float(t_star),
                "point_ft": [float(c) for c in point],
                "v_in_ft_s": [float(c) for c in v_in_3d],
                "v_out_ft_s": [float(c) for c in v_out_3d],
                "arc_rms_px": [float(previous.rms_px), float(current.rms_px)],
            }
            events.append(event)
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ballistic.py -q`
Expected: PASS. If the segmentation test finds 3 arcs instead of 2, the greedy threshold is too tight for the synthetic noise — check `rms_px` passed in the test matches the implementation's usage before touching tolerances.

- [ ] **Step 5: Commit**

```bash
git add ballistic.py tests/test_ballistic.py
git commit -m "ballistic.py: 3D segmentation, closest-approach impact, contact events"
```

---

### Task 7: Wire the ballistic source into `detect_events_fused` + job flag

**Files:**
- Modify: `event_engine.py` (`detect_events_fused`, `FUSION_DEFAULTS`)
- Modify: `job_runner.py` (fusion branch ~line 771; payload assembly ~line 582)
- Test: `tests/test_event_engine.py`

**Interfaces:**
- Consumes: `ballistic.arc_boundary_events`, `court_model.solve_camera_model`.
- Produces:
  - `detect_events_fused(rows, audio_windows=None, calibration=None, wall_x_range=None, config=None, max_gap=MAX_GAP_FRAMES, camera=None)` — new keyword-only-style trailing param, default `None` (2D behavior byte-identical).
  - `FUSION_DEFAULTS` gains `"arc3d_rms_px": 3.0`.
  - Hits produced in 3D mode carry `hit["contact_3d"]` when their event had one.
  - `job_runner`: when `engine == "fusion"` and `job.get("fusion_3d")` is truthy, solve the camera and pass it; payload gains `"camera_model"` (dict) on success or `"camera_warning"` (info dict) on failure.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_event_engine.py` (reuse the file's existing row-building helpers for the first test):

```python
def test_camera_none_is_default_behavior():
    # Passing camera=None must equal not passing it at all.
    rows = _rows_for_simple_bounce()  # use/adapt an existing fixture helper
    baseline = detect_events_fused(rows)
    explicit = detect_events_fused(rows, camera=None)
    assert baseline == explicit


def test_ballistic_source_used_with_camera():
    import numpy as np
    from synthetic3d import make_camera
    from tests_ballistic_helpers import make_bounce_rows  # see Step 3
    camera = make_camera()
    rows, expected_frame = make_bounce_rows(camera)
    hits = detect_events_fused(rows, camera=camera)
    assert any("ballistic" in hit["methods"] for hit in hits)
    matched = [h for h in hits if abs(h["hit_frame"] - expected_frame) <= 3]
    assert matched and "contact_3d" in matched[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_event_engine.py -q`
Expected: FAIL — `detect_events_fused` has no `camera` parameter.

- [ ] **Step 3: Implement the engine change**

In `event_engine.py`:

1. `FUSION_DEFAULTS` — add after `"arc_min_points": 4,`:

```python
    "arc3d_rms_px": 3.0,         # 3D reprojection tolerance before an arc breaks
```

2. `detect_events_fused` signature (`event_engine.py:436`): add trailing `camera=None` parameter and docstring line "camera: optional court_model.CameraModel; when present the parabola trajectory source is replaced by 3D ballistic segmentation and events gain contact_3d."

3. Replace the trajectory-source block (currently `events = merge_trajectory_events(parabolic_arc_events(...), derivative_events(...), cfg["merge_gap_s"])`) with:

```python
        if camera is not None:
            from ballistic import arc_boundary_events

            trajectory_events = arc_boundary_events(
                frames, timestamps, positions, tracks, camera, cfg)
        else:
            trajectory_events = parabolic_arc_events(
                frames, timestamps, positions, tracks, cfg)
        events = merge_trajectory_events(
            trajectory_events,
            derivative_events(frames, timestamps, positions, tracks, cfg),
            cfg["merge_gap_s"],
        )
```

Note `merge_trajectory_events` prefers the representative whose methods contain `"parabola"`; extend that preference to `"ballistic"` (both are fitted, stabler velocities): change the condition to check `{"parabola", "ballistic"} & event["methods"]`.

4. In the hit-assembly loop, right after `hit["signals"] = signals`, carry the 3D payload through:

```python
        if event.get("contact_3d"):
            hit["contact_3d"] = event["contact_3d"]
```

- [ ] **Step 4: Add the test fixture helper**

Create `tests/tests_ballistic_helpers.py`:

```python
"""Row-shaped synthetic data for engine-level 3D tests."""
import numpy as np

from ballistic import GRAVITY_VEC


def make_bounce_rows(camera, fps=60.0):
    """CSV-row dicts for a drive that bounces off the front wall.
    Returns (rows, bounce_frame)."""
    position = np.array([10.0, 24.0, 4.0])
    velocity = np.array([2.0, -70.0, 6.0])
    dt = 1.0 / fps
    rows, bounce_frame = [], None
    for frame in range(40):
        u, v = camera.project(position)
        rows.append({
            "source_frame": frame, "timestamp_seconds": frame * dt,
            "detected": "true", "x": u, "y": v, "width": 8.0, "height": 8.0,
        })
        velocity_next = velocity + GRAVITY_VEC * dt
        position_next = position + velocity * dt + 0.5 * GRAVITY_VEC * dt * dt
        if position_next[1] <= 0.0 and bounce_frame is None:
            bounce_frame = frame + 1
            position_next[1] = -position_next[1]
            velocity_next[1] = -0.7 * velocity_next[1]
        position, velocity = position_next, velocity_next
    return rows, bounce_frame
```

Adapt the row keys to exactly what `load_detected_positions_from_rows` (`detect_wall_hits.py`) expects — check its parsing and mirror the real CSV column names (`source_frame`, `timestamp_seconds`, `detected`, ball center x/y, width/height). This helper must produce rows indistinguishable from real tracking rows.

- [ ] **Step 5: Run engine tests**

Run: `.venv/bin/python -m pytest tests/test_event_engine.py -q`
Expected: PASS, including the untouched 2D tests.

- [ ] **Step 6: Wire the job flag in `job_runner.py`**

In the fusion branch (around line 771):

```python
                if engine == "fusion":
                    update_job(run_id, stage="judging", message="Judging wall hits...")
                    camera = camera_info = None
                    if job.get("fusion_3d") and calibration:
                        camera, camera_info = court_model.solve_camera_model(calibration)
                    classified = detect_events_fused(
                        sorted_rows(results),
                        audio_windows=audio_windows,
                        calibration=calibration,
                        wall_x_range=wall_x_range,
                        config=job.get("fusion"),
                        max_gap=max(MAX_GAP_FRAMES, frame_stride),
                        camera=camera,
                    )
```

Thread `camera`/`camera_info` to the payload assembly (the function that writes `detected_hits.json` around line 582–585; pass them as parameters or attach to the hits container the same way `floor_zones` flows):

```python
    if camera is not None:
        payload["camera_model"] = camera.to_dict()
    elif camera_info is not None:
        payload["camera_warning"] = camera_info
```

The flag default is **off** (`job.get("fusion_3d")` falsy) until Task 11 flips it.

- [ ] **Step 7: Full suite + commit**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

```bash
git add event_engine.py job_runner.py tests/test_event_engine.py tests/tests_ballistic_helpers.py
git commit -m "Fusion engine: 3D ballistic trajectory source behind fusion_3d flag"
```

---

### Task 8: 3D emission scores (plane distances + velocity reflection)

**Files:**
- Modify: `event_engine.py` (`FUSION_DEFAULTS`, `_emission_scores`, new `_emission_scores_3d`, `detect_events_fused`)
- Test: `tests/test_event_engine.py`

**Interfaces:**
- Consumes: `hit`/`event["contact_3d"]` payload (Task 6/7), `CameraModel.depth_ft`, `CameraModel.ray`.
- Produces:
  - `event_engine._surface_geometry(point_ft)` → list of `(state, normal np.ndarray, distance_ft)` for `("floor", (0,0,1), z)`, `("wall", (0,1,0), y)`, `("side", (±1,0,0), min(x, 21-x))`.
  - `event_engine._emission_scores_3d(event, audio_available, camera, cfg) -> dict` — same key set as `_emission_scores` (`wall/floor/side/racket`).
  - `_emission_scores` refactored: audio scoring extracted into `_audio_scores(event, audio_available, cfg) -> dict` used by both paths (behavior identical in 2D — existing tests prove it).
  - New `FUSION_DEFAULTS` keys (exact values below).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_event_engine.py`:

```python
def _contact_event(point_ft, v_in, v_out):
    return {
        "x": 900.0, "y": 500.0, "time": 1.0, "frame": 60, "index": 10,
        "v_in": np.array([0.0, 0.0]), "v_out": np.array([0.0, 0.0]),
        "speed_before": 1.0, "speed_after": 1.0,
        "methods": {"ballistic"}, "audio_window": None, "size_ratio": None,
        "contact_3d": {
            "time": 1.0, "point_ft": list(point_ft),
            "v_in_ft_s": list(v_in), "v_out_ft_s": list(v_out),
            "arc_rms_px": [0.5, 0.5],
        },
    }


def test_3d_emissions_floor_bounce():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 15.0, 0.2), (5.0, -20.0, -18.0), (4.0, -16.0, 12.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["floor"] == max(scores.values())


def test_3d_emissions_front_wall_bounce():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 0.4, 6.0), (2.0, -60.0, 4.0), (1.5, 40.0, 1.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["wall"] == max(scores.values())


def test_3d_emissions_racket_interior_energy_gain():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 25.0, 4.0), (3.0, 30.0, -4.0), (2.0, -70.0, 10.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["racket"] == max(scores.values())


def test_3d_emissions_side_wall():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((0.3, 12.0, 5.0), (-30.0, -30.0, 2.0), (22.0, -26.0, 1.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["side"] == max(scores.values())
```

(Import `_emission_scores_3d` and `merge_fusion_config` at the top of the test file alongside the existing event_engine imports; add `import numpy as np` if absent.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_event_engine.py -q`
Expected: FAIL with ImportError on `_emission_scores_3d`.

- [ ] **Step 3: Implement**

In `event_engine.py`:

1. Add to `FUSION_DEFAULTS` (after the side-wall block):

```python
    # 3D evidence mode (camera present). Distances in feet; sigma scales
    # with per-event depth resolution so far contacts get honest tolerance.
    "plane_sigma_px": 3.0,        # pixel noise driving positional sigma
    "plane_sigma_min_ft": 0.4,
    "surface_near_bonus": 1.5,
    "surface_far_penalty": 1.0,
    "reflection_bonus": 0.75,     # velocity mirrors about the surface normal
    "reflection_penalty": 0.5,
    "interior_clearance_ft": 2.5, # farther than this from every surface
    "interior_racket_bonus": 1.25,
```

2. Extract the audio block of `_emission_scores` (the `if audio_available:` section, lines ~331–348) into:

```python
def _audio_scores(event, audio_available, cfg):
    scores = {"wall": 0.0, "floor": 0.0, "side": 0.0, "racket": 0.0}
    # ... body moved verbatim from _emission_scores ...
    return scores
```

and have `_emission_scores` start from `scores = _audio_scores(event, audio_available, cfg)`. No behavior change — the 2D tests prove it.

3. Add the 3D scorer:

```python
def _surface_geometry(point_ft):
    x, y, z = (float(c) for c in point_ft)
    side_normal = np.array([1.0, 0.0, 0.0]) if x <= 10.5 else np.array([-1.0, 0.0, 0.0])
    return [
        ("floor", np.array([0.0, 0.0, 1.0]), z),
        ("wall", np.array([0.0, 1.0, 0.0]), y),
        ("side", side_normal, min(x, 21.0 - x)),
    ]


def _positional_sigma_ft(camera, point_ft, normal, cfg):
    """Pixel noise mapped to feet at this point, inflated when the surface
    normal is nearly parallel to the viewing ray (poorly observed axis)."""
    point = np.asarray(point_ft, dtype=float)
    transverse = cfg["plane_sigma_px"] * camera.depth_ft(point) / camera.focal_px
    _, direction = camera.ray(camera.project(point))
    perpendicular = normal - (normal @ direction) * direction
    observability = max(0.2, float(np.linalg.norm(perpendicular)))
    return max(cfg["plane_sigma_min_ft"], transverse / observability)


def _emission_scores_3d(event, audio_available, camera, cfg):
    scores = _audio_scores(event, audio_available, cfg)
    contact = event["contact_3d"]
    point = np.asarray(contact["point_ft"], dtype=float)
    v_in = np.asarray(contact["v_in_ft_s"], dtype=float)
    v_out = np.asarray(contact["v_out_ft_s"], dtype=float)
    speed_in = float(np.linalg.norm(v_in))
    speed_out = float(np.linalg.norm(v_out))

    min_distance = None
    try:
        surfaces = [
            (state, normal, distance, _positional_sigma_ft(camera, point, normal, cfg))
            for state, normal, distance in _surface_geometry(point)
        ]
    except ValueError:  # contact point projected behind the camera: no 3D vote
        return scores
    for state, normal, distance, sigma in surfaces:
        near = float(np.exp(-0.5 * (distance / sigma) ** 2))
        scores[state] += cfg["surface_near_bonus"] * near
        scores[state] -= cfg["surface_far_penalty"] * (1.0 - near)
        if speed_in > 1e-6 and speed_out > 1e-6:
            reflected = v_in - 2.0 * float(v_in @ normal) * normal
            alignment = float(reflected @ v_out) / (
                np.linalg.norm(reflected) * speed_out)
            restitution = speed_out / speed_in
            if alignment > 0 and restitution <= 1.05:
                scores[state] += cfg["reflection_bonus"] * near * alignment
            else:
                scores[state] -= cfg["reflection_penalty"] * near
        min_distance = distance if min_distance is None else min(min_distance, distance)

    if min_distance is not None and min_distance > cfg["interior_clearance_ft"]:
        scores["racket"] += cfg["interior_racket_bonus"]
    if speed_in > 1e-6 and speed_out / speed_in >= cfg["racket_speed_gain"]:
        scores["racket"] += cfg["racket_gain_bonus"]
        scores["floor"] -= cfg["floor_gain_penalty"]

    # Debuggability contract from the spec: the evidence that scored this
    # event must survive into the hit's signals (see step below).
    event["evidence_3d"] = {
        "mode": "3d",
        "plane_distance_ft": {state: round(distance, 3)
                              for state, _, distance, _ in surfaces},
        "sigma_ft": {state: round(sigma, 3)
                     for state, _, _, sigma in surfaces},
        "restitution": (round(speed_out / speed_in, 3)
                        if speed_in > 1e-6 else None),
    }
    return scores
```

5. In `detect_events_fused`'s hit loop, where Task 7 added the `contact_3d` carry-through (right after `hit["signals"] = signals`), also persist the evidence:

```python
        if event.get("contact_3d"):
            hit["contact_3d"] = event["contact_3d"]
            hit["signals"]["evidence_3d"] = event.get("evidence_3d")
```

4. In `detect_events_fused`, replace the emission construction:

```python
    emissions = [
        _emission_scores_3d(event, audio_available, camera, cfg)
        if camera is not None and event.get("contact_3d")
        else _emission_scores(event, audio_available, wall_region, cfg)
        for event in events
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass — the 4 new 3D emission tests and every existing 2D test.

- [ ] **Step 5: Commit**

```bash
git add event_engine.py tests/test_event_engine.py
git commit -m "3D emissions: plane-distance + reflection evidence with honest sigma"
```

---

### Task 9: Downstream handoff — metric wall impacts and floor positions

**Files:**
- Modify: `event_engine.py` (hit assembly in `detect_events_fused`)
- Modify: `job_runner.py` (`judge_hits`, around lines 480 and 573)
- Test: `tests/test_event_engine.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `hit["contact_3d"]`, `CameraModel.project`, `court_model.distort_point`, `court_model.floor_zone_for_point`.
- Produces:
  - Wall-labeled hits in 3D mode carry `impact_x`/`impact_y` (raw-pixel space, distorted back so `judge_ball` compares against the calibration lines correctly), `impact_time`, and `impact_height_ft`.
  - Floor-labeled hits in 3D mode carry `court_position_ft: {"x": ..., "y": ...}`.
  - `judge_hits` uses an engine-supplied `court_position_ft` if present instead of recomputing from the floor homography.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_event_engine.py`:

```python
def test_3d_wall_hit_carries_metric_impact():
    from synthetic3d import make_camera
    from tests_ballistic_helpers import make_bounce_rows
    camera = make_camera()
    rows, expected_frame = make_bounce_rows(camera)
    hits = detect_events_fused(rows, camera=camera)
    wall_hits = [h for h in hits if h["event_type"] == "wall"
                 and abs(h["hit_frame"] - expected_frame) <= 3]
    assert wall_hits
    hit = wall_hits[0]
    assert "impact_x" in hit and "impact_y" in hit
    assert "impact_height_ft" in hit
    assert 0.0 < hit["impact_height_ft"] < 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_event_engine.py -q`
Expected: FAIL on the missing `impact_height_ft` (3D wall hits currently only get the 2D two-stage fit, gated on `label == "wall" and calibration`).

- [ ] **Step 3: Implement the engine side**

In `detect_events_fused`'s hit loop, extend the wall-impact block. The existing block runs `detect_bounce_two_stage` for `label == "wall"`; make the 3D path take priority:

```python
        if label == "wall" and event.get("contact_3d") and camera is not None:
            contact = event["contact_3d"]
            point = np.asarray(contact["point_ft"], dtype=float)
            wall_point = point.copy()
            wall_point[1] = 0.0  # snap onto the front-wall plane for judging
            try:
                pixel = camera.project(wall_point)
                from court_model import distort_point
                hit["impact_x"], hit["impact_y"] = distort_point(
                    pixel, camera.distortion)
                hit["impact_time"] = contact["time"]
                hit["impact_height_ft"] = float(point[2])
            except ValueError:
                pass  # fall through to the 2D impact fit below
        if label == "wall" and calibration and event["index"] is not None \
                and "impact_x" not in hit:
            # ... existing detect_bounce_two_stage block unchanged ...
```

And for floor hits, right after the label is known:

```python
        if label == "floor" and event.get("contact_3d"):
            point = event["contact_3d"]["point_ft"]
            hit["court_position_ft"] = {"x": float(point[0]), "y": float(point[1])}
```

- [ ] **Step 4: Implement the judge side**

In `job_runner.judge_hits`, where floor entries currently compute `entry["court_position_ft"]` from `floor_map.image_to_court` (around line 573), prefer the engine's metric value:

```python
                if hit.get("court_position_ft"):
                    entry["court_position_ft"] = hit["court_position_ft"]
                    x_ft, y_ft = hit["court_position_ft"]["x"], hit["court_position_ft"]["y"]
                    entry["floor_zone"] = court_model.floor_zone_for_point(x_ft, y_ft)
                elif floor_map is not None:
                    # ... existing homography path unchanged ...
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass (pipeline tests confirm `judge_hits` still works in 2D mode).

- [ ] **Step 6: Commit**

```bash
git add event_engine.py job_runner.py tests/test_event_engine.py
git commit -m "3D handoff: metric wall impact height + floor court positions"
```

---

### Task 10: Offline A/B harness — `rerun_detection.py`

The eval set bakes in detector output at build time, so measuring the 3D delta requires re-running detection over stored runs. This harness replays fusion over each run's `ball_coordinates.csv` + `calibration.json` into a **mirror directory** (never touching real run artifacts), with and without the camera. Audio windows aren't persisted in run dirs, so both replay arms run without audio — an apples-to-apples comparison of exactly the trajectory/emission change.

**Files:**
- Create: `rerun_detection.py`
- Test: `tests/test_ballistic.py` (smoke test via tmp dir)

**Interfaces:**
- Consumes: `detect_events_fused`, `solve_camera_model`, run-dir layout (`ball_coordinates.csv`, `calibration.json`, `ground_truth.json`, `corrections.json`).
- Produces: CLI `rerun_detection.py --runs-dir ui_runs --out-dir <mirror> [--use-3d]`. For every run dir containing `ball_coordinates.csv` + `ground_truth.json` (or `corrections.json`): a mirror dir with the label files copied and a fresh `detected_hits.json` `{"hits": [...judge-shaped entries with "frame" and "event_type"...]}`. Prints per-run status and a final count; skipped runs are listed, never silent.

- [ ] **Step 1: Write the script**

Create `rerun_detection.py`:

```python
"""Replay fusion detection over stored runs into a mirror dir for eval A/B.

Real run artifacts are never modified. Audio windows are not persisted in
run dirs, so replays run without audio evidence in BOTH arms - this
compares the trajectory + emission change, holding everything else fixed.

Usage:
  python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval2d
  python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval3d --use-3d
Then: python build_eval_set.py --runs-dir <out-dir> ... and eval_line_calls.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import court_model
from event_engine import detect_events_fused


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["source_frame"]))
    return rows


def replay_run(run_dir, out_dir, use_3d):
    rows = load_rows(run_dir / "ball_coordinates.csv")
    calibration = None
    calibration_path = run_dir / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera = camera_info = None
    if use_3d and calibration is not None:
        camera, camera_info = court_model.solve_camera_model(calibration)
    hits = detect_events_fused(rows, audio_windows=None,
                               calibration=calibration, camera=camera)
    for hit in hits:
        hit["frame"] = hit["hit_frame"]  # build_eval_set matches on "frame"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ground_truth.json", "corrections.json", "calibration.json"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    payload = {"hits": hits, "replay": {"use_3d": bool(use_3d and camera),
                                        "camera_info": camera_info}}
    (out_dir / "detected_hits.json").write_text(
        json.dumps(payload, default=float), encoding="utf-8")
    return len(hits), camera is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("ui_runs"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--use-3d", action="store_true")
    args = parser.parse_args()

    replayed, skipped = 0, []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        has_labels = (run_dir / "ground_truth.json").exists() or \
                     (run_dir / "corrections.json").exists()
        if not has_labels or not (run_dir / "ball_coordinates.csv").exists():
            skipped.append(run_dir.name)
            continue
        try:
            count, solved = replay_run(run_dir, args.out_dir / run_dir.name,
                                       args.use_3d)
        except Exception as error:  # a bad run must not sink the sweep
            print(f"  ERROR {run_dir.name}: {error}")
            skipped.append(run_dir.name)
            continue
        print(f"  {run_dir.name}: {count} hits"
              + (" [3D]" if solved else " [2D]"))
        replayed += 1
    print(f"{replayed} runs replayed, {len(skipped)} skipped: {skipped}")


if __name__ == "__main__":
    main()
```

If `detect_events_fused`'s row loader (`load_detected_positions_from_rows`) needs typed values rather than CSV strings, mirror whatever `job_runner.sorted_rows(results)` produces — check that function first and convert in `load_rows` to match.

- [ ] **Step 2: Write a smoke test**

Append to `tests/test_ballistic.py`:

```python
def test_rerun_detection_smoke(tmp_path):
    import json as jsonlib
    from rerun_detection import replay_run
    from tests_ballistic_helpers import make_bounce_rows
    camera = make_camera()
    rows, _ = make_bounce_rows(camera)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    with open(run_dir / "ball_coordinates.csv", "w", newline="") as handle:
        import csv as csvlib
        writer = csvlib.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "ground_truth.json").write_text(jsonlib.dumps(
        {"events": [{"frame": 20, "type": "wall"}]}))
    count, solved = replay_run(run_dir, tmp_path / "out" / "run1", use_3d=False)
    assert (tmp_path / "out" / "run1" / "detected_hits.json").exists()
    assert not solved
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ballistic.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add rerun_detection.py tests/test_ballistic.py
git commit -m "rerun_detection.py: offline fusion replay harness for eval A/B"
```

---

### Task 11: Eval gate — measure the 3D delta, flip the default

**Files:**
- Create: `eval_set/RESULTS-3d-contact.md`
- Modify: `job_runner.py` (one line, only if the gate passes)

**Interfaces:**
- Consumes: Tasks 1–10.
- Produces: recorded A/B numbers; `fusion_3d` default flipped to on iff the gate passes.

- [ ] **Step 1: Run the A/B**

```bash
.venv/bin/python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval-2d
.venv/bin/python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval-3d --use-3d
.venv/bin/python build_eval_set.py --runs-dir /tmp/eval-2d --out /tmp/eval-2d-set
.venv/bin/python build_eval_set.py --runs-dir /tmp/eval-3d --out /tmp/eval-3d-set
.venv/bin/python eval_line_calls.py --eval-set /tmp/eval-2d-set/cases.jsonl --verbose
.venv/bin/python eval_line_calls.py --eval-set /tmp/eval-3d-set/cases.jsonl --verbose
```

(Adjust `--out` to build_eval_set's actual CLI shape, same as Task 1.) Note how many runs actually solved a camera (`[3D]` lines) — if zero solve, stop: fix `solve_camera_model` against real calibrations before any comparison is meaningful.

- [ ] **Step 2: Record results**

Write `eval_set/RESULTS-3d-contact.md`: date, commit, runs replayed, cameras solved, and a table of the axes (missed-bounce rate, type accuracy/confusion, position error, timing) for 2D vs 3D. State the gate verdict explicitly: **pass = missed-bounce and type axes improve, no axis regresses** (position/timing within noise counts as no regression; say so with numbers).

- [ ] **Step 3: Flip the default only on a pass**

If the gate passes, in `job_runner.py` change `job.get("fusion_3d")` to `job.get("fusion_3d", True)` so calibrated runs default to 3D (uncalibrated runs still degrade automatically — `solve_camera_model` returns None). If the gate fails, do NOT flip; file the failure modes in `RESULTS-3d-contact.md` and stop this plan after committing the results — tuning (Task 12) may rescue it.

- [ ] **Step 4: Full suite + golden-run check**

```bash
.venv/bin/python -m pytest tests -q
```

Expected: all pass. Then re-run the replay for the golden run only and eyeball the hit sequence against the known rally (`ui_runs/1784236711057`, the fusion docstring's reference footage):

```bash
.venv/bin/python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/golden-3d --use-3d
.venv/bin/python -c "
import json
hits = json.load(open('/tmp/golden-3d/1784236711057/detected_hits.json'))['hits']
print([(h['frame'], h['event_type']) for h in hits])"
```

- [ ] **Step 5: Commit**

```bash
git add eval_set/RESULTS-3d-contact.md job_runner.py
git commit -m "Eval gate: record 2D vs 3D contact detection A/B; flip fusion_3d default"
```

---

### Task 12: Emission-weight tuning from corrections (phase 4)

**Files:**
- Create: `tune_fusion_weights.py`
- Test: none (offline tool; correctness = reproducible eval numbers)

**Interfaces:**
- Consumes: `rerun_detection.replay_run`, `build_eval_set.build_eval_set`, `eval_line_calls.evaluate_cases` / `load_eval_set`.
- Produces: CLI that grid-searches selected `FUSION_DEFAULTS` weights and prints the best config as JSON (to be pasted into `FUSION_DEFAULTS` or passed as the job's `fusion` config).

- [ ] **Step 1: Write the tool**

Create `tune_fusion_weights.py`:

```python
"""Grid-search fusion emission weights against the labeled eval runs.

Replays detection per candidate config (rerun_detection machinery), rebuilds
eval cases in-memory, scores the type axis, and reports the best config.
Deterministic features, learned weights - the "ML assists" hook from the
3D contact detection spec.

Usage: python tune_fusion_weights.py --runs-dir ui_runs [--use-3d]
"""
import argparse
import itertools
import json
import tempfile
from pathlib import Path

import build_eval_set
import eval_line_calls
from rerun_detection import replay_run

GRID = {
    "surface_near_bonus": [1.0, 1.5, 2.0],
    "reflection_bonus": [0.5, 0.75, 1.0],
    "interior_racket_bonus": [1.0, 1.25, 1.75],
    "none_score": [0.4, 0.6, 0.8],
}


def score_config(runs_dir, config, use_3d):
    """Type-axis accuracy for one config; higher is better."""
    with tempfile.TemporaryDirectory() as scratch:
        out_root = Path(scratch) / "runs"
        for run_dir in sorted(Path(runs_dir).iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "ball_coordinates.csv").exists():
                continue
            try:
                replay_run(run_dir, out_root / run_dir.name, use_3d,
                           fusion_config=config)
            except Exception:
                continue
        cases, _ = build_eval_set.build_eval_set(out_root)
        report = eval_line_calls.evaluate_cases(cases)
        checked = report.get("type_checked") or 0
        if not checked:
            return None
        return report["type_correct"] / checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("ui_runs"))
    parser.add_argument("--use-3d", action="store_true")
    args = parser.parse_args()

    names = sorted(GRID)
    results = []
    for values in itertools.product(*(GRID[name] for name in names)):
        config = dict(zip(names, values))
        accuracy = score_config(args.runs_dir, config, args.use_3d)
        if accuracy is None:
            continue
        results.append((accuracy, config))
        print(f"{accuracy:.3f}  {config}")
    if not results:
        raise SystemExit("No config produced scorable cases.")
    best_accuracy, best = max(results, key=lambda item: item[0])
    print(f"\nBest ({best_accuracy:.3f}):\n{json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Thread `fusion_config` through `replay_run`**

In `rerun_detection.py`, add a `fusion_config=None` parameter to `replay_run` and pass it as `config=fusion_config` into `detect_events_fused`. Adapt the call signatures — `build_eval_set.build_eval_set` and `eval_line_calls.evaluate_cases` return shapes were confirmed in Tasks 1 and 11; match them here.

- [ ] **Step 3: Run it and record**

```bash
.venv/bin/python tune_fusion_weights.py --runs-dir ui_runs --use-3d
```

The 81-config grid re-runs detection per config; expect minutes, not hours (no video decoding). Append the winning config and its accuracy to `eval_set/RESULTS-3d-contact.md`. Only fold winning values into `FUSION_DEFAULTS` if the improvement holds on the type axis without regressing the others (re-run the Task 11 comparison with the new defaults).

- [ ] **Step 4: Full suite + commit**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

```bash
git add tune_fusion_weights.py rerun_detection.py eval_set/RESULTS-3d-contact.md
git commit -m "tune_fusion_weights.py: grid-search emission weights on eval runs"
```
