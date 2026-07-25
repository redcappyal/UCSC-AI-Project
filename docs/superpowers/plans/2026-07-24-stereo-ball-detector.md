# Stereo Ball Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the trained YOLOX-Tiny ball detector in the stereo path as a swappable, self-describing TorchScript artifact, without changing the single-camera pipeline or the stereo geometry.

**Architecture:** Three new root-level modules. `ball_model.py` owns the manifest and TorchScript loading; `ball_detector.py` owns the tiled native-resolution sweep, coordinate mapping and cross-tile NMS; `export_ball_model.py` is a training-side tool that produces the artifact. They meet the existing seam — a `frame -> list[dict]` callable — at exactly one call site, `stereo_offline._build_infer`.

**Tech Stack:** Python 3.12, numpy, OpenCV, PyTorch (TorchScript, lazily imported). YOLOX only in the training environment, never at runtime.

Spec: `docs/superpowers/specs/2026-07-24-stereo-ball-detector-design.md`

## Global Constraints

- **The default test suite must never need torch.** `requirements-test.txt` is deliberately minimal; its comment: "if a test starts needing the real model runtime, that test belongs behind a marker, not in the default CI job." Every heavy import is lazy, inside a function.
- **Do not modify** `stereo_engine.py`, `inference_engine.py`, `tracking_common.py`, `requirements.txt`, or `requirements-test.txt`.
- **Baseline is 271 passing tests in ~8 s.** After every task the suite must still be green, with only additions to the count.
- Run tests with `.venv/Scripts/python.exe -m pytest tests/ -q` (Windows; the venv is already created in this worktree from `requirements-test.txt`).
- Predictions are dicts with keys `x, y, width, height, confidence, class, class_name` — **centre-based, full-frame pixel coordinates**.
- Class name must be `"ball"` — one of `tracking_common.BALL_CLASS_NAMES`.
- A `PostToolUse` hook auto-runs `tests/test_<name>.py` when `<name>.py` is edited. A failure comes back as a *blocked edit*, not a warning.
- Manifest `conf_threshold` must stay **≤** the caller's `confidence` (0.4 in `fuse_clips`), or the caller's threshold becomes unreachable dead code.

---

### Task 1: `ball_model.py` — manifest loading and validation

Pure stdlib. No torch, no cv2. This is the "which model" half.

**Files:**
- Create: `ball_model.py`
- Test: `tests/test_ball_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelManifest` (frozen dataclass with fields `schema_version, name, version, input_size, decode, conf_threshold, nms_iou, class_names, tile_overlap_px, max_batch_tiles, artifact_sha256, source_checkpoint, trained_commit, val_ap50_95, notes`); `load_manifest(model_dir) -> ModelManifest`; `describe(model_dir=None) -> dict | None`; `model_dir_from_env() -> Path`; `SCHEMA_VERSION = "ball-model-v1"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ball_model.py
import json
import hashlib
import pytest

import ball_model


def _write_model(tmp_path, **overrides):
    artifact = tmp_path / "model.torchscript"
    artifact.write_bytes(b"not-a-real-torchscript")
    manifest = {
        "schema_version": "ball-model-v1",
        "name": "crosscourt-ball-416",
        "version": 1,
        "input_size": [416, 416],
        "decode": "in_graph",
        "conf_threshold": 0.25,
        "nms_iou": 0.65,
        "class_names": ["ball"],
        "tile_overlap_px": 64,
        "max_batch_tiles": 32,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "source_checkpoint": "best_ckpt.pth (epoch 100)",
        "trained_commit": "2968a89",
        "val_ap50_95": 0.4034,
        "notes": "val is diagnostic only",
    }
    manifest.update(overrides)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_load_manifest_reads_all_fields(tmp_path):
    manifest = ball_model.load_manifest(_write_model(tmp_path))
    assert manifest.name == "crosscourt-ball-416"
    assert manifest.input_size == (416, 416)
    assert manifest.conf_threshold == 0.25
    assert manifest.nms_iou == 0.65
    assert manifest.class_names == ("ball",)
    assert manifest.tile_overlap_px == 64


def test_load_manifest_rejects_unknown_schema_version(tmp_path):
    model_dir = _write_model(tmp_path, schema_version="ball-model-v99")
    with pytest.raises(ValueError, match="ball-model-v99"):
        ball_model.load_manifest(model_dir)


def test_load_manifest_rejects_sha256_mismatch(tmp_path):
    model_dir = _write_model(tmp_path, artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ball_model.load_manifest(model_dir)


def test_load_manifest_missing_dir_names_the_path_and_export_script(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError) as excinfo:
        ball_model.load_manifest(missing)
    message = str(excinfo.value)
    assert "nope" in message
    assert "export_ball_model.py" in message


def test_load_manifest_rejects_overlap_not_smaller_than_tile(tmp_path):
    model_dir = _write_model(tmp_path, tile_overlap_px=416)
    with pytest.raises(ValueError, match="tile_overlap_px"):
        ball_model.load_manifest(model_dir)


def test_describe_returns_none_when_unavailable(tmp_path):
    assert ball_model.describe(tmp_path / "nope") is None


def test_describe_summarises_without_loading_torch(tmp_path):
    summary = ball_model.describe(_write_model(tmp_path))
    assert summary["name"] == "crosscourt-ball-416"
    assert summary["version"] == 1
    assert summary["artifact_sha256"].startswith(
        __import__("hashlib").sha256(b"not-a-real-torchscript").hexdigest()[:8])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ball_model'`

- [ ] **Step 3: Implement `ball_model.py`**

```python
"""Manifest + artifact loading for the on-device ball detector's server twin.

Everything that varies between model iterations lives in manifest.json, so a
future model is a directory drop-in and one env var -- never a code change.

torch is imported lazily inside load_detector(): the default test suite runs
without it (see requirements-test.txt), and nothing here needs it to read a
manifest.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "ball-model-v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "crosscourt-ball-416-v1"
_EXPORT_HINT = (
    "Produce it in the training environment with export_ball_model.py "
    "(see ios/MODEL.md §2)."
)

_MANIFEST_CACHE = {}
_DETECTOR_CACHE = {}


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    name: str
    version: int
    input_size: tuple
    decode: str
    conf_threshold: float
    nms_iou: float
    class_names: tuple
    tile_overlap_px: int
    max_batch_tiles: int
    artifact_sha256: str
    source_checkpoint: str
    trained_commit: str
    val_ap50_95: float
    notes: str
    model_dir: Path

    @property
    def artifact_path(self):
        return self.model_dir / "model.torchscript"


def model_dir_from_env():
    configured = os.environ.get("BALL_MODEL_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_MODEL_DIR


def load_manifest(model_dir=None):
    model_dir = Path(model_dir) if model_dir is not None else model_dir_from_env()
    cached = _MANIFEST_CACHE.get(model_dir)
    if cached is not None:
        return cached

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No ball-model manifest at {manifest_path}. {_EXPORT_HINT}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest at {manifest_path}: {exc}") from exc

    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {schema!r} at {manifest_path}; "
            f"this build understands {SCHEMA_VERSION!r}. Fields are not guessed.")

    artifact = model_dir / "model.torchscript"
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Manifest at {manifest_path} but no model.torchscript beside it. "
            f"{_EXPORT_HINT}")

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != raw.get("artifact_sha256"):
        raise ValueError(
            f"artifact sha256 mismatch for {artifact}: manifest says "
            f"{raw.get('artifact_sha256')!r}, file is {digest!r}. A mismatched "
            f"artifact makes every result unattributable -- re-run "
            f"export_ball_model.py to regenerate both together.")

    input_size = tuple(int(v) for v in raw["input_size"])
    overlap = int(raw["tile_overlap_px"])
    if not 0 <= overlap < min(input_size):
        raise ValueError(
            f"tile_overlap_px must be in [0, {min(input_size)}), got {overlap}")

    manifest = ModelManifest(
        schema_version=schema,
        name=str(raw["name"]),
        version=int(raw["version"]),
        input_size=input_size,
        decode=str(raw["decode"]),
        conf_threshold=float(raw["conf_threshold"]),
        nms_iou=float(raw["nms_iou"]),
        class_names=tuple(str(c) for c in raw["class_names"]),
        tile_overlap_px=overlap,
        max_batch_tiles=int(raw["max_batch_tiles"]),
        artifact_sha256=digest,
        source_checkpoint=str(raw.get("source_checkpoint", "")),
        trained_commit=str(raw.get("trained_commit", "")),
        val_ap50_95=float(raw.get("val_ap50_95", 0.0)),
        notes=str(raw.get("notes", "")),
        model_dir=model_dir,
    )
    _MANIFEST_CACHE[model_dir] = manifest
    return manifest


def describe(model_dir=None):
    """Provenance summary for stamping into output, or None if unavailable.

    Best-effort on purpose: this is reporting, not inference. Inference still
    fails loudly (load_detector) when the model is missing.
    """
    try:
        manifest = load_manifest(model_dir)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    return {
        "name": manifest.name,
        "version": manifest.version,
        "artifact_sha256": manifest.artifact_sha256,
        "conf_threshold": manifest.conf_threshold,
        "nms_iou": manifest.nms_iou,
        "input_size": list(manifest.input_size),
        "source_checkpoint": manifest.source_checkpoint,
        "trained_commit": manifest.trained_commit,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_model.py -q`
Expected: 7 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `278 passed` (271 baseline + 7)

- [ ] **Step 6: Commit**

```bash
git add ball_model.py tests/test_ball_model.py
git commit -m "feat(ball-model): manifest loading with schema and sha256 validation"
```

---

### Task 2: `ball_detector.py` — tiling and cross-tile NMS

Pure functions, no torch. This is the part most likely to hide a real bug, so it is tested hardest.

**Files:**
- Create: `ball_detector.py`
- Test: `tests/test_ball_detector.py`

**Interfaces:**
- Consumes: nothing from Task 1 yet.
- Produces: `tile_windows(frame_w, frame_h, tile, overlap) -> list[tuple[int, int]]`; `iou(a, b) -> float`; `merge_detections(dets, iou_threshold) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ball_detector.py
import pytest

import ball_detector


def _box(x, y, w=10.0, h=10.0, confidence=0.9):
    return {"x": x, "y": y, "width": w, "height": h,
            "confidence": confidence, "class": "ball", "class_name": "ball"}


def test_tile_windows_4k_grid_is_11_by_6():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    xs = sorted({x for x, _ in windows})
    ys = sorted({y for _, y in windows})
    assert len(xs) == 11
    assert len(ys) == 6
    assert len(windows) == 66


def test_tile_windows_last_tile_is_clamped_in_bounds():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    assert max(x for x, _ in windows) + 416 == 3840
    assert max(y for _, y in windows) + 416 == 2160


def test_tile_windows_covers_every_pixel_column():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    covered = set()
    for x, _ in windows:
        covered.update(range(x, x + 416))
    assert covered == set(range(3840))


def test_tile_windows_exact_tile_size_frame_is_single_window():
    assert ball_detector.tile_windows(416, 416, 416, 64) == [(0, 0)]


def test_tile_windows_frame_smaller_than_tile_is_single_origin():
    assert ball_detector.tile_windows(300, 200, 416, 64) == [(0, 0)]


def test_tile_windows_rejects_overlap_at_or_above_tile():
    with pytest.raises(ValueError, match="overlap"):
        ball_detector.tile_windows(3840, 2160, 416, 416)


def test_tile_windows_stride_exceeds_max_ball_width():
    # Overlap must exceed the p90 ball width (24 px) so a ball is wholly
    # contained in at least one tile rather than clipped across a seam.
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    xs = sorted({x for x, _ in windows})
    assert 416 - (xs[1] - xs[0]) == 64


def test_iou_identical_boxes_is_one():
    assert ball_detector.iou(_box(100, 100), _box(100, 100)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert ball_detector.iou(_box(100, 100), _box(500, 500)) == 0.0


def test_merge_detections_drops_duplicate_across_tile_seam():
    kept = ball_detector.merge_detections(
        [_box(100, 100, confidence=0.7), _box(101, 100, confidence=0.9)], 0.65)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_merge_detections_keeps_two_distinct_balls():
    kept = ball_detector.merge_detections([_box(100, 100), _box(900, 900)], 0.65)
    assert len(kept) == 2


def test_merge_detections_empty_input():
    assert ball_detector.merge_detections([], 0.65) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ball_detector'`

- [ ] **Step 3: Implement the pure functions**

```python
"""Tiled native-resolution ball detection.

ios/MODEL.md §6: never downscale the whole frame. The model was trained on
416 px windows cut at native source resolution, where the ball is 7-24 px.
Resizing a 4K frame to 960 px wide makes a p50 10.7 px ball ~2.7 px -- outside
anything the model has seen. So we slide the trained window over the frame at
native scale instead.

The window strategy is deliberately a seam: §6's velocity-extrapolated single
crop can replace tile_windows later without touching the model adapter.
"""


def tile_windows(frame_w, frame_h, tile, overlap):
    """Top-left origins of tiles covering the frame, clamped in-bounds.

    Overlap must exceed the largest expected ball (p90 is 24 px) so no ball is
    clipped across a seam in every tile that sees it.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, {tile}), got {overlap}")

    stride = tile - overlap

    def origins(extent):
        if extent <= tile:
            return [0]
        starts = list(range(0, extent - tile + 1, stride))
        if starts[-1] != extent - tile:
            starts.append(extent - tile)
        return starts

    return [(x, y) for y in origins(frame_h) for x in origins(frame_w)]


def _corners(box):
    half_w = box["width"] / 2.0
    half_h = box["height"] / 2.0
    return (box["x"] - half_w, box["y"] - half_h,
            box["x"] + half_w, box["y"] + half_h)


def iou(a, b):
    ax0, ay0, ax1, ay1 = _corners(a)
    bx0, by0, bx1, by1 = _corners(b)
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    union = (a["width"] * a["height"]) + (b["width"] * b["height"]) - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections, iou_threshold):
    """Greedy NMS across tiles. Overlapping tiles see the same ball twice."""
    kept = []
    for detection in sorted(detections,
                            key=lambda d: d.get("confidence", 0.0),
                            reverse=True):
        if all(iou(detection, k) <= iou_threshold for k in kept):
            kept.append(detection)
    return kept
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_detector.py -q`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add ball_detector.py tests/test_ball_detector.py
git commit -m "feat(ball-detector): native-resolution tiling and cross-tile NMS"
```

---

### Task 3: `detect_frame` — coordinate mapping from tile space to full frame

The off-by-tile bug is the likeliest real defect. This task exists to make it impossible, and it needs no model: the detector is injected.

**Files:**
- Modify: `ball_detector.py`
- Test: `tests/test_ball_detector.py`

**Interfaces:**
- Consumes: `tile_windows`, `merge_detections` from Task 2; `ModelManifest` from Task 1.
- Produces: `detect_frame(runner, frame, manifest) -> list[dict]`, where `runner` is any callable `run_batch(crops) -> list[list[tuple]]`, each inner tuple `(cx, cy, w, h, score, class_index)` in **tile-local** pixels.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ball_detector.py
import numpy as np

import ball_model


def _manifest(tmp_path, **overrides):
    fields = {
        "schema_version": "ball-model-v1", "name": "t", "version": 1,
        "input_size": (416, 416), "decode": "in_graph", "conf_threshold": 0.25,
        "nms_iou": 0.65, "class_names": ("ball",), "tile_overlap_px": 64,
        "max_batch_tiles": 32, "artifact_sha256": "x", "source_checkpoint": "",
        "trained_commit": "", "val_ap50_95": 0.0, "notes": "",
        "model_dir": tmp_path,
    }
    fields.update(overrides)
    return ball_model.ModelManifest(**fields)


class _FakeRunner:
    """Emits one box at a fixed FULL-FRAME point, expressed tile-locally."""

    def __init__(self, windows, target_xy, score=0.9):
        self.windows = windows
        self.target_xy = target_xy
        self.score = score
        self.batch_sizes = []

    def run_batch(self, crops):
        self.batch_sizes.append(len(crops))
        out = []
        for crop in crops:
            index = len(out) + sum(self.batch_sizes[:-1])
            x0, y0 = self.windows[index]
            tx, ty = self.target_xy
            local_x, local_y = tx - x0, ty - y0
            h, w = crop.shape[:2]
            if 0 <= local_x < w and 0 <= local_y < h:
                out.append([(float(local_x), float(local_y), 10.0, 10.0,
                             self.score, 0)])
            else:
                out.append([])
        return out


def test_detect_frame_maps_tile_local_box_to_full_frame():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    manifest = _manifest(None)
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    runner = _FakeRunner(windows, target_xy=(1500, 900))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(1500.0)
    assert detections[0]["y"] == pytest.approx(900.0)


def test_detect_frame_labels_class_ball_so_tracking_common_accepts_it():
    from tracking_common import is_ball_prediction

    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    manifest = _manifest(None)
    runner = _FakeRunner([(0, 0)], target_xy=(200, 200))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert detections[0]["class"] == "ball"
    assert detections[0]["class_name"] == "ball"
    # candidate_ball_predictions silently falls back to ALL predictions when
    # nothing matches BALL_CLASS_NAMES, so a mislabel would not fail loudly.
    assert is_ball_prediction(detections[0])


def test_detect_frame_drops_boxes_below_manifest_conf_threshold():
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    manifest = _manifest(None, conf_threshold=0.5)
    runner = _FakeRunner([(0, 0)], target_xy=(200, 200), score=0.4)

    assert ball_detector.detect_frame(runner, frame, manifest) == []


def test_detect_frame_respects_max_batch_tiles():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    manifest = _manifest(None, max_batch_tiles=16)
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    runner = _FakeRunner(windows, target_xy=(1500, 900))

    ball_detector.detect_frame(runner, frame, manifest)

    assert max(runner.batch_sizes) <= 16
    assert sum(runner.batch_sizes) == 66


def test_detect_frame_pads_frame_smaller_than_tile():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    manifest = _manifest(None)
    runner = _FakeRunner([(0, 0)], target_xy=(100, 100))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(100.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_detector.py -q`
Expected: FAIL — `AttributeError: module 'ball_detector' has no attribute 'detect_frame'`

- [ ] **Step 3: Implement `detect_frame`**

Append to `ball_detector.py`:

```python
def _crop(frame, x0, y0, tile):
    """Tile-sized crop, zero-padded when the frame is smaller than one tile.

    Padding keeps the network input shape static, which the traced TorchScript
    graph requires.
    """
    import numpy as np

    patch = frame[y0:y0 + tile, x0:x0 + tile]
    height, width = patch.shape[:2]
    if height == tile and width == tile:
        return patch
    padded = np.zeros((tile, tile, patch.shape[2]), dtype=patch.dtype)
    padded[:height, :width] = patch
    return padded


def detect_frame(runner, frame, manifest):
    """Full-frame detections from a tiled sweep.

    `runner.run_batch(crops)` returns, per crop, a list of
    (cx, cy, w, h, score, class_index) in TILE-LOCAL pixels. This function owns
    the tile-local -> full-frame mapping and the cross-tile merge.
    """
    tile = manifest.input_size[0]
    frame_h, frame_w = frame.shape[:2]
    windows = tile_windows(frame_w, frame_h, tile, manifest.tile_overlap_px)

    detections = []
    batch = max(1, manifest.max_batch_tiles)
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        crops = [_crop(frame, x0, y0, tile) for x0, y0 in chunk]
        for (x0, y0), boxes in zip(chunk, runner.run_batch(crops)):
            for cx, cy, width, height, score, class_index in boxes:
                if score < manifest.conf_threshold:
                    continue
                name = (manifest.class_names[int(class_index)]
                        if int(class_index) < len(manifest.class_names)
                        else str(class_index))
                detections.append({
                    "x": float(cx) + x0,
                    "y": float(cy) + y0,
                    "width": float(width),
                    "height": float(height),
                    "confidence": float(score),
                    "class": name,
                    "class_name": name,
                })

    return merge_detections(detections, manifest.nms_iou)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_detector.py -q`
Expected: 17 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `295 passed`

- [ ] **Step 6: Commit**

```bash
git add ball_detector.py tests/test_ball_detector.py
git commit -m "feat(ball-detector): tile-local to full-frame coordinate mapping"
```

---

### Task 4: TorchScript runner in `ball_model.load_detector`

The only torch-touching runtime code. Import is lazy; the test is marked and skipped by default.

**Files:**
- Modify: `ball_model.py`
- Test: `tests/test_ball_model.py`

**Interfaces:**
- Consumes: `load_manifest` from Task 1.
- Produces: `TorchScriptRunner` with `.manifest` and `.run_batch(crops) -> list[list[tuple]]`; `load_detector(model_dir=None) -> TorchScriptRunner` (cached per directory).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ball_model.py
def test_load_detector_missing_model_raises_and_does_not_fall_back(tmp_path):
    # Silent fallback to RF-DETR would make the stereo/single-camera detector
    # split invisible; the spec requires this to be loud.
    with pytest.raises(FileNotFoundError, match="export_ball_model.py"):
        ball_model.load_detector(tmp_path / "nope")


def test_ball_model_imports_without_torch():
    import sys
    # The default CI job has no torch installed; importing ball_model must not
    # need it. If torch is absent this is trivially true, so assert the module
    # did not pull it in either way.
    before = "torch" in sys.modules
    import importlib
    importlib.reload(ball_model)
    assert ("torch" in sys.modules) == before


@pytest.mark.requires_model
def test_torchscript_runner_returns_boxes_for_real_model():
    runner = ball_model.load_detector()
    import numpy as np
    crops = [np.zeros((416, 416, 3), dtype=np.uint8)]
    result = runner.run_batch(crops)
    assert len(result) == 1
    assert isinstance(result[0], list)
```

- [ ] **Step 2: Register the marker so the suite does not warn**

Create `pytest.ini` **only if it does not already exist**; otherwise add the
`markers` line to the existing config.

```ini
[pytest]
markers =
    requires_model: needs the exported TorchScript ball model and torch (deselect with '-m "not requires_model"')
addopts = -m "not requires_model"
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_model.py -q`
Expected: FAIL — `AttributeError: module 'ball_model' has no attribute 'load_detector'`

- [ ] **Step 4: Implement the runner**

Append to `ball_model.py`:

```python
class TorchScriptRunner:
    """Runs the traced ball model over a batch of tile crops.

    Traced with decode_in_inference=True, so the graph emits decoded
    [batch, n_anchors, 5 + n_classes] and this class only thresholds by
    objectness and reshapes. The Core ML artifact (ios/MODEL.md §4) is traced
    the other way -- raw head outputs, decode in Swift for ANE residency --
    which is what manifest["decode"] distinguishes.
    """

    def __init__(self, module, manifest, torch_module):
        self._module = module
        self._torch = torch_module
        self.manifest = manifest

    def run_batch(self, crops):
        import numpy as np

        torch = self._torch
        # HWC uint8 BGR -> NCHW float32. YOLOX trains on raw 0-255 BGR as
        # cv2 delivers it (ValTransform applies no mean/std and no channel
        # swap), so pass the channels through unchanged and do not rescale.
        stacked = np.stack(crops).astype("float32")
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            raw = self._module(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        output = raw.detach().cpu().numpy()

        # Vectorised threshold before touching Python. A 416 crop yields ~3549
        # anchors; at 66 tiles that is ~234k rows per frame, and a per-row
        # Python loop over all of them would dominate the runtime.
        results = []
        for row in output:
            objectness = row[:, 4]
            class_scores = row[:, 5:]
            if class_scores.shape[1]:
                class_index = class_scores.argmax(axis=1)
                score = objectness * class_scores[
                    np.arange(class_scores.shape[0]), class_index]
            else:
                class_index = np.zeros(row.shape[0], dtype=int)
                score = objectness
            keep = np.nonzero(score >= self.manifest.conf_threshold)[0]
            results.append([
                (float(row[i, 0]), float(row[i, 1]),
                 float(row[i, 2]), float(row[i, 3]),
                 float(score[i]), int(class_index[i]))
                for i in keep
            ])
        return results


def load_detector(model_dir=None):
    """Load the traced model. Raises loudly; never falls back to another detector."""
    model_dir = Path(model_dir) if model_dir is not None else model_dir_from_env()
    cached = _DETECTOR_CACHE.get(model_dir)
    if cached is not None:
        return cached

    manifest = load_manifest(model_dir)

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The stereo ball detector needs torch. The default test env is "
            "deliberately torch-free; install the full requirements.txt."
        ) from exc

    module = torch.jit.load(str(manifest.artifact_path), map_location="cpu")
    module.eval()
    runner = TorchScriptRunner(module, manifest, torch)
    _DETECTOR_CACHE[model_dir] = runner
    return runner
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ball_model.py -q`
Expected: 9 passed, 1 deselected

- [ ] **Step 6: Commit**

```bash
git add ball_model.py tests/test_ball_model.py pytest.ini
git commit -m "feat(ball-model): TorchScript runner with lazy torch and a loud missing-model error"
```

---

### Task 5: `export_ball_model.py` — training-side artifact producer

Runs only in `ball-detector-train/.venv`. This is the boundary that keeps YOLOX out of the serving path. No unit test: it imports YOLOX, which the CI env does not have. It is verified by running it in Task 7.

**Files:**
- Create: `export_ball_model.py`

**Interfaces:**
- Consumes: `ball_model.SCHEMA_VERSION`.
- Produces: a `models/<name>-v<n>/` directory containing `model.torchscript` and `manifest.json`.

- [ ] **Step 1: Write the script**

```python
"""Trace the trained YOLOX ball checkpoint to TorchScript + manifest.

Run this in the TRAINING environment (ball-detector-train/.venv), never in the
app env. That is the whole point: yolox_ball_exp.py's docstring requires YOLOX
stay a training dependency, so the serving path gets a self-contained traced
artifact and never imports YOLOX.

    python export_ball_model.py \
        --exp yolox_ball_exp.py \
        --ckpt .../YOLOX_outputs/crosscourt-ball-416/best_ckpt.pth \
        --out models/crosscourt-ball-416-v1 \
        --version 1
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA_VERSION = "ball-model-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="path to yolox_ball_exp.py")
    parser.add_argument("--ckpt", required=True, help="path to best_ckpt.pth")
    parser.add_argument("--out", required=True, help="output model directory")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--name", default="crosscourt-ball-416")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--tile-overlap-px", type=int, default=64)
    parser.add_argument("--max-batch-tiles", type=int, default=32)
    parser.add_argument("--val-ap50-95", type=float, default=0.0)
    parser.add_argument("--notes", default="val is diagnostic only -- shares a "
                                           "rig/session with train")
    return parser.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    import torch
    from yolox.exp import get_exp

    exp = get_exp(args.exp, None)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Decode inside the graph: the server then only thresholds and NMSes, so
    # the grid math lives in one place. The Core ML export (ios/MODEL.md §4)
    # needs the opposite -- raw outputs, decode in Swift for ANE residency.
    model.head.decode_in_inference = True

    size = exp.test_size[0]
    example = torch.randn(1, 3, size, size)
    traced = torch.jit.trace(model, example)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "model.torchscript"
    traced.save(str(artifact))

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": args.name,
        "version": args.version,
        "input_size": [size, size],
        "decode": "in_graph",
        "conf_threshold": args.conf_threshold,
        "nms_iou": float(exp.nmsthre),
        "class_names": ["ball"],
        "tile_overlap_px": args.tile_overlap_px,
        "max_batch_tiles": args.max_batch_tiles,
        "artifact_sha256": digest,
        "source_checkpoint": str(Path(args.ckpt).name),
        "trained_commit": _git_commit(),
        "val_ap50_95": args.val_ap50_95,
        "notes": args.notes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {artifact} ({artifact.stat().st_size} bytes)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it at least parses and shows help**

Run: `.venv/Scripts/python.exe export_ball_model.py --help`
Expected: argparse usage text, exit 0 (the YOLOX/torch imports are inside `main`, so `--help` works without them)

- [ ] **Step 3: Ignore model artifacts**

Append to `.gitignore`:

```
# Exported detector artifacts: large binaries, reproducible from the training
# checkpoint via export_ball_model.py. The manifest records provenance.
models/
```

- [ ] **Step 4: Commit**

```bash
git add export_ball_model.py .gitignore
git commit -m "feat(export): trace the trained YOLOX checkpoint to TorchScript + manifest"
```

---

### Task 6: Wire into `stereo_offline` with provenance

**Files:**
- Modify: `stereo_offline.py:57-71` (`_build_infer`), `stereo_offline.py:133-182` (`fuse_clips`)
- Test: `tests/test_stereo_offline.py`

**Interfaces:**
- Consumes: `ball_model.load_detector`, `ball_model.describe`, `ball_detector.detect_frame`.
- Produces: `fuse_clips(...)` result gains a `"detector"` key.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_stereo_offline.py
def test_fuse_clips_reports_injected_detector(monkeypatch):
    # Injected infer means no model is loaded; provenance must say so rather
    # than reading a manifest that is not there. This is what keeps the
    # existing synthetic tests working with no model on disk.
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline, "_iter_frames",
                        lambda video_path: iter(_fake_frames(5)))

    result = stereo_offline.fuse_clips(
        "a.mp4", calibration_a, "b.mp4", calibration_b,
        infer_a=lambda frame: [], infer_b=lambda frame: [])

    assert result["detector"] == {"backend": "injected"}


def test_build_infer_defaults_to_yolox_for_stereo(monkeypatch):
    calls = {}

    class _Runner:
        manifest = "MANIFEST"

    monkeypatch.setattr(stereo_offline, "_load_ball_detector",
                        lambda: _Runner())
    monkeypatch.setattr(
        stereo_offline, "_detect_frame",
        lambda runner, frame, manifest: calls.setdefault(
            "args", (runner, frame, manifest)) or [])

    infer = stereo_offline._build_infer(None, 0.4)
    infer("FRAME")

    assert calls["args"][1] == "FRAME"
    assert calls["args"][2] == "MANIFEST"


def test_build_infer_honours_rfdetr_override(monkeypatch):
    monkeypatch.setenv("STEREO_DETECTOR", "rfdetr")
    seen = {}

    def _fake_import():
        def _get_model():
            return "RFDETR_MODEL"

        def _infer(model, frame, confidence):
            seen["model"] = model
            return []

        return _get_model, _infer

    monkeypatch.setattr(stereo_offline, "_import_rfdetr", _fake_import)
    infer = stereo_offline._build_infer(None, 0.4)
    infer("FRAME")

    assert seen["model"] == "RFDETR_MODEL"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stereo_offline.py -q`
Expected: FAIL — `KeyError: 'detector'` and `AttributeError: ... '_load_ball_detector'`

- [ ] **Step 3: Replace `_build_infer` (lines 57-71)**

```python
STEREO_DETECTOR_DEFAULT = "yolox"


def _import_rfdetr():
    """Seam for tests; keeps the heavy import lazy."""
    from inference_engine import get_tracking_model, infer_frame_predictions
    return get_tracking_model, infer_frame_predictions


def _load_ball_detector():
    """Seam for tests; keeps torch out of import time."""
    import ball_model
    return ball_model.load_detector()


def _detect_frame(runner, frame, manifest):
    """Seam for tests."""
    import ball_detector
    return ball_detector.detect_frame(runner, frame, manifest)


def selected_detector():
    import os
    return os.environ.get("STEREO_DETECTOR", STEREO_DETECTOR_DEFAULT).strip().lower()


def _build_infer(model, confidence):
    """Build the stereo path's per-frame callable.

    Defaults to the locally trained YOLOX detector (STEREO_DETECTOR=yolox);
    STEREO_DETECTOR=rfdetr restores the hosted RF-DETR the single-camera
    pipeline still uses. Imported lazily so a clip with zero decoded frames
    never loads a model -- the CLI smoke test depends on that.

    Never falls back between detectors: a missing model raises, because a
    silent swap would make the stereo/single-camera split invisible.
    """
    backend = selected_detector()

    if backend == "rfdetr":
        get_tracking_model, infer_frame_predictions = _import_rfdetr()
        if model is None:
            model = get_tracking_model()

        def _infer(frame):
            return infer_frame_predictions(model, frame, confidence)

        return _infer

    if backend != "yolox":
        raise ValueError(
            f"Unknown STEREO_DETECTOR {backend!r}; expected 'yolox' or 'rfdetr'")

    runner = model if model is not None else _load_ball_detector()

    def _infer(frame):
        return _detect_frame(runner, frame, runner.manifest)

    return _infer
```

- [ ] **Step 4: Add provenance to `fuse_clips`**

Replace the `return` block at the end of `fuse_clips`:

```python
    if infer_a is not None and infer_b is not None:
        detector_info = {"backend": "injected"}
    else:
        backend = selected_detector()
        if backend == "rfdetr":
            detector_info = {"backend": "rfdetr"}
        else:
            import ball_model
            detector_info = {"backend": "yolox", "model": ball_model.describe()}

    return {
        "impacts": [_impact_to_dict(impact) for impact in impacts],
        "pair_agreement": agreement,
        "sample_counts": {"a": len(samples_a), "b": len(samples_b)},
        "detector": detector_info,
    }
```

Also extend the docstring's documented return shape to include
`"detector": {"backend": ..., "model": {...}}`.

- [ ] **Step 5: Run the stereo tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stereo_offline.py -q`
Expected: all pass, including the 3 new ones

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `298 passed` (295 + 3), 1 deselected

- [ ] **Step 7: Commit**

```bash
git add stereo_offline.py tests/test_stereo_offline.py
git commit -m "feat(stereo): default the stereo path to the local YOLOX detector"
```

---

### Task 7: Export the real model, then measure

Turns the plan's one unmeasured assumption — tile-sweep cost — into a number.

**Files:**
- Create: `models/crosscourt-ball-416-v1/` (gitignored)
- Modify: `docs/superpowers/specs/2026-07-24-stereo-ball-detector-design.md` (record measurements)

- [ ] **Step 1: Export the artifact from the training env**

```bash
cd C:/Users/alann/Code/ball-detector-train
.venv/Scripts/python.exe C:/Users/alann/Code/UCSC-AI-Project/.claude/worktrees/stereo-ball-detector/export_ball_model.py --exp C:/Users/alann/Code/UCSC-AI-Project/.claude/worktrees/stereo-ball-detector/yolox_ball_exp.py --ckpt YOLOX_outputs/crosscourt-ball-416/best_ckpt.pth --out C:/Users/alann/Code/UCSC-AI-Project/.claude/worktrees/stereo-ball-detector/models/crosscourt-ball-416-v1 --version 1 --val-ap50-95 0.4034
```

Expected: prints the artifact size and sha256.

- [ ] **Step 2: Verify the manifest round-trips**

```bash
.venv/Scripts/python.exe -c "import ball_model; m = ball_model.load_manifest('models/crosscourt-ball-416-v1'); print(m.name, m.input_size, m.artifact_sha256[:12])"
```

Expected: `crosscourt-ball-416 (416, 416) <12 hex chars>` — proving the sha256 written by the exporter matches what the loader computes.

- [ ] **Step 3: Run the marked real-model test**

Run this against an environment that has torch (the training venv, with the worktree on `PYTHONPATH`):

```bash
.venv/Scripts/python.exe -m pytest tests/test_ball_model.py -q -m requires_model
```

Expected: 1 passed

- [ ] **Step 4: Benchmark one 4K frame, CPU and GPU**

```python
# scratch benchmark, not committed
import time, numpy as np, ball_model, ball_detector
runner = ball_model.load_detector("models/crosscourt-ball-416-v1")
frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
ball_detector.detect_frame(runner, frame, runner.manifest)     # warm up
start = time.perf_counter()
for _ in range(5):
    ball_detector.detect_frame(runner, frame, runner.manifest)
print("per-frame seconds:", (time.perf_counter() - start) / 5)
```

- [ ] **Step 5: Record the measurements in the spec**

Add a "Measured" subsection to the spec's Risks table row for tile cost, with
the real per-frame seconds for CPU and GPU. If CPU exceeds ~2 s/frame, note
that a deployment should raise `tile_overlap_px` stride or move to §6's
tracker-guided crop, and say so explicitly rather than leaving it implied.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-24-stereo-ball-detector-design.md
git commit -m "docs(spec): record measured tile-sweep cost for the stereo detector"
```

---

## Self-Review

**Spec coverage:** `ball_model.py` → Tasks 1, 4. `ball_detector.py` → Tasks 2, 3. `export_ball_model.py` → Task 5. Manifest schema → Task 1. Error-handling table → Tasks 1 and 4 (every row has a test or an explicit raise). Data flow and provenance → Task 6. Testing section → Tasks 1–3, 6; marked real-model test → Task 4; golden invariance → covered by running the full suite, which already contains `tests/test_stereo_goldens.py`. Risks/measurement → Task 7.

**Placeholders:** none. Every code step carries complete code.

**Type consistency:** `run_batch(crops) -> list[list[tuple]]` with `(cx, cy, w, h, score, class_index)` is produced by `TorchScriptRunner` (Task 4) and consumed by `detect_frame` (Task 3) and `_FakeRunner` (Task 3) identically. `ModelManifest` field names match across Tasks 1, 3, 4 and the exporter's JSON in Task 5. `detect_frame(runner, frame, manifest)` has the same three-argument signature at its definition (Task 3) and both call sites (Tasks 3, 6).

**Known gap, deliberate:** Task 4's `run_batch` assumes the traced graph emits `[batch, n_anchors, 5 + n_classes]`. That shape is only confirmed when Task 7 runs the real export. If it differs, Task 7 Step 3 fails and `run_batch` is corrected there — which is why the real-model test exists and is run before anything depends on the output.
