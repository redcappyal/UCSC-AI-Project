# WASB Temporal Ball Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-frame ball detector with a WASB-style 3-frame temporal model (transfer-learned from pretrained sports-ball weights) served through the existing manifest/TorchScript seam, targeting a large recall gain over the ~35% rally-scale baseline.

**Architecture:** The serving seam stays exactly where it is — `ball_model.py` loads a TorchScript artifact described by a manifest, `ball_detector.py` runs the tiled native-resolution sweep, `ball_track_offline.py` drives clips. We extend the manifest schema to v2 (`frames_per_input: 3`, heatmap decode), teach the tiler to cut co-located tiles from a 3-frame stack, and add centered-window iteration to the offline runner. **Windows are CENTERED, not causal:** analysis is offline, so frame t is detected from (t−1, t, t+1) — both past and future context — which matters most at wall/floor contacts, where a trailing window has never seen the post-bounce direction. WASB's MIMO output (one heatmap per input frame) makes this a decode-index choice, not an architecture change. Training happens in the separate CUDA training environment (like YOLOX today): fine-tune WASB (MIT license, HRNet, 9-channel input, MIMO heatmaps, pretrained on 5 sports) on sequence crops emitted by an extended `prepare_ball_dataset.py`. Cloud-only: no Core ML, no ANE work.

**Tech Stack:** Python, numpy, cv2, torch (serving loads TorchScript only), WASB-SBDT (training env only — never a runtime dependency, same rule as YOLOX), pytest.

## Global Constraints

- Venv: everything on the Mac runs through `.venv` (`.venv/bin/python -m pytest tests/ -q`; green = "283 passed, 1 deselected" before this plan; each task adds to that count). System `python3` has no flask/cv2.
- The default test env is torch-free and must stay that way: tests never import torch; heavy imports stay lazy inside functions (existing convention in `ball_model.py` / `ball_track_offline.py`).
- Editing a `*.py` with a paired `tests/test_*.py` auto-runs that file via PostToolUse hook; failures block the edit.
- Training box: `C:\Users\alann\Code\ball-detector-train\.venv` (cv2 + torch, **no pytest**). Training-side tasks are verified by smoke runs, not unit tests.
- **Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite`** — use `_imread_unicode`/`_imwrite_unicode` in `prepare_ball_dataset.py`. Crop filenames stay ASCII via `ascii_slug()`. `cv2.VideoCapture` is exempt.
- **Never downscale the whole frame** (ios/MODEL.md §6): the model sees 416 px windows at native resolution only.
- WASB-SBDT is MIT (verified in research sweep). Do NOT pull BlurBall or TrackNetV4 weights/data without checking their licenses first (unresolved).
- One detection path per backend — no new engine forks. `BALL_DETECTOR` selects `local` (canonical; `yolox` kept as a compatible alias) or `rfdetr`; the *manifest* decides how the local model runs.
- Do not extend or import from `archive/` (stereo, fusion-engine).
- Any claim of improvement goes through the `/eval` skill against the newest `eval_set/BASELINE-*.md` (judge metrics must not regress) plus the detection-rate comparison in Task 8.

## Out of Scope (deliberate)

- **RF-DETR modifications** — this plan builds its replacement; RF-DETR serves only as the eval baseline.
- **Pipeline (`job_runner.py`) integration** — swapping the Flask pipeline's detector is its own plan, gated on Task 8's numbers.
- **BlurBall streak/(θ, length) head** — the v2 dataset already carries streak metrics per annotation, so nothing is lost by deferring; do not block the recall win on a new head with unresolved license questions.
- **Velocity-extrapolated single-crop inference** (MODEL.md §6 fine pass) — the tiled sweep is fine on a server GPU; the seam for this already exists in `ball_detector.py`.
- **ANE/Core ML anything** — dropped 2026-07-27; analysis is cloud-only.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `ball_model.py` | modify | Manifest v2 parsing (`frames_per_input`, `decode`, `heatmap_stride`, `nominal_ball_px`), `decode_heatmap()` pure function, `HeatmapRunner` |
| `ball_detector.py` | modify | `detect_frame_stack()` — co-located multi-frame tiling; existing `detect_frame` untouched |
| `ball_track_offline.py` | modify | `local` backend alias; centered-window iteration (one-frame lookahead) for temporal manifests |
| `prepare_ball_dataset.py` | modify | `--clips-dir` / `--seq-frames` sequence-crop emission, frame-alignment verification, `ball-crops-v2` schema |
| `export_wasb_model.py` | create | Training-env boundary: WASB checkpoint → TorchScript + v2 manifest |
| `docs/WASB-TRAIN.md` | create | Training runbook for the CUDA box (dataset adapter, fine-tune, export commands) |
| `eval_set/RESULTS-wasb-v1.md` | create (Task 8) | Recorded eval outcome vs gates |
| `tests/test_ball_model.py` | modify | v2 manifest + decode tests |
| `tests/test_ball_detector.py` | modify | stack-tiling tests |
| `tests/test_ball_track_offline.py` | modify | ring-buffer + alias tests |
| `tests/test_prepare_ball_dataset.py` | modify | sequence-crop tests |

Interface contracts used throughout (defined in Task 1–3, consumed everywhere after):

- Manifest v2 adds: `frames_per_input: int` (v1 manifests report 1; must be **odd** for `heatmap_peak` — the target frame sits in the middle), `decode: "heatmap_peak"`, `heatmap_stride: int`, `nominal_ball_px: float`. `conf_threshold` doubles as the heatmap peak threshold — one knob, not two.
- **Centered-window convention (used everywhere):** a stack is oldest-first `(t−1, t, t+1)`; detections/labels/decodes all refer to the **middle** frame, channel index `frames_per_input // 2`. Clip edges pad by repeating the first/last frame — identically in the dataset builder (Task 5) and the runner (Task 4), so training and serving see the same edge distribution.
- `decode_heatmap(heatmap, threshold, stride, nominal_px) -> list[(cx, cy, score)]` — pure numpy, sub-pixel, full-input-resolution coordinates.
- `HeatmapRunner.run_batch(stacks) -> list[list[(cx, cy, w, h, score, class_index)]]` — same tuple shape `TorchScriptRunner` emits, so `ball_detector`'s mapping/merge code is reused verbatim. `stacks` are HWC uint8 with C = 3 × frames_per_input, oldest frame first; the runner decodes the **middle** MIMO channel.
- `detect_frame_stack(runner, frames, manifest) -> list[detection dict]` — `frames` oldest-first, length == `manifest.frames_per_input`; detections are for the **middle** frame; same dict shape as `detect_frame`.

---

### Task 1: Manifest schema v2 in `ball_model.py`

**Files:**
- Modify: `ball_model.py`
- Test: `tests/test_ball_model.py`

**Interfaces:**
- Consumes: existing `load_manifest`, `ModelManifest`, `SCHEMA_VERSION` (`"ball-model-v1"`).
- Produces: `SCHEMA_VERSION_V2 = "ball-model-v2"`; `ModelManifest` gains `frames_per_input`, `decode`, `heatmap_stride`, `nominal_ball_px` (v1 loads report `frames_per_input=1`, `decode="in_graph"`, `heatmap_stride=0`, `nominal_ball_px=0.0`).

- [ ] **Step 1: Write the failing tests**

Follow the existing fixture pattern in `tests/test_ball_model.py` (write a `manifest.json` plus a dummy `model.torchscript` whose sha256 goes into the manifest). Add:

```python
def _v2_manifest_dict(artifact_sha):
    return {
        "schema_version": "ball-model-v2",
        "name": "crosscourt-wasb-416",
        "version": 1,
        "input_size": [416, 416],
        "frames_per_input": 3,
        "decode": "heatmap_peak",
        "heatmap_stride": 2,
        "nominal_ball_px": 12.0,
        "conf_threshold": 0.1,
        "nms_iou": 0.45,
        "class_names": ["ball"],
        "tile_overlap_px": 64,
        "max_batch_tiles": 32,
        "artifact_sha256": artifact_sha,
        "source_checkpoint": "wasb_best.pth",
        "trained_commit": "abc1234",
        "val_ap50_95": 0.0,
        "notes": "",
    }


def test_v2_manifest_loads_temporal_fields(tmp_path):
    model_dir = _write_model_dir(tmp_path, _v2_manifest_dict)  # reuse/extend the existing helper
    manifest = ball_model.load_manifest(model_dir)
    assert manifest.frames_per_input == 3
    assert manifest.decode == "heatmap_peak"
    assert manifest.heatmap_stride == 2
    assert manifest.nominal_ball_px == 12.0


def test_v1_manifest_reports_single_frame_defaults(tmp_path):
    model_dir = _write_model_dir(tmp_path)  # existing v1 helper
    manifest = ball_model.load_manifest(model_dir)
    assert manifest.frames_per_input == 1


def test_v2_manifest_rejects_missing_frames_per_input(tmp_path):
    def broken(sha):
        d = _v2_manifest_dict(sha)
        del d["frames_per_input"]
        return d
    model_dir = _write_model_dir(tmp_path, broken)
    with pytest.raises(KeyError):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_nonpositive_heatmap_stride(tmp_path):
    def broken(sha):
        d = _v2_manifest_dict(sha)
        d["heatmap_stride"] = 0
        return d
    model_dir = _write_model_dir(tmp_path, broken)
    with pytest.raises(ValueError, match="heatmap_stride"):
        ball_model.load_manifest(model_dir)
```

If the existing file's fixture helper doesn't take a dict-builder argument, add a `_write_model_dir(tmp_path, manifest_builder=None)` variant rather than duplicating fixture code.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: new tests FAIL (`Unsupported manifest schema_version 'ball-model-v2'`).

- [ ] **Step 3: Implement v2 parsing**

In `ball_model.py`: add `SCHEMA_VERSION_V2 = "ball-model-v2"`, extend the dataclass, and branch in `load_manifest`:

```python
SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V2)

# in ModelManifest:
    frames_per_input: int = 1
    heatmap_stride: int = 0
    nominal_ball_px: float = 0.0
# `decode` already exists in v1 ("in_graph"); v2 sets "heatmap_peak".

# in load_manifest, replace the exact-version check:
    if schema not in SCHEMA_VERSIONS:
        raise ValueError(...)  # keep the existing loud message, list both versions

    if schema == SCHEMA_VERSION_V2:
        frames_per_input = int(raw["frames_per_input"])   # KeyError on absence is the contract
        heatmap_stride = int(raw["heatmap_stride"])
        nominal_ball_px = float(raw["nominal_ball_px"])
        if frames_per_input < 1:
            raise ValueError(f"frames_per_input must be >= 1, got {frames_per_input}")
        if frames_per_input % 2 == 0:
            raise ValueError(
                f"frames_per_input must be odd for heatmap_peak decode (the "
                f"detected frame sits in the middle of a centered window), "
                f"got {frames_per_input}")
        if heatmap_stride < 1:
            raise ValueError(f"heatmap_stride must be >= 1, got {heatmap_stride}")
        if nominal_ball_px <= 0:
            raise ValueError(f"nominal_ball_px must be > 0, got {nominal_ball_px}")
        if str(raw["decode"]) != "heatmap_peak":
            raise ValueError(
                f"ball-model-v2 only supports decode='heatmap_peak', got {raw['decode']!r}")
    else:
        frames_per_input, heatmap_stride, nominal_ball_px = 1, 0, 0.0
```

Pass the three new fields into the `ModelManifest(...)` construction.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: PASS (all, including pre-existing v1 tests).

- [ ] **Step 5: Commit**

```bash
git add ball_model.py tests/test_ball_model.py
git commit -m "feat(ball-model): manifest schema v2 with temporal + heatmap fields"
```

---

### Task 2: Heatmap decode + `HeatmapRunner` in `ball_model.py`

**Files:**
- Modify: `ball_model.py`
- Test: `tests/test_ball_model.py`

**Interfaces:**
- Consumes: v2 `ModelManifest` from Task 1.
- Produces: `decode_heatmap(heatmap, threshold, stride, nominal_px)` (pure numpy → `[(cx, cy, score), ...]`, coordinates in input pixels); `HeatmapRunner` with `.manifest` and `.run_batch(stacks)` matching `TorchScriptRunner`'s return shape; `load_detector` returns a `HeatmapRunner` when `manifest.decode == "heatmap_peak"`.

- [ ] **Step 1: Write the failing decode tests** (pure numpy — no torch)

```python
import numpy as np

def test_decode_heatmap_finds_subpixel_peak():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[50, 100] = 0.9
    hm[50, 101] = 0.6          # pulls the sub-pixel x positively
    peaks = ball_model.decode_heatmap(hm, threshold=0.1, stride=2, nominal_px=12.0)
    assert len(peaks) == 1
    cx, cy, score = peaks[0]
    assert score == pytest.approx(0.9)
    assert cy == pytest.approx(100.0, abs=1.0)          # 50 * stride
    assert 200.0 < cx < 204.0                           # 100 * stride, nudged right


def test_decode_heatmap_below_threshold_is_empty():
    hm = np.full((208, 208), 0.05, dtype=np.float32)
    assert ball_model.decode_heatmap(hm, 0.1, 2, 12.0) == []


def test_decode_heatmap_two_separated_peaks_both_found():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[20, 20] = 0.8
    hm[150, 150] = 0.5
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 2


def test_decode_heatmap_plateau_emits_single_peak():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[30, 30] = hm[30, 31] = 0.7    # two equal neighbours must not double-fire
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 1


def test_heatmap_runner_decodes_only_the_middle_channel():
    # Output [B, frames, Hh, Wh]; runner must decode ONLY the middle
    # (centre-frame) channel and emit TorchScriptRunner-shaped tuples.
    out = np.zeros((1, 3, 208, 208), dtype=np.float32)
    out[:, 1, 50, 100] = 0.9   # middle frame — the one being detected
    out[:, 0, 10, 10] = 0.9    # past-frame channel — must be ignored
    out[:, 2, 90, 90] = 0.9    # future-frame channel — must be ignored
    runner = _runner_with_manifest(frames_per_input=3, conf_threshold=0.1,
                                   heatmap_stride=2, nominal_ball_px=12.0)
    results = runner._decode_output(out)
    assert len(results) == 1 and len(results[0]) == 1
    cx, cy, w, h, score, class_index = results[0][0]
    assert (cy, score, class_index) == (pytest.approx(100.0, abs=1.0), pytest.approx(0.9), 0)
    assert w == h == 12.0
```

Note for the implementer: `run_batch` needs torch to run the module, so the runner-level test targets `HeatmapRunner._decode_output(output_array)` — the numpy-pure method that slices the middle channel and calls `decode_heatmap`. `_runner_with_manifest` builds a `HeatmapRunner(module=None, manifest=SimpleNamespace(...), torch_module=None)`; `_decode_output` must not touch `self._module`/`self._torch`, which keeps torch out of the tests entirely.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: FAIL (`module 'ball_model' has no attribute 'decode_heatmap'`).

- [ ] **Step 3: Implement**

```python
def decode_heatmap(heatmap, threshold, stride, nominal_px):
    """Sub-pixel peaks of one heatmap, in input-pixel coordinates.

    Local maxima (8-neighbourhood, strict against later-scanned equals so a
    flat 2-px plateau fires once) above `threshold`, refined by a 3x3
    centre-of-mass. Returns [(cx, cy, score)] sorted by score descending.
    `nominal_px` is not used here (it sizes the reported box in the runner);
    it is in the signature so callers hold the full decode contract in one
    place.
    """
    import numpy as np

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"decode_heatmap expects a 2D heatmap, got shape {hm.shape}")
    padded = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
    neighbourhoods = np.stack([
        padded[dy:dy + hm.shape[0], dx:dx + hm.shape[1]]
        for dy in range(3) for dx in range(3) if not (dy == 1 and dx == 1)
    ])
    is_peak = (hm >= neighbourhoods.max(axis=0)) & (hm >= threshold)
    # break plateau ties: keep only the first occurrence in scan order
    ys, xs = np.nonzero(is_peak)
    peaks, taken = [], np.zeros_like(hm, dtype=bool)
    for y, x in zip(ys, xs):
        if taken[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
            continue
        taken[y, x] = True
        window = hm[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        wy, wx = np.mgrid[max(y - 1, 0):min(y + 2, hm.shape[0]),
                          max(x - 1, 0):min(x + 2, hm.shape[1])]
        weight = np.clip(window, 0, None)
        total = float(weight.sum())
        cy = float((weight * wy).sum() / total) if total > 0 else float(y)
        cx = float((weight * wx).sum() / total) if total > 0 else float(x)
        peaks.append((cx * stride, cy * stride, float(hm[y, x])))
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


class HeatmapRunner:
    """Runs the traced WASB-style model over co-located multi-frame tile stacks.

    The graph emits [batch, frames_per_input, Hh, Wh] MIMO heatmaps; only the
    MIDDLE (centre-frame) channel is decoded — analysis is offline, so every
    frame is detected with both its past and future neighbour in view, which
    is what carries the model through direction reversals at wall and floor
    contacts. Training supervises all frames; serving answers "where is the
    ball in the CENTRE frame".
    """

    def __init__(self, module, manifest, torch_module):
        self._module = module
        self._torch = torch_module
        self.manifest = manifest

    def _decode_output(self, output):
        results = []
        size = float(self.manifest.nominal_ball_px)
        middle = int(self.manifest.frames_per_input) // 2
        for heatmaps in output:                      # [frames, Hh, Wh]
            peaks = decode_heatmap(
                heatmaps[middle], self.manifest.conf_threshold,
                self.manifest.heatmap_stride, size)
            results.append([(cx, cy, size, size, score, 0)
                            for cx, cy, score in peaks])
        return results

    def run_batch(self, stacks):
        import numpy as np
        torch = self._torch
        stacked = np.stack(stacks).astype("float32")           # [B, H, W, 3*F]
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            raw = self._module(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        return self._decode_output(raw.detach().cpu().numpy())
```

In `load_detector`, after loading the module:

```python
    if manifest.decode == "heatmap_peak":
        runner = HeatmapRunner(module, manifest, torch)
    else:
        runner = TorchScriptRunner(module, manifest, torch)
```

(Channel order convention, document in the `HeatmapRunner` docstring: stacks are BGR frames concatenated oldest-first along the channel axis, raw 0–255, matching the training adapter in Task 6/`docs/WASB-TRAIN.md`. No mean/std, consistent with the existing BGR-raw convention.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ball_model.py tests/test_ball_model.py
git commit -m "feat(ball-model): heatmap peak decode + HeatmapRunner for v2 manifests"
```

---

### Task 3: `detect_frame_stack` in `ball_detector.py`

**Files:**
- Modify: `ball_detector.py`
- Test: `tests/test_ball_detector.py`

**Interfaces:**
- Consumes: `tile_windows`, `_crop`, `_tile_edge`, `_class_name`, `merge_detections` (all existing); `HeatmapRunner.run_batch` contract from Task 2.
- Produces: `detect_frame_stack(runner, frames, manifest) -> list[detection dict]` — frames oldest-first, len == `manifest.frames_per_input`, detections for the **middle** frame, same dict shape as `detect_frame`.

- [ ] **Step 1: Write the failing tests**

Use the existing fake-runner pattern (`test_detect_frame_maps_tile_local_box_to_full_frame` etc.):

```python
def _stack_manifest(**overrides):
    # mirror the existing _manifest() helper but with v2 fields
    values = dict(input_size=(416, 416), tile_overlap_px=64, max_batch_tiles=32,
                  conf_threshold=0.1, nms_iou=0.45, class_names=("ball",),
                  frames_per_input=3, decode="heatmap_peak",
                  heatmap_stride=2, nominal_ball_px=12.0)
    values.update(overrides)
    return SimpleNamespace(**values)


class _StackRecordingRunner:
    """Asserts every stack it sees has 9 channels; fires one detection in the
    tile that contains (target_x, target_y)."""
    def __init__(self, target):
        self.target = target
        self.stacks_seen = 0

    def run_batch(self, stacks):
        results = []
        for stack in stacks:
            assert stack.shape[2] == 9, stack.shape
            self.stacks_seen += 1
            results.append([])
        return results  # extend per-test to place a hit, as the v1 fakes do


def test_detect_frame_stack_concatenates_frames_oldest_first():
    manifest = _stack_manifest()
    frames = [np.full((416, 416, 3), fill, dtype=np.uint8) for fill in (10, 20, 30)]

    class Probe:
        def run_batch(self, stacks):
            (stack,) = stacks
            assert stack[0, 0, 0] == 10 and stack[0, 0, 3] == 20 and stack[0, 0, 6] == 30
            return [[(100.0, 50.0, 12.0, 12.0, 0.9, 0)]]

    detections = ball_detector.detect_frame_stack(Probe(), frames, manifest)
    assert detections[0]["x"] == pytest.approx(100.0)
    assert detections[0]["class"] == "ball"


def test_detect_frame_stack_rejects_wrong_frame_count():
    manifest = _stack_manifest()
    frames = [np.zeros((416, 416, 3), dtype=np.uint8)] * 2
    with pytest.raises(ValueError, match="frames_per_input"):
        ball_detector.detect_frame_stack(_StackRecordingRunner((0, 0)), frames, manifest)


def test_detect_frame_stack_rejects_mismatched_frame_shapes():
    manifest = _stack_manifest()
    frames = [np.zeros((416, 416, 3), dtype=np.uint8),
              np.zeros((416, 416, 3), dtype=np.uint8),
              np.zeros((832, 416, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="shape"):
        ball_detector.detect_frame_stack(_StackRecordingRunner((0, 0)), frames, manifest)


def test_detect_frame_stack_maps_tile_local_peak_to_full_frame():
    # 4K stack, peak in a non-first tile — port of the v1 mapping test.
    manifest = _stack_manifest()
    frames = [np.zeros((2160, 3840, 3), dtype=np.uint8)] * 3

    class OneHit:
        def run_batch(self, stacks):
            out = [[] for _ in stacks]
            out[0] = [(10.0, 20.0, 12.0, 12.0, 0.9, 0)]
            return out

    # run once with batch covering all 66 tiles to keep the fake simple
    manifest = _stack_manifest(max_batch_tiles=66)
    detections = ball_detector.detect_frame_stack(OneHit(), frames, manifest)
    assert detections[0]["x"] == pytest.approx(10.0)
    assert detections[0]["y"] == pytest.approx(20.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_detector.py -q`
Expected: FAIL (`no attribute 'detect_frame_stack'`).

- [ ] **Step 3: Implement**

```python
def detect_frame_stack(runner, frames, manifest):
    """Full-frame detections for the MIDDLE frame of an oldest-first stack.

    Cuts co-located tiles from every frame, concatenates them along the
    channel axis (oldest first — the temporal ordering the model was trained
    on), and reuses the same tile-local -> full-frame mapping and cross-tile
    merge as detect_frame. The runner decodes the centre MIMO channel, so the
    returned detections locate the ball in frames[len(frames) // 2].
    """
    import numpy as np

    expected = int(manifest.frames_per_input)
    if len(frames) != expected:
        raise ValueError(
            f"detect_frame_stack got {len(frames)} frames but "
            f"manifest.frames_per_input is {expected}")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"stack frames disagree on shape: {sorted(shapes)}")

    tile = _tile_edge(manifest.input_size)
    frame_h, frame_w = frames[-1].shape[:2]
    windows = tile_windows(frame_w, frame_h, tile, manifest.tile_overlap_px)

    detections = []
    batch = max(1, manifest.max_batch_tiles)
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        stacks = [
            np.concatenate([_crop(f, x0, y0, tile) for f in frames], axis=2)
            for x0, y0 in chunk
        ]
        for (x0, y0), boxes in zip(chunk, runner.run_batch(stacks), strict=True):
            for cx, cy, width, height, score, class_index in boxes:
                if score < manifest.conf_threshold:
                    continue
                name = _class_name(manifest.class_names, class_index)
                detections.append({
                    "x": float(cx) + x0, "y": float(cy) + y0,
                    "width": float(width), "height": float(height),
                    "confidence": float(score),
                    "class": name, "class_name": name,
                })
    return merge_detections(detections, manifest.nms_iou)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_detector.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ball_detector.py tests/test_ball_detector.py
git commit -m "feat(ball-detector): co-located multi-frame tile stacks via detect_frame_stack"
```

---

### Task 4: Centered-window iteration + `local` alias in `ball_track_offline.py`

**Files:**
- Modify: `ball_track_offline.py`
- Test: `tests/test_ball_track_offline.py`

**Interfaces:**
- Consumes: `detect_frame_stack` (Task 3), `manifest.frames_per_input` (Task 1); existing seams `_load_ball_detector`, `_detect_frame`, `selected_detector`, `_build_infer`, `_iter_frames`.
- Produces: `selected_detector()` accepts `"local"` (canonical), `"yolox"` (alias → same branch), `"rfdetr"`; `_centered_windows(indexed_frames)` generator yielding `(frame_index, [prev, cur, nxt])` with edge padding; `detections_to_track_samples` routes temporal manifests through it. Detection for frame t uses (t−1, t, t+1) — **one frame of lookahead, correct timestamps** — because analysis is offline and future context is free.

- [ ] **Step 1: Write the failing tests**

```python
def test_selected_detector_accepts_local_alias(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "local")
    assert ball_track_offline.selected_detector() == "local"


def test_selected_detector_yolox_still_accepted(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "yolox")
    assert ball_track_offline.selected_detector() == "local"   # normalised


def test_centered_windows_pads_both_clip_edges():
    frames = [(i, np.full((2, 2, 3), v, dtype=np.uint8))
              for i, v in enumerate((10, 20, 30, 40))]
    out = [(i, [int(f[0, 0, 0]) for f in window])
           for i, window in ball_track_offline._centered_windows(iter(frames))]
    assert out == [(0, [10, 10, 20]),   # left edge: first frame repeated
                   (1, [10, 20, 30]),
                   (2, [20, 30, 40]),
                   (3, [30, 40, 40])]   # right edge: last frame repeated


def test_centered_windows_single_frame_clip():
    frames = [(0, np.full((2, 2, 3), 7, dtype=np.uint8))]
    out = [(i, [int(f[0, 0, 0]) for f in w])
           for i, w in ball_track_offline._centered_windows(iter(frames))]
    assert out == [(0, [7, 7, 7])]


def test_centered_windows_empty_clip():
    assert list(ball_track_offline._centered_windows(iter([]))) == []


def test_temporal_manifest_routes_through_centered_windows(monkeypatch):
    calls = []

    def fake_detect_stack(runner, frames, manifest):
        calls.append([int(f[0, 0, 0]) for f in frames])
        return [{"x": 1.0, "y": 2.0, "width": 3.0, "height": 3.0,
                 "confidence": 0.9, "class": "ball", "class_name": "ball"}]

    runner = SimpleNamespace(manifest=SimpleNamespace(
        conf_threshold=0.1, frames_per_input=3))
    monkeypatch.setattr(ball_track_offline, "_detect_frame_stack", fake_detect_stack)
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda path: 60.0)
    monkeypatch.setattr(ball_track_offline, "_iter_frames", lambda path: iter(
        (i, np.full((2, 2, 3), v, dtype=np.uint8))
        for i, v in enumerate((10, 20, 30))))

    samples = ball_track_offline.detections_to_track_samples(
        "fake.mp4", model=runner, confidence=0.4)

    assert calls == [[10, 10, 20], [10, 20, 30], [20, 30, 30]]
    # timestamps: frame t's sample uses t/fps even though t+1 was decoded first
    assert [s.t_s for s in samples] == pytest.approx([0.0, 1 / 60, 2 / 60])


def test_temporal_manifest_rejects_stride(monkeypatch):
    runner = SimpleNamespace(manifest=SimpleNamespace(
        conf_threshold=0.1, frames_per_input=3))
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda path: 60.0)
    monkeypatch.setattr(ball_track_offline, "_iter_frames", lambda path: iter(
        [(0, np.zeros((2, 2, 3), dtype=np.uint8))]))
    with pytest.raises(ValueError, match="stride"):
        ball_track_offline.detections_to_track_samples(
            "fake.mp4", model=runner, confidence=0.4, stride=2)


def test_single_frame_manifest_keeps_v1_path(monkeypatch):
    # frames_per_input == 1 must keep going through _detect_frame per frame
    seen = []
    monkeypatch.setattr(ball_track_offline, "_detect_frame",
                        lambda runner, frame, manifest: seen.append(1) or [])
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda path: 60.0)
    monkeypatch.setattr(ball_track_offline, "_iter_frames", lambda path: iter(
        [(0, np.zeros((2, 2, 3), dtype=np.uint8))]))
    runner = SimpleNamespace(manifest=SimpleNamespace(
        conf_threshold=0.1, frames_per_input=1))
    ball_track_offline.detections_to_track_samples(
        "fake.mp4", model=runner, confidence=0.4)
    assert seen == [1]
```

Note: existing tests monkeypatch `_detect_frame`/`_load_ball_detector`/`_iter_frames`; follow the same seam style and add `_detect_frame_stack` as a module-level seam.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_track_offline.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _detect_frame_stack(runner, frames, manifest):
    """Seam for tests."""
    import ball_detector
    return ball_detector.detect_frame_stack(runner, frames, manifest)


def _centered_windows(indexed_frames):
    """(frame_index, [prev, cur, nxt]) for every frame of an (index, frame)
    iterator — the centred 3-frame window the temporal detector consumes.

    Emission lags decode by one frame: frame t's window needs t+1, which is
    fine offline. Clip edges pad by repeating the first/last frame, mirroring
    the dataset builder's edge padding so serving matches training. Supports
    exactly 3-frame windows; detections_to_track_samples rejects other
    frames_per_input values loudly.
    """
    previous = None       # frame t-1 (None at the left edge)
    current = None        # (index, frame t) awaiting its right neighbour
    for index, frame in indexed_frames:
        if current is not None:
            cur_index, cur_frame = current
            yield cur_index, [previous if previous is not None else cur_frame,
                              cur_frame, frame]
            previous = cur_frame
        current = (index, frame)
    if current is not None:
        cur_index, cur_frame = current
        yield cur_index, [previous if previous is not None else cur_frame,
                          cur_frame, cur_frame]
```

In `selected_detector()`: accept `{"local", "yolox", "rfdetr"}`, return `"local"` for both `local` and `yolox` (update the error message and the docstring; `yolox` stays accepted because MODEL.md §2b documents it).

In `detections_to_track_samples`: the rfdetr and single-frame local paths keep the existing per-frame loop untouched. For the local backend, resolve the runner first (it is needed to know `frames_per_input`; keep the lazy-load property by resolving on the first decoded frame, as `_build_infer` is built today). When `frames_per_input > 1`:

```python
    # temporal manifests: centred windows, consecutive frames only
    if frames_needed != 3:
        raise ValueError(
            f"frames_per_input={frames_needed} not supported; the centred "
            f"window path implements exactly 3 (WASB). Widen _centered_windows "
            f"deliberately if a future model needs 5.")
    if stride != 1:
        raise ValueError(
            f"stride={stride} is invalid with a temporal manifest: the model "
            f"consumes consecutive frames; non-adjacent frames were never in "
            f"its training distribution.")
    for frame_index, window in _centered_windows(_iter_frames(video_path)):
        predictions = _detect_frame_stack(runner, window, runner.manifest)
        ball = select_ball_prediction(predictions, confidence)
        if ball is None:
            continue
        t_s = frame_index / fps + offset_s
        samples.append(TrackSample(t_s=t_s, px=(float(ball["x"]), float(ball["y"]))))
    return samples
```

Keep the manifest `conf_threshold <= confidence` guard from `_build_infer` applied to this path too (same rationale: the manifest floor must not silently swallow the caller's threshold). Update the module and function docstrings: the temporal path uses one frame of lookahead (double-sided context) and emits correct per-frame timestamps.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_track_offline.py -q`
Expected: PASS (including all pre-existing tests — the yolox alias must not break them).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, 1 deselected.

- [ ] **Step 6: Commit**

```bash
git add ball_track_offline.py tests/test_ball_track_offline.py
git commit -m "feat(offline): centered 3-frame windows for temporal manifests; BALL_DETECTOR=local alias"
```

---

### Task 5: Sequence crops in `prepare_ball_dataset.py`

**Files:**
- Modify: `prepare_ball_dataset.py`
- Test: `tests/test_prepare_ball_dataset.py`

**Interfaces:**
- Consumes: existing `load_export`, `plan_crops`, `render_split`, `ascii_slug`, `crop_file_name`, `_imread_unicode`, `_imwrite_unicode`, `FRAME_RE`.
- Produces: CLI flags `--clips-dir PATH` and `--seq-frames N` (default 1 = today's behaviour, byte-identical output; N must be odd). With `N > 1`: schema `"ball-crops-v2"`, each COCO image entry gains `"sequence": [file_tm1, file_t, file_tp1]` (oldest first, **labeled frame in the middle** — the centered-window convention), sequence files named `<crop_stem>.tm1.jpg` / `<crop_stem>.tp1.jpg` (ASCII by construction — derived from the existing ASCII crop stem). New functions: `find_clip_videos(clips_dir, records) -> dict[clip_slug, Path]`, `verify_frame_alignment(video_frame, export_image) -> float` (mean abs diff), `decode_frames(video_path, indices) -> dict[int, ndarray]`.

- [ ] **Step 1: Write the failing tests**

The existing tests build fake COCO exports on disk; extend the fixtures. Video decode must be seam-injected so tests never touch cv2 video I/O:

```python
def test_seq_frames_requires_clips_dir(tmp_path):
    with pytest.raises(SystemExit, match="clips-dir"):
        prepare_ball_dataset.build(
            source=..., out=tmp_path / "out", crop=416, positives=1, negatives=0,
            jitter=0.3, min_visible=0.5, seq_frames=3, clips_dir=None, ...)


def test_find_clip_videos_matches_by_ascii_slug(tmp_path):
    (tmp_path / "Bay Club Rally 1.mp4").write_bytes(b"")
    records = [{"clip": prepare_ball_dataset.ascii_slug("Bay Club Rally 1"), ...}]
    found = prepare_ball_dataset.find_clip_videos(tmp_path, records)
    assert set(found) == {records[0]["clip"]}


def test_find_clip_videos_missing_clip_is_fatal(tmp_path):
    records = [{"clip": "nonexistent-clip", ...}]
    with pytest.raises(SystemExit, match="nonexistent-clip"):
        prepare_ball_dataset.find_clip_videos(tmp_path, records)


def test_sequence_render_writes_three_aligned_crops(tmp_path, monkeypatch):
    # monkeypatch decode_frames to return synthetic frames whose pixel values
    # encode the frame index; assert the .tm1/.tp1 files land beside the
    # anchor crop, contain the right frame's pixels (t-1 and t+1), and the
    # COCO "sequence" lists [tm1, anchor, tp1] oldest-first.
    ...


def test_sequence_pads_at_clip_start(tmp_path, monkeypatch):
    # labeled frame index 0: tm1 must repeat frame 0 (matches the runtime
    # edge padding in ball_track_offline._centered_windows), not crash or
    # go negative; tp1 is the real frame 1.
    ...


def test_sequence_pads_at_clip_end(tmp_path, monkeypatch):
    # labeled frame == last frame of the clip: tp1 repeats the last frame
    # (decode_frames raising "ended at frame N" must not happen for this).
    ...


def test_alignment_failure_is_fatal(tmp_path, monkeypatch):
    # decode_frames returns a frame that does NOT match the export image ->
    # SystemExit naming the clip; a silently shifted sequence poisons training.
    ...
```

(Fill the `...` bodies following the file's existing fixture helpers — they already synthesize COCO exports and images; reuse them rather than inventing new fixture machinery.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_ball_dataset.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Key pieces:

```python
SCHEMA_VERSION_SEQ = "ball-crops-v2"
ALIGNMENT_MAX_ABS_DIFF = 12.0   # JPEG recompression tolerance, mean |diff| over pixels
VIDEO_EXTENSIONS = (".mp4", ".mov", ".MP4", ".MOV")


def find_clip_videos(clips_dir, records):
    """Map each record clip slug to its source video, or die listing the gaps."""
    clips_dir = Path(clips_dir)
    by_slug = {}
    for path in sorted(clips_dir.iterdir()):
        if path.suffix in VIDEO_EXTENSIONS:
            by_slug.setdefault(ascii_slug(path.stem), path)
    needed = {r["clip"] for r in records}
    missing = sorted(needed - set(by_slug))
    if missing:
        raise SystemExit(
            f"--clips-dir {clips_dir} has no video for clip(s): {missing}. "
            f"Sequence crops need the source footage for neighbour frames.")
    return {slug: by_slug[slug] for slug in needed}


def decode_frames(video_path, indices):
    """{index: frame_bgr} for the requested indices, one sequential pass.

    Sequential read, not CAP_PROP_POS_FRAMES seeks: HEVC seek lands on
    keyframes unreliably across OpenCV builds, and a silently wrong frame is
    exactly the poison this mode must not produce.
    """
    import cv2
    wanted = sorted(set(indices))
    out = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")
        index = 0
        for target in wanted:
            while index <= target:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit(
                        f"{video_path} ended at frame {index}; needed {target}")
                current, index = frame, index + 1
            out[target] = current
    finally:
        cap.release()
    return out


def verify_frame_alignment(video_frame, export_image):
    """Mean absolute pixel difference; resizes the export image if the export
    was scaled. Caller compares against ALIGNMENT_MAX_ABS_DIFF and dies loudly
    on mismatch, naming the clip and frame."""
    import cv2
    import numpy as np
    if video_frame.shape[:2] != export_image.shape[:2]:
        export_image = cv2.resize(
            export_image, (video_frame.shape[1], video_frame.shape[0]))
    return float(np.mean(np.abs(
        video_frame.astype(np.int16) - export_image.astype(np.int16))))
```

In `render_split` (or a wrapper around it — keep the single-frame path byte-identical): when `seq_frames > 1`, for each record decode `{t-1, t, t+1}` — clamped into the clip at both ends (frame 0's `t-1` repeats frame 0; the last frame's `t+1` repeats the last frame — `decode_frames` must therefore learn the clip's frame count or treat one-past-the-end as "repeat last" rather than dying). Verify alignment of decoded frame `t` against the export image once per record (fatal on failure — check both the parsed index and, if that fails, index±1 to auto-detect a uniform off-by-one *per clip*, applying it clip-wide only if consistent, else die), apply the same per-clip scale factor and the same crop window to every frame in the sequence, and write neighbours with `_imwrite_unicode` as `<stem>.tm1.jpg`, `<stem>.tp1.jpg`. Add `"sequence"` (`[tm1, anchor, tp1]`, oldest-first, labeled frame in the middle) to the per-image COCO entry and set the top-level schema field to `ball-crops-v2` when sequences are present. Decode each video once per split with all needed indices batched (one sequential pass), not once per record.

Also update `parse_args`: `--clips-dir` (Path, default None), `--seq-frames` (int, default 1); `build(...)` validates `seq_frames > 1 and not clips_dir` → `SystemExit` mentioning `--clips-dir`, and `seq_frames % 2 == 0` → `SystemExit` ("sequence length must be odd: the labeled frame sits in the middle of a centered window").

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prepare_ball_dataset.py -q`
Expected: PASS, including all pre-existing single-frame tests.

- [ ] **Step 5: Commit**

```bash
git add prepare_ball_dataset.py tests/test_prepare_ball_dataset.py
git commit -m "feat(dataset): aligned 3-frame sequence crops (ball-crops-v2) from source clips"
```

---

### Task 6: Training runbook + dataset adapter (`docs/WASB-TRAIN.md`)

**Files:**
- Create: `docs/WASB-TRAIN.md`
- Reference: `prepare_ball_dataset.py` (Task 5 output format), WASB-SBDT upstream (`github.com/nttcom/WASB-SBDT`, MIT)

This task is documentation plus training-environment work — the CUDA box has no pytest, so verification is by smoke run. The adapter script itself lives in the training repo (`C:\Users\alann\Code\ball-detector-train`), like the YOLOX exp file does today; `docs/WASB-TRAIN.md` in *this* repo is the runbook and the contract.

- [ ] **Step 1: Write `docs/WASB-TRAIN.md`** covering, concretely:

1. **Setup**: clone WASB-SBDT at a pinned commit into the training env; record the commit hash in the doc. Download the pretrained checkpoints (tennis + badminton are the closest regimes; both are small-fast-ball racquet sports at broadcast-ish scale).
2. **Dataset adapter contract** (script `wasb_crops_dataset.py` in the training repo): reads a `ball-crops-v2` COCO directory; each sample = the 3 sequence files `[t−1, t, t+1]` loaded BGR raw 0–255 oldest-first, concatenated to 9 channels; targets = one Gaussian heatmap per input frame at `heatmap_stride` resolution, σ = 2.0 heatmap px, centred on that frame's ball centre. **Only the middle frame `t` has a label** — for `t−1`/`t+1` supervise with the anchor's centre when the sequence was edge-padded (identical frames), otherwise mask those channels out of the loss (unlabeled ≠ empty; an empty target would actively teach "no ball"). Negatives (crops with no ball) supervise all frames with empty heatmaps. **Hard negatives are mandatory, not optional:** wall and floor scuff marks are the incumbent detector's dominant false-positive class, and the temporal input only suppresses them if training shows static ball-lookalikes with empty targets. Mine negative crop windows preferentially from (a) locations where the RF-DETR baseline false-fired on the eval clips (its stationary-penalty rejections are already logged in eval runs) and (b) player-adjacent wall regions, where revealed-background ghosting — a mark "appearing" as the player moves off it — creates transient motion energy at a mark's location. Both cases must appear in training with empty heatmap targets.
3. **Fine-tune recipe**: init from the tennis checkpoint; adapt the input conv from 9ch pretrained (WASB is already 3-frame/9-channel — no conv surgery needed, which is half the reason it was chosen); train 416×416 to match the tile size (WASB's default input differs — set the config's input size; HRNet is fully convolutional so pretrained weights transfer); freeze nothing (dataset is small but domain gap is real); early stop on val heatmap F1.
4. **Commands** (adapt paths on the box), including the dataset-regeneration command on the Mac side:

```bash
# Mac: regenerate crops with sequences (paths per your local layout)
.venv/bin/python prepare_ball_dataset.py \
    --source <roboflow-coco-export-dir> \
    --clips-dir <dir-with-source-clips> \
    --seq-frames 3 \
    --out ball-crops-seq-v1 \
    --val-clips <same clip split as the last crop build> \
    --test-clips <same>
```

5. **Smoke gates before a full run**: (a) one batch overfits to near-zero loss in <200 steps; (b) a rendered predicted-heatmap contact sheet over 16 val samples visually peaks on the ball; (c) input-channel order round-trip check — feed a sequence where the ball only appears in the middle frame and confirm the **middle** output channel responds, not the outer ones; (d) static-clutter check — a sequence of 3 identical frames containing a ball-sized dark mark must produce a near-empty middle heatmap (the wall-mark rejection the temporal input exists to buy).

- [ ] **Step 2: Execute the runbook on the CUDA box** (user-owned hardware; the fine-tune itself is run by whoever sits at that machine). Deliverable back into this plan: `wasb_best.pth` + its val metrics.

- [ ] **Step 3: Commit the runbook**

```bash
git add docs/WASB-TRAIN.md
git commit -m "docs(train): WASB fine-tune runbook and sequence-dataset adapter contract"
```

---

### Task 7: `export_wasb_model.py` — checkpoint → TorchScript + v2 manifest

**Files:**
- Create: `export_wasb_model.py`
- Test: `tests/test_export_wasb_model.py` (arg/manifest logic only — the trace path needs torch + WASB and runs only in the training env)

**Interfaces:**
- Consumes: `wasb_best.pth` (Task 6), v2 manifest contract (Task 1).
- Produces: `models/crosscourt-wasb-416-v1/{model.torchscript, manifest.json}` loadable by `ball_model.load_detector` on the Mac.

- [ ] **Step 1: Write the failing test** for the pure part — manifest assembly:

```python
def test_build_manifest_dict_v2_fields():
    manifest = export_wasb_model.build_manifest(
        name="crosscourt-wasb-416", version=1, input_size=416,
        frames_per_input=3, heatmap_stride=2, nominal_ball_px=12.0,
        conf_threshold=0.1, nms_iou=0.45, tile_overlap_px=64,
        max_batch_tiles=32, artifact_sha256="deadbeef",
        source_checkpoint="wasb_best.pth", trained_commit="abc1234",
        val_metric=0.0, notes="")
    assert manifest["schema_version"] == "ball-model-v2"
    assert manifest["decode"] == "heatmap_peak"
    assert manifest["frames_per_input"] == 3
    assert manifest["class_names"] == ["ball"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_export_wasb_model.py -q`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement** — mirror `export_ball_model.py`'s structure exactly (argparse, `_git_commit`, sha256, manifest write), with:

- `build_manifest(...)` as a pure function (tested above), emitting every v2 field from Task 1.
- Trace with `example = torch.randn(1, 3 * args.frames_per_input, size, size)`; model loaded via the WASB repo's model factory (import inside `main()`, training-env only — module import stays WASB-free, same rule as `export_ball_model.py` and YOLOX).
- **Verify the traced output shape before writing**: run the traced module on the example; assert output is `[1, frames_per_input, size // heatmap_stride, size // heatmap_stride]`, else die naming the actual shape — a mis-shaped graph would otherwise surface as garbage decodes on the Mac.
- Default `--conf-threshold 0.1` — deliberately low (BYTE-style rescue: weak peaks flow into `tracking_common`'s motion-consistency scorer, which promotes moving candidates +0.30 and suppresses stationary ones −0.40; the selection bar stays at the caller's 0.4).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_export_wasb_model.py -q`
Expected: PASS.

- [ ] **Step 5: Export on the training box, load-check on the Mac**

Training box: `python export_wasb_model.py --ckpt ...\wasb_best.pth --out models/crosscourt-wasb-416-v1 --version 1`
Mac check:

```bash
BALL_MODEL_DIR=models/crosscourt-wasb-416-v1 .venv/bin/python -c "import ball_model; r = ball_model.load_detector(); print(r.manifest.name, r.manifest.frames_per_input)"
```

Expected: `crosscourt-wasb-416 3` with no traceback. (Needs the full requirements env with torch; the manifest checks alone run without it.)

- [ ] **Step 6: Commit**

```bash
git add export_wasb_model.py tests/test_export_wasb_model.py
git commit -m "feat(export): WASB checkpoint -> TorchScript + ball-model-v2 manifest"
```

---

### Task 8: Recall eval against gates

**Files:**
- Create: `eval_set/RESULTS-wasb-v1.md`
- Uses: `ball_track_offline.detections_to_track_samples`, `local_model_eval.py` (RF-DETR baseline), `/eval` skill

No new library code — this task is measurement, honestly recorded whichever way it goes (precedent: `eval_set/RESULTS-3d-contact.md` records a *failed* gate).

- [ ] **Step 1: Produce the RF-DETR baseline detection rates** on the labeled eval clips (the clips behind `bayclub_wall_hits.csv` / `matchplay_ep3_wall_hits.csv` and the 95-label eval set), via `local_model_eval.py` if a baseline run isn't already on disk.

- [ ] **Step 2: Run the WASB model over the same clips**

```bash
BALL_DETECTOR=local BALL_MODEL_DIR=models/crosscourt-wasb-416-v1 \
  .venv/bin/python -c "
from ball_track_offline import detections_to_track_samples
samples = detections_to_track_samples('<clip>.mp4', confidence=0.4)
print(len(samples))"
```

(Or a small throwaway driver in the scratchpad — per-frame detected/total and per-labeled-event hit/miss; do not commit a new eval framework, the numbers go in the RESULTS file.)

- [ ] **Step 3: Score against the gates.** All three must hold to call this shipped:

1. **Rally-scale detection recall beats RF-DETR by ≥ 15 points** on the labeled eval clips (baseline ≈ 35%; the entire premise of the temporal model is a large recall gain — parity is a failure).
2. **`/eval` judge suite** against the newest `eval_set/BASELINE-*.md`: no regression on judge metrics when the pipeline inputs are regenerated with the new detector track (where applicable at this stage; the pipeline itself still runs RF-DETR — this gate protects the baseline bookkeeping and catches surprises in shared code paths touched by Tasks 1–5).
3. **Confidence-threshold sensitivity + wall-mark precision**: repeat the recall measurement at selection confidence 0.3/0.4/0.5 — the low manifest threshold (0.1) plus motion-consistency rescue must not collapse precision. Spot-check ≥ 50 detections on frames *without* a labeled ball across 2 clips, and classify each false fire by cause (wall/floor mark, racket head, shadow, reflection, other). Wall marks get their own line in the RESULTS table: they are the incumbent's dominant false-positive class and the specific clutter the temporal input is claimed to suppress — include several sequences where a player moves off a marked wall patch (revealed-background ghosting), the hardest static-clutter case.

- [ ] **Step 4: Write `eval_set/RESULTS-wasb-v1.md`** — per-clip tables (RF-DETR vs WASB detection rate, event recall), thresholds tried, false-fire spot-check, verdict against each gate, and the exact model manifest sha256 so the numbers stay attributable.

- [ ] **Step 5: Commit**

```bash
git add eval_set/RESULTS-wasb-v1.md
git commit -m "eval: WASB temporal detector vs RF-DETR baseline (gates + verdict)"
```

- [ ] **Step 6: Update docs to match reality** — `CLAUDE.md`'s `ball_track_offline.py` note and `ios/MODEL.md` §2b (`BALL_DETECTOR=local`, v2 manifest existence, WASB-TRAIN.md pointer). Commit with the docs change:

```bash
git add CLAUDE.md ios/MODEL.md
git commit -m "docs: local detector backend covers v2 temporal manifests"
```

---

## After this plan

- **If the Task 8 gates pass:** write the follow-up plan for pipeline integration (`job_runner.py` swaps RF-DETR for the local temporal model — which also finally retires the 960 px downscale) and for the BlurBall streak head (license check first).
- **If they fail:** the RESULTS file is the deliverable; next lever per the research memory is the labeling bootstrap (PadelTracker100-style: detector+tracker proposes, human verifies) to grow past ~698 instances before re-training.
- **5-frame ablation trigger (why 3 and not 5/7):** 3 frames is the pretrained-weight shape (zero-surgery transfer, which is the point of WASB at our dataset size), and at 200–550 px/frame the ball is usually outside a co-located 416 px tile beyond t±1 — the published 5>3 results (TOTNet occlusion, SPIRIT) come from full-frame inputs where the ball stays in view. But occlusion and slow-lob-apex frames are genuine exceptions where t±2 adds signal while the ball remains in the tile. If Task 8's miss breakdown concentrates there rather than in plain no-fire, run a 5-frame ablation (`frames_per_input: 5` is already legal in the manifest; widen `_centered_windows` and the dataset `--seq-frames` deliberately) against the 3-frame baseline.

## Self-Review Notes

- Spec coverage: manifest/serving (T1–2), tiling (T3), runner (T4), data (T5), training (T6), export boundary (T7), eval gate (T8). ByteTrack decision folded into T7 (low threshold) + T8 gate 3. RF-DETR untouched by design.
- Type consistency: `frames_per_input` spelled identically in manifest JSON, `ModelManifest`, `detect_frame_stack`, and `_centered_windows`; `decode_heatmap` returns `(cx, cy, score)` and `HeatmapRunner` widens to the 6-tuple `ball_detector` consumes; the centered convention is identical everywhere — stacks oldest-first `[t−1, t, t+1]`, label/decode on the middle element (`frames_per_input // 2`), edge padding by repetition in both the dataset builder and `_centered_windows`.
- Double-sided context: analysis is offline, so windows are centered (past AND future), not causal — chosen deliberately for direction reversals at wall/floor contacts; enforced by odd-`frames_per_input` validation (T1), middle-channel decode (T2/T3), lookahead iteration (T4), and middle-labeled sequences (T5/T6).
- Wall-mark false positives: attacked at the source (static input ≈ no motion signal), backed by mandatory hard negatives incl. revealed-background ghosting (T6), a static-clutter smoke gate before training completes (T6), and a per-cause false-fire breakdown in the eval (T8 gate 3).
- Known open risk (stated, not hidden): frame-index alignment between Roboflow export filenames and video frame numbering — T5's alignment check turns it from silent poison into a loud per-clip failure with off-by-one auto-detection.
