# WASB Ball Detector in job_runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Analysis jobs run the committed WASB temporal ball detector by default (native-resolution 3-frame windows), with the hosted RF-DETR kept switchable for A/B eval, GPU auto on CUDA, and per-run detector attribution.

**Architecture:** `job_runner` gains a backend seam (`BALL_DETECTOR` env, resolved by `ball_track_offline.selected_detector()`). `decode_segments_to_queue` grows a temporal mode that emits `(center, [prev, cur, nxt])` windows; `track_segments` branches per backend and applies the manifest's confidence floor on the local branch. `ball_model` gains `BALL_DEVICE` resolution and its default dir moves to the committed WASB artifact. Everything downstream of the ball CSV rows is untouched.

**Tech Stack:** Python 3.12 (`.venv`), pytest, cv2, torch (runtime only — the test suite never imports it), TorchScript.

**Spec:** `docs/superpowers/specs/2026-07-29-wasb-job-runner-design.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest tests/... -q` — system python has no cv2/flask.
- Editing a `*.py` with a paired `tests/test_*.py` auto-runs that file (PostToolUse hook); a failure comes back as a blocked edit.
- The default test suite must never import torch (`requirements-test.txt` excludes it); torch-touching tests are `@pytest.mark.requires_model` or use fake torch namespaces.
- No silent fallback between detectors anywhere — every failure raises with its cause.
- The rfdetr branch must stay byte-identical in behavior (eval zero-drift proves it in Task 7).
- `frames_per_input` support is exactly {1, 3}; any other value raises.
- Auto device never picks MPS; MPS is opt-in via `BALL_DEVICE=mps`.
- Commit after every task with the message given in its final step.

---

### Task 1: `BALL_DEVICE` device resolution in ball_model

**Files:**
- Modify: `ball_model.py` (add `_resolve_device`; thread `device` through `load_detector`, `TorchScriptRunner`, `HeatmapRunner`)
- Test: `tests/test_ball_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ball_model._resolve_device(torch_module) -> str`; `TorchScriptRunner.__init__(self, module, manifest, torch_module, device="cpu")` and same for `HeatmapRunner`, both exposing `self.device`; `load_detector()` returns a runner whose `.device` reflects `BALL_DEVICE`. Task 5's `ball_backend_summary` reads `model.device`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ball_model.py`)

```python
class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeMps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def _fake_torch(cuda=False, mps=False):
    import types
    return types.SimpleNamespace(
        cuda=_FakeCuda(cuda),
        backends=types.SimpleNamespace(mps=_FakeMps(mps)),
    )


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.delenv("BALL_DEVICE", raising=False)
    assert ball_model._resolve_device(_fake_torch(cuda=True)) == "cuda"


def test_resolve_device_auto_without_cuda_is_cpu_never_mps(monkeypatch):
    # MPS available but auto must not pick it (8 GB unified-memory machines).
    monkeypatch.delenv("BALL_DEVICE", raising=False)
    assert ball_model._resolve_device(_fake_torch(mps=True)) == "cpu"


def test_resolve_device_explicit_mps_honored(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "mps")
    assert ball_model._resolve_device(_fake_torch(mps=True)) == "mps"


def test_resolve_device_explicit_mps_unavailable_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "mps")
    with pytest.raises(RuntimeError, match="MPS"):
        ball_model._resolve_device(_fake_torch(mps=False))


def test_resolve_device_explicit_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="CUDA|cuda"):
        ball_model._resolve_device(_fake_torch(cuda=False))


def test_resolve_device_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "tpu")
    with pytest.raises(ValueError, match="tpu"):
        ball_model._resolve_device(_fake_torch())


def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "cpu")
    assert ball_model._resolve_device(_fake_torch(cuda=True)) == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q -k resolve_device`
Expected: 7 failures, `AttributeError: module 'ball_model' has no attribute '_resolve_device'`

- [ ] **Step 3: Implement** in `ball_model.py`, above `load_detector`:

```python
def _resolve_device(torch_module):
    """BALL_DEVICE -> torch device string.

    Auto (unset or "auto") picks CUDA when available, else CPU. It never
    picks MPS: HRNet over 32-tile batches on an 8 GB unified-memory machine
    is a realistic memory-pressure panic, so MPS is opt-in only
    (BALL_DEVICE=mps). An explicitly requested device that is unavailable
    raises rather than falling back -- a silent CPU fallback would make a
    "fast" run quietly 50x slower.
    """
    configured = os.environ.get("BALL_DEVICE", "").strip().lower()
    if configured in ("", "auto"):
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cpu":
        return "cpu"
    if configured == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("BALL_DEVICE=cuda but torch reports no CUDA device")
        return "cuda"
    if configured == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("BALL_DEVICE=mps but torch reports MPS unavailable")
        return "mps"
    raise ValueError(
        f"Unknown BALL_DEVICE {configured!r}; expected auto, cpu, cuda or mps")
```

Change both runner constructors (`TorchScriptRunner.__init__` and `HeatmapRunner.__init__`) from `(self, module, manifest, torch_module)` to `(self, module, manifest, torch_module, device="cpu")` and add `self.device = device` in each. In both `run_batch` methods, after `tensor = torch.from_numpy(...).permute(0, 3, 1, 2).contiguous()`, add:

```python
        if self.device != "cpu":
            tensor = tensor.to(self.device)
```

(Results already come back via `.detach().cpu()` — no change on the output side.)

In `load_detector`, replace the module-load block with:

```python
    device = _resolve_device(torch)
    module = torch.jit.load(str(manifest.artifact_path), map_location="cpu")
    module.eval()
    if device != "cpu":
        module = module.to(device)
    if manifest.decode == "heatmap_peak":
        runner = HeatmapRunner(module, manifest, torch, device=device)
    else:
        runner = TorchScriptRunner(module, manifest, torch, device=device)
```

Add to `load_detector`'s docstring: `BALL_DEVICE is read once per model_dir (the runner is cached); changing it mid-process has no effect.`

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: all pass, 1 deselected (`requires_model`)

- [ ] **Step 5: Commit**

```bash
git add ball_model.py tests/test_ball_model.py
git commit -m "feat(ball_model): BALL_DEVICE device resolution — CUDA auto, MPS opt-in"
```

---

### Task 2: Default model dir → the committed WASB artifact

**Files:**
- Modify: `ball_model.py:19` (`DEFAULT_MODEL_DIR`)
- Modify: `tests/test_ball_model.py` (the `requires_model` test + new default-manifest test)

**Interfaces:**
- Produces: `ball_model.load_manifest()` / `load_detector()` with no args now resolve to `models/crosscourt-wasb-416-v1`. Task 5 relies on `load_detector()` finding the committed artifact by default.

- [ ] **Step 1: Write the failing test** (append to `tests/test_ball_model.py`)

```python
def test_default_model_dir_is_the_committed_wasb_artifact(monkeypatch):
    """The default must point at an artifact that ships with the repo --
    the YOLOX dir is gitignored and absent on every fresh clone."""
    monkeypatch.delenv("BALL_MODEL_DIR", raising=False)
    manifest = ball_model.load_manifest()
    assert manifest.name == "crosscourt-wasb-416"
    assert manifest.schema_version == "ball-model-v2"
    assert manifest.frames_per_input == 3
    assert manifest.decode == "heatmap_peak"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q -k default_model_dir`
Expected: FAIL with `FileNotFoundError: No ball-model manifest at .../models/crosscourt-ball-416-v1/manifest.json`

- [ ] **Step 3: Implement**

In `ball_model.py` line 19:

```python
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "crosscourt-wasb-416-v1"
```

Update the `requires_model` test (`test_torchscript_runner_returns_boxes_for_real_model`, `tests/test_ball_model.py:251`) to be schema-aware — the default artifact is now temporal, so the crop's channel count comes from the manifest:

```python
@pytest.mark.requires_model
def test_runner_returns_boxes_for_real_model():
    runner = ball_model.load_detector()
    channels = 3 * runner.manifest.frames_per_input
    crops = [np.zeros((416, 416, channels), dtype=np.uint8)]
    result = runner.run_batch(crops)
    assert len(result) == 1
    assert isinstance(result[0], list)
```

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_ball_model.py -q`
Expected: all pass, 1 deselected. Also run the deselected one once on this machine (torch + artifact both present): `.venv/bin/python -m pytest tests/test_ball_model.py -q -m requires_model` — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add ball_model.py tests/test_ball_model.py
git commit -m "feat(ball_model): default model dir is the committed WASB artifact"
```

---

### Task 3: Temporal mode for the decode producer

**Files:**
- Modify: `job_runner.py` (`decode_segments_to_queue`, around line 300)
- Test: `tests/test_job_runner.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `decode_segments_to_queue(video_path, segments, frame_queue, stop_event, decode_errors, temporal=False)`. With `temporal=True` every queue item is `(center_idx, [prev, cur, nxt])` (3 BGR frames, consecutive, edge-padded by repetition); with `temporal=False` items stay `(frame_idx, frame)`. Task 4's consumer branches on this payload shape.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_job_runner.py`; `_write_clip` already writes frame `i` as solid value `i * 20`, so pixel value identifies the frame)

```python
def _drain_decode(video_path, segments, temporal):
    import queue as queue_module
    import threading

    frame_queue = queue_module.Queue()
    stop_event = threading.Event()
    errors = []
    job_runner.decode_segments_to_queue(
        video_path, segments, frame_queue, stop_event, errors,
        temporal=temporal,
    )
    assert errors == []
    items = []
    while True:
        item = frame_queue.get_nowait()
        if item is None:
            return items
        items.append(item)


def _frame_value(frame):
    return int(frame[0, 0, 0])


def test_temporal_decode_stride1_sliding_windows_and_edge_padding(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=5)
    items = _drain_decode(video, [(0, 4, 1)], temporal=True)
    assert [idx for idx, _ in items] == [0, 1, 2, 3, 4]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[0] == [0, 0, 20]        # left edge pads prev with cur
    assert values[2] == [20, 40, 60]      # interior: true neighbours
    assert values[4] == [60, 80, 80]      # right edge pads nxt with cur


def test_temporal_decode_stride4_centers_get_true_neighbours(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=10)
    items = _drain_decode(video, [(0, 9, 4)], temporal=True)
    assert [idx for idx, _ in items] == [0, 4, 8]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[0] == [0, 0, 20]          # first center: padded prev, true nxt
    assert values[4] == [60, 80, 100]       # strided center: TRUE t-1/t+1, not t-4/t+4
    assert values[8] == [140, 160, 180]


def test_temporal_decode_stride2_shared_neighbours(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=5)
    items = _drain_decode(video, [(0, 4, 2)], temporal=True)
    assert [idx for idx, _ in items] == [0, 2, 4]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[2] == [20, 40, 60]
    assert values[4] == [60, 80, 80]


def test_temporal_decode_resets_across_segments(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=12)
    items = _drain_decode(video, [(0, 3, 1), (8, 11, 1)], temporal=True)
    assert [idx for idx, _ in items] == [0, 1, 2, 3, 8, 9, 10, 11]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[3] == [40, 60, 60]        # segment end pads, never crosses
    assert values[8] == [160, 160, 180]     # new segment starts padded


def test_non_temporal_decode_payload_unchanged(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=6)
    items = _drain_decode(video, [(0, 5, 2)], temporal=False)
    assert [idx for idx, _ in items] == [0, 2, 4]
    assert all(isinstance(frame, np.ndarray) for _, frame in items)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q -k temporal_decode or non_temporal_decode`
Expected: FAIL with `TypeError: decode_segments_to_queue() got an unexpected keyword argument 'temporal'`

- [ ] **Step 3: Implement** — replace `decode_segments_to_queue` in `job_runner.py`:

```python
def decode_segments_to_queue(video_path, segments, frame_queue, stop_event,
                             decode_errors, temporal=False):
    """Producer thread: decode (start, end, stride) segments into frame_queue.

    temporal=False enqueues (frame_idx, frame) -- the single-frame payload
    the rfdetr backend consumes. temporal=True enqueues
    (center_idx, [prev, cur, nxt]) for each strided center: CONSECUTIVE
    frames (the temporal model was trained on t-1/t/t+1, never t-s/t/t+s),
    decoded with read(); frames no window needs are skipped with grab()
    exactly as before. Segment edges pad by repeating the first/last frame,
    mirroring ball_track_offline._centered_windows so serving matches
    training, and state resets per segment so a window never spans a
    segment boundary.
    """

    def enqueue(item):
        while not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def emit_window(center_idx, prev, center, nxt):
        enqueue((center_idx, [prev if prev is not None else center,
                              center,
                              nxt if nxt is not None else center]))

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        for seg_start, seg_end, stride in segments:
            if stop_event.is_set():
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            read_count = seg_start
            last_decoded = None   # (idx, frame) -- prev-adjacency check
            pending = None        # (center_idx, prev, frame) awaiting its nxt

            while read_count <= seg_end and not stop_event.is_set():
                offset = (read_count - seg_start) % stride
                is_center = offset == 0
                is_neighbour = stride >= 2 and offset in (1, stride - 1)
                wanted = is_center or (temporal and (stride == 1 or is_neighbour))
                if not wanted:
                    if not cap.grab():
                        break
                    read_count += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                if not temporal:
                    enqueue((read_count, frame))
                    read_count += 1
                    continue

                if pending is not None:
                    center_idx, prev, center = pending
                    nxt = frame if read_count == center_idx + 1 else None
                    emit_window(center_idx, prev, center, nxt)
                    pending = None

                if is_center:
                    prev = None
                    if last_decoded is not None and last_decoded[0] == read_count - 1:
                        prev = last_decoded[1]
                    pending = (read_count, prev, frame)

                last_decoded = (read_count, frame)
                read_count += 1

            if temporal and pending is not None:
                center_idx, prev, center = pending
                emit_window(center_idx, prev, center, None)
    except Exception as error:
        decode_errors.append(error)
    finally:
        cap.release()
        enqueue(None)
```

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q`
Expected: all pass (existing tests unaffected — default `temporal=False`)

- [ ] **Step 5: Commit**

```bash
git add job_runner.py tests/test_job_runner.py
git commit -m "feat(job_runner): temporal decode mode — strided centers with true neighbours"
```

---

### Task 4: Backend branch in track_segments

**Files:**
- Modify: `job_runner.py` (`track_segments`, line ~356; imports at top)
- Test: `tests/test_job_runner.py`

**Interfaces:**
- Consumes: Task 3's payload shapes; `ball_detector.detect_frame(runner, frame, manifest)` and `detect_frame_stack(runner, frames, manifest)` (both return normalized prediction dicts).
- Produces: `track_segments(model, video_path, segments, inference_width, source_fps, results, on_frame, frame_observer=None, backend="rfdetr")`. On `backend="local"`, `model` is a ball_model runner (`.manifest` present): temporal manifests (frames_per_input 3) use `detect_frame_stack`, single-frame manifests use `detect_frame`, `inference_width` is ignored (native-res tiling), and the selection floor is `model.manifest.conf_threshold`. Task 5's call sites pass `backend=`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_job_runner.py`)

```python
class _StubBallRunner:
    """A ball_model-runner stand-in: manifest + recording run_batch."""

    class _Manifest:
        frames_per_input = 3
        conf_threshold = 0.1
        input_size = (416, 416)
        tile_overlap_px = 64
        max_batch_tiles = 32
        nms_iou = 0.45
        class_names = ("ball",)
        name = "stub-wasb"
        version = 1
        artifact_sha256 = "deadbeef"

    def __init__(self):
        self.manifest = self._Manifest()
        self.batches = []
        self.device = "cpu"

    def run_batch(self, stacks):
        self.batches.append([s.shape for s in stacks])
        return [[] for _ in stacks]


def test_track_segments_local_temporal_feeds_stacks_and_observes_centers(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=8)
    runner = _StubBallRunner()
    results = {}
    observed = []
    job_runner.track_segments(
        runner, video, [(0, 7, 4)], 640, 30.0, results,
        on_frame=lambda idx: None,
        frame_observer=lambda idx, frame: observed.append(
            (idx, int(frame[0, 0, 0]))),
        backend="local",
    )
    # Observer fired once per CENTER frame with the center frame itself.
    assert observed == [(0, 0), (4, 80)]
    # Every stack reaching the runner is one 9-channel tile (64x48 clip
    # is smaller than one 416 tile, zero-padded).
    for batch in runner.batches:
        for shape in batch:
            assert shape == (416, 416, 9)
    # No detections -> empty rows for the two centers.
    assert sorted(results) == [0, 4]
    assert all(results[idx]["detected"] is False for idx in results)


def test_track_segments_local_uses_manifest_confidence_floor(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=4)
    runner = _StubBallRunner()
    floors = []

    def spy_select(predictions_by_frame, confidence_threshold, **kwargs):
        floors.append(confidence_threshold)
        return {frame: None for frame in predictions_by_frame}

    monkeypatch.setattr(
        job_runner, "select_motion_consistent_ball_predictions", spy_select)
    job_runner.track_segments(
        runner, video, [(0, 3, 1)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="local")
    assert floors == [pytest.approx(0.1)]

    job_runner.track_segments(
        object(), video, [(0, 3, 1)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="rfdetr")
    assert floors[-1] == pytest.approx(0.40)


def test_track_segments_local_single_frame_manifest_uses_detect_frame(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=4)
    runner = _StubBallRunner()
    runner.manifest.frames_per_input = 1
    calls = []
    monkeypatch.setattr(
        job_runner, "detect_frame",
        lambda model, frame, manifest: calls.append(frame.shape) or [])
    job_runner.track_segments(
        runner, video, [(0, 3, 2)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="local")
    assert len(calls) == 2          # frames 0 and 2 (stride 2), full frames


def test_track_segments_rejects_unsupported_frames_per_input(tmp_path):
    runner = _StubBallRunner()
    runner.manifest.frames_per_input = 5
    with pytest.raises(ValueError, match="frames_per_input"):
        job_runner.track_segments(
            runner, "unused.mp4", [(0, 3, 1)], 640, 30.0, {},
            on_frame=lambda idx: None, backend="local")


def test_track_segments_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        job_runner.track_segments(
            object(), "unused.mp4", [(0, 3, 1)], 640, 30.0, {},
            on_frame=lambda idx: None, backend="coreml")
```

Note for the rfdetr spy case: `infer_frame_predictions` is monkeypatched by `_stub_pipeline` in other tests; here the rfdetr call runs with the real function against `object()` — to keep it inert, monkeypatch it in that test before the second `track_segments` call:

```python
    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
```

(add this line just above the second `track_segments` call in `test_track_segments_local_uses_manifest_confidence_floor`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q -k track_segments`
Expected: FAIL with `TypeError: track_segments() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Implement**

Add to `job_runner.py` imports (after line 28's bounce imports):

```python
from ball_detector import detect_frame, detect_frame_stack
```

Replace `track_segments` (keeping the existing docstring's decode-overlap paragraph, extended):

```python
def track_segments(model, video_path, segments, inference_width, source_fps, results, on_frame,
                   frame_observer=None, backend="rfdetr"):
    """Consumer loop: infer frames from the decode queue into `results`.

    Decode runs on its own thread so it overlaps inference, which dominates.
    `frame_observer(frame_idx, frame)`, when given, is called for every
    decoded CENTER frame in this call -- callers wire it only on the coarse
    pass so the person-detection cadence rides the coarse decode, never
    refine or audio-rescue (spec §4.2). Neighbour frames on the temporal
    path are invisible to observers.

    backend "rfdetr" (default) is the historical per-frame path. backend
    "local" runs a ball_model runner (`model` has .manifest) at NATIVE
    resolution -- inference_width is deliberately ignored, tiling owns scale
    (ios/MODEL.md §6) -- and a temporal manifest (frames_per_input == 3)
    consumes the 3-frame windows the temporal producer emits. The selection
    floor becomes the manifest's own conf_threshold: the motion-consistency
    selector is the low-confidence rescue mechanism (the ByteTrack idea,
    absorbed), so the 0.40 rfdetr floor would silently discard the recall
    the model was fine-tuned to recover.
    """
    temporal = False
    confidence_floor = CONFIDENCE_THRESHOLD
    if backend == "local":
        frames_per_input = int(model.manifest.frames_per_input)
        if frames_per_input not in (1, 3):
            raise ValueError(
                f"job_runner supports frames_per_input 1 or 3, got "
                f"{frames_per_input}; widening the window is a deliberate "
                f"change, not a default.")
        temporal = frames_per_input == 3
        confidence_floor = float(model.manifest.conf_threshold)
    elif backend != "rfdetr":
        raise ValueError(f"Unknown ball backend {backend!r}")

    # A temporal item holds 3 frames; shrink the queue so worst-case
    # buffered frame count stays in the same ballpark.
    queue_size = max(2, DECODE_QUEUE_SIZE // 3) if temporal else DECODE_QUEUE_SIZE
    frame_queue = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    decode_errors = []
    decoder = threading.Thread(
        target=decode_segments_to_queue,
        args=(video_path, segments, frame_queue, stop_event, decode_errors),
        kwargs={"temporal": temporal},
        daemon=True,
    )
    decoder.start()

    raw_predictions = {}
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break

            frame_idx, payload = item
            center = payload[1] if temporal else payload
            if frame_observer is not None:
                frame_observer(frame_idx, center)
            if backend == "local":
                if temporal:
                    predictions = detect_frame_stack(model, payload, model.manifest)
                else:
                    predictions = detect_frame(model, payload, model.manifest)
            else:
                predictions = infer_frame_predictions(
                    model,
                    payload,
                    CONFIDENCE_THRESHOLD,
                    inference_width,
                )
            raw_predictions[frame_idx] = predictions
            on_frame(frame_idx)
    finally:
        stop_event.set()
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        decoder.join(timeout=5)

    if decode_errors:
        raise decode_errors[0]

    selected_predictions = select_motion_consistent_ball_predictions(
        raw_predictions,
        confidence_floor,
        window_frames=scaled_window_frames(source_fps),
    )
    for frame_idx, ball_prediction in selected_predictions.items():
        results[frame_idx] = ball_csv_row(frame_idx, source_fps, ball_prediction)
```

(The `frames_per_input`/backend validation runs before the decoder thread starts, so the two `rejects_*` tests never touch the fake video path.)

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add job_runner.py tests/test_job_runner.py
git commit -m "feat(job_runner): local ball backend branch in track_segments — native res, manifest floor"
```

---

### Task 5: Backend selection, call-site wiring, attribution

**Files:**
- Modify: `job_runner.py` (imports; new `load_ball_backend` + `ball_backend_summary`; the load site at ~line 1564; `backend=` on the three `track_segments` calls at ~1579/1639/1686)
- Test: `tests/test_job_runner.py` (including `_stub_pipeline`)

**Interfaces:**
- Consumes: `ball_track_offline.selected_detector() -> "local" | "rfdetr"` (raises on unknown); `ball_model.load_detector()`; Task 4's `backend=` parameter; `inference_engine.DEFAULT_MODEL_ID`.
- Produces: `job_runner.load_ball_backend() -> (backend: str, model)`; `job_runner.ball_backend_summary(backend, model) -> dict` — the exact block Task 6 passes through to report-v1: local → `{"backend","name","version","artifact_sha256","device"}`, rfdetr → `{"backend","model_id"}`. The job dict gains key `ball_backend`.

- [ ] **Step 1: Update `_stub_pipeline` and write the failing tests**

`_stub_pipeline` keeps every existing test on the historical path — add one line at its top:

```python
def _stub_pipeline(monkeypatch):
    """No ball model, no audio -- both are off-topic for this test, and
    keeping them out avoids pulling in real model weights."""
    monkeypatch.setenv("BALL_DETECTOR", "rfdetr")
    ...existing body unchanged...
```

Append the new tests:

```python
def test_load_ball_backend_local_never_touches_roboflow(monkeypatch):
    monkeypatch.delenv("BALL_DETECTOR", raising=False)   # default is local
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    stub = _StubBallRunner()
    monkeypatch.setattr(job_runner.ball_model, "load_detector", lambda: stub)

    def explode():
        raise AssertionError("rfdetr path must not load on the local backend")

    monkeypatch.setattr(job_runner, "get_tracking_model", explode)
    backend, model = job_runner.load_ball_backend()
    assert backend == "local"
    assert model is stub


def test_load_ball_backend_rfdetr_branch(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "rfdetr")
    sentinel = object()
    monkeypatch.setattr(job_runner, "get_tracking_model", lambda: sentinel)
    backend, model = job_runner.load_ball_backend()
    assert backend == "rfdetr"
    assert model is sentinel


def test_load_ball_backend_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "coreml")
    with pytest.raises(ValueError, match="coreml"):
        job_runner.load_ball_backend()


def test_ball_backend_summary_shapes():
    local = job_runner.ball_backend_summary("local", _StubBallRunner())
    assert local == {
        "backend": "local", "name": "stub-wasb", "version": 1,
        "artifact_sha256": "deadbeef", "device": "cpu",
    }
    hosted = job_runner.ball_backend_summary("rfdetr", object())
    assert hosted["backend"] == "rfdetr"
    assert hosted["model_id"]          # squashai/1 or ROBOFLOW_MODEL_ID


def test_run_tracking_job_local_backend_end_to_end(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-local-ball"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)

    monkeypatch.delenv("BALL_DETECTOR", raising=False)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setenv("PERSON_DETECTOR", "none")
    stub = _StubBallRunner()
    monkeypatch.setattr(job_runner.ball_model, "load_detector", lambda: stub)
    monkeypatch.setattr(
        job_runner, "extract_audio_candidates",
        lambda video_path, start_frame, end_frame, fps: [])

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["ball_backend"]["backend"] == "local"
    assert job["ball_backend"]["name"] == "stub-wasb"
    assert stub.batches            # the stub actually ran
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q -k "load_ball_backend or ball_backend_summary or local_backend_end_to_end"`
Expected: FAIL with `AttributeError: module 'job_runner' has no attribute 'load_ball_backend'` (and `ball_model` attribute errors)

- [ ] **Step 3: Implement**

Imports in `job_runner.py` — extend the existing block:

```python
import ball_model
from ball_track_offline import selected_detector
from inference_engine import DEFAULT_MODEL_ID, get_tracking_model, infer_frame_predictions
```

Add near `track_segments`:

```python
def load_ball_backend():
    """Resolve BALL_DETECTOR into (backend, model) for this job.

    "local" (the default) loads the committed WASB artifact via
    ball_model.load_detector(); "rfdetr" keeps the hosted tracking model --
    the only branch that needs ROBOFLOW_API_KEY. Loud on every failure;
    never falls back, because a silent swap would make the local/rfdetr
    split invisible in every downstream number.
    """
    backend = selected_detector()
    if backend == "rfdetr":
        return backend, get_tracking_model()
    return backend, ball_model.load_detector()


def ball_backend_summary(backend, model):
    """The attribution block stored on the job and report (spec §7):
    a run's numbers are attributable to the detector and device that
    produced them, like players_v2.backend."""
    if backend == "local":
        manifest = model.manifest
        return {
            "backend": "local",
            "name": manifest.name,
            "version": manifest.version,
            "artifact_sha256": manifest.artifact_sha256,
            "device": getattr(model, "device", "cpu"),
        }
    return {
        "backend": "rfdetr",
        "model_id": os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID),
    }
```

At the load site (line ~1563), replace:

```python
            update_job(run_id, status="running", stage="coarse", message="Loading local model...")
            model = get_tracking_model()
```

with:

```python
            update_job(run_id, status="running", stage="coarse", message="Loading ball detector...")
            ball_backend, model = load_ball_backend()
            summary = ball_backend_summary(ball_backend, model)
            label = (
                f"local ({summary['name']} v{summary['version']}, {summary['device']})"
                if ball_backend == "local"
                else f"rfdetr ({summary['model_id']})"
            )
            update_job(run_id, ball_backend=summary,
                       message=f"Ball detector: {label}")
```

Add `backend=ball_backend` to all three `track_segments(...)` calls (coarse ~1579, refine ~1639, audio rescue ~1686). On the coarse call, extend the width comment:

```python
            # A pass at stride > 1 only needs to be good enough to locate hit
            # candidates, so it can also run at a reduced inference width.
            # (rfdetr only: the local backend tiles at native resolution and
            # ignores inference_width entirely -- MODEL.md §6.)
```

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_job_runner.py -q`
Expected: all pass (existing tests ride `_stub_pipeline`'s new `BALL_DETECTOR=rfdetr` pin)

- [ ] **Step 5: Commit**

```bash
git add job_runner.py tests/test_job_runner.py
git commit -m "feat(job_runner): BALL_DETECTOR backend seam, default local WASB, run attribution"
```

---

### Task 6: report-v1 passthrough

**Files:**
- Modify: `match_report.py` (`build_report`, line ~127)
- Test: `tests/test_match_report.py`

**Interfaces:**
- Consumes: Task 5's `ball_backend` job key.
- Produces: report-v1 dict gains `"ball_backend"` (dict or None). Web/iOS clients may render it; absence on legacy runs is expressed as None, matching the report's legacy tolerance.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_match_report.py`, following that file's existing job.json fixture pattern — write a minimal `job.json` into `tmp_path` and call `build_report(tmp_path)`)

```python
def test_report_carries_ball_backend(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({
        "run_id": "r1", "status": "complete",
        "capabilities": {},
        "ball_backend": {"backend": "local", "name": "crosscourt-wasb-416",
                          "version": 1, "artifact_sha256": "abc", "device": "cuda"},
    }), encoding="utf-8")
    report = match_report.build_report(tmp_path)
    assert report["ball_backend"]["backend"] == "local"
    assert report["ball_backend"]["name"] == "crosscourt-wasb-416"


def test_report_ball_backend_none_for_legacy_runs(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({
        "run_id": "r0", "status": "complete",
    }), encoding="utf-8")
    report = match_report.build_report(tmp_path)
    assert report["ball_backend"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_match_report.py -q -k ball_backend`
Expected: FAIL with `KeyError: 'ball_backend'`

- [ ] **Step 3: Implement** — in `build_report`'s return dict, after `"detection_coverage"`:

```python
        "ball_backend": job.get("ball_backend"),
```

- [ ] **Step 4: Run the file's tests**

Run: `.venv/bin/python -m pytest tests/test_match_report.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add match_report.py tests/test_match_report.py
git commit -m "feat(match_report): carry ball_backend attribution into report-v1"
```

---

### Task 7: Full suite, env docs, eval zero-drift gate

**Files:**
- Modify: `.env.example`, `README.md` (pipeline/env section)
- No pipeline code changes in this task.

**Interfaces:** none — verification and documentation.

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, 2 skipped, 1 deselected (the count grows from 605 by the new tests; the `requires_model` deselection is expected)

- [ ] **Step 2: Document the env contract** — append to `.env.example`:

```bash
# Ball detector for analysis jobs. local (default) = the committed WASB
# temporal model in models/crosscourt-wasb-416-v1; rfdetr = the hosted
# Roboflow RF-DETR (needs ROBOFLOW_API_KEY). No silent fallback.
#BALL_DETECTOR=local
# Device for the local ball detector: auto (default; CUDA if available,
# else CPU — never auto-MPS), cpu, cuda, or mps (opt-in).
#BALL_DEVICE=auto
# Override the local ball model directory (defaults to the committed WASB).
#BALL_MODEL_DIR=
```

Update README.md's pipeline description to state: analysis jobs default to the local WASB detector (`BALL_DETECTOR=local`); `ROBOFLOW_API_KEY` is only needed for `BALL_DETECTOR=rfdetr`; every run records `ball_backend` in job.json and report-v1. Add — verbatim, until the eval is run — the sentence: **"The WASB backend is wired but not yet measured; the line-call eval against `eval_set/BASELINE-2026-07-23.md` is the gate for any recall claim."**

- [ ] **Step 3: Eval zero-drift check (rfdetr path untouched)**

Invoke the `/eval` skill and run the line-call axis with `BALL_DETECTOR=rfdetr` against `eval_set/BASELINE-2026-07-23.md`.
Expected: **identical numbers — zero drift.** Any drift is a Task 4/5 regression on the rfdetr branch; stop and fix before proceeding.

- [ ] **Step 4: Record what is and is not measured**

The local-backend eval (the actual measurement) is CPU-hours on this machine — it runs on the CUDA box (`BALL_DEVICE` auto-picks cuda there). Do not run it here; do not claim improvement anywhere. The README sentence from Step 2 is the standing statement.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md
git commit -m "docs: BALL_DETECTOR/BALL_DEVICE env contract; WASB wired, not yet measured"
```

---

## Self-Review Notes

- **Spec coverage:** §1 backend seam → Task 5; §2 default dir → Task 2; §3 temporal decode → Tasks 3–4; §4 native res → Task 4 (width ignored on local); §5 confidence floor → Task 4; §6 device → Task 1; §7 attribution → Tasks 5–6; §8 failure modes → Tasks 1/4/5 raise-tests; §9 test matrix → distributed per task; §10 eval gate → Task 7.
- **Type consistency:** `load_ball_backend() -> (str, model)`; `ball_backend_summary` keys match Task 6's test expectations; `_StubBallRunner` defined in Task 4, reused in Task 5 (same file, defined earlier in it).
- **Suite compatibility:** every pre-existing job_runner test flows through `_stub_pipeline`, which now pins `BALL_DETECTOR=rfdetr`; `select_motion_consistent_ball_predictions` keeps positional `confidence_threshold`, so the spy signature matches.
