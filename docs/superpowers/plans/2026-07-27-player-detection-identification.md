# Player Detection & Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Record any divergence in `## Deviations` at the bottom.

**Goal:** Observed player attribution — RF-DETR Keypoint person detection riding the
coarse decode, anonymous A/B tracking, serve-anchored hit attribution with strict
within-rally alternation, winner chain from next-rally serves, and post-hoc naming —
per `docs/superpowers/specs/2026-07-27-player-detection-identification-design.md`.

**Architecture:** Three new pure modules (`person_model.py`, `player_tracker.py`,
`player_attribution.py`) plus a frame-observer seam in `job_runner.track_segments`
and a `serve_resolver` parameter threaded through `assign_front_wall_hit_players`.
The pipeline stays name-agnostic (tracks "A"/"B"); names are post-hoc run metadata
via one new POST route. No calibration/homography dependency anywhere in this slice.

**Tech Stack:** Python 3.12 in `.venv` (torch 2.13 + MPS, cv2 4.10, numpy, flask),
new pinned `rfdetr` dependency (Apache-2.0, **lazy-imported only**), vanilla JS in
`index.html`.

## Global Constraints

- Branch: work on the current worktree branch. One commit per task. Never push.
- Every command runs through `.venv/bin/python` — **the venv lives in the main
  checkout**, so from this worktree use
  `VENV=$(git rev-parse --path-format=absolute --git-common-dir)/../.venv` or the
  literal path `/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv`.
  System python has no flask/cv2.
- Full suite `<venv>/bin/python -m pytest tests/ -q` green before every commit.
  Expected shape: "N passed, 1 deselected" (the deselection is `requires_model`,
  expected — CLAUDE.md).
- **The test suite must never import `rfdetr`** (or torch beyond what it already
  does). All rfdetr imports are lazy inside function/constructor bodies, pattern:
  `yolo_model_eval.py:63`.
- Editing a `*.py` with a paired `tests/test_*.py` auto-runs that pair (PostToolUse
  hook); failures come back as a blocked edit. `job_runner.py` has no paired test
  file — its assignment tests live in `tests/test_pipeline.py`; run that explicitly.
- Existing `public_job` keys keep their shapes; additions only. `players_v1` value
  semantics: observed mode may change `player_number`/`server_player_number`
  *values* — that is the point — and `attribution_backend` says which mode ran.
- Do not touch `ios/`, `archive/`, or judged line-call outputs. `/eval` must show
  zero drift (this slice never touches `judge_call.py` or calibration).
- New thresholds are named module-level constants, each exercised by a test.
- app.py route style is `@app.get(...)`/`@app.post(...)`, error helper
  `error_response(...)`, run dirs resolved as `RUNS_DIR / secure_filename(run_id)`.
- UI work: DESIGN.md is binding; verify both themes at a phone viewport via the
  `/verify` skill before checking the UI task off.
- Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite` (CLAUDE.md).
  All new artifact filenames here are fixed ASCII literals.

---

## Task 1: `person_model.py` — RF-DETR Keypoint seam

**Files:**
- Create: `person_model.py`
- Create: `tests/test_person_model.py`
- Create: `docs/PERSON_MODEL.md`
- Modify: `requirements.txt` (add pinned `rfdetr`)

**Interfaces:**
- Produces (later tasks import these exact names):
  ```python
  PERSON_SCHEMA_VERSION = "person-model-v1"
  PERSON_CONFIDENCE_THRESHOLD = 0.5
  COCO_KEYPOINT_NAMES  # tuple of 17 names, index-aligned with model output

  @dataclass(frozen=True)
  class PersonDetection:
      x: float          # bbox CENTER x, px (matches the ball prediction convention)
      y: float          # bbox CENTER y, px
      width: float
      height: float
      confidence: float
      keypoints: tuple  # 17 × (x_px, y_px, confidence), or () when unavailable

      @property
      def foot_px(self) -> tuple[float, float]   # (x, y + height / 2)

  def keypoints_result_to_detections(xyxy, det_conf, kp_xy, kp_conf,
                                     threshold=PERSON_CONFIDENCE_THRESHOLD) -> list[PersonDetection]
  def available_backend() -> str                 # "rfdetr" | "none"
  def load_person_detector()                     # RFDETRPersonDetector | None
  def save_person_crop(video_path, frame_idx, detection, out_path,
                       pad_ratio=CROP_PAD_RATIO) -> bool
  ```

- [ ] **Step 1: Install and pin rfdetr**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/pip" install rfdetr
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/pip" show rfdetr | head -2
```

Append to `requirements.txt` (replace `X.Y.Z` with the version `pip show` printed):

```
# Person detection (player attribution). Apache-2.0. Lazy-imported in
# person_model.py only — the test suite must run without it.
rfdetr==X.Y.Z
```

- [ ] **Step 2: Write the failing tests**

`tests/test_person_model.py`:

```python
"""person_model: RF-DETR adapter, backend gating, crop helper.

No rfdetr import may happen at test time — the adapter math and gating are
exercised pure, and detection loading is stubbed."""

import numpy as np
import pytest

import person_model
from person_model import (
    PERSON_CONFIDENCE_THRESHOLD,
    PersonDetection,
    keypoints_result_to_detections,
)


def test_adapter_converts_xyxy_to_center_boxes_and_keypoints():
    xyxy = np.array([[100.0, 200.0, 180.0, 420.0]])
    det_conf = np.array([0.9])
    kp_xy = np.zeros((1, 17, 2)) + 5.0
    kp_conf = np.ones((1, 17)) * 0.7
    detections = keypoints_result_to_detections(xyxy, det_conf, kp_xy, kp_conf)
    assert len(detections) == 1
    det = detections[0]
    assert det.x == pytest.approx(140.0)
    assert det.y == pytest.approx(310.0)
    assert det.width == pytest.approx(80.0)
    assert det.height == pytest.approx(220.0)
    assert det.confidence == pytest.approx(0.9)
    assert len(det.keypoints) == 17
    assert det.keypoints[0] == (5.0, 5.0, 0.7)
    assert det.foot_px == (pytest.approx(140.0), pytest.approx(420.0))


def test_adapter_filters_below_threshold_and_handles_missing_keypoints():
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 60.0]])
    det_conf = np.array([PERSON_CONFIDENCE_THRESHOLD - 0.1,
                         PERSON_CONFIDENCE_THRESHOLD + 0.1])
    detections = keypoints_result_to_detections(xyxy, det_conf, None, None)
    assert len(detections) == 1
    assert detections[0].keypoints == ()


def test_available_backend_is_none_when_rfdetr_missing(monkeypatch):
    monkeypatch.setattr(person_model, "_import_rfdetr", lambda: None)
    assert person_model.available_backend() == "none"
    assert person_model.load_person_detector() is None


def test_available_backend_env_kill_switch(monkeypatch):
    monkeypatch.setenv("PERSON_DETECTOR", "none")
    assert person_model.available_backend() == "none"


def test_save_person_crop(tmp_path):
    import cv2
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), i * 10, dtype=np.uint8)
        frame[100:200, 140:180] = 255
        writer.write(frame)
    writer.release()

    det = PersonDetection(x=160.0, y=150.0, width=40.0, height=100.0,
                          confidence=0.9, keypoints=())
    out_path = tmp_path / "crop.jpg"
    assert person_model.save_person_crop(video, 5, det, out_path) is True
    crop = cv2.imread(str(out_path))
    assert crop is not None
    assert crop.shape[0] > 100  # padded beyond the raw bbox height
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_person_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'person_model'`.

- [ ] **Step 4: Implement `person_model.py`**

```python
"""Person detection seam for player attribution.

RF-DETR Keypoint (Apache-2.0) is the only real backend. rfdetr is imported
lazily so the test suite and any environment without the package still import
this module. When the backend is unavailable the pipeline falls back to the
assumed-alternation attribution and says so (spec §4.1 — never silently).
"""

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

PERSON_SCHEMA_VERSION = "person-model-v1"
PERSON_CONFIDENCE_THRESHOLD = 0.5
CROP_PAD_RATIO = 0.25  # padding added around a bbox crop, fraction of each side

COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class PersonDetection:
    x: float          # bbox center x, px
    y: float          # bbox center y, px
    width: float
    height: float
    confidence: float
    keypoints: tuple  # 17 x (x_px, y_px, confidence), or () when unavailable

    @property
    def foot_px(self):
        return (self.x, self.y + self.height / 2.0)


def keypoints_result_to_detections(xyxy, det_conf, kp_xy, kp_conf,
                                   threshold=PERSON_CONFIDENCE_THRESHOLD):
    """sv.KeyPoints arrays -> PersonDetection list. Pure numpy; no rfdetr."""
    detections = []
    if xyxy is None or det_conf is None:
        return detections
    xyxy = np.asarray(xyxy, dtype=float)
    det_conf = np.asarray(det_conf, dtype=float)
    for index in range(xyxy.shape[0]):
        confidence = float(det_conf[index])
        if confidence < threshold:
            continue
        x_min, y_min, x_max, y_max = (float(v) for v in xyxy[index])
        keypoints = ()
        if kp_xy is not None and kp_conf is not None:
            kp_xy_arr = np.asarray(kp_xy, dtype=float)
            kp_conf_arr = np.asarray(kp_conf, dtype=float)
            keypoints = tuple(
                (float(kp_xy_arr[index, k, 0]),
                 float(kp_xy_arr[index, k, 1]),
                 float(kp_conf_arr[index, k]))
                for k in range(kp_xy_arr.shape[1])
            )
        detections.append(PersonDetection(
            x=(x_min + x_max) / 2.0,
            y=(y_min + y_max) / 2.0,
            width=x_max - x_min,
            height=y_max - y_min,
            confidence=confidence,
            keypoints=keypoints,
        ))
    return detections


def _import_rfdetr():
    """Lazy rfdetr import. Returns the model class or None."""
    try:
        from rfdetr import RFDETRKeypointPreview  # noqa: PLC0415 — deliberate lazy import
    except Exception:
        return None
    return RFDETRKeypointPreview


class RFDETRPersonDetector:
    backend = "rfdetr"

    def __init__(self, model_class):
        # First construction downloads the COCO checkpoint (docs/PERSON_MODEL.md).
        self._model = model_class()

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._model.predict(frame_rgb, threshold=PERSON_CONFIDENCE_THRESHOLD)
        xyxy = None
        data = getattr(result, "data", None) or {}
        if "xyxy" in data:
            xyxy = data["xyxy"]
        return keypoints_result_to_detections(
            xyxy,
            getattr(result, "detection_confidence", None),
            getattr(result, "xy", None),
            getattr(result, "keypoint_confidence", None),
        )


def available_backend():
    if os.getenv("PERSON_DETECTOR", "").strip().lower() == "none":
        return "none"
    return "rfdetr" if _import_rfdetr() is not None else "none"


def load_person_detector():
    if os.getenv("PERSON_DETECTOR", "").strip().lower() == "none":
        return None
    model_class = _import_rfdetr()
    if model_class is None:
        return None
    return RFDETRPersonDetector(model_class)


def save_person_crop(video_path, frame_idx, detection, out_path,
                     pad_ratio=CROP_PAD_RATIO):
    """Seek one frame and write the padded bbox crop. ASCII out_path only
    (CLAUDE.md: cv2.imwrite must never see a non-ASCII path)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return False

    height, width = frame.shape[:2]
    pad_x = detection.width * pad_ratio
    pad_y = detection.height * pad_ratio
    x_min = max(0, int(detection.x - detection.width / 2 - pad_x))
    x_max = min(width, int(detection.x + detection.width / 2 + pad_x))
    y_min = max(0, int(detection.y - detection.height / 2 - pad_y))
    y_max = min(height, int(detection.y + detection.height / 2 + pad_y))
    if x_max <= x_min or y_max <= y_min:
        return False

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame[y_min:y_max, x_min:x_max]))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_person_model.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Write `docs/PERSON_MODEL.md`**

```markdown
# Person model (player attribution)

Backend: **RF-DETR Keypoint** — `rfdetr` pip package (Apache-2.0), class
`RFDETRKeypointPreview`, COCO-pretrained, 17 body keypoints per person.
Pinned in `requirements.txt`; imported lazily in `person_model.py` only.
The test suite must run without the package installed.

## Checkpoint provisioning

The first `RFDETRKeypointPreview()` construction downloads the pretrained
checkpoint to the rfdetr cache (per-user cache dir). On a server, warm it
once after installing requirements:

    .venv/bin/python -c "from rfdetr import RFDETRKeypointPreview; RFDETRKeypointPreview()"

Record the installed `rfdetr` version here when bumping the pin, and re-run
`eval_set/BASELINE-ATTRIBUTION-*.md` (see eval_attribution.py) before
trusting a new checkpoint.

## Disabling

`PERSON_DETECTOR=none` disables person detection entirely; runs fall back to
assumed-alternation attribution and the payload reports
`attribution_backend: "assumed"` (spec §4.1).

## Conventions

`PersonDetection` boxes are **center-based** (x, y = bbox center), matching
the ball prediction dicts in `tracking_common.py`. Foot point =
(x, y + height/2). Keypoints are stored on every detection from day one but
v1 consumes only boxes (spec §4.1 — do not strip them).
```

- [ ] **Step 7: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add person_model.py tests/test_person_model.py docs/PERSON_MODEL.md requirements.txt
git commit -m "feat: RF-DETR person-detector seam (lazy import, env kill-switch)"
```

---

## Task 2: `player_tracker.py` — two anonymous tracks + person pass

**Files:**
- Create: `player_tracker.py`
- Create: `tests/test_player_tracker.py`

**Interfaces:**
- Consumes: `person_model.PersonDetection` (Task 1).
- Produces (Tasks 3/5 rely on these exact names):
  ```python
  COAST_MAX_S = 1.0
  AMBIGUITY_MARGIN = 0.2
  PERSON_DETECT_HZ = 4.0

  @dataclass
  class TrackSample:
      t_s: float
      frame_idx: int
      foot_px: tuple      # (x, y)
      bbox: tuple         # (x_center, y_center, width, height)
      confidence: float
      coasted: bool

  class TwoPlayerTracker:
      def update(self, t_s, frame_idx, detections) -> None
      def samples(self) -> dict          # {"A": [TrackSample...], "B": [...]}
      def ambiguity_times(self) -> list  # [t_s, ...] one per ambiguous assignment
      def stats(self) -> dict            # {"updates", "ambiguous_assignments"}

  class PersonFramePass:                 # glue used by job_runner (Task 5)
      def __init__(self, detector, source_fps, frame_stride)
      def observe(self, frame_idx, frame_bgr) -> None
      tracker: TwoPlayerTracker          # attribute
      detect_every: int                  # attribute (computed cadence)
  ```

**Tracker rules (implement exactly):**
- Keep the top-2 detections by confidence per update; ignore the rest
  (spectators through the back glass are real).
- Seeding: the first update with ≥1 detection creates track "A" from the
  leftmost detection (smallest `x`) and, if a second detection exists, "B"
  from the other. A second track seeds later from the first unmatched
  detection once "A" exists. (The letters are arbitrary; naming anchors
  meaning later — spec §4.4.)
- Assignment with 2 live tracks and 2 detections: evaluate both pairings by
  total euclidean foot-point distance; take the cheaper. **Ambiguous** iff
  `min_total >= (1 - AMBIGUITY_MARGIN) * max_total` (costs within 20%);
  record `t_s` in `ambiguity_times`. With 1 detection: assign to the nearer
  track. Never create a third track.
- A track with no detection this update **coasts**: emit a `TrackSample` at
  its last foot position with `coasted=True`, `confidence=0.0`, until the
  gap since its last real sample exceeds `COAST_MAX_S`; after that emit
  nothing for it (no fabrication).

- [ ] **Step 1: Write the failing tests**

`tests/test_player_tracker.py`:

```python
"""player_tracker: two-track assignment, coasting, ambiguity accounting."""

import numpy as np

from person_model import PersonDetection
from player_tracker import (
    AMBIGUITY_MARGIN,
    COAST_MAX_S,
    PersonFramePass,
    TwoPlayerTracker,
)


def det(x, y, confidence=0.9):
    return PersonDetection(x=x, y=y, width=40.0, height=100.0,
                           confidence=confidence, keypoints=())


def test_separated_walkers_keep_identity():
    tracker = TwoPlayerTracker()
    for step in range(50):
        t = step * 0.25
        tracker.update(t, step, [det(100 + step * 2, 300),
                                 det(900 - step * 2, 300)])
    samples = tracker.samples()
    assert len(samples["A"]) == 50 and len(samples["B"]) == 50
    a_x = [s.foot_px[0] for s in samples["A"]]
    b_x = [s.foot_px[0] for s in samples["B"]]
    assert all(later > earlier for earlier, later in zip(a_x, a_x[1:]))
    assert all(later < earlier for earlier, later in zip(b_x, b_x[1:]))
    assert tracker.ambiguity_times() == []


def test_dropout_coasts_then_stops_then_reacquires():
    tracker = TwoPlayerTracker()
    tracker.update(0.0, 0, [det(100, 300), det(900, 300)])
    # B drops out for 2 s at 4 Hz: first COAST_MAX_S emits coasted samples.
    steps = 8
    for step in range(1, steps + 1):
        tracker.update(step * 0.25, step, [det(100 + step, 300)])
    samples = tracker.samples()
    coasted = [s for s in samples["B"] if s.coasted]
    assert coasted, "B should coast after dropout"
    assert all(s.foot_px == samples["B"][0].foot_px for s in coasted)
    assert max(s.t_s for s in samples["B"]) <= COAST_MAX_S + 1e-9
    # Reacquire near the old position.
    tracker.update((steps + 1) * 0.25, steps + 1, [det(101 + steps, 300), det(905, 300)])
    live_b = [s for s in tracker.samples()["B"] if not s.coasted]
    assert live_b[-1].foot_px[0] == 905.0


def test_crossing_records_ambiguity():
    tracker = TwoPlayerTracker()
    # Both players converge on the same point, then swap-like geometry.
    tracker.update(0.0, 0, [det(400, 300), det(600, 300)])
    tracker.update(0.25, 1, [det(499, 300), det(501, 300)])
    assert len(tracker.ambiguity_times()) >= 1
    stats = tracker.stats()
    assert stats["ambiguous_assignments"] >= 1
    assert stats["updates"] == 2


def test_third_detection_ignored_top2_by_confidence():
    tracker = TwoPlayerTracker()
    tracker.update(0.0, 0, [det(100, 300, 0.95), det(900, 300, 0.9),
                            det(500, 100, 0.3)])
    samples = tracker.samples()
    xs = {samples["A"][0].foot_px[0], samples["B"][0].foot_px[0]}
    assert xs == {100.0, 900.0}


def test_person_frame_pass_cadence_and_wiring():
    calls = []

    class StubDetector:
        backend = "stub"

        def detect(self, frame_bgr):
            calls.append(1)
            return [det(100, 300), det(900, 300)]

    # 60 fps at stride 4 -> 15 Hz coarse cadence -> detect every 4th frame
    # for PERSON_DETECT_HZ = 4.0 (round(15/4) = 4).
    person_pass = PersonFramePass(StubDetector(), source_fps=60.0, frame_stride=4)
    assert person_pass.detect_every == 4
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(8):
        person_pass.observe(i * 4, frame)
    assert len(calls) == 2  # observed frames 0..7 -> detected on 0 and 4
    assert len(person_pass.tracker.samples()["A"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_tracker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'player_tracker'`.

- [ ] **Step 3: Implement `player_tracker.py`**

```python
"""Two-player tracking over person detections.

Anonymous tracks "A"/"B" — naming is a post-hoc relabel (spec §4.4). Pure
logic: no cv2/torch/rfdetr imports; PersonFramePass only counts frames and
delegates to the injected detector.
"""

from dataclasses import dataclass
import math

COAST_MAX_S = 1.0        # coast a dropped track at its last position this long
AMBIGUITY_MARGIN = 0.2   # pairings within 20% total cost are ambiguous
PERSON_DETECT_HZ = 4.0   # target person-detection cadence in video seconds


@dataclass
class TrackSample:
    t_s: float
    frame_idx: int
    foot_px: tuple
    bbox: tuple            # (x_center, y_center, width, height)
    confidence: float
    coasted: bool


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _Track:
    def __init__(self):
        self.samples = []
        self.last_live_t = None
        self.last_foot = None
        self.last_bbox = None

    def add_live(self, t_s, frame_idx, detection):
        foot = detection.foot_px
        bbox = (detection.x, detection.y, detection.width, detection.height)
        self.samples.append(TrackSample(t_s, frame_idx, foot, bbox,
                                        detection.confidence, coasted=False))
        self.last_live_t = t_s
        self.last_foot = foot
        self.last_bbox = bbox

    def add_coast(self, t_s, frame_idx):
        if self.last_live_t is None:
            return
        if t_s - self.last_live_t > COAST_MAX_S:
            return
        self.samples.append(TrackSample(t_s, frame_idx, self.last_foot,
                                        self.last_bbox, 0.0, coasted=True))


class TwoPlayerTracker:
    def __init__(self):
        self._tracks = {"A": _Track(), "B": _Track()}
        self._ambiguity_times = []
        self._updates = 0

    def update(self, t_s, frame_idx, detections):
        self._updates += 1
        top2 = sorted(detections, key=lambda d: -d.confidence)[:2]
        a, b = self._tracks["A"], self._tracks["B"]

        if a.last_foot is None:
            # Seed: leftmost -> "A", the other (if any) -> "B".
            ordered = sorted(top2, key=lambda d: d.x)
            if ordered:
                a.add_live(t_s, frame_idx, ordered[0])
            if len(ordered) > 1:
                b.add_live(t_s, frame_idx, ordered[1])
            return

        if b.last_foot is None and len(top2) > 1:
            # Second track seeds from the detection farther from A.
            ordered = sorted(top2, key=lambda d: _distance(d.foot_px, a.last_foot))
            a.add_live(t_s, frame_idx, ordered[0])
            b.add_live(t_s, frame_idx, ordered[1])
            return

        if len(top2) == 0:
            a.add_coast(t_s, frame_idx)
            b.add_coast(t_s, frame_idx)
            return

        if len(top2) == 1 or b.last_foot is None:
            detection = top2[0]
            if b.last_foot is None:
                a.add_live(t_s, frame_idx, detection)
                return
            cost_a = _distance(detection.foot_px, a.last_foot)
            cost_b = _distance(detection.foot_px, b.last_foot)
            if cost_a <= cost_b:
                a.add_live(t_s, frame_idx, detection)
                b.add_coast(t_s, frame_idx)
            else:
                b.add_live(t_s, frame_idx, detection)
                a.add_coast(t_s, frame_idx)
            return

        d1, d2 = top2
        straight = (_distance(d1.foot_px, a.last_foot)
                    + _distance(d2.foot_px, b.last_foot))
        crossed = (_distance(d2.foot_px, a.last_foot)
                   + _distance(d1.foot_px, b.last_foot))
        low, high = min(straight, crossed), max(straight, crossed)
        if high > 0 and low >= (1.0 - AMBIGUITY_MARGIN) * high:
            self._ambiguity_times.append(t_s)
        if straight <= crossed:
            a.add_live(t_s, frame_idx, d1)
            b.add_live(t_s, frame_idx, d2)
        else:
            a.add_live(t_s, frame_idx, d2)
            b.add_live(t_s, frame_idx, d1)

    def samples(self):
        return {key: list(track.samples) for key, track in self._tracks.items()}

    def ambiguity_times(self):
        return list(self._ambiguity_times)

    def stats(self):
        return {
            "updates": self._updates,
            "ambiguous_assignments": len(self._ambiguity_times),
        }


class PersonFramePass:
    """Coarse-pass frame observer: detect every Nth coarse frame, feed the
    tracker. job_runner passes .observe as track_segments' frame_observer."""

    def __init__(self, detector, source_fps, frame_stride):
        self.detector = detector
        self.source_fps = float(source_fps) or 30.0
        coarse_hz = self.source_fps / max(1, int(frame_stride))
        self.detect_every = max(1, round(coarse_hz / PERSON_DETECT_HZ))
        self.tracker = TwoPlayerTracker()
        self._seen = 0

    def observe(self, frame_idx, frame_bgr):
        index = self._seen
        self._seen += 1
        if index % self.detect_every != 0:
            return
        detections = self.detector.detect(frame_bgr)
        self.tracker.update(frame_idx / self.source_fps, frame_idx, detections)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_tracker.py -v
```

Expected: 5 passed. (If `test_dropout_coasts_then_stops_then_reacquires`
fails on the reacquire assertion, check the 1-detection branch: the single
detection must go to the *nearer* track and the far detection of a later
2-detection update must reattach to the stale track — that is plain 2×2
assignment, no special casing.)

- [ ] **Step 5: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add player_tracker.py tests/test_player_tracker.py
git commit -m "feat: two-player tracker with coasting and ambiguity accounting"
```

---

## Task 3: `player_attribution.py` — serve resolver + players_v1 builder

**Files:**
- Create: `player_attribution.py`
- Create: `tests/test_player_attribution.py`

**Interfaces:**
- Consumes: `player_tracker.TrackSample` lists (Task 2); ball rows shaped like
  `tracking_common.ball_csv_row` output (dicts with `source_frame`,
  `timestamp_seconds`, `detected`, `x_center`, `y_center` — numeric fields
  may be strings, `detected` may be bool or `"True"`); rally hit lists
  (dicts with `timestamp_seconds`).
- Produces (Tasks 4/5 rely on these exact names):
  ```python
  SERVE_LOOKBACK_S = 2.0        # ball row search window before a rally's first hit
  MAX_SERVE_TRACK_GAP_S = 0.75  # a track vote needs a live sample this fresh

  def build_serve_resolver(samples_by_track, ball_rows) -> callable
      # returned resolver: (rally_hits) -> "A" | "B" | None
  def rally_identity_confidences(ambiguity_times, rallies) -> dict
      # {rally_number: float in [0,1] | None}
  def build_players_v1(assignment, tracker_stats, detector_backend,
                       serve_crop_relpath=None, player_names=None) -> dict
  def serve_crop_target(assignment, samples_by_track) -> tuple | None
      # (frame_idx, TrackSample) for the rally-1 observed server, else None
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_player_attribution.py`:

```python
"""player_attribution: serve resolver, identity confidence, players_v1."""

import pytest

from player_tracker import TrackSample
from player_attribution import (
    MAX_SERVE_TRACK_GAP_S,
    SERVE_LOOKBACK_S,
    build_players_v1,
    build_serve_resolver,
    rally_identity_confidences,
    serve_crop_target,
)


def sample(t_s, x, y, coasted=False):
    return TrackSample(t_s=t_s, frame_idx=int(t_s * 60), foot_px=(x, y),
                       bbox=(x, y - 50.0, 40.0, 100.0),
                       confidence=0.0 if coasted else 0.9, coasted=coasted)


def ball_row(frame, t_s, x, y):
    return {"source_frame": frame, "timestamp_seconds": f"{t_s:.6f}",
            "detected": True, "x_center": str(x), "y_center": str(y)}


def hit(t_s):
    return {"timestamp_seconds": t_s, "frame": int(t_s * 60)}


def test_resolver_picks_track_nearest_serve_ball():
    samples = {
        "A": [sample(t, 300, 500) for t in (9.0, 9.5, 10.0)],
        "B": [sample(t, 900, 500) for t in (9.0, 9.5, 10.0)],
    }
    # Ball detected near A 0.4 s before the first front-wall hit at t=10.4.
    rows = [ball_row(600, 10.0, 320, 480)]
    resolver = build_serve_resolver(samples, rows)
    assert resolver([hit(10.4), hit(11.5)]) == "A"


def test_resolver_none_without_ball_or_fresh_samples():
    samples = {"A": [sample(9.9, 300, 500)], "B": [sample(9.9, 900, 500)]}
    resolver = build_serve_resolver(samples, [])
    assert resolver([hit(10.4)]) is None  # no ball rows at all

    stale = {"A": [sample(1.0, 300, 500)], "B": [sample(1.0, 900, 500)]}
    rows = [ball_row(600, 10.0, 320, 480)]
    resolver = build_serve_resolver(stale, rows)
    assert resolver([hit(10.4)]) is None  # both tracks stale at ball time

    coasting = {"A": [sample(10.0, 300, 500, coasted=True)],
                "B": [sample(1.0, 900, 500)]}
    resolver = build_serve_resolver(coasting, rows)
    assert resolver([hit(10.4)]) is None  # coasted samples don't vote


def test_resolver_ignores_ball_rows_outside_lookback():
    samples = {"A": [sample(10.0, 300, 500)], "B": [sample(10.0, 900, 500)]}
    rows = [ball_row(60, 1.0, 320, 480)]  # far outside SERVE_LOOKBACK_S
    resolver = build_serve_resolver(samples, rows)
    assert resolver([hit(10.4)]) is None


def test_rally_identity_confidences_window_before_rally():
    rallies = [
        {"rally_number": 1, "start_time_seconds": 10.0, "end_time_seconds": 20.0},
        {"rally_number": 2, "start_time_seconds": 30.0, "end_time_seconds": 40.0},
        {"rally_number": 3, "start_time_seconds": 50.0, "end_time_seconds": 60.0},
    ]
    # Two ambiguous events during the break before rally 2, none before 3.
    confidences = rally_identity_confidences([21.0, 25.0], rallies)
    assert confidences[1] is None            # no break precedes rally 1
    assert confidences[2] == pytest.approx(0.0)
    assert confidences[3] == pytest.approx(1.0)


def test_build_players_v1_shape():
    assignment = {
        "method": "rally_gap_observed_serves",
        "rally_count": 2,
        "rallies": [
            {"rally_number": 1, "server_player_number": 1, "server_track": "A",
             "server_source": "observed", "winner_player_number": 2,
             "winner_source": "next_serve", "winner_crosscheck_agrees": True,
             "start_time_seconds": 10.0, "end_time_seconds": 20.0},
            {"rally_number": 2, "server_player_number": 2, "server_track": "B",
             "server_source": "observed", "winner_player_number": None,
             "winner_source": None, "winner_crosscheck_agrees": None,
             "start_time_seconds": 30.0, "end_time_seconds": 40.0},
        ],
    }
    block = build_players_v1(assignment, {"updates": 100, "ambiguous_assignments": 3},
                             detector_backend="rfdetr",
                             serve_crop_relpath="players/serve_rally1.jpg")
    assert block["attribution_backend"] == "observed"
    assert block["detector_backend"] == "rfdetr"
    assert block["serve_crop"] == "players/serve_rally1.jpg"
    assert block["player_names"] == {"A": None, "B": None}
    assert block["tracker"] == {"updates": 100, "ambiguous_assignments": 3}
    assert [r["rally_number"] for r in block["rallies"]] == [1, 2]
    scores = [r["score_after"] for r in block["rallies"]]
    assert scores[0] == {"1": 0, "2": 1}   # player 2 won rally 1
    assert scores[1] == {"1": 0, "2": 1}   # rally 2 winner unknown -> carried

    assumed = build_players_v1({"method": "rally_gap_server_alternation",
                                "rally_count": 0, "rallies": []},
                               None, detector_backend="none")
    assert assumed["attribution_backend"] == "assumed"


def test_serve_crop_target_uses_rally1_observed_server():
    assignment = {"rallies": [
        {"rally_number": 1, "server_track": "A", "server_source": "observed",
         "start_time_seconds": 10.0},
    ]}
    samples = {"A": [sample(9.8, 300, 500), sample(10.6, 310, 500)],
               "B": [sample(9.8, 900, 500)]}
    frame_idx, chosen = serve_crop_target(assignment, samples)
    assert chosen.foot_px == (300.0, 500.0)
    assert frame_idx == chosen.frame_idx

    no_obs = {"rallies": [{"rally_number": 1, "server_track": None,
                           "server_source": "propagated",
                           "start_time_seconds": 10.0}]}
    assert serve_crop_target(no_obs, samples) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_attribution.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'player_attribution'`.

- [ ] **Step 3: Implement `player_attribution.py`**

```python
"""Serve-anchored player attribution (spec §4.4).

Within a rally, front-wall hits strictly alternate (squash rule), so per-hit
identity is unnecessary: the only question is which track served each rally.
The resolver answers it from raw pixels — the track whose live sample is
nearest the last detected ball position shortly before the rally's first
front-wall hit. No homography, no calibration.
"""

import math

SERVE_LOOKBACK_S = 2.0        # ball row search window before a rally's first hit
MAX_SERVE_TRACK_GAP_S = 0.75  # a track vote needs a live sample this fresh


def _row_time(row):
    return float(row.get("timestamp_seconds", 0.0))


def _row_detected(row):
    detected = row.get("detected")
    if isinstance(detected, str):
        return detected.strip().lower() == "true"
    return bool(detected)


def build_serve_resolver(samples_by_track, ball_rows):
    """-> resolver(rally_hits) -> "A" | "B" | None.

    Stateless with respect to rallies: matches by the first hit's timestamp,
    so it can be called from any assign_front_wall_hit_players invocation.
    """
    detected_rows = sorted(
        (row for row in ball_rows if _row_detected(row)), key=_row_time
    )
    live = {
        track: [s for s in samples if not s.coasted]
        for track, samples in samples_by_track.items()
    }

    def resolver(rally_hits):
        if not rally_hits:
            return None
        first_hit_t = float(rally_hits[0].get("timestamp_seconds", 0.0))
        window = [row for row in detected_rows
                  if first_hit_t - SERVE_LOOKBACK_S <= _row_time(row) < first_hit_t]
        if not window:
            return None
        ball = window[-1]
        ball_t = _row_time(ball)
        ball_xy = (float(ball["x_center"]), float(ball["y_center"]))

        best_track, best_distance = None, None
        for track in ("A", "B"):
            candidates = [s for s in live.get(track, [])
                          if abs(s.t_s - ball_t) <= MAX_SERVE_TRACK_GAP_S]
            if not candidates:
                return None  # a missing vote makes the comparison meaningless
            nearest = min(candidates, key=lambda s: abs(s.t_s - ball_t))
            distance = math.hypot(nearest.foot_px[0] - ball_xy[0],
                                  nearest.foot_px[1] - ball_xy[1])
            if best_distance is None or distance < best_distance:
                best_track, best_distance = track, distance
        return best_track

    return resolver


def rally_identity_confidences(ambiguity_times, rallies):
    """Confidence that identity survived the break BEFORE each rally.

    confidence = 1 - min(1, ambiguous_events_in_break); rally 1 has no
    preceding break -> None. Deliberately blunt for v1 (spec: collected
    silently, debugged later)."""
    confidences = {}
    previous_end = None
    for rally in rallies:
        number = rally["rally_number"]
        start = float(rally.get("start_time_seconds", 0.0))
        if previous_end is None:
            confidences[number] = None
        else:
            count = sum(1 for t in ambiguity_times if previous_end < t <= start)
            confidences[number] = max(0.0, 1.0 - float(count))
        previous_end = float(rally.get("end_time_seconds", start))
    return confidences


def build_players_v1(assignment, tracker_stats, detector_backend,
                     serve_crop_relpath=None, player_names=None):
    observed = any(
        rally.get("server_source") == "observed"
        for rally in assignment.get("rallies", [])
    )
    score = {"1": 0, "2": 0}
    rallies = []
    for rally in assignment.get("rallies", []):
        winner = rally.get("winner_player_number")
        if winner in (1, 2):
            score[str(winner)] += 1
        rallies.append({
            "rally_number": rally.get("rally_number"),
            "server_player_number": rally.get("server_player_number"),
            "server_track": rally.get("server_track"),
            "server_source": rally.get("server_source"),
            "winner_player_number": winner,
            "winner_source": rally.get("winner_source"),
            "winner_crosscheck_agrees": rally.get("winner_crosscheck_agrees"),
            "identity_confidence": rally.get("identity_confidence"),
            "score_after": dict(score),
        })
    return {
        "attribution_backend": "observed" if observed else "assumed",
        "detector_backend": detector_backend,
        "tracker": tracker_stats,
        "rallies": rallies,
        "serve_crop": serve_crop_relpath,
        "player_names": dict(player_names) if player_names else {"A": None, "B": None},
    }


def serve_crop_target(assignment, samples_by_track):
    """-> (frame_idx, TrackSample) for the rally-1 observed server, or None."""
    rallies = assignment.get("rallies") or []
    if not rallies:
        return None
    first = rallies[0]
    track = first.get("server_track")
    if first.get("server_source") != "observed" or track not in ("A", "B"):
        return None
    serve_t = float(first.get("start_time_seconds", 0.0))
    live = [s for s in samples_by_track.get(track, []) if not s.coasted]
    if not live:
        return None
    nearest = min(live, key=lambda s: abs(s.t_s - serve_t))
    return nearest.frame_idx, nearest
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_attribution.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add player_attribution.py tests/test_player_attribution.py
git commit -m "feat: serve resolver, identity confidence, players_v1 builder"
```

---

## Task 4: observed serves in `assign_front_wall_hit_players`

**Files:**
- Modify: `job_runner.py:560-655` (`assign_front_wall_hit_players`) and
  `job_runner.py` `judge_hits` (the `assign_front_wall_hit_players(hits)` call
  at `job_runner.py:867` — signature gains `serve_resolver=None`)
- Test: extend `tests/test_pipeline.py`

**Interfaces:**
- Produces: `assign_front_wall_hit_players(hits, serve_resolver=None)` and
  `judge_hits(run_dir, results, classified, audio_available=None, serve_resolver=None)`.
  With a resolver, each rally summary dict gains: `server_track`
  (`"A"|"B"|None`), `server_source` (`"observed"|"propagated"`),
  `winner_source` (`"next_serve"|"est"|None`), `winner_crosscheck_agrees`
  (`bool|None`). Top-level: `method` becomes `"rally_gap_observed_serves"`
  when ≥1 rally is observed, and `observed_serve_count` is added. Without a
  resolver (or when it never resolves), output is **identical to today's**
  (plus `server_source: "propagated"`, `server_track: None`,
  `winner_source: "est"|None`, `winner_crosscheck_agrees: None` on each rally).

**Semantics (implement exactly — spec §4.4):**
1. Segment rallies as today. Call `serve_resolver(rally_hits)` per rally →
   `resolved[rally_index] = "A"|"B"|None`.
2. **Anchor:** at the first rally index with a non-None resolved track, bind
   `track_player_map[track] = ` that rally's *propagated* server (the chain
   value at that point: `first_server` env default for rally 1, else the
   est-winner chain). The other track maps to `other_player(...)`. In the
   normal case (rally 1 observed) this yields track↦player = server↦1.
3. **Servers, forward pass:** rally server = `track_player_map[resolved]`
   when resolved (`server_source: "observed"`), else the propagated chain
   value (`server_source: "propagated"`). Per-hit `player_number` =
   alternation from the rally server (unchanged code).
4. Compute the existing est winner per rally (`job_runner.py:606-620`
   logic, unchanged) — call it `est_winner`.
5. **Winner back-fill pass:** for rally N (not last): if rally N+1's server
   is observed, `winner = server_{N+1}`, `winner_source = "next_serve"`,
   `winner_crosscheck_agrees = (est_winner == winner)` when `est_winner` is
   not None else None. Otherwise `winner = est_winner`,
   `winner_source = "est" if est_winner else None`,
   `winner_crosscheck_agrees = None`. The **last rally** always uses
   `est_winner` (`winner_source "est"`/None) — it has no next serve.
6. The propagation chain (`server = winner` step at `job_runner.py:644-645`)
   uses, in order of preference: next-serve winner when already knowable in
   the forward pass is NOT required — keep the forward chain on `est_winner`
   exactly as today (observed rallies override their own server anyway, and
   the back-fill pass rewrites the winner fields afterward). This keeps the
   diff small and the unobserved behavior bit-identical.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`;
  it uses `from job_runner import ...` — add `import job_runner` to its
  imports so the snippet below works as written)

```python
import job_runner


def _attribution_hit(t_s, call="IN", is_serve=False):
    return {
        "timestamp_seconds": t_s,
        "frame": int(t_s * 60),
        "event_type": "wall",
        "target_zone": {"zone": 4, "side": "center", "x": 0.5, "y": 0.5},
        "call": call,
        "is_serve": is_serve,
    }


def _three_rally_hits():
    # Rally 1: t=10.0 (serve), 11.0, 12.0 ; rally 2: t=30.0, 31.0 ;
    # rally 3: t=60.0. Gaps >> RALLY_GAP defaults.
    return (
        [_attribution_hit(10.0, is_serve=True), _attribution_hit(11.0),
         _attribution_hit(12.0)]
        + [_attribution_hit(30.0, is_serve=True), _attribution_hit(31.0)]
        + [_attribution_hit(60.0, is_serve=True)]
    )


def test_assign_without_resolver_matches_legacy_and_labels_sources():
    hits = _three_rally_hits()
    assignment = job_runner.assign_front_wall_hit_players(hits)
    assert assignment["method"] == "rally_gap_server_alternation"
    assert assignment["observed_serve_count"] == 0
    for rally in assignment["rallies"]:
        assert rally["server_track"] is None
        assert rally["server_source"] == "propagated"
        assert rally["winner_crosscheck_agrees"] is None
    # Legacy alternation: rally 1 server=1, winner=1 (last hit IN by player 1).
    assert assignment["rallies"][0]["server_player_number"] == 1
    assert assignment["rallies"][0]["winner_player_number"] == 1
    assert assignment["rallies"][0]["winner_source"] == "est"


def test_assign_with_resolver_observed_servers_and_next_serve_winners():
    hits = _three_rally_hits()
    by_first_hit_t = {10.0: "A", 30.0: "B", 60.0: "B"}

    def resolver(rally_hits):
        return by_first_hit_t.get(float(rally_hits[0]["timestamp_seconds"]))

    assignment = job_runner.assign_front_wall_hit_players(hits, serve_resolver=resolver)
    assert assignment["method"] == "rally_gap_observed_serves"
    assert assignment["observed_serve_count"] == 3
    rallies = assignment["rallies"]
    # Anchor: rally 1 observed "A" -> A=player 1 (propagated first server).
    assert rallies[0]["server_player_number"] == 1
    assert rallies[0]["server_track"] == "A"
    assert rallies[0]["server_source"] == "observed"
    # Rally 2 served by B -> player 2; so rally 1 winner = 2 via next_serve.
    assert rallies[1]["server_player_number"] == 2
    assert rallies[0]["winner_player_number"] == 2
    assert rallies[0]["winner_source"] == "next_serve"
    # est winner of rally 1 was player 1 (last IN hit) -> crosscheck disagrees.
    assert rallies[0]["winner_crosscheck_agrees"] is False
    # Rally 3 also served by B: winner of rally 2 = player 2 via next_serve.
    # Rally 2's LAST hit (t=31.0, IN) was by the receiver (player 1), so the
    # est winner is 1 and the cross-check disagrees.
    assert rallies[1]["winner_player_number"] == 2
    assert rallies[1]["winner_source"] == "next_serve"
    assert rallies[1]["winner_crosscheck_agrees"] is False
    # Last rally: est only.
    assert rallies[2]["winner_source"] in ("est", None)
    # Per-hit alternation from the observed servers.
    rally2_hits = [h for h in hits if h.get("rally_number") == 2]
    assert [h["player_number"] for h in rally2_hits] == [2, 1]


def test_assign_with_partial_resolver_falls_back_to_propagation():
    hits = _three_rally_hits()

    def resolver(rally_hits):
        # Only rally 2 observed.
        return "B" if float(rally_hits[0]["timestamp_seconds"]) == 30.0 else None

    assignment = job_runner.assign_front_wall_hit_players(hits, serve_resolver=resolver)
    rallies = assignment["rallies"]
    assert rallies[0]["server_source"] == "propagated"
    assert rallies[1]["server_source"] == "observed"
    # Anchor is rally 2: its propagated server is the est winner of rally 1
    # (player 1), so B -> player 1 there.
    assert rallies[1]["server_player_number"] == 1
    assert assignment["observed_serve_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_pipeline.py -v -k "assign_with or assign_without"
```

Expected: FAIL — `observed_serve_count` KeyError / unexpected keyword
`serve_resolver`.

- [ ] **Step 3: Implement in `job_runner.py`**

Rewrite `assign_front_wall_hit_players` (keep every existing field and the
existing est-winner logic verbatim; the diff adds the resolver plumbing and
the back-fill pass):

```python
def assign_front_wall_hit_players(hits, serve_resolver=None):
    for hit in hits:
        for key in (
            "front_wall_sequence",
            "rally_number",
            "rally_hit_sequence",
            "server_player_number",
            "player_number",
        ):
            hit.pop(key, None)

    assignable_hits = [
        hit for hit in hits if is_player_assignable_front_wall_hit(hit)
    ]
    rallies, rally_gap_seconds, rally_gap_method = segment_front_wall_hits_into_rallies(
        assignable_hits
    )
    try:
        first_server = int(
            env_float("PLAYER_ASSIGNMENT_FIRST_SERVER", DEFAULT_FIRST_SERVER_PLAYER)
        )
    except (OverflowError, ValueError):
        first_server = DEFAULT_FIRST_SERVER_PLAYER
    if first_server not in (1, 2):
        first_server = DEFAULT_FIRST_SERVER_PLAYER

    resolved_tracks = [
        serve_resolver(rally_hits) if serve_resolver is not None else None
        for rally_hits in rallies
    ]

    sequence = 0
    server = first_server
    track_player_map = None
    rally_summaries = []
    for rally_index, rally_hits in enumerate(rallies, start=1):
        resolved = resolved_tracks[rally_index - 1]
        if resolved in ("A", "B") and track_player_map is None:
            # Anchor: the first observed rally binds its track to the
            # propagated server at this point in the chain (spec §4.4).
            track_player_map = {
                resolved: server,
                ("B" if resolved == "A" else "A"): other_player(server),
            }
        if resolved in ("A", "B") and track_player_map is not None:
            rally_server = track_player_map[resolved]
            server_source = "observed"
        else:
            rally_server = server
            server_source = "propagated"

        for rally_sequence, hit in enumerate(rally_hits, start=1):
            sequence += 1
            player_number = (
                rally_server
                if rally_sequence % 2 == 1
                else other_player(rally_server)
            )
            hit["front_wall_sequence"] = sequence
            hit["rally_number"] = rally_index
            hit["rally_hit_sequence"] = rally_sequence
            hit["server_player_number"] = rally_server
            hit["player_number"] = player_number

        serve_hit = rally_hits[0]
        last_hit = rally_hits[-1]
        last_player = int(last_hit.get("player_number") or rally_server)
        last_call = last_hit.get("call")
        serve_call = serve_hit.get("call") if serve_hit.get("is_serve") else None
        if serve_call == "OUT":
            winner = other_player(rally_server)
            winner_reason = "serve_out"
        elif last_call == "OUT":
            winner = other_player(last_player)
            winner_reason = "last_front_wall_hit_out"
        elif last_call == "IN":
            winner = last_player
            winner_reason = "last_front_wall_hit_in_winner"
        else:
            winner = None
            winner_reason = "last_front_wall_hit_unjudged"

        rally_start_time = float(rally_hits[0].get("timestamp_seconds", 0.0))
        rally_end_time = float(rally_hits[-1].get("timestamp_seconds", 0.0))
        rally_summaries.append({
            "rally_number": rally_index,
            "start_frame": int(rally_hits[0].get("frame", 0) or 0),
            "end_frame": int(rally_hits[-1].get("frame", 0) or 0),
            "start_time_seconds": rally_start_time,
            "end_time_seconds": rally_end_time,
            "duration_seconds": round(max(0.0, rally_end_time - rally_start_time), 3),
            "front_wall_hit_count": len(rally_hits),
            "server_player_number": rally_server,
            "server_track": resolved if resolved in ("A", "B") else None,
            "server_source": server_source,
            "winner_player_number": winner,
            "winner_reason": winner_reason,
            "winner_source": "est" if winner is not None else None,
            "winner_crosscheck_agrees": None,
            "last_call": last_call,
            "last_player_number": last_player,
            "serve_frame": (
                int(serve_hit.get("frame", 0) or 0)
                if serve_hit.get("is_serve")
                else None
            ),
            "serve_call": serve_call,
        })
        if winner in (1, 2):
            server = winner

    # Winner back-fill: winner of rally N := observed server of rally N+1
    # (squash rule — the winner serves next). The est winner stays as a
    # silent cross-check (spec §4.4). Last rally keeps est.
    for index in range(len(rally_summaries) - 1):
        next_summary = rally_summaries[index + 1]
        if next_summary["server_source"] != "observed":
            continue
        summary = rally_summaries[index]
        est_winner = summary["winner_player_number"]
        observed_winner = next_summary["server_player_number"]
        summary["winner_player_number"] = observed_winner
        summary["winner_source"] = "next_serve"
        summary["winner_crosscheck_agrees"] = (
            (est_winner == observed_winner) if est_winner is not None else None
        )

    observed_serve_count = sum(
        1 for summary in rally_summaries if summary["server_source"] == "observed"
    )
    return {
        "method": (
            "rally_gap_observed_serves"
            if observed_serve_count
            else "rally_gap_server_alternation"
        ),
        "rally_gap_seconds": rally_gap_seconds,
        "rally_gap_method": rally_gap_method,
        "first_server_player_number": first_server,
        "rally_count": len(rally_summaries),
        "assigned_front_wall_hit_count": len(assignable_hits),
        "observed_serve_count": observed_serve_count,
        "rallies": rally_summaries,
    }
```

Then thread the parameter through `judge_hits`: change its signature to
`def judge_hits(run_dir, results, classified, audio_available=None, serve_resolver=None):`
and the internal call at `job_runner.py:867` to
`assign_front_wall_hit_players(hits, serve_resolver=serve_resolver)`.

- [ ] **Step 4: Run the pipeline tests**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_pipeline.py -v
```

Expected: all pass — including every pre-existing test (the no-resolver path
must be value-identical; only the three additive keys per rally and
`observed_serve_count`/`winner_source` appear).

- [ ] **Step 5: Full suite + /eval zero-drift, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" eval_line_calls.py --eval-set eval_set/cases.jsonl
```

Expected: suite green; eval numbers identical to the newest
`eval_set/BASELINE-*.md` (this task cannot touch judging — confirm, don't
assume).

```bash
git add job_runner.py tests/test_pipeline.py
git commit -m "feat: observed-serve attribution with next-serve winner chain"
```

---

## Task 5: pipeline wiring — person pass, players_v1, serve crop

**Files:**
- Modify: `job_runner.py` — `track_segments` (`:183`), `run_tracking_job`
  (`:901-1121`)
- Test: extend `tests/test_player_tracker.py` (observer seam) and
  `tests/test_player_attribution.py` (persistence helper)

**Interfaces:**
- `track_segments(model, video_path, segments, inference_width, source_fps,
  results, on_frame, frame_observer=None)` — observer called as
  `frame_observer(frame_idx, frame)` for every decoded coarse frame.
- `run_tracking_job` emits `players_v1` via `update_job` and writes
  `players/track_samples.json` + `players/serve_rally1.jpg` under the run dir.
- New helper in `job_runner.py`:
  ```python
  def build_person_pass(source_fps, frame_stride)   # PersonFramePass | None
  def write_track_samples(run_dir, samples_by_track, ambiguity_times) -> None
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_player_tracker.py`:

```python
def test_track_segments_frame_observer_sees_coarse_frames(tmp_path, monkeypatch):
    import cv2
    import job_runner

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
    observed = []
    job_runner.track_segments(
        model=None, video_path=video, segments=[(0, 11, 3)],
        inference_width=64, source_fps=30.0, results={},
        on_frame=lambda idx: None,
        frame_observer=lambda idx, frame: observed.append((idx, frame.shape)),
    )
    assert [idx for idx, _ in observed] == [0, 3, 6, 9]
    assert all(shape == (48, 64, 3) for _, shape in observed)
```

Append to `tests/test_player_attribution.py`:

```python
def test_write_track_samples_round_trip(tmp_path):
    import json
    import job_runner

    samples = {"A": [sample(1.0, 300, 500)], "B": []}
    job_runner.write_track_samples(tmp_path, samples, [2.5])
    payload = json.loads((tmp_path / "players" / "track_samples.json").read_text())
    assert payload["schema"] == "player-tracks-v1"
    assert payload["ambiguity_times"] == [2.5]
    entry = payload["tracks"]["A"][0]
    assert entry == {"t_s": 1.0, "frame_idx": 60, "foot_px": [300.0, 500.0],
                     "bbox": [300.0, 450.0, 40.0, 100.0],
                     "confidence": 0.9, "coasted": False}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_tracker.py tests/test_player_attribution.py -v -k "observer or round_trip"
```

Expected: FAIL — unexpected keyword `frame_observer` /
`write_track_samples` AttributeError.

- [ ] **Step 3: Implement in `job_runner.py`**

(a) `track_segments` — add the parameter and one call in the consumer loop
(`job_runner.py:183-213`):

```python
def track_segments(model, video_path, segments, inference_width, source_fps,
                   results, on_frame, frame_observer=None):
    ...
            frame_idx, frame = item
            if frame_observer is not None:
                frame_observer(frame_idx, frame)
            predictions = infer_frame_predictions(
```

(b) New helpers near the top of the job section (after `persist_job`); add
imports `import player_attribution` and
`from player_tracker import PersonFramePass` alongside the existing top
imports, and `import person_model`:

```python
def build_person_pass(source_fps, frame_stride):
    """PersonFramePass when the detector backend is available, else None.
    Detector load failures must never kill a tracking job."""
    try:
        detector = person_model.load_person_detector()
    except Exception:
        return None
    if detector is None:
        return None
    return PersonFramePass(detector, source_fps, frame_stride)


def write_track_samples(run_dir, samples_by_track, ambiguity_times):
    payload = {
        "schema": "player-tracks-v1",
        "ambiguity_times": [float(t) for t in ambiguity_times],
        "tracks": {
            track: [
                {
                    "t_s": s.t_s,
                    "frame_idx": s.frame_idx,
                    "foot_px": [s.foot_px[0], s.foot_px[1]],
                    "bbox": [s.bbox[0], s.bbox[1], s.bbox[2], s.bbox[3]],
                    "confidence": s.confidence,
                    "coasted": s.coasted,
                }
                for s in samples
            ]
            for track, samples in samples_by_track.items()
        },
    }
    players_dir = Path(run_dir) / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    (players_dir / "track_samples.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
```

(c) Wire `run_tracking_job`:

- After `model = get_tracking_model()` (`:935`), add:
  ```python
  person_pass = build_person_pass(source_fps, frame_stride)
  if person_pass is not None:
      update_job(run_id, message="Person detector: rfdetr")
  ```
- The **coarse** `track_segments` call (`:942-950`) gains
  `frame_observer=person_pass.observe if person_pass is not None else None`.
  The refine and audio-rescue calls are NOT changed (observer is
  coarse-pass-only — spec §4.2).
- Before the `judge_hits` call, build the resolver:
  ```python
  serve_resolver = None
  samples_by_track = None
  ambiguity_times = []
  if person_pass is not None:
      samples_by_track = person_pass.tracker.samples()
      ambiguity_times = person_pass.tracker.ambiguity_times()
      serve_resolver = player_attribution.build_serve_resolver(
          samples_by_track, sorted_rows(results)
      )
  ```
- Pass `serve_resolver=serve_resolver` to `judge_hits` (`:1079`) and to both
  `assign_front_wall_hit_players` calls (`:1085`, `:1091`).
- **After the try/except block** (both paths set `player_assignment`; insert
  just before `job = get_job(run_id) or {}` at `:1096`), attach identity
  confidences and build the block:
  ```python
  players_v1 = None
  if person_pass is not None:
      confidences = player_attribution.rally_identity_confidences(
          ambiguity_times, player_assignment["rallies"]
      )
      for rally in player_assignment["rallies"]:
          rally["identity_confidence"] = confidences.get(rally["rally_number"])
      write_track_samples(run_dir, samples_by_track, ambiguity_times)
      serve_crop_relpath = None
      crop_target = player_attribution.serve_crop_target(
          player_assignment, samples_by_track
      )
      if crop_target is not None:
          crop_frame_idx, crop_sample = crop_target
          crop_detection = person_model.PersonDetection(
              x=crop_sample.bbox[0], y=crop_sample.bbox[1],
              width=crop_sample.bbox[2], height=crop_sample.bbox[3],
              confidence=crop_sample.confidence, keypoints=(),
          )
          if person_model.save_person_crop(
              video_path, crop_frame_idx, crop_detection,
              run_dir / "players" / "serve_rally1.jpg",
          ):
              serve_crop_relpath = "players/serve_rally1.jpg"
      players_v1 = player_attribution.build_players_v1(
          player_assignment,
          person_pass.tracker.stats(),
          detector_backend=person_pass.detector.backend,
          serve_crop_relpath=serve_crop_relpath,
      )
  else:
      players_v1 = player_attribution.build_players_v1(
          player_assignment, None, detector_backend="none"
      )
  ```
- Add `players_v1=players_v1` to the completion `update_job` call (`:1101`).
  In the `except` path (`:1089-1094`) leave behavior as-is except passing
  `serve_resolver=serve_resolver` (hits may be empty; that is fine — the
  resolver is only consulted per rally).

- [ ] **Step 4: Run tests to verify they pass**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_player_tracker.py tests/test_player_attribution.py tests/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 5: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add job_runner.py tests/test_player_tracker.py tests/test_player_attribution.py
git commit -m "feat: person pass riding the coarse decode; players_v1 in the job payload"
```

---

## Task 6: API — `players_v1` exposure + naming route

**Files:**
- Modify: `app.py` — `public_job` key loop (`app.py:132-145`), new route after
  `save_ground_truth` (`app.py:1403`)
- Create: `tests/test_players_api.py`

**Interfaces:**
- `public_job` passes through `players_v1` when present.
- `POST /api/runs/<run_id>/players` body `{"A": "Ian", "B": "Alvin"}` (values
  string or null, ≤ 40 chars, stripped). Writes `player_names.json` in the
  run dir, merges into the job's `players_v1.player_names` via
  `job_runner.update_job`, returns `{"ok": True, "player_names": {...}}`.
- The serve crop needs no new route — `GET /api/runs/<run_id>/<path:filename>`
  (`app.py:1364`) already serves run-dir files, including `players/...` paths.

- [ ] **Step 1: Write the failing tests**

`tests/test_players_api.py`:

```python
"""POST /api/runs/<id>/players and players_v1 passthrough."""

import json


def make_client():
    import app as app_module
    return app_module.app.test_client()


def make_run(runs_dir, run_id="run-players"):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def test_players_v1_passes_through_public_job():
    import app as app_module
    payload = app_module.public_job({
        "status": "complete",
        "players_v1": {"attribution_backend": "observed"},
    })
    assert payload["players_v1"] == {"attribution_backend": "observed"}


def test_post_players_names(runs_dir):
    import job_runner
    client = make_client()
    run_dir = make_run(runs_dir)
    job_runner.update_job(
        "run-players",
        run_dir=str(run_dir),
        status="complete",
        players_v1={"attribution_backend": "observed",
                    "player_names": {"A": None, "B": None}},
    )

    response = client.post("/api/runs/run-players/players",
                           json={"A": "  Ian ", "B": "Alvin"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["player_names"] == {"A": "Ian", "B": "Alvin"}

    stored = json.loads((run_dir / "player_names.json").read_text())
    assert stored == {"A": "Ian", "B": "Alvin"}
    job = job_runner.get_job("run-players")
    assert job["players_v1"]["player_names"] == {"A": "Ian", "B": "Alvin"}


def test_post_players_validation(runs_dir):
    client = make_client()
    make_run(runs_dir)
    assert client.post("/api/runs/run-players/players",
                       json={"A": "x" * 41}).status_code == 400
    assert client.post("/api/runs/run-players/players",
                       json={"C": "nope"}).status_code == 400
    assert client.post("/api/runs/missing-run/players",
                       json={"A": "Ian"}).status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_players_api.py -v
```

Expected: FAIL — `players_v1` missing from `public_job` output, then 404s
for the unregistered route.

- [ ] **Step 3: Implement in `app.py`**

(a) Add `"players_v1",` to the passthrough tuple in `public_job`
(`app.py:132-145`, after `"player_assignment",`).

(b) New route after `save_ground_truth`:

```python
PLAYER_NAME_MAX_CHARS = 40


@app.post("/api/runs/<run_id>/players")
def save_player_names(run_id):
    """Post-hoc naming: map anonymous tracks A/B to typed names. Pure run
    metadata — analysis never re-runs (spec §4.5)."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True) or {}
    if any(key not in ("A", "B") for key in data):
        return error_response("Only tracks A and B can be named.")

    names = {}
    for track in ("A", "B"):
        value = data.get(track)
        if value is None:
            names[track] = None
            continue
        value = str(value).strip()
        if not value or len(value) > PLAYER_NAME_MAX_CHARS:
            return error_response(
                f"Names must be 1-{PLAYER_NAME_MAX_CHARS} characters."
            )
        names[track] = value

    (run_dir / "player_names.json").write_text(
        json.dumps(names, indent=2), encoding="utf-8"
    )
    job = job_runner.get_job(run_id)
    if job and isinstance(job.get("players_v1"), dict):
        players_v1 = dict(job["players_v1"])
        players_v1["player_names"] = names
        job_runner.update_job(run_id, players_v1=players_v1)
    return jsonify({"ok": True, "player_names": names})
```

(Check `app.py`'s existing imports — `job_runner` is already imported for
job access; if it is imported as functions, follow the existing style.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_players_api.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add app.py tests/test_players_api.py
git commit -m "feat: players_v1 in the public payload; post-hoc naming route"
```

---

## Task 7: results-page Players card + name propagation

**Files:**
- Modify: `index.html` — Players card near the player report buttons
  (`index.html:937`), name-aware labels at the render sites listed below.

**Blueprint (DESIGN.md is binding — use existing card/token classes, no new
colors, both themes):**

1. **Players card** renders on the results page when `S.run.players_v1`
   exists. Contents:
   - Backend line: observed → "Players detected automatically (RF-DETR)".
     assumed → "Player attribution is assumed serve alternation — person
     detection was unavailable for this run." (honest capability copy,
     spec §4.6).
   - Two labeled text inputs: "Player A name", "Player B name".
   - When `players_v1.serve_crop` is set: the crop image
     (`/api/runs/<RUN_ID>/players/serve_rally1.jpg`) with the question
     "Who is this? (served the first rally)" and two buttons that fill from
     the typed names; picking one assigns that name to the crop's track
     (track letter comes from `players_v1.rallies[0].server_track`) and the
     other name to the other track.
   - Save button → `POST /api/runs/<RUN_ID>/players` with `{A, B}`;
     on success update `S.run.players_v1.player_names` and re-render.
   - Markup sketch (adapt to the page's existing card classes — do not
     invent new tokens):
     ```html
     <section class="card" id="playersCard" hidden>
       <strong>Players</strong>
       <p id="playersBackendNote"></p>
       <label>Player A name <input id="playerNameA" maxlength="40"></label>
       <label>Player B name <input id="playerNameB" maxlength="40"></label>
       <div id="serveCropBlock" hidden>
         <img id="serveCropImg" alt="First server at the rally 1 serve">
         <p>Who is this? (served the first rally)</p>
         <button type="button" id="cropIsA"></button>
         <button type="button" id="cropIsB"></button>
       </div>
       <button type="button" id="savePlayersBtn">Save players</button>
     </section>
     ```
   - Note: the per-player *shot filter* the spec names as the headline win
     (front-wall impacts of only your hits) **already exists** — the
     `target_zones_by_player` wall maps and Player 1/Player 2 report pages
     are keyed by `player_number`, which observed attribution now makes
     real. Do NOT build a new filter surface; wiring names into those
     existing views completes the loop.
2. **Name propagation:** add one JS helper and use it at every site that
   renders "Player 1"/"Player 2" labels on the results page:
   ```javascript
   function playerDisplayName(playerNumber){
     const v1 = S.run && S.run.players_v1;
     const names = v1 && v1.player_names;
     const rally1 = v1 && v1.rallies && v1.rallies[0];
     // player 1 = rally-1 server; its track letter maps name -> number.
     let track = null;
     if (rally1 && rally1.server_track) {
       track = playerNumber === 1 ? rally1.server_track
                                  : (rally1.server_track === 'A' ? 'B' : 'A');
     }
     const name = track && names ? names[track] : null;
     return name || ('Player ' + playerNumber);
   }
   ```
   Wire it into: the player report tab labels (`index.html:937`, `:947`,
   `:971`, phase labels at `:1248`), `playerAssignmentText` (`:4660`), the
   per-player summaries render (`:4968`), and the rally table render
   (`:5013`, `:5077`). Grep for `'Player ' +` and `Player 1`/`Player 2`
   literals in the results-page sections to catch stragglers — leave the
   calibration wizard and non-results copy alone.
3. **Score line:** where the rally table renders (`:5013` area), if
   `players_v1.rallies[n].score_after` exists, append a running score column
   using `playerDisplayName` headers. `winner_source === "est"` rows show the
   existing "est." labeling — do not assert observed confidence on them.

- [ ] **Step 1: Implement the card + helper + propagation** (single step —
  UI work in this file is not test-driven; the `/verify` skill is the gate)

- [ ] **Step 2: Verify via the `/verify` skill**

Launch the app (`PORT=5188 <venv>/bin/python app.py` per the skill), load an
existing completed run (any `ui_runs/` run served through the app, or run a
short clip), and check at a phone viewport in **both themes**:
- Players card renders in observed and assumed states (toggle by editing the
  run's `job.json` `players_v1.attribution_backend` if only one state is
  reachable).
- Names save, persist across reload, and propagate to the player report
  labels and rally table.
- No layout overflow at 390×844.

- [ ] **Step 3: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add index.html
git commit -m "feat: players card with post-hoc naming and name propagation"
```

---

## Task 8: attribution eval axis + golden-clip proof

**Files:**
- Create: `eval_attribution.py`
- Create: `tests/test_eval_attribution.py`
- Create: `eval_set/attribution-labels-SquashAnalytics.template.json`
- Create: `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md` (generated)

**Interfaces:**
- CLI: `python eval_attribution.py --run-dir <dir> [--labels <json>]
  [--output <md>]` — reads the run's `job.json` `players_v1`, compares
  observed servers against labels, writes the baseline markdown.
- Labels schema:
  ```json
  {"schema": "attribution-labels-v1", "clip": "SquashAnalytics.mp4",
   "rallies": [{"rally_number": 1, "server": 1}, {"rally_number": 2, "server": 2}]}
  ```
  `server` is a player number under the spec convention (player 1 = rally-1
  server). `null` = human could not tell.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval_attribution.py`:

```python
"""eval_attribution: scoring observed servers against human labels."""

from eval_attribution import score_attribution


def test_score_attribution_counts_matches_and_coverage():
    players_v1 = {"rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
        {"rally_number": 2, "server_player_number": 2, "server_source": "observed"},
        {"rally_number": 3, "server_player_number": 1, "server_source": "propagated"},
    ]}
    labels = {"rallies": [
        {"rally_number": 1, "server": 1},
        {"rally_number": 2, "server": 1},
        {"rally_number": 3, "server": 1},
        {"rally_number": 4, "server": 2},
    ]}
    report = score_attribution(players_v1, labels)
    assert report["labeled_rallies"] == 4
    assert report["observed_rallies"] == 2
    assert report["scored_rallies"] == 2       # observed AND labeled
    assert report["correct"] == 1              # rally 1 right, rally 2 wrong
    assert report["accuracy"] == 0.5
    assert report["observed_coverage"] == 0.5  # 2 observed of 4 labeled


def test_score_attribution_skips_null_labels():
    players_v1 = {"rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
    ]}
    labels = {"rallies": [{"rally_number": 1, "server": None}]}
    report = score_attribution(players_v1, labels)
    assert report["scored_rallies"] == 0
    assert report["accuracy"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_eval_attribution.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `eval_attribution.py`**

```python
"""Score observed serve attribution against human rally labels.

Usage:
    python eval_attribution.py --run-dir runs/<id> \
        --labels eval_set/attribution-labels-SquashAnalytics.json \
        --output eval_set/BASELINE-ATTRIBUTION-<date>.md

The labels file is human-produced (see the template). This script is the
only path to claiming attribution "improved" (spec §5)."""

import argparse
import json
from pathlib import Path


def score_attribution(players_v1, labels):
    observed = {
        r["rally_number"]: r["server_player_number"]
        for r in players_v1.get("rallies", [])
        if r.get("server_source") == "observed"
    }
    labeled = {
        r["rally_number"]: r["server"]
        for r in labels.get("rallies", [])
        if r.get("server") in (1, 2)
    }
    all_labeled = [r for r in labels.get("rallies", [])]
    scored = [n for n in labeled if n in observed]
    correct = sum(1 for n in scored if observed[n] == labeled[n])
    return {
        "labeled_rallies": len(all_labeled),
        "observed_rallies": len(observed),
        "scored_rallies": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "observed_coverage": (
            len(scored) / len(all_labeled) if all_labeled else None
        ),
        "mismatches": sorted(n for n in scored if observed[n] != labeled[n]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    job = json.loads((args.run_dir / "job.json").read_text(encoding="utf-8"))
    players_v1 = job.get("players_v1") or {}
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = score_attribution(players_v1, labels)

    lines = [
        "# Serve-attribution baseline",
        "",
        f"- Run: `{args.run_dir}`",
        f"- Labels: `{args.labels}` ({report['labeled_rallies']} rallies)",
        f"- Detector backend: {players_v1.get('detector_backend')}",
        f"- Observed rallies: {report['observed_rallies']}",
        f"- Scored (observed AND labeled): {report['scored_rallies']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']}",
        f"- Observed coverage of labeled rallies: {report['observed_coverage']}",
        f"- Mismatched rally numbers: {report['mismatches']}",
        "",
        "Rallies without observed serves are excluded from accuracy —",
        "coverage reports them honestly (spec §7: no pre-hit ball track ->",
        "`server_track: null`, never a guess).",
    ]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass, write the labels template**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/test_eval_attribution.py -v
```

`eval_set/attribution-labels-SquashAnalytics.template.json`:

```json
{
  "schema": "attribution-labels-v1",
  "clip": "SquashAnalytics.mp4",
  "note": "HUMAN GATE: watch the clip, fill server per rally (1 = the rally-1 server, 2 = the other player, null = cannot tell), save without .template",
  "rallies": [
    {"rally_number": 1, "server": 1}
  ]
}
```

- [ ] **Step 5: Golden-clip run + baseline + perf number**

Run the full pipeline on the golden clip twice via the CLI runner — once
with the detector, once without — and record wall-clock:

```bash
# Prepare a run dir by uploading SquashAnalytics.mp4 through the web UI (or
# reuse an existing runs/<id> whose job.json points at it), then:
time "/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" job_runner.py runs/<id>
PERSON_DETECTOR=none time "/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" job_runner.py runs/<id2>
```

Then generate the baseline (labels may still be the template — the doc must
state so honestly):

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" eval_attribution.py \
  --run-dir runs/<id> \
  --labels eval_set/attribution-labels-SquashAnalytics.template.json \
  --output eval_set/BASELINE-ATTRIBUTION-2026-07-27.md
```

Append to `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md` by hand:
- The two wall-clock numbers (with/without person pass) and
  `PERSON_DETECT_HZ` / computed `detect_every`.
- Whether labels are real or template ("HUMAN GATE pending" if template).
- `/eval` line-call result: run
  `<venv>/bin/python eval_line_calls.py --eval-set eval_set/cases.jsonl`
  and state zero drift vs the newest `eval_set/BASELINE-*.md`.

- [ ] **Step 6: Full suite, then commit**

```bash
"/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.venv/bin/python" -m pytest tests/ -q
git add eval_attribution.py tests/test_eval_attribution.py eval_set/attribution-labels-SquashAnalytics.template.json eval_set/BASELINE-ATTRIBUTION-2026-07-27.md
git commit -m "feat: serve-attribution eval axis with golden-clip baseline"
```

---

## Human gates (the intended stopping line)

1. **Label the golden clip:** copy the template to
   `eval_set/attribution-labels-SquashAnalytics.json`, watch the clip, fill
   the true server per rally, re-run `eval_attribution.py`, update the
   baseline doc.
2. **Checkpoint warm on the server:** run the one-liner in
   `docs/PERSON_MODEL.md` wherever the pipeline deploys (first construction
   downloads the checkpoint).
3. **Review the perf delta** and decide whether `PERSON_DETECT_HZ = 4.0`
   stays (tuning it is a one-constant change + baseline re-run).

## Deviations

(record here during execution)
