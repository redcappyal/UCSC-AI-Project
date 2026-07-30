# Tripod Match Analysis MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Record any divergence in `## Deviations` at the bottom. The kickoff prompt's operating rules override any skill ceremony that conflicts with autonomous execution.

**Goal:** Grow the line-calling pipeline into a tiered match-analysis product: rally
timeline from audio+motion, player movement stats behind a swappable person-detector
seam, automatic court solve, honest capability gating, and a real Matches report — per
`docs/superpowers/specs/2026-07-27-tripod-match-analysis-design.md`.

**Architecture:** All analysis stays in the Python pipeline (`job_runner.py` job graph).
New pure modules (`media_probe.py`, `capabilities.py`, `rally_segmenter.py`,
`court_lines.py`, `auto_solve.py`, `person_model.py`, `player_tracker.py`,
`movement_stats.py`, `match_report.py`) keep one responsibility each and get paired
`tests/test_<module>.py` files (the PostToolUse hook auto-runs the pair on every edit).
**The MVP never depends on the full 3D camera solve** — auto-solve acceptance and
movement mapping run on the floor homography + 2D line fits, which are chirality-safe
and already proven on real footage (the 3D `solve_camera_model` currently fails real
calibrations — a known, separately-tracked defect; do not put it on the critical path).
`index.html` gains the confirm-overlay and report surfaces. No iOS changes.

**Tech stack:** Existing only — Python 3.12 in `.venv` (cv2 4.10, numpy, scipy, PyAV
14.2, torch 2.13, flask), vanilla JS in `index.html`. **No new dependencies.** `ffmpeg`
is NOT on PATH and `librosa`/`soundfile` are absent — audio goes through
`audio_events.audio_to_mono_float` (PyAV/afconvert chain) and video transcode through
PyAV (Task 18).

## Global Constraints

- Branch: `claude/tripod-mvp` created **from clean `main`** (see kickoff Step 0). One
  commit per task. Never push.
- Every command runs through `.venv/bin/python` (system python has no flask/cv2).
- Full suite `.venv/bin/python -m pytest tests/ -q` green before every commit. Gate on
  "collection completes with zero errors and zero failures" — the absolute test count
  grows with this plan and may shift if the in-flight stereo-archive work lands first.
- Judging/calibration changes: run
  `.venv/bin/python eval_line_calls.py --eval-set eval_set/cases.jsonl` per the `/eval`
  skill. **Scope honesty:** that eval replays stored cases through `judge_call` only —
  it cannot see detector/tracking wiring. Where a task needs detector-path equivalence
  (Task 3), the task specifies a hit-replay diff on stored CSVs as the real check.
- Line-call result compatibility: existing `public_job` keys keep their shapes;
  additions only. Do not touch `ios/` sources. Stereo/peer code: don't modify it; if a
  concurrent archive has moved it (`archive/stereo/`), adapt imports per what you find
  and note it in Deviations.
- UI: DESIGN.md is binding (dark-first tokens; verify both themes at 390×844 via the
  `/verify` skill before checking a UI task off). Flask runs as
  `.venv/bin/python app.py` (port 5188) during verification.
- Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite` (CLAUDE.md).
- New thresholds are named module-level constants; every one is exercised by a test or
  eval named in its task.
- app.py route style is `@app.get(...)`/`@app.post(...)` (not `@app.route`).
- Flask tests: `tests/test_app.py` does not exist yet — Task 10 **creates** it. Pattern:
  use the existing `conftest.py` `runs_dir` fixture for anything touching uploads/runs,
  and an inline `client = app_module.app.test_client()` per test (there is no shared
  client fixture; don't invent one).

---

## Phase 1 — Probe and capability model

### Task 1: `media_probe.py`

**Files:**
- Create: `media_probe.py`
- Create: `tests/test_media_probe.py`

**Interfaces:**
- Produces: `def probe_video(video_path) -> dict` →
  `{"fps": float, "width": int, "height": int, "frame_count": int,
    "duration_s": float, "sharpness": float | None, "has_audio": bool}`.
  Sharpness = median over ≤16 evenly-sampled frames of
  `cv2.Laplacian(gray, cv2.CV_64F).var()` on a center crop (half width/height);
  `None` if no frame decodes. `has_audio` via PyAV: open container, any
  `stream.type == "audio"` (wrap in try/except → False).

- [ ] **Step 1: Failing tests** — follow the synthetic-video pattern from
  `tests/test_pipeline.py:474` (`cv2.VideoWriter`, fourcc `mp4v`, tmp_path):

```python
import cv2, numpy as np
import media_probe

def _write_clip(path, fps=30.0, size=(64, 48), frames=30, blur=False):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "mp4v codec unavailable?"
    rng = np.random.default_rng(7)
    for _ in range(frames):
        frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        if blur:
            frame = cv2.GaussianBlur(frame, (15, 15), 6.0)
        writer.write(frame)
    writer.release()

def test_probe_reports_fps_size_and_no_audio(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    p = media_probe.probe_video(clip)
    assert round(p["fps"]) == 30 and (p["width"], p["height"]) == (64, 48)
    assert p["frame_count"] == 30 and p["has_audio"] is False

def test_sharpness_orders_sharp_above_blurred(tmp_path):
    sharp, soft = tmp_path / "sharp.mp4", tmp_path / "soft.mp4"
    _write_clip(sharp); _write_clip(soft, blur=True)
    assert media_probe.probe_video(sharp)["sharpness"] > media_probe.probe_video(soft)["sharpness"]
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_media_probe.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** per the contract (cv2.VideoCapture for fps/dims/count; sample indices via `np.linspace`; guard zero fps with 30.0 like `app.py:91-110`).
- [ ] **Step 4: Pass + full suite.**
- [ ] **Step 5: Commit** — `feat: media probe (fps/size/sharpness/audio) for capability gating`

### Task 2: `capabilities.py` + pipeline threading

**Files:**
- Create: `capabilities.py`
- Create: `tests/test_capabilities.py`, `tests/test_job_runner.py` (both new)
- Modify: `app.py` (`track_clip` ~:1127 stores probe in job; whitelist tuple at ~:139), `job_runner.py` (`run_tracking_job` ~:1243: compute + honor + emit)

**Interfaces:**
- Produces:
  ```python
  BALL_MIN_FPS = 50.0
  BALL_MIN_WIDTH_PX = 1600
  BALL_MIN_SHARPNESS = 40.0     # initial; Task 19 validates on the real pair
  MOVEMENT_MIN_FPS = 12.0

  def compute_capabilities(probe: dict, *, court_solved: bool) -> dict
  ```
  → `{"rally_structure"|"player_movement"|"ball_tracking"|"line_calls":
      {"enabled": bool, "reason": str | None}}` — reason set iff disabled, exact
  strings asserted in tests (e.g. `"needs >=50 fps (got 30)"`, `"no solved court"`).
  `court_solved` = a floor homography exists (wizard or auto calibration).
  `rally_structure` is **always enabled** (audio absence is runtime information —
  it surfaces as `rally_timeline.audio_available` in Task 5, never as a capability
  input). `player_movement` needs `court_solved` and fps ≥ `MOVEMENT_MIN_FPS`.
  `ball_tracking` needs fps/width/sharpness gates AND `court_solved`. `line_calls`
  needs `ball_tracking` enabled (it judges ball hits).
- Job payload gains `"probe"` and `"capabilities"`. Append to the `public_job`
  pass-through tuple at `app.py:139`: `"probe"`, `"capabilities"`,
  `"detection_coverage"`, `"rally_timeline"`, `"players_v2"` (emitted by later tasks;
  `public_job` copies only keys present, so early whitelisting is harmless).
- `run_tracking_job` behavior when `ball_tracking` is disabled: skip model load,
  coarse/refine ball passes, and judging entirely — but **still write a headers-only
  `ball_coordinates.csv`** (so anything that later reads the CSV finds a valid empty
  file) and **do not fire any stereo-fuse trigger** for such runs. Emit `hits=[]`
  plus `capabilities`. When enabled, also emit `"detection_coverage"` =
  detected-rows / processed-rows from the results.

- [ ] **Step 1: Failing pure-function tests** (`tests/test_capabilities.py`):

```python
import capabilities as cap

PROBE_60 = {"fps": 60.0, "width": 1920, "height": 1080, "frame_count": 600,
            "duration_s": 10.0, "sharpness": 120.0, "has_audio": True}
PROBE_30 = dict(PROBE_60, fps=30.0)

def test_ball_tier_gated_by_fps():
    caps = cap.compute_capabilities(PROBE_30, court_solved=True)
    assert caps["ball_tracking"]["enabled"] is False
    assert "50 fps" in caps["ball_tracking"]["reason"]
    assert caps["rally_structure"]["enabled"] is True
    assert caps["player_movement"]["enabled"] is True

def test_no_court_disables_movement_and_ball():
    caps = cap.compute_capabilities(PROBE_60, court_solved=False)
    assert caps["player_movement"]["enabled"] is False
    assert caps["ball_tracking"]["enabled"] is False
    assert caps["line_calls"]["enabled"] is False
    assert caps["rally_structure"]["enabled"] is True
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement. Step 4: Pass.**
- [ ] **Step 5: Thread through** — probe at track start into `job["probe"]`;
  capabilities computed in `run_tracking_job` after calibration resolution; stage
  skip + headers-only CSV + no stereo-fuse on disabled ball tier. Integration test
  in the new `tests/test_job_runner.py`: a 30 fps probe job completes with
  `hits == []`, `capabilities` present, and no model load attempted (monkeypatch
  `job_runner.get_tracking_model` to raise if called).
- [ ] **Step 6: Full suite + /eval** (judge-path unchanged ⇒ expect zero movement).
- [ ] **Step 7: Commit** — `feat: capability gating with honest reasons; ball stages skipped on unqualified footage`

---

## Phase 2 — fps-normalize time-domain constants (equivalence refactor)

### Task 3: Scaled hit/track windows via existing kwargs

**Files:**
- Create: `tests/test_detect_wall_hits.py` (new — creating it activates the pairing hook for `detect_wall_hits.py`)
- Modify: `detect_wall_hits.py` (add `scaled_hit_kwargs` only), `tracking_common.py` (add `scaled_window_frames` only), `job_runner.py` (three `detect_hits_from_rows` call sites at ~:1324, ~:1418, ~:1452; audio pads ~:1330-1402; `window_frames` pass-through at the `select_motion_consistent_ball_predictions` call ~:560-563)

**Interfaces:**
- Produces in `detect_wall_hits.py`:
  `def scaled_hit_kwargs(fps: float) -> dict` →
  `{"max_gap", "min_gap", "smooth"}`, each `max(1, round(CONST * fps / 60.0))`.
  Constants themselves untouched (60 fps reference; current corpus is ~60 fps so
  scaling is identity there).
- Produces in `tracking_common.py`: `def scaled_window_frames(fps: float) -> int` =
  `max(2, round(MOTION_TRACK_WINDOW_FRAMES * fps / 60.0))`.
  `select_motion_consistent_ball_predictions` **already has** a `window_frames`
  parameter — pass the scaled value through at the job_runner call site; no
  signature change.
- **Preserve the deliberate stride floor:** all three job_runner call sites currently
  pass `max_gap=max(MAX_GAP_FRAMES, frame_stride)` (comment at ~:1321-1326 explains
  stride > 3 would split every track). The wiring must be
  `max_gap=max(scaled_hit_kwargs(fps)["max_gap"], frame_stride)` at all three sites.
- Audio pad frames (`AUDIO_WINDOW_PAD_FRAMES`, `AUDIO_RESCUE_PAD_FRAMES`) scaled the
  same way at use.
- Pixel-domain constants (`MOTION_TRACK_MAX_STEP_PX`, `MAX_JUMP_PX_PER_FRAME`, …) are
  **deliberately not scaled**: the ball-tier gate (Task 2) only admits ≥1600 px-wide
  ≥50 fps footage. Add a comment block above the constants saying exactly this.

- [ ] **Step 1: Failing tests** (`tests/test_detect_wall_hits.py`):

```python
from detect_wall_hits import MAX_GAP_FRAMES, MIN_GAP_FRAMES, SMOOTH_WINDOW, scaled_hit_kwargs

def test_scaled_kwargs_identity_at_reference_fps():
    assert scaled_hit_kwargs(60.0) == {"max_gap": MAX_GAP_FRAMES, "min_gap": MIN_GAP_FRAMES, "smooth": SMOOTH_WINDOW}

def test_scaled_kwargs_halve_at_30fps():
    k = scaled_hit_kwargs(30.0)
    assert k["min_gap"] == 5 and k["max_gap"] == 2

def test_stride_floor_survives_scaling():
    # mirrors the job_runner wiring rule: stride wins when larger
    assert max(scaled_hit_kwargs(60.0)["max_gap"], 4) == 4
```

- [ ] **Step 2: FAIL → implement helpers → PASS.**
- [ ] **Step 3: Wire call sites** with the stride floor; add a `tests/test_job_runner.py`
  case asserting the wired kwargs at fps=60/stride=4 equal the pre-refactor values.
- [ ] **Step 4: Real equivalence check (the actual gate for this task):** replay
  `detect_hits_from_rows` over 2-3 stored `ui_runs/*/ball_coordinates.csv` (pick
  runs that also have `detected_hits.json`, e.g. `ui_runs/1784583924415`) before and
  after the refactor and diff the detected hit frames — must be identical. Run
  /eval too (judge path — expect zero movement). Record both results in the commit
  message.
- [ ] **Step 5: Commit** — `refactor: fps-normalize time-domain hit/track windows (identity at 60fps, replay-verified)`

---

## Phase 3 — Rally structure upstream of the ball

### Task 4: `rally_segmenter.py` core (pure)

**Files:**
- Create: `rally_segmenter.py`
- Create: `tests/test_rally_segmenter.py`

**Interfaces:**
- Produces:
  ```python
  MIN_RALLY_S = 2.0
  DEFAULT_GAP_S = 5.0          # mirrors job_runner.DEFAULT_RALLY_GAP_SECONDS
  MIN_GAP_S = 4.0              # mirrors job_runner.RALLY_GAP_MIN_SPLIT_SECONDS

  def segment_rallies(impact_times_s: list[float],
                      motion: list[tuple[float, float]],   # (t_s, energy >= 0)
                      duration_s: float,
                      min_rally_s: float = MIN_RALLY_S,
                      gap_s: float | None = None) -> list[dict]
  ```
  Returns `[{"start_s","end_s","impact_count","source","confidence"}]`, sorted,
  non-overlapping. Algorithm: (1) adaptive gap from largest ratio jump in sorted
  inter-impact gaps ≥ `MIN_GAP_S` (generalize the idea of
  `job_runner.infer_rally_gap_seconds` over plain floats; do not import
  job_runner — keep this module dependency-free); (2) cluster impacts into runs
  split on gaps > gap_s; (3) motion normalized by `median + 3*MAD`; contiguous
  above-threshold spans extend/merge overlapping runs; motion-only spans ≥
  `min_rally_s` become rallies `source="motion"`, impact-backed ones
  `"audio+motion"` or `"audio"`; (4) drop runs < `min_rally_s`; pad ±0.5 s clamped
  to `[0, duration_s]` and neighbors; confidence = `min(1.0, impact_count / 4)`
  floored at `0.3` for motion-only.

- [ ] **Step 1: Failing tests** (deterministic synthetic series):

```python
import numpy as np
from rally_segmenter import segment_rallies

def _motion(spans, duration=60.0, dt=0.2, hi=8.0, lo=0.4):
    ts = np.arange(0.0, duration, dt)
    e = np.full(ts.shape, lo)
    for a, b in spans:
        e[(ts >= a) & (ts <= b)] = hi
    return list(zip(ts.tolist(), e.tolist()))

def test_two_rallies_from_impacts_and_motion():
    impacts = [5.0, 5.8, 6.9, 8.0, 9.2,   30.0, 30.7, 31.9, 33.0]
    rallies = segment_rallies(impacts, _motion([(4.5, 9.5), (29.5, 33.5)]), 60.0)
    assert len(rallies) == 2
    assert rallies[0]["start_s"] <= 5.0 and rallies[0]["end_s"] >= 9.2
    assert rallies[0]["impact_count"] == 5 and rallies[0]["source"] == "audio+motion"

def test_motion_only_rally_low_confidence():
    rallies = segment_rallies([], _motion([(10.0, 16.0)]), 30.0)
    assert len(rallies) == 1 and rallies[0]["source"] == "motion"
    assert rallies[0]["confidence"] == 0.3

def test_short_blips_dropped():
    assert segment_rallies([12.0], _motion([(12.0, 12.6)]), 30.0) == []
```

- [ ] **Step 2: FAIL → implement → PASS → full suite.**
- [ ] **Step 3: Commit** — `feat: audio+motion rally segmenter (pure, ball-independent)`

### Task 5: Motion hook + pipeline integration + `completion_payload`

**Files:**
- Modify: `job_runner.py` — consumer loop of `track_segments` (~:517), motion-only decode path, `run_tracking_job`, completion `update_job` call (~:1491-1504)
- Modify: `rally_segmenter.py` (add `motion_energy_step` + `build_rally_timeline`)
- Test: extend `tests/test_job_runner.py`

**Interfaces:**
- Produces in `job_runner.py`:
  ```python
  def completion_payload(*, hits, target_zones, target_zones_by_player,
                         player_assignment, hits_error, processed_frames=None,
                         rows=None, message="Tracking complete.", extra_fields=None) -> dict
  ```
  — extraction of the kwargs currently built inline at ~:1491-1504 (which also pass
  `status`/`stage`; keep those inside the function). New keys it adds:
  `"rally_timeline"`, `"players_v2"`, `"detection_coverage"` — whenever provided via
  `extra_fields`.
- Produces in `rally_segmenter.py`:
  - `def motion_energy_step(prev_small, frame_bgr) -> tuple[np.ndarray, float]` —
    downscale to width 160 gray, `cv2.absdiff` mean vs previous.
  - `def build_rally_timeline(impact_times_s, motion_series, duration_s, player_assignment) -> dict`
    → `{"rallies": [...segment_rallies...], "gap_s": float, "audio_available": bool,
       "agrees_with_hits": bool | None}` — agreement = every hit-based rally midpoint
    (`player_assignment["rallies"]`) inside some timeline rally; None when no
    hit-based rallies exist.
- Motion accumulation rules: **coarse pass only** (single contiguous full-range
  segment) — never during refine/audio-rescue re-tracks; reset `prev_small` at each
  segment start; key samples by `frame_idx` so re-decodes can't duplicate. When the
  ball tier is disabled (no coarse pass), decode for motion only: reuse
  `decode_segments_to_queue` (~:473) with stride `max(2, round(source_fps / 6))`
  (~6 samples/s) and no inference.
- Impacts: `extract_audio_candidates(video_path, start_frame, end_frame, fps,
  max_peaks=512)` (`audio_events.py:191`; returns None → `audio_available=False`,
  motion-only).

- [ ] **Step 1: Failing integration test** — monkeypatch `extract_audio_candidates`
  to fixed peaks; inject a motion series through the accumulator seam; assert the
  completion payload contains `rally_timeline` with `agrees_with_hits` computed,
  and that `completion_payload(...)` round-trips the existing keys (hits,
  player_assignment, processed_frames, rows).
- [ ] **Step 2: FAIL → implement → PASS.**
- [ ] **Step 3: Full suite + /eval (judge path, expect zero movement) + replay-diff
  from Task 3 Step 4 on one stored CSV to prove hit detection untouched.**
- [ ] **Step 4: Commit** — `feat: rally timeline from audio+motion, emitted independent of ball tier`

### Task 6: Rally-boundary eval axis

**Files:**
- Create: `eval_rally_boundaries.py`, `tools/seed_rally_labels.py`
- Create: `tests/test_eval_rally_boundaries.py`
- Create: `eval_set/rally_labels.jsonl` + `eval_set/BASELINE-RALLY-2026-07-27.md`

**Interfaces:**
- Produces: `def score_rallies(predicted: list[dict], labeled: list[dict], tol_s: float = 1.5) -> dict`
  → `{"tp","fp","fn","precision","recall","f1","count_delta"}` (greedy 1-1 match on
  midpoints within tol). CLI:
  `.venv/bin/python eval_rally_boundaries.py --labels eval_set/rally_labels.jsonl`
  re-runs `segment_rallies` per labeled video (live audio+motion decode) and prints
  per-video + aggregate F1.
- `tools/seed_rally_labels.py` discovery rule (inline, self-contained): glob
  `ui_runs/*/calibration.json`, keep runs whose `job.json` `video_path` still exists
  on disk (~35 expected of 46); skip-and-count the rest. **Stored
  `detected_hits.json` files do not contain `player_assignment`/`rallies`** — the
  seeder recomputes them deterministically:
  `job_runner.assign_front_wall_hit_players(payload["hits"])["rallies"]`. Rows:
  `{"video_path", "rallies": [{"start_s","end_s"}], "provenance": "silver-hit-based", "verified": false}`.
  Header comment marks human spot-check as a gate that lives in spec §7 (not this
  plan).
- Test exercises `score_rallies` on synthetic cases (exact match → f1 1.0; one
  missed → recall 2/3).

- [ ] **Step 1: Failing scorer test → implement → pass.**
- [ ] **Step 2: Seed labels; commit `rally_labels.jsonl` (text, small).**
- [ ] **Step 3: Baseline** — run the CLI; write `eval_set/BASELINE-RALLY-2026-07-27.md`
  (aggregate F1, per-video table, commit hash, segmenter constants). Target ≥0.8 F1
  vs silver labels; tune `rally_segmenter` constants if below and log iterations in
  the doc. State plainly that silver labels are hit-derived (agreement metric, not
  absolute truth).
- [ ] **Step 4: Full suite. Commit** — `feat: rally-boundary eval axis + silver labels + baseline`

---

## Phase 4 — Automatic court solve (homography-gated; the 3D solve stays off the critical path)

### Task 7: `court_lines.py` segment detector

**Files:**
- Create: `court_lines.py`
- Create: `tests/test_court_lines.py`

**Interfaces:**
- Produces:
  ```python
  def detect_line_segments(frame_bgr) -> list[dict]
      # {"x1","y1","x2","y2","angle_deg" (in [-90, 90)), "length_px"}
  def merge_collinear(segments, angle_tol_deg=3.0, perp_tol_px=6.0, gap_px=40.0) -> list[dict]
  ```
  Gray → `cv2.Canny` (thresholds from median: `0.66*m`/`1.33*m`) →
  `cv2.HoughLinesP(rho=1, theta=np.pi/360, threshold=60, minLineLength=frame_w//12, maxLineGap=12)`
  → merge. No color assumptions (the wizard's flood-fill stays the color-aware
  fallback).

- [ ] **Step 1: Failing test** — deterministic synthetic frames:

```python
import numpy as np, cv2
from court_lines import detect_line_segments, merge_collinear

def _frame_with_lines(w=960, h=540):
    img = np.full((h, w, 3), 60, np.uint8)
    cv2.line(img, (40, 120), (920, 118), (230, 230, 230), 3)
    cv2.line(img, (40, 300), (920, 305), (230, 230, 230), 3)
    cv2.line(img, (480, 60), (470, 520), (230, 230, 230), 3)
    return img

def test_detects_three_long_lines():
    merged = merge_collinear(detect_line_segments(_frame_with_lines()))
    horiz = [s for s in merged if abs(s["angle_deg"]) < 10]
    vert = [s for s in merged if abs(s["angle_deg"]) > 80]
    assert len(horiz) >= 2 and len(vert) >= 1
    assert max(s["length_px"] for s in horiz) > 700
```

- [ ] **Step 2: FAIL → implement → PASS → full suite → commit** — `feat: court line-segment detection (Canny+HoughLinesP+merge)`

### Task 8: `auto_solve.py` — segments → calibration-v2 (homography acceptance)

**Files:**
- Create: `auto_solve.py`
- Create: `tests/test_auto_solve.py`

**Interfaces:**
- Produces: `def auto_solve(frame_bgr) -> tuple[dict | None, dict]` — on success a
  `squash-calibration-v2` dict exactly as the wizard's `buildJson()` emits
  (`index.html:2617-2681`): three `lines[]` entries named
  `out_line_lower_edge`/`service_line_top_edge`/`tin_top_edge` with
  endpoints/slope/intercept — **endpoints ordered left-then-right (endpoint[0] is
  the court x=0 side)**, `planes.wall.corners` (4), `planes.floor.landmarks` — each
  entry carrying `{"id", "court_ft"` (copied from `court_model.FLOOR_LANDMARKS`),
  `"refined_px", "tap_px", "method": "auto"}` (consumers silently drop entries
  without `court_ft`), `frame_width/height`, plus top-level `"method": "auto"` and
  `"auto_solve": {"score": float, "n_segments": int}`. Info:
  `{"status": "ok" | "no_candidates" | "rejected", "floor_rms_px": float | None, "tried": int}`.
- **Acceptance gate (chirality-safe, 2D only):** fit the floor homography with
  `court_model.fit_homography` over the derived floor-landmark correspondences and
  accept iff `floor_rms_px <= 4.0 * frame_width / 1920.0` (the same scaled-gate form
  as `CAMERA_MAX_RMS_PX`) AND the three wall lines are vertically ordered
  out < service < tin (image y increasing) AND both wall corners sit on the out
  line within tolerance. **Do NOT call `solve_camera_model` as a gate** — it
  currently rejects real-footage correspondences (known chirality defect, tracked
  separately).
- Candidate search: near-horizontal segments in the upper ~2/3 by y-order →
  (out, service, tin) triples; floor landmarks from intersections of short-line /
  half-court / service-box segments in the lower region; wall corners from out-line
  ∩ side-wall segments and floor-seam intersections. Cap `tried ≤ 200`, keep best
  by `floor_rms_px`.
- Consumes: `court_lines` (Task 7), `court_model.fit_homography` / `FLOOR_LANDMARKS`.

- [ ] **Step 1: Failing round-trip test** — synthetic render with **real image
  chirality**, following the repo's own `SyntheticCamera` pattern in
  `tests/test_court_model.py:44-53` (its "Court y = 32 - Yp" convention matches
  real footage; do NOT use `tests/synthetic3d.make_camera`, which renders mirrored
  frames). Render an explicit 3D segment list — there is no ready wireframe
  helper: `court_model.FLOOR_WIREFRAME` is 2D floor-only, so lift it to z=0 and add
  the front-wall lines yourself:

```python
import numpy as np, cv2, court_model
from auto_solve import auto_solve

def _segments_3d():
    segs = [((ax, ay, 0.0), (bx, by, 0.0)) for (ax, ay), (bx, by) in court_model.FLOOR_WIREFRAME]
    for h in (court_model.OUT_LINE_HEIGHT_FT, court_model.SERVICE_LINE_HEIGHT_FT, court_model.TIN_TOP_HEIGHT_FT):
        segs.append(((0.0, 0.0, h), (court_model.COURT_WIDTH_FT, 0.0, h)))
    for x in (0.0, court_model.COURT_WIDTH_FT):        # side-wall out-line verticals at the front plane
        segs.append(((x, 0.0, court_model.TIN_TOP_HEIGHT_FT), (x, 0.0, court_model.OUT_LINE_HEIGHT_FT)))
    return segs

def test_auto_solve_recovers_synthetic_court():
    cam = SyntheticCamera(...)   # copy the construction from tests/test_court_model.py:44-53
    img = np.full((720, 1280, 3), 70, np.uint8)
    for a, b in _segments_3d():
        pa, pb = cam.project(a), cam.project(b)
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (235, 235, 235), 2)
    calib, info = auto_solve(img)
    assert info["status"] == "ok" and calib is not None
    by_id = {l["id"]: l for l in calib["planes"]["floor"]["landmarks"]}
    for lm in court_model.FLOOR_LANDMARKS:
        if lm.get("optional") or lm["id"] not in by_id:
            continue
        want = cam.project((*lm["court_ft"], 0.0))
        got = by_id[lm["id"]]["refined_px"]
        assert abs(want[0] - got[0]) < 8 and abs(want[1] - got[1]) < 8
```

  (Exact constant/attribute names: wall heights at `court_model.py:516-525`,
  `FLOOR_WIREFRAME` at `:163`, landmark dicts at `:74-158` with `"court_ft"` keys.
  If names differ in detail, follow the code and note it in Deviations.)
- [ ] **Step 2: FAIL → implement → PASS.** Add a laterally-offset second pose to
  prevent centered-pose overfitting.
- [ ] **Step 3: Full suite → commit** — `feat: automatic court solve, homography-gated (synthetic round-trip, real chirality)`

### Task 9: Auto-solve validation against stored wizard calibrations

**Files:**
- Create: `eval_auto_solve.py`
- Create: `tests/test_eval_auto_solve.py` (scorer only)
- Create: `eval_set/BASELINE-AUTOSOLVE-2026-07-27.md`

**Interfaces:**
- Population reality (verified 2026-07-27): 46 stored calibrations = **7
  squash-calibration-v2** (floor landmarks present) + 27 v1 + 12 unversioned
  (v1/unversioned carry only the three wall `lines`); ~38 runs still resolve a
  `job.json` `video_path` on disk. So the CLI reports **solve rate over all
  frame-recoverable calibrations (~38)**, **landmark deltas over the v2 subset
  (7)**, and a **line fallback metric for the rest**: per wall line, mean |y_auto −
  y_wizard| sampled at x = 0.25/0.5/0.75 × frame_width.
- Produces: CLI `.venv/bin/python eval_auto_solve.py` — per run: seek the
  calibration frame (`frame_time_s`, via `cv2.CAP_PROP_POS_MSEC`), run `auto_solve`,
  score `def landmark_delta(auto_calib, wizard_calib) -> dict`
  (`{"matched","median_px","max_px"}` over shared landmark ids, `refined_px` falling
  back to `tap_px`) and `def line_delta(auto_calib, wizard_calib, frame_width) -> dict`
  (`{"out","service","tin"}` mean y-deltas). Prints per-run rows + aggregates.
- Consumes: `auto_solve` (Task 8).

- [ ] **Step 1: Failing scorer tests** (literal calibration dict stubs for both
  metrics) → implement → pass.
- [ ] **Step 2: Run on real data; iterate.** Spec §8 target: accepted solutions on
  ≥70% of frame-recoverable calibrations, median landmark delta ≤ 15 px on the v2
  subset. Log every iteration in the baseline doc. If the ceiling can't be met,
  record best numbers + failure taxonomy (glass glare, occluded floor, …) — the
  wizard fallback keeps the product whole.
- [ ] **Step 3: Write the baseline doc. Full suite. Commit** — `feat: auto-solve eval vs stored wizard calibrations + baseline`

### Task 10: Calibration becomes optional at the API

**Files:**
- Modify: `app.py` — `track_clip` empty-calibration branch (the current 400 comes
  from `json.loads("")` at ~:1136-1139; the "effectively required today" comment is
  at ~:1186-1188), plus new `POST /api/auto-solve`
- Create: `tests/test_app.py` (new file; see Global Constraints for the pattern)

**Interfaces:**
- Produces: `track_clip` with **empty/absent** `calibration_json` proceeds with
  `calibration = None` and `job["calibration_source"] = "none"` — guarding the two
  crash points on that path: skip `validate_floor_calibration` when None (it does
  `calibration.get(...)` at ~:1112, called ~:1223) and skip the unconditional
  `calibration.json` write (~:1231-1232). Malformed non-empty JSON keeps the 400.
  (No embedded server-side auto-solve here: the web client submits the auto-solve
  result through the existing `calibration_json` field — single solve entry point.
  `calibration_source` is derived from the submitted calibration's `method` field:
  `"auto"`, `"wizard"` when absent, `"none"` when no calibration.)
- Produces: `POST /api/auto-solve` `{video_id, time_s}` →
  `{ok, status, calibration | null, floor_rms_px, skeleton_px: [[x1,y1,x2,y2], ...]}`
  — skeleton = the court wireframe mapped through the solved floor homography
  (floor segments) plus the three fitted wall lines' endpoints (no 3D solve).
- Consumes: `auto_solve` (Task 8), capabilities threading (Task 2:
  `court_solved = calibration is not None`).

- [ ] **Step 1: Failing Flask tests** (`tests/test_app.py`, new): (a) `/api/track`
  with empty calibration on a tiny synthetic upload → 200, job created,
  `calibration_source == "none"`; (b) malformed JSON → 400; (c) `/api/auto-solve`
  on an uploaded synthetic-court clip (reuse Task 8's renderer written to mp4 via
  `cv2.VideoWriter`) returns `status == "ok"` and a nonempty skeleton.
- [ ] **Step 2: FAIL → implement → PASS → full suite + /eval (zero movement) → commit** — `feat: calibration optional at /api/track; auto-solve endpoint for the confirm flow`

### Task 11: Web confirm overlay (Accept / Adjust / Skip)

**Files:**
- Modify: `index.html` — after `ensureVideoUploaded()` (~:4330) call
  `/api/auto-solve`; draw `skeleton_px` on the shared canvas (pattern of
  `drawFitLine` ~:2465 / `drawWallOverlay` ~:2524); Accept submits the returned
  calibration through the existing `calibration_json` form field (exactly like the
  wizard path at ~:4392); Adjust drops into the wizard (`setPhase('tap_out')`)
  pre-seeded by writing auto landmarks where `snapLandmark` (~:2946) would, then
  `refitFloor()` (~:2989); **Skip** ("Analyze without court lines") submits with
  empty `calibration_json` — Task 10 accepts it and the report will show
  movement/line-call tiers disabled with reasons.
- Verification: `/verify` skill run (no JS unit rig exists).

- [ ] **Step 1: Implement per DESIGN.md** — §8 components, tokens only; skeleton uses
  the `courtLine` styling; Accept primary, Adjust secondary, Skip tertiary;
  auto-solve failure routes into the wizard flow with Skip still offered.
- [ ] **Step 2: Verify** — `/verify`: load `SquashAnalytics.mp4` via
  `setInputFiles('#fileIn', ...)`; assert (a) overlay renders when auto-solve
  succeeds, (b) Accept reaches `p-analyze` with zero wizard taps, (c) Skip reaches
  `p-analyze` with empty calibration. Screenshots both themes at 390×844.
- [ ] **Step 3: Commit** — `feat(web): auto-solve confirm overlay with wizard and skip fallbacks`

---

## Phase 5 — Player movement tier

### Task 12: `person_model.py` seam (mirrors ball_model.py)

**Files:**
- Create: `person_model.py`, `docs/PERSON_MODEL.md`
- Create: `tests/test_person_model.py`

**Interfaces:**
- Produces:
  ```python
  PERSON_SCHEMA_VERSION = "person-model-v1"
  DEFAULT_PERSON_MODEL_DIR = <repo>/models/crosscourt-person-v1   # env PERSON_MODEL_DIR

  @dataclass(frozen=True)
  class PersonDetection: x: float; y: float; width: float; height: float; confidence: float

  class MotionBlobPersonDetector:      # backend "motion" — works today, no weights
      def detect(self, frame_bgr) -> list[PersonDetection]
  class TorchPersonDetector:           # backend "torch" — manifest + model.torchscript
      def detect(self, frame_bgr) -> list[PersonDetection]
  def available_backend() -> str       # "torch" iff manifest+artifact load, else "motion"
  def load_person_detector(backend: str | None = None)   # .detect + .backend attr
  ```
  Manifest handling copies `ball_model.py` (`load_manifest` :56 — schema check,
  artifact presence, loud raises, per-dir cache); torch runner follows
  `TorchScriptRunner` (:142) but single full-frame letterboxed pass (no tiling).
  MotionBlob: `cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32,
  detectShadows=True)`, morph-open, contours with bbox height ≥ `frame_h * 0.12`,
  top-2 by area.
- `docs/PERSON_MODEL.md` (write fully): weights-dir contract mirroring
  `ball-model-v1` (`class_names: ["person"]`), recommended source (COCO-pretrained
  YOLOX-Nano person class, Apache-2.0 — matches `yolox_ball_exp.py` /
  `export_ball_model.py` infra), export steps patterned on `ios/MODEL.md` §2b, and:
  "Provisioning these weights is a human gate (spec §7); the motion backend keeps
  the tier functional meanwhile."

- [ ] **Step 1: Failing tests** — (a) `available_backend() == "motion"` without a
  weights dir; (b) MotionBlob on a synthetic sequence (gray background, one moving
  40×90 bright rectangle over 30 frames) yields ≥1 detection overlapping the
  rectangle in later frames; (c) manifest validation raises on wrong
  schema_version (temp dir, bad manifest.json).
- [ ] **Step 2: FAIL → implement → PASS → full suite → commit** — `feat: person-detector seam with motion-blob fallback (weights are a human gate)`

### Task 13: `player_tracker.py` (pure logic)

**Files:**
- Create: `player_tracker.py`
- Create: `tests/test_player_tracker.py`

**Interfaces:**
- Produces:
  ```python
  COAST_MAX_S = 1.0

  @dataclass
  class TrackSample: t_s: float; foot_px: tuple[float, float]; court_xy: tuple[float, float] | None; confidence: float

  class TwoPlayerTracker:
      def __init__(self, px_to_court=None)     # callable (x_px, y_px) -> (x_ft, y_ft) | None
      def update(self, t_s: float, detections: list[PersonDetection]) -> None
      def tracks(self) -> tuple[list[TrackSample], list[TrackSample]]
      def identity_confidence(self) -> float   # 1 - ambiguous_assignments / assignments
  ```
  Foot point = bbox bottom-center. **Distance-only** greedy min-cost assignment
  (court distance when mapped, else px distance / frame diagonal) over ≤2 tracks ×
  K detections; unmatched tracks coast at last position ≤ `COAST_MAX_S`, then
  samples stop (no fabrication). An assignment is "ambiguous" when the two possible
  pairings' costs differ by < 20% — crossing players surface as reduced
  `identity_confidence`, not as a guess. (Deliberate MVP cut: no appearance/kit
  re-ID — the motion-blob backend merges crossing players into one blob anyway;
  appearance re-ID arrives with real detector weights, spec §7.)
- Consumes: `PersonDetection` (Task 12).

- [ ] **Step 1: Failing tests** — scripted detections, no video: (a) two
  well-separated walkers keep identity over 50 steps (assert per-track
  monotonicity); (b) 0.5 s dropout → coast then reacquire, sample count reflects
  the gap; (c) crossing paths → `identity_confidence() < 1.0` and both tracks
  still emit ≤1 sample per update.
- [ ] **Step 2: FAIL → implement → PASS → full suite → commit** — `feat: two-player tracker with coasting and ambiguity-aware identity confidence`

### Task 14: `movement_stats.py` + pipeline integration

**Files:**
- Create: `movement_stats.py`
- Create: `tests/test_movement_stats.py`
- Modify: `job_runner.py` (person pass on the decode paths from Task 5; emit `players_v2`); extend `tests/test_job_runner.py`

**Interfaces:**
- Produces:
  ```python
  T_RADIUS_FT = 6.0
  SPEED_MAX_FTPS = 30.0
  SMOOTH_WINDOW_S = 0.5
  GRID_X, GRID_Y = 7, 8

  def movement_stats(samples: list[TrackSample], rallies: list[dict]) -> dict
  ```
  → `{"distance_ft","avg_speed_ftps","p95_speed_ftps","t_time_pct","front_pct",
     "back_pct","heatmap","sample_coverage"}` — rally-scoped samples only (inside
  `rally_timeline` windows), positions smoothed over `SMOOTH_WINDOW_S`, per-step
  speed clamped to `SPEED_MAX_FTPS`, T at the `t_point` entry of
  `court_model.FLOOR_LANDMARKS` (court_ft (10.5, ≈17.918)), front/back split at
  y = 16.0 ft, heatmap `GRID_Y×GRID_X` row-major normalized to sum 1.0 (or zeros),
  `sample_coverage` = fraction of rally time with a live (non-coasted) sample.
- Pipeline: person detection every
  `PERSON_DETECT_STRIDE = max(1, int((source_fps / stride_used) / 4))` processed
  frames (floor semantics — ≥4 detections/s on the ball-tier decode; ~6/s on the
  sparse motion-only decode where `stride_used` is Task 5's motion stride).
  Feet map through the **floor homography** from the active calibration
  (invert `planes.floor.homography_image_from_court`, exactly what `refitFloor`
  does in JS — build `px_to_court` from `court_model.fit_homography` output or the
  stored matrix; no 3D solve). Emit
  `"players_v2": {"backend", "identity_confidence", "player_a": stats, "player_b": stats}`
  via `completion_payload`. Tier honors `capabilities["player_movement"]`.
- Consumes: Tasks 4/5 (rally windows), 12, 13, capabilities (Task 2).

- [ ] **Step 1: Failing tests** — synthetic tracks: (a) stationary-at-T →
  `t_time_pct == 1.0`, `distance_ft ≈ 0`; (b) back-corner shuttle → known distance
  within 5%, `front_pct == 0.0`; (c) heatmap sums to 1.0, peaks in the correct
  cell; (d) samples outside rallies excluded (`sample_coverage` reflects it).
- [ ] **Step 2: FAIL → implement → PASS.**
- [ ] **Step 3: Integration** — extend the Task 5 integration test with a stub
  detector via `load_person_detector` monkeypatch; assert `players_v2` in the
  completion payload with `backend == "motion"`.
- [ ] **Step 4: Full suite + /eval (zero movement) → commit** — `feat: per-player movement stats behind capability gate`

---

## Phase 6 — Match report

### Task 15: `match_report.py` + endpoints

**Files:**
- Create: `match_report.py`
- Create: `tests/test_match_report.py`
- Modify: `app.py` (add `GET /api/runs`, `GET /api/runs/<run_id>/report`); extend `tests/test_app.py`

**Interfaces:**
- Produces: `def build_report(run_dir: Path, coach_builder=None) -> dict` — schema
  `"report-v1"`:
  `{"schema","run_id","created_ms","probe","capabilities","detection_coverage",
    "rally_timeline","players_v2","shots": {...} | None,
    "video": {"duration_s","fps","width","height"}}` from `job.json` +
  `detected_hits.json`. **Tolerant assembly** — every new-key read is
  `job.get(..., None)`: ~38 legacy runs on disk predate these keys; they render
  with a `capabilities = None` → "legacy run — analyzed before capability gating"
  card, never a KeyError. `created_ms` = the numeric run-dir name (dirs are
  epoch-ms; hence the numeric-only listing rule). `shots.coach` comes from the
  injected `coach_builder` — the app endpoint passes the same
  `build_coaching_analytics` helper the `/coach` route uses (`app.py:434-482`);
  None when the ball tier was off. Missing `job.json` → `FileNotFoundError`
  (endpoint 404s).
- Produces: `GET /api/runs` → `{"ok": true, "runs": [{run_id, created_ms, status,
  duration_s, tiers_enabled: [str]}]}` — numeric-named dirs with `job.json` only,
  newest-first, cap 100. `GET /api/runs/<run_id>/report` → report JSON.
- Consumes: payload keys from Tasks 2/5/14.

- [ ] **Step 1: Failing tests** — three fixture run dirs in tmp_path: full-tier,
  rally-only, and **legacy** (`job.json` with only hits/target_zones/
  player_assignment-era keys) → assert shapes for all three; Flask client asserts
  list ordering + 404 on unknown run.
- [ ] **Step 2: FAIL → implement → PASS → full suite → commit** — `feat: report-v1 assembly + runs index/report endpoints (legacy-tolerant)`

### Task 16: Matches tab becomes the report surface

**Files:**
- Modify: `index.html` — replace the `#p-matches` placeholder (~:1064) with a list
  view (fetch `/api/runs`) + a new `#p-report` section (fetch report JSON).
  Registration: add `report` to `STEP_META` (~:1253) and to `setPhase`'s hide list
  (~:1334-1336); decide `SECTION_ROOTS` membership for back-button behavior
  (~:1287); deep-link `#report=<run_id>` in `applyDeepLink()` (~:6460). **Do not
  touch `stepSequence()`** — it is the numbered wizard ordering, not tab routing.
  While in the file: point the one `|| 30` fps fallback at ~:4281 at the same
  helper the other two sites use (they default 60 — make all three consistent).
- Verification: `/verify` skill run.

- [ ] **Step 1: Implement per DESIGN.md** — list = existing card components; report =
  capability cards (disabled tiers show their `reason` verbatim; ball card shows
  `detection_coverage` as "ball detected in N% of processed frames"; legacy runs
  get the legacy card), rally strip (reuse `.rallySegment` CSS ~:233-241 pattern),
  per-player heatmap over `courtWireframeSvg('courtLine')` (~:3048) as an SVG cell
  grid (pattern of `renderTargetZones` ~:4541 + `targetZoneFill` ~:4458), stat
  tiles (distance, speeds, T-time %, front/back), shots section only when present,
  "winner (est.)" labeling on inferred rally winners, and a `backend: motion`
  badge when `players_v2.backend != "torch"`. Load the dataviz skill for
  heatmap/tile design choices; DESIGN.md §12 wins on conflict.
- [ ] **Step 2: Verify** — `/verify`: run a cheap tracked clip (0.6 s window,
  `frame_stride=10`, `inference_width=640` per the skill), open `#tab=matches`,
  assert the run lists and its report renders; screenshots both themes 390×844.
- [ ] **Step 3: Commit** — `feat(web): real Matches tab — run list + report-v1 surface`

---

## Phase 7 — Ingest robustness + end-to-end proof

### Task 17: Chunked upload

**Files:**
- Modify: `app.py` (`POST /api/upload/init`, `POST /api/upload/chunk/<upload_id>`,
  `POST /api/upload/complete/<upload_id>`; reuse `upload_video`'s by-hash store
  ~:925), `index.html` (`ensureVideoUploaded` ~:4330 branches to chunked when
  `file.size > 512 * 1024 * 1024`)
- Extend: `tests/test_app.py`

**Interfaces:**
- Produces: init `{filename, size}` → `{upload_id}`; **init persists the sanitized
  filename suffix** (part-file named `<upload_id><ext>`, default `.mp4` exactly like
  `app.py:931`) because `video_path_for_id` resolves ids via
  `BY_HASH_DIR.glob(f"{video_id}.*")` (`app.py:921`) — an extensionless assembled
  file would be unresolvable. Chunk = raw body ≤ 32 MB with `?index=N` strictly
  sequential (409 on gap/repeat, 413 oversize); complete → sha256 the assembly,
  move to `by-hash/<sha256><ext>` (dedupe like `upload_video`), return the same
  `{video_id, fps, ...}` shape as `/api/upload`. **The partial dir must be computed
  from patched app state** (e.g. `BY_HASH_DIR.parent / "partial"`) so the
  `runs_dir` test fixture sandboxes it — no module-level absolute constant.
- [ ] **Step 1: Failing Flask tests** — 3-chunk roundtrip assembles byte-identical
  content and the returned video_id resolves; out-of-order chunk → 409; oversized
  chunk → 413; partials land under the tmp runs tree (fixture assertion).
- [ ] **Step 2: FAIL → implement → PASS.**
- [ ] **Step 3: Web client** — `uploadFileChunked(file, progressLabel)` (16 MB
  slices, sequential, progress via the existing `postFormRequest` progress UI);
  small files keep the legacy path. `/verify` smoke: normal-size upload still
  works end-to-end.
- [ ] **Step 4: Full suite → commit** — `feat: chunked upload past the 2GB single-POST wall`

### Task 18: PyAV proxy tool (the 1080p30 test asset)

**Files:**
- Create: `tools/make_proxy.py`
- Create: `tests/test_make_proxy.py`

**Interfaces:**
- Produces: `def make_proxy(src, dst, target_fps=30.0, target_width=1920) -> dict`
  (returns `media_probe.probe_video(dst)`) — PyAV decode → drop frames to hit
  `target_fps` (pts-accumulator), scale to `target_width` keeping AR
  (`frame.reformat`), encode `libx264` (`crf=23`) + `aac` audio when the source has
  an audio stream. **ffmpeg is not on PATH; PyAV (`av` 14.2) is the only sanctioned
  transcode path.** CLI:
  `.venv/bin/python tools/make_proxy.py <src> <dst> --fps 30 --width 1920`.
- [ ] **Step 1: Failing test** — source = **1 s, 60 fps, 640×360** synthetic clip
  (cv2 writer); `make_proxy(src, dst, target_fps=30.0, target_width=320)`; assert
  `27 <= probe["frame_count"] <= 33`, `probe["width"] == 320`, and
  `round(probe["fps"]) == 30`. (Source has no audio → `has_audio is False`; the
  audio path is exercised on real footage in Task 19.)
- [ ] **Step 2: FAIL → implement → PASS → full suite → commit** — `feat: PyAV proxy tool for camera-roll-grade test footage`

### Task 19: End-to-end pair + baselines + handoff (final gate)

**Files:**
- Create: `docs/HANDOFF-tripod-mvp.md`
- Modify: the two new baseline docs (final numbers), this plan's `## Deviations`

- [ ] **Step 1: Build the pair** — `SquashAnalytics.mp4` (repo root; **1080p60**,
  1920×1080 @ 60 fps, 311.9 s — it passes the ball gate: 1920 ≥ 1600, 60 ≥ 50) and
  its proxy `tools/make_proxy.py SquashAnalytics.mp4 ui_runs/uploads/e2e-proxy-1080p30.mp4 --fps 30`.
  Verify the proxy kept audio (`media_probe` → `has_audio is True`) — this is the
  real-footage audio test for Task 18.
- [ ] **Step 2: Run both through the full stack** (server per `/verify`; real
  `/api/track`, no monkeypatches). Calibration: submit a stored wizard calibration
  for this video (several `ui_runs` were made from it) OR the auto-solve result if
  it accepts — the point of this step is tier behavior, not auto-solve quality
  (that's Task 9's baseline). Assert via `/api/runs/<id>/report`:
  - 60 fps run: `ball_tracking.enabled true`, nonempty `rally_timeline`,
    `players_v2.backend == "motion"`, shots present, `detection_coverage` > 0.
  - 30 fps proxy run: `ball_tracking.enabled false` with the fps reason string,
    nonempty `rally_timeline`, movement stats present (court was solved), report
    renders.
  This validates the Task 2 gate constants against reality — tune them there if
  either side lands wrong, and note it in Deviations.
- [ ] **Step 3: `/verify` final pass** — both reports screenshotted in both themes
  at 390×844; console clean.
- [ ] **Step 4: Full suite + /eval final** — line-call axes unchanged vs newest
  `eval_set/BASELINE-*.md`; rally + autosolve baselines finalized with commit hash.
- [ ] **Step 5: Write `docs/HANDOFF-tripod-mvp.md`** with exactly these sections:
  *What shipped* (tier table + endpoints + surfaces), *Eval baselines* (three
  files, headline numbers), *Human gate 1: person weights* (contract from
  `docs/PERSON_MODEL.md`; effect: replaces the `backend: motion` badge),
  *Human gate 2 (optional): ball retrain* (standing recall headline 71/109),
  *Human gate 3: verify silver rally labels* (`eval_set/rally_labels.jsonl`,
  `verified:false` rows), *Deviations summary*, *How to demo* (exact commands).
- [ ] **Step 6: Commit** — `docs: tripod-MVP handoff — human gates are the only remaining work`

---

## Deviations

(Executor: append dated entries here when the code contradicts the plan; the code wins.)
