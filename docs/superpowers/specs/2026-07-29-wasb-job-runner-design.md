# WASB ball detector in job_runner — design

Date: 2026-07-29
Status: approved, awaiting plan

## Problem

The analysis pipeline's ball detector is still the hosted Roboflow RF-DETR (`squashai/1`)
via `inference_engine.get_tracking_model()`, with ~35% recall at rally scale (71/109
bounces missed). The recall replacement — the fine-tuned WASB temporal model — is trained,
exported, and committed (`models/crosscourt-wasb-416-v1`, val F1 0.710, P 1.000 / R 0.551
on the shared-rig val split), and `ball_model.py` / `ball_detector.py` /
`ball_track_offline.py` already know how to run it. But nothing in `app.py` or
`job_runner.py` references it: analysis jobs cannot use the model the project has been
investing in. Detection recall gates every downstream statistic, so this wiring is the
single highest-leverage pipeline change available.

## Goals

1. Analysis jobs run the local WASB detector by default; the hosted RF-DETR stays
   reachable for A/B eval runs.
2. The temporal model is fed what it was trained on: consecutive 3-frame windows at
   native resolution, on both the strided coarse pass and the stride-1 refine passes.
3. GPU support where it matters (CUDA auto), without risking this 8 GB Air (MPS opt-in).
4. Every run says which detector produced its numbers.
5. Zero behavior change downstream of the ball CSV rows: hits, refine, audio rescue,
   bounce GB model, judging, person/motion passes, and the report pipeline are untouched.

## Non-goals

- No recall/accuracy claims. That is /eval's job, after the wiring lands
  (CLAUDE.md: scored against the newest `eval_set/BASELINE-*.md`).
- No retraining, no export changes, no manifest schema changes.
- No removal of the rfdetr path (that is a later decision, informed by eval).
- No live/on-device inference changes; `ios/` is untouched.

## 1. Backend seam

`job_runner` adopts `ball_track_offline`'s backend contract verbatim: the `BALL_DETECTOR`
env var, values `local` (default; alias `yolox` accepted) or `rfdetr`. `job_runner`
imports `selected_detector()` from `ball_track_offline` — that module is deliberately
import-light (all heavy deps lazy), and one source of truth for the env contract beats
two copies drifting.

At the single call site (`job_runner.py:1564`):

- `local` → `ball_model.load_detector()` — WASB TorchScript + manifest.
- `rfdetr` → `get_tracking_model()` — unchanged hosted path.

The job status message names the backend (`"Loading ball detector: local
(crosscourt-wasb-416 v1)"` / `"... rfdetr (squashai/1)"`).

**No silent fallback in either direction.** A missing artifact, missing torch, or bad
manifest raises and fails the job with the loader's own message. This mirrors
`_build_infer`'s documented philosophy: a silent swap would make the local/rfdetr split
invisible.

With the default now `local`, analysis jobs no longer require `ROBOFLOW_API_KEY` for ball
tracking. The key check moves inside the rfdetr branch. (The person detector uses the
`rfdetr` *package* with its own downloaded weights — no Roboflow API involvement.)

## 2. Default model dir

`ball_model.DEFAULT_MODEL_DIR` changes from `models/crosscourt-ball-416-v1` (YOLOX,
gitignored, absent on every fresh clone) to `models/crosscourt-wasb-416-v1` (committed,
ships with the repo). `BALL_MODEL_DIR` still overrides, so the YOLOX artifact remains
reachable for anyone who has exported it.

Check the one `requires_model` test and any MODEL.md references: if they assume the
YOLOX default dir, they are updated to pin their dir explicitly via `BALL_MODEL_DIR`
or a `model_dir=` argument rather than riding the default.

## 3. Temporal decode — strided centers + neighbors

The model was trained on consecutive frames, so a strided coarse pass may not feed it
`[t−s, t, t+s]`. Instead, the stride controls which frames get *detected*, and each
detected center gets its true neighbors:

`decode_segments_to_queue` grows a temporal mode (active only for the local backend).
For each center frame t in the segment's stride pattern it enqueues `(t, [f(t−1), f(t),
f(t+1)])`:

- Frames adjacent to a center are decoded with `read()`; frames not needed by any window
  are skipped with `grab()` exactly as today.
- At stride 1 (refine and audio-rescue segments) this degenerates to a sliding 3-buffer:
  every frame decoded once, reused across up to three windows.
- At stride 2, t+1 of one window is t−1 of the next; the buffer handles reuse, no frame
  is decoded twice.
- Segment edges pad by repeating the first/last frame, mirroring
  `ball_track_offline._centered_windows`, so serving matches training everywhere.

On the rfdetr branch the queue payload is unchanged — `(frame_idx, frame)`, exactly
today's shape. Only the local branch enqueues `(frame_idx, [prev, cur, nxt])`.
`track_segments` branches on backend:

- local → `ball_detector.detect_frame_stack(runner, frames, runner.manifest)`
- rfdetr → `infer_frame_predictions(model, frame, ...)` unchanged.

`frame_observer` (person pass, motion accumulator) fires once per **center** frame, with
the center frame only. Person-detection cadence and motion sampling are byte-identical to
today; neighbor frames are invisible to observers. Memory note: the temporal queue holds
3 frames per item; `DECODE_QUEUE_SIZE` is revisited (likely divided by 3) so worst-case
buffered frame count does not triple.

`detect_frame_stack` returns the same normalized prediction dicts
(`{"x","y","width","height","confidence","class"}`) the rest of `track_segments` already
consumes; `select_motion_consistent_ball_predictions` and `ball_csv_row` run unchanged.

## 4. Native resolution

The local path never resizes: `inference_width` / `COARSE_INFERENCE_WIDTH` apply only to
the rfdetr branch. Tiling owns scale (`ios/MODEL.md` §6: the model was trained on 416 px
windows cut at native resolution; downscaling makes a 10 px ball ~3 px). Coarse-pass cost
control is the stride, not the width. This is the main runtime cost of the change:
~15 tiles/frame at 1080p, ~66 at 4K, batched at `max_batch_tiles` (32).

## 5. Confidence floor per backend

`select_motion_consistent_ball_predictions`' evidence floor stays `CONFIDENCE_THRESHOLD`
(0.40) for rfdetr but becomes the manifest's `conf_threshold` (0.1) for the local
backend. This is deliberate, not incidental: the temporal-detection research concluded
the motion-consistency scorer *is* the low-confidence rescue mechanism (the ByteTrack
idea, absorbed). The manifest's floor is what the detector already applied; passing 0.40
would silently discard the recall the model was fine-tuned to recover. The
`_build_infer` guard ("manifest conf_threshold must not exceed the requested
confidence") carries over to the job_runner path.

## 6. Device support

`ball_model.load_detector()` reads a new `BALL_DEVICE` env var:

- unset/auto → `cuda` when `torch.cuda.is_available()`, else `cpu`. MPS is **not**
  auto-selected: on an 8 GB unified-memory machine, HRNet over 32-tile batches is a
  realistic memory-pressure panic (one happened this morning).
- explicit `cuda` / `mps` / `cpu` → used as given; an unavailable explicit device raises
  rather than falling back.

The module stays `torch.jit.load(..., map_location="cpu")` then `.to(device)`; both
runners move input tensors to the module's device and results come back via
`.detach().cpu()` as today. CPU-only environments see zero change.

## 7. Run attribution

The job record (`job.json`) and `report-v1` gain a `ball_backend` block:

```json
{"backend": "local", "name": "crosscourt-wasb-416", "version": 1,
 "artifact_sha256": "6eb5a6d3...", "device": "cuda"}
```

(for rfdetr: `{"backend": "rfdetr", "model_id": "squashai/1"}`). Same honesty contract
as `players_v2.backend`: a run's numbers are attributable to the detector and device
that produced them. `ball_model.manifest_summary()` already exposes the fields.

## 8. Failure modes

| Failure | Behavior |
|---|---|
| Artifact dir missing (local) | Job fails at load with the manifest loader's error |
| torch missing (local) | Job fails at load with the ImportError message |
| Explicit `BALL_DEVICE` unavailable | Job fails at load, names the device |
| `BALL_DETECTOR` unknown value | `selected_detector()` raises ValueError |
| No `ROBOFLOW_API_KEY` (rfdetr only) | Existing RuntimeError, now only on that branch |
| Decode error mid-segment | Unchanged: first decode error re-raised after join |

No fallback chains anywhere: every failure is loud and names its cause.

## 9. Testing

TDD per task, in the paired `tests/test_*.py` files (the PostToolUse hook enforces
green):

- **Temporal producer:** stack shape and ordering (oldest-first), edge padding at
  segment start/end, stride-1 sliding reuse, stride-2 boundary reuse, stride-4 grab
  pattern, multi-segment reset.
- **track_segments branch:** local path calls `detect_frame_stack` with the manifest
  (stub runner), rfdetr path byte-identical to today (existing tests keep passing).
- **Observer cadence:** observer sees exactly the center frames, in order, on both
  backends.
- **Backend selection:** monkeypatched loaders; unknown value raises; rfdetr branch
  still requires the API key, local branch does not.
- **Confidence floor:** local uses manifest conf_threshold, rfdetr uses 0.40; the
  guard raises when manifest floor exceeds the requested confidence.
- **Device:** env parsing, auto behavior with cuda mocked present/absent, explicit-MPS
  honored, explicit-unavailable raises. No real GPU in CI — all mocked.
- **Attribution:** `ball_backend` present and correct in job.json and report-v1 on both
  branches; legacy runs without the field still render (match_report is
  legacy-tolerant).

## 10. Eval gate (required before any improvement claim)

After the wiring is green: run the line-call eval (`/eval`) against
`eval_set/BASELINE-2026-07-23.md` twice — `BALL_DETECTOR=rfdetr` (expect zero drift,
proving the rfdetr path is untouched) and `BALL_DETECTOR=local` (the actual
measurement). The local run on this Air will be slow (CPU, native-res tiles); the CUDA
box is the realistic venue for the full run. Until that eval is recorded, the PR
description and any docs say "wired, not yet measured" — recall numbers from the
training val split are not pipeline claims.

## Out of scope, explicitly deferred

- Velocity-extrapolated single-crop windowing (the `tile_windows` seam) — cuts ~66
  tiles/frame to ~1 once a track exists; big speedup, separate change.
- Retiring the rfdetr path and `inference_engine` — after eval says WASB wins.
- Batching stacks across tiles *and* frames on GPU — perf work, after measurement.
