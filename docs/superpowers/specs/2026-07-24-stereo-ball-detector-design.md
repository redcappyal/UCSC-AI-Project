# Stereo Ball Detector — Design

**Date:** 2026-07-24
**Status:** Approved in conversation (Ian), pending written-spec review
**Branch:** `worktree-stereo-ball-detector`, based on `origin/main` (bcea937)
**Baseline:** 271 tests passing in 7.6 s before any change

Puts the newly trained YOLOX-Tiny ball detector behind the stereo path's detector
seam, as a swappable, self-describing artifact — without altering the
single-camera pipeline or the stereo geometry.

## Decision log (rulings that shaped this design)

1. **Target today's committed stereo path, in an isolated worktree** (Ian). A
   concurrent session is building "phase 5 cloud fusion" in the same checkout.
   `stereo_offline.py`, `inference_engine.py`, `stereo_engine.py` and
   `tracking_common.py` are byte-identical on `origin/main` and
   `origin/claude/phase4-ui`, so basing on `main` keeps this change independently
   reviewable instead of entangled with ~6,900 lines of their in-flight work.
2. **Tiled native-resolution sweep, not whole-frame resize** (Ian). The model was
   trained on 416 px windows cut at native resolution. `infer_frame_predictions`
   resizes to 960 px wide, which on 4K is 0.25× — a p50 10.7 px ball becomes
   ~2.7 px, outside anything the model has seen. `ios/MODEL.md` §6 exists to
   prevent exactly this.
3. **YOLOX becomes the default for the stereo path only** (Ian). The single-camera
   production path stays on RF-DETR. Accepted risk: two detectors in one system.
   Mitigated by stamping detector provenance into stereo output and by refusing to
   fall back silently (§Error handling).
4. **TorchScript artifact, exported once** (Ian). `yolox_ball_exp.py`'s docstring
   requires YOLOX stay "a training dependency, not a runtime one … out of
   requirements.txt and out of the test environment." Loading `best_ckpt.pth`
   directly would need YOLOX's model classes at serving time. Tracing to
   TorchScript needs only torch, which already arrives transitively via
   `inference`.
5. **Manifest-driven model directory** (Ian). Everything that varies between
   iterations — input size, thresholds, class map, provenance — is data, not code.
6. **Do not pre-optimise the tile count** (Claude's call, delegated). Ship the full
   batched sweep and *measure* CPU and GPU per-frame cost rather than assert it.
   `stereo_offline.py` is an offline batch tool; §6's 16 ms budget governs the
   Swift path, not this one. Optimising throughput for a model that has not passed
   §3's acceptance gate spends complexity on something that may not survive
   evaluation.

## Problem

The trained detector exists (`best_ckpt.pth`, epoch 100, val AP[.5:.95] 0.4034)
but nothing can run it. Three obstacles:

- **No runtime loader.** The checkpoint is a YOLOX state dict; rebuilding the
  network needs YOLOX imported, which the repo forbids at runtime.
- **Wrong input scale.** The only existing inference entry point downscales whole
  frames, destroying the small-object evidence the crops were built to preserve.
- **No swap path.** `inference_engine.py` hardcodes a Roboflow model id
  (`squashai/1`) and raises without an API key. There is no way to point the
  pipeline at a local model, and no record of which model produced a result.

## Goals

- Run the trained model over the two clips in the stereo path, at the resolution
  it was trained for.
- Make a future model iteration a directory drop-in plus one config value — no
  code change even if input size or thresholds differ.
- Change nothing about the single-camera path, the stereo geometry, or the CI
  test environment.
- Record which detector produced any given stereo result.

## Non-goals

- **Passing `ios/MODEL.md` §3's acceptance gate.** This makes the gate
  *satisfiable* — a `frame → dicts` callable is what an eval harness needs — but
  scoring YOLOX against the RF-DETR baseline is the follow-up, not this change.
- Any claim that YOLOX beats RF-DETR. Per the dataset limits, no number available
  today can support that.
- Swift / Core ML work (`MODEL.md` §4–5 are macOS-only).
- Changing RF-DETR behaviour anywhere.
- The velocity-extrapolated single-crop loop from §6. The window strategy is a
  seam so it can land later without touching the model adapter.

## Architecture

The detector seam already exists and is narrow: a **`frame → list[dict]`**
callable, each dict `{x, y, width, height, confidence, class, class_name}` in
full-frame, centre-based pixel coordinates. This is what `normalize_prediction`
emits and what `tracking_common.select_ball_prediction` consumes. Both detectors
satisfy it, so the integration is a substitution at one call site.

| File | Change | Consequence |
| --- | --- | --- |
| `stereo_engine.py` | none | Stereo math provably unaffected; goldens do not move; no `generate_stereo_goldens.py` run; no Swift twin drift. |
| `inference_engine.py` | none | `job_runner.py`, `modelEval.py`, `local_model_eval.py`, `app.py` keep RF-DETR unchanged. |
| `tracking_common.py` | none | Prediction contract unchanged. |
| `stereo_offline.py` | `_build_infer` gains detector selection | The one place the stereo path picks a detector. |
| `requirements*.txt` | none | TorchScript adds no dependency; suite stays torch-free. |

That `stereo_engine.py` is untouched is the strongest available "nothing breaks"
guarantee: only the *input* to the geometry changes, never the geometry.

## Components

### `ball_model.py` — which model

Owns the manifest and artifact loading. No torch or cv2 at import time; the torch
import is lazy inside the loader, matching `inference_engine.py`'s and
`stereo_offline.py`'s existing pattern.

```
models/crosscourt-ball-416-v1/     # gitignored; located via BALL_MODEL_DIR
  model.torchscript
  manifest.json
```

```json
{
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
  "artifact_sha256": "<sha256 of model.torchscript>",
  "source_checkpoint": "crosscourt-ball-416 best_ckpt.pth (epoch 100)",
  "trained_commit": "2968a89",
  "val_ap50_95": 0.4034,
  "notes": "val is diagnostic only — shares a rig/session with train"
}
```

Public surface: `ModelManifest` (frozen dataclass), `load_manifest(dir)`,
`load_detector(dir=None)` returning a cached loaded module. `conf_threshold`,
`nms_iou`, `input_size`, `tile_overlap_px` and `max_batch_tiles` are read from the
manifest, never hardcoded — that is what makes a differently-shaped future model a
data change.

**Two thresholds already exist in this path and must not be confused.** The
manifest's `conf_threshold` is the *detector's* floor: below it a box is not
emitted at all. Separately, `detections_to_track_samples(confidence=0.4)` passes a
threshold to `tracking_common.select_ball_prediction`, which picks the
highest-confidence ball among what it was given. They compose as
detector-floor-then-selection, so the manifest value must stay **at or below** the
caller's, or the caller's threshold silently becomes dead and unreachable. The
design keeps the detector floor deliberately lower (0.25 vs 0.4) so tuning the
selection threshold in the stereo path continues to have an effect.

Note `conf_threshold: 0.25` is a production operating point, not inherited from
training: the exp sets `test_conf = 0.01`, which is an eval-time setting chosen to
maximise AP by admitting almost everything. It is a manifest field precisely so it
can be tuned without a code change.

### `ball_detector.py` — how it is run

Composes a window strategy with a model. The pure functions carry the logic most
likely to harbour a real bug, and none of them need torch:

- `tile_windows(frame_w, frame_h, tile, overlap) -> [(x0, y0), ...]` — pure.
- `merge_detections(dets, iou) -> [dict]` — cross-tile NMS, pure numpy.
- `detect_frame(detector, frame) -> [dict]` — full-frame pixel dicts.

**Overlap must exceed the largest ball.** p90 ball width is 24 px; overlap is
64 px, giving stride 352. Any ball is then wholly contained within at least one
tile rather than clipped across a seam. On 4K that is 11 × 6 = 66 tiles per frame,
submitted in batches of `max_batch_tiles`.

Tiles at the right and bottom edges are clamped to the frame and therefore overlap
their neighbour by more than 64 px; `merge_detections` deduplicates the extra.

### `export_ball_model.py` — training-side tool

Runs in the **training** environment only (`ball-detector-train/.venv`), never in
the app env. Loads `best_ckpt.pth` through YOLOX, traces to TorchScript, computes
the sha256, writes `manifest.json`. This is the boundary that keeps YOLOX out of
the serving path.

Traced with `decode_in_inference = True`, so the graph emits decoded
`[1, N, 6]` and Python only thresholds and NMSes — one less place to get grid math
wrong. `MODEL.md` §4's Core ML export needs the opposite (raw head outputs, decode
in Swift, for ANE residency), so the manifest's `decode` field distinguishes the
two artifacts produced by the same script.

## Data flow

```
stereo_offline.fuse_clips(video_a, calib_a, video_b, calib_b)
  └─ detections_to_track_samples(video, model=None)   # already supports injection
       └─ _build_infer()                               # ← only changed logic
            ├─ stereo default: ball_detector.detect_frame(ball_model.load_detector())
            └─ override:       inference_engine RF-DETR (unchanged path)
       └─ tracking_common.select_ball_prediction(...)  # unchanged contract
  └─ stereo_engine.build_track3d / detect_impacts      # untouched
```

`fuse_clips`' output JSON gains a `detector` block copied from the manifest
(name, version, sha256, thresholds). Every stereo result then records the exact
model that produced it — the mitigation for running two detectors in one system.

## Error handling

**No silent fallback to RF-DETR.** A missing model directory, missing manifest, or
unreadable artifact raises, naming the expected path and pointing at
`export_ball_model.py`. Silent fallback would make the stereo/single-camera
detector split invisible, which is precisely the risk the split rollout carries.

| Condition | Behaviour |
| --- | --- |
| `BALL_MODEL_DIR` unset and no default dir | Raise, naming the path and the export script |
| `manifest.json` missing or unparseable | Raise with the offending path |
| `schema_version` unrecognised | Raise; do not guess field meanings |
| `artifact_sha256` mismatch | Raise — a mismatched artifact makes results unattributable, and regenerating the manifest is a one-line fix |
| torch not importable | Raise: stereo needs the full `requirements.txt` |
| Frame with no detections | Normal; return `[]`. The tracker already treats gaps as gaps |

## Testing

Shaped so the default CI job never needs torch, per `requirements-test.txt`'s
instruction that anything needing the real model runtime "belongs behind a marker,
not in the default CI job."

- `tests/test_ball_model.py` — manifest parsing, schema-version rejection, sha256
  mismatch, missing-file errors. No torch.
- `tests/test_ball_detector.py` — `tile_windows` coverage and overlap invariants
  (every pixel covered; overlap ≥ max ball width; edge tiles clamped in-bounds);
  `merge_detections` NMS behaviour including the duplicate-across-tiles case.
- **Coordinate round-trip — the test that matters most.** Synthetic frame, a disc
  at a known full-frame position, a fake model returning a tile-local box; assert
  the merged detection lands at the known full-frame position. Off-by-tile mapping
  is the likeliest real defect and this catches it with no model involved.
- **Golden invariance.** Assert this change does not move `tests/stereo_goldens.json`.
  Phrased as "unchanged by this change", not as fixed bytes — the other branch has
  already regenerated those goldens, so the bytes are branch-dependent.
- Real-model smoke test behind a pytest marker, skipped by default.

The full suite must still report **271 passed**, plus the new tests.

## Risks

| Risk | Mitigation |
| --- | --- |
| 66 tiles/frame is too slow on a CPU-only Flask box | Measure before optimising (Decision 6). `tile_overlap_px` and `max_batch_tiles` are manifest knobs, so a deployment can trade recall for speed explicitly rather than silently. The window strategy is a seam for §6's tracker-guided crop. |
| Two detectors in one system become confusable | Manifest stamped into output; no silent fallback |
| Tile seams cause missed or duplicated balls | Overlap > p90 ball width; cross-tile NMS; round-trip test |
| Model is unevaluated and may be worse than RF-DETR | Confined to the stereo path, which is itself unevaluated. §3's gate is the follow-up, and this change is what makes it runnable |
| Concurrent phase-5 session conflicts | Isolated worktree; `stereo_offline.py` untouched on their branch, so this should merge cleanly |

## Follow-ups (not this change)

1. **Give `yolo_model_eval.py` a YOLOX loader** and score both detectors on the
   same clip. This is `MODEL.md` §3's gate and the prerequisite for promoting
   YOLOX beyond the stereo path.
2. Benchmark tile-sweep cost; decide whether the tracker-guided crop is needed.
3. Retrain with `max_epoch ≈ 120` — the run converged at 100 of 300.
