# Serve-attribution baseline

- Run: `ui_runs/attr-eval-with-person`
- Labels: `eval_set/attribution-labels-SquashAnalytics.template.json` (1 rallies)
- Detector backend: rfdetr
- Observed rallies: 1
- Scored (observed AND labeled): 1
- Correct: 1
- Accuracy: 1.0
- Observed coverage of labeled rallies: 1.0
- Mismatched rally numbers: []

Rallies without observed serves are excluded from accuracy —
coverage reports them honestly (spec §7: no pre-hit ball track ->
`server_track: null`, never a guess).

## Labels status — HUMAN GATE PENDING

The labels file scored above is the **template**
(`eval_set/attribution-labels-SquashAnalytics.template.json`), not human-verified
labels. It contains one placeholder rally (`rally_number: 1, server: 1`) that was
never checked against the clip. **The accuracy=1.0 / coverage=1.0 figures above are
not a real accuracy claim** — they only show that the pipeline's rally 1 observed
server matches whatever placeholder value shipped in the template, which is true by
construction for anyone who hasn't done the human labeling pass yet. To get a real
number: copy the template to `eval_set/attribution-labels-SquashAnalytics.json`,
watch the clip, fill `server` per rally (1/2/null) for every rally the run produced,
then re-run:

```
.venv/bin/python eval_attribution.py --run-dir ui_runs/attr-eval-with-person \
  --labels eval_set/attribution-labels-SquashAnalytics.json \
  --output eval_set/BASELINE-ATTRIBUTION-2026-07-27.md
```

## Golden-clip proof run

Clip: `SquashAnalytics.mp4` (worktree root, 1080p60, 18713 frames / 311.9s total).
Both runs use the same 60-second window — `start_frame=0 end_frame=3599`
(3600 selected frames), `frame_stride=4` (default), `inference_width=960`
(default) — so `total_frames=900` coarse-pass samples, refined to 1445 rows
after the hit-candidate refine pass. Both runs share the same
`ui_runs/golden-coarse2fine/calibration.json` (lines only — no floor plane, no
wall corners, no service line — matching what that pre-existing golden run
already proved works for front-wall IN/OUT judging).

Run dirs were constructed by hand (no Flask server in this environment): a
`job.json` written to match exactly what `/api/track`'s `create_job()` would
produce (run_id, run_dir, video_path, start/end frame, fps, frame_stride,
inference_width, processed_frames, total_frames, csv_url, and the
`model_provenance()` fields `model_id=squashai/1`, `tracking_backend=torch`,
`device=mps`, `app_version=wall-corner-calibration-2026-07-20-1`), then run via
the CLI: `python job_runner.py <run_dir>`.

| Run | run dir | `PERSON_DETECTOR` | wall-clock | status |
|---|---|---|---|---|
| 1 (with person pass, default) | `ui_runs/attr-eval-with-person` | (unset — default) | **173.97s (2m 54s)** — 02:01:28Z → 02:04:21Z | complete |
| 2 (no person pass) | `ui_runs/attr-eval-no-person` | `none` | **104.92s (1m 45s)** — 02:12:26Z → 02:14:10Z | complete |

Delta: the person-detection pass added **+69.05s (+65.8%)** wall-clock to this
60-second/1445-row clip.

`PERSON_DETECT_HZ = 4.0` (`player_tracker.py`). For this run:
`coarse_hz = source_fps / frame_stride = 60.0 / 4 = 15.0`;
`detect_every = round(coarse_hz / PERSON_DETECT_HZ) = round(15.0 / 4.0) = 4` —
the detector runs on every 4th coarse-pass frame. Observed:
`players_v1.tracker.updates = 225` in run 1, which is exactly `900 / 4` —
matches the formula.

### Run 1 (with person detector) result

`players_v1`: `attribution_backend: "observed"`, `detector_backend: "rfdetr"`,
`tracker: {updates: 225, ambiguous_assignments: 16, detect_failures: 0}`,
`serve_crop: "players/serve_rally1.jpg"` (written, 312x713px). Two rallies
were segmented from the 60s window; rally 1's server was **observed** (track A
→ player 1), rally 2's was **propagated** (no observed track — track_samples
had no confident pairing crossing that rally's serve).

### Run 2 (`PERSON_DETECTOR=none`) result

`players_v1`: `attribution_backend: "assumed"`, `detector_backend: "none"`,
`tracker: null`, `serve_crop: null`, both rallies `server_source: "propagated"`
— the documented fallback (docs/PERSON_MODEL.md, spec §4.1). Ball-tracking
output (8 front-wall hits, same 2 rallies) is identical between runs, as
expected — person detection is a fully additive frame_observer riding the
coarse decode and does not touch ball tracking or judging.

### Concern: RF-DETR Keypoint Preview checkpoint quality

The first `RFDETRKeypointPreview()` construction downloaded
`rf-detr-keypoint-preview-xlarge.pth` (156MB) fresh to `~/.roboflow/models/`
(this machine had no prior download; the cache is a per-user `$HOME` dir, so
it is already shared across checkouts — no per-repo symlink needed for it,
unlike the ball-tracker's `.roboflow-cache`, see below). Load emitted several
warnings beyond the keypoint-head one `docs/PERSON_MODEL.md` already
documents:

```
Using a different number of positional encodings than DINOv2, which means
we're not loading DINOv2 backbone weights.
Using patch size 12 instead of 14, which means we're not loading DINOv2
backbone weights.
Checkpoint has 1 classes but model is configured for 90. Using checkpoint
class count (1).
load_pretrain_weights: checkpoint lacks args.num_queries / args.group_detr;
falling back to flat slice.
Pretrained weights loaded only partially — 4 checkpoint key(s) not consumed:
[keypoint_head.keypoint_proj.0.weight, keypoint_head.keypoint_proj.0.bias,
keypoint_head.keypoint_proj.2.weight, keypoint_head.keypoint_proj.2.bias]
```

The first two lines mean the DINOv2 **backbone** weights are not loading
either — this is broader than the keypoint-head mismatch `docs/PERSON_MODEL.md`
already calls out, and looks like a version skew between the installed
`rfdetr` package and this "Preview" checkpoint's expected architecture.

Checking `players/track_samples.json` from run 1 for whether this actually
degrades box quality: `detection.confidence` values are **not in [0,1]** —
observed range 0.50–4.02 (track A) / 0.50–3.92 (track B), mean ~2.7–2.8 —
inconsistent with `PERSON_CONFIDENCE_THRESHOLD = 0.5` being a normalized
probability cutoff. Bounding-box sizes are also inconsistent in quality: mean
box size for track A is 559×816px and reaches 1065×1074px at times in a
1920×1080 frame (more than half the frame), though the actual serve crop
written to disk (`serve_rally1.jpg`, 312×713px, 16% frame width / 66% frame
height) looks like a plausible single-person box. Foot positions (`foot_px`)
stay within frame bounds and move smoothly frame-to-frame for both tracks, and
the pass completed with `detect_failures: 0`, so tracking is *functionally*
coherent (rally 1 got a clean observed-server result) despite the anomalous
confidence scale and occasional oversized boxes.

**This needs a human decision before the checkpoint is trusted further**:
confirm the installed `rfdetr` version matches what `rf-detr-keypoint-preview-
xlarge.pth` expects (patch size 14, correct positional-encoding count), or
pin/re-export a matching checkpoint. v1 only consumes boxes, not keypoints, so
this proof run's attribution result stands, but box/keypoint quality should be
verified before leaning on this detector for anything more precision-sensitive.

### What was symlinked

`.roboflow-cache/` (gitignored, ball-tracker Roboflow model cache,
`MODEL_CACHE_DIR` in `inference_engine.py`) did not exist in this worktree.
Symlinked from the main checkout instead of copied:

```
ln -s "/Users/Ian2/Desktop/UCSC AI Project/UCSC-AI-Project/.roboflow-cache" .roboflow-cache
```

In practice this directory only holds small metadata (`model_type.json`,
`usage.db`); the ball-tracker's actual torch weights are fetched over the
network by the `inference` package on first `get_tracking_model()` call
(confirmed by a smoke test: cold load took ~36s, backend `torch`, device
`mps`, `InferenceModelsObjectDetectionAdapter`). The RF-DETR person-detector
checkpoint uses a separate, already-shared `~/.roboflow/models/` cache (not
per-repo) — nothing to symlink there.

### `/eval` zero-drift check

```
.venv/bin/python eval_line_calls.py --eval-set eval_set/cases.jsonl
```

Output is **byte-for-byte identical** to the "Full report" block in
`eval_set/BASELINE-2026-07-23.md` (the newest existing baseline; confirmed by
`git log -- eval_set/`): 113 cases, IN/OUT accuracy 2/2 = 100%, drift 0/3,
missed bounces 71/109. Expected — this task adds only `eval_attribution.py`,
its test, and eval-set/doc files; it does not touch `judge_call.py`,
calibration handling, detection, or tracking.

Reproduce this whole run:

```
.venv/bin/python -m pytest tests/test_eval_attribution.py -v
.venv/bin/python eval_attribution.py --run-dir ui_runs/attr-eval-with-person \
  --labels eval_set/attribution-labels-SquashAnalytics.template.json \
  --output eval_set/BASELINE-ATTRIBUTION-2026-07-27.md
.venv/bin/python eval_line_calls.py --eval-set eval_set/cases.jsonl
.venv/bin/python -m pytest tests/ -q
```
