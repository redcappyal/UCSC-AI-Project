# Data-flywheel work log

Originally built overnight 2026-07-18 as one-tap IN/OUT call corrections;
re-scoped 2026-07-20 with Ian to target the bounce detector directly.

## Why the re-scope

IN/OUT is deterministic geometry given a ball point and calibration — an
IN/OUT label can't say *what* went wrong upstream and gives the models
nothing to learn from. The real error sources are the perception layer:
ball position, hit-type classification, and bounce timing. Corrections
now capture those directly.

## Branch stack (merge in order)

1. `overnight/2026-07-18-call-corrections` — capture side: correction
   panel UI + `/api/runs/<id>/corrections` (schema `corrections-v2`).
2. `overnight/2026-07-18-eval-set` — distillation side: sweeps
   corrections + ground_truth into a git-tracked eval set and replays
   multi-axis evals against it. Stacked on branch 1.

## corrections-v2 schema

One entry per frame in `<run_dir>/corrections.json`, top level
`{"schema_version": "corrections-v2", "corrections": [...]}`:

- `corrected.type` — `wall | side_wall | floor | racket | none`
  (same vocabulary as `GROUND_TRUTH_TYPES` / the event engine; `none`
  = detector false positive, replacing the old `NOT_A_HIT` call).
- `corrected.call` — `IN | OUT`, wall hits only (elsewhere null).
- `corrected.ball` — human-corrected ball position, video pixels.
  Direct RF-DETR retraining signal (position + video sha + frame =
  an image/ball-center pair).
- `corrected.frame_is_bounce` + `corrected.frame` — bounce-timing
  supervision for `estimate_impact`; the corrected frame is captured
  when the human says the detected frame is wrong.
- `predicted.{type,call,source,margin_px,ball}` — label-time model
  snapshot; `agrees.{type,call,frame}` derived server-side.
- Undo = POST `"corrected": null`. `type: none` forces ball/timing null.

## Capture UI (track phase)

- Correction panel under the verdict card: 5-way type row, In/Out row
  (wall only), bounce-frame row ("This frame" / "Different frame…" →
  scrub, then "Use this frame").
- Event cards (racket/floor/side-wall) now get the panel too — type
  corrections apply to every detected hit, not just wall verdicts.
  Their ball dot seeds from the hit's `impact` fit when present, else
  the nearest detected center in `ball_coordinates.csv` (loaded
  client-side; non-wall hits carry no coordinates in detected_hits).
- Ball dot drawn on the track canvas at the draft position; drag to
  correct (display→video-pixel conversion reuses the calibration-tap
  pattern; drag is additive, gated on track phase + an active target).
- Tap the saved type again to undo. Every completed interaction POSTs
  immediately — no separate save button.

## Eval set (branch 2)

- `build_eval_set.py` → `eval_set/cases.jsonl` + `manifest.json`
  (schema `bounce-eval-v2`), deterministic, deduped one case per
  physical moment. Two case kinds:
  - `correction` — v2 corrections verbatim + calibration/video identity.
  - `ground_truth_event` — labeling-mode events swept from
    `ground_truth.json` (previously written-but-unconsumed), matched
    against `detected_hits.json` at build time → missed-bounce
    (false-negative) axis without needing ui_runs at eval time.
- `eval_line_calls.py` axes: IN/OUT accuracy vs human ball (+ conf.
  matrix, `--fail-under` gates this), drift vs label-time call, type
  classification confusion + FP rate, position error (px, split by
  judge source), timing distribution, missed-bounce FN rate per type.

## Housekeeping

- `ui_runs/corr-ui-test/` is a fabricated no-model fixture (calibration
  + CSV + empty detected_hits) used by verification; its corrections
  were regenerated as v2 and distilled into the committed `eval_set/`.
- ground_truth.json labeling mode is untouched UI-wise; it just feeds
  the eval set now.

## Next queue

1. Retraining scaffolding + model versioning (model registry,
   `ROBOFLOW_MODEL_ID` provenance in job.json).
2. Perception layer per roadmap (rally segmentation first).
