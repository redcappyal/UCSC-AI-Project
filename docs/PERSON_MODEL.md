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

## Known checkpoint issue (rfdetr 1.8.3)

The pinned `rfdetr==1.8.3` loading `rf-detr-keypoint-preview-xlarge.pth`
loads only **partially**: `keypoint_head.keypoint_proj.*` keys go unconsumed,
and DINOv2 backbone weights don't load either (positional-encoding count and
patch-size mismatches against the checkpoint) — a version skew between the
installed package and this "Preview" checkpoint's expected architecture, not
just the keypoint head.

Detection confidences are observed at **0.5–4.0**, not the normalized [0,1]
range `PERSON_CONFIDENCE_THRESHOLD = 0.5` assumes as a probability cutoff —
so the threshold barely filters anything, and the tracker's top-2-by-
confidence pick (`player_tracker.py`) orders candidates by an uncalibrated
score, not a true confidence ranking. Boxes are also sometimes oversized —
up to ~half the frame in a 1920x1080 source.

Despite this, tracking is coherent on the golden window: foot positions stay
in-bounds and move smoothly frame-to-frame, `detect_failures: 0`, and rally 1
resolved a clean observed server. v1 only consumes boxes, not keypoints, so
this doesn't block the current feature — but box/keypoint quality should be
verified before leaning on this detector for anything more precision-
sensitive. Full measurements: `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md`.

**Open human gate**: confirm the installed `rfdetr` version matches what
`rf-detr-keypoint-preview-xlarge.pth` expects, or re-pin/re-export a matching
checkpoint. Not yet decided.
