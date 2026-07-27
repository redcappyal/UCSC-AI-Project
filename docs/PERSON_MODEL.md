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
