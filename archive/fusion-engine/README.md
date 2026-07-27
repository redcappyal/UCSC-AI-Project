# Archived: the `fusion` bounce engine and 3D contact detection

**Archived 2026-07-27. Restore point: `git tag archive/fusion-engine-v1`.**

Nothing here is deleted, imported, compiled, collected, or linted. Every file in
this directory arrived by `git mv`, so `git log --follow` reaches its full
history, and the tag above is a complete, working snapshot as built.

Same rules as [`archive/stereo/`](../stereo/README.md): **do not extend this, and
do not import from it.** If it comes back, it comes back through the gate below.

---

## What it was

A second bounce-detection engine, selectable per run against the default
`gb_model` detector.

| Piece | File | What it did |
|---|---|---|
| **Fusion engine** | `python/event_engine.py` | `detect_events_fused` — audio repetition × trajectory derivative peaks × parabolic arc fits, resolved into surface labels by a squash sequence grammar (Viterbi over racket → wall → floor). Emitted its own `event_type`, so it replaced both the detector *and* `classify_events`. |
| **3D contact detection** | `python/ballistic.py` | The `fusion_3d` flag. Solved a camera model from the calibration, then fitted gravity-constrained 3D arcs (`X(t) = X0 + V0·tau + ½g·tau²`) in court feet and took contacts where consecutive arcs met — producing `court_position_ft` directly rather than through the floor homography. |
| **A/B replay tool** | `python/rerun_detection.py` | Replayed stored `ui_runs` through the engine into a mirror dir, so 2D and 3D arms could be scored against the same labels without re-tracking. |

34 tests came with it (`tests/test_event_engine.py`, `tests/test_ballistic.py`,
and the shared `tests/tests_ballistic_helpers.py`).

## What state it reached

**Built and tested. Evaluated once, and it failed its gate.**

`eval_set/RESULTS-3d-contact.md` (2026-07-21) is the record, and it is worth
reading before anyone revives this. The short version:

- **The 3D arm was never measurable.** All 14 ground-truth events sat on a v1
  calibration with no floor plane, so both arms ran 2D and scored identically on
  every axis. The corpus could not see the 3D delta at all.
- **Where it *did* engage, it fabricated geometry.** Ballistic contacts clustered
  at y ≈ 24.7–30 ft near the solved camera position, including two grammar-forced
  "wall" labels at y ≈ 26–28 ft — a front-wall contact must be y ≈ 0. Those then
  got wall impact pixels and an `impact_height_ft` synthesized from bad geometry.
  Root causes were depth degeneracy in the arc fit on short noisy arcs, and the
  sequence grammar overriding the plane-distance penalty.
- **After the cheirality fix, its one solvable calibration stopped solving.** The
  addendum records `1784583924415` going from `status: ok` (with an improper
  `det_rotation = -1`) to `implausible_geometry / camera_center_below_floor`. The
  earlier "ok" had been a mirror solution. That left `fusion_3d` with zero real
  calibrations it could run on.

`fusion_3d` stayed off by default from the day it landed and was never flipped on.

## Why it was archived

The product direction set 2026-07-27 (see `CLAUDE.md`) is a training feedback
loop whose measurables come from **2D**: detect the ball and its bounces per
frame, then map contacts to court feet with the floor homography
(`court_model.FloorMap`). There is no 3D simulation in that architecture, so the
`fusion_3d` flag had nothing left to switch on, and a second selectable bounce
engine was a fork in a pipeline that needs one well-measured path.

The engine's own idea — fusing several weak signals instead of trusting one
detector — is not disproven. It was never scored against `gb_model` on the
recall axis that actually gates the product.

## Revisit gate

Same shape as stereo's: **do not restore this to buy recall.** Restore it only
when there is something to measure it with.

1. Labeled, **v2-calibrated** footage from the agreed mount, with a camera solve
   that actually passes `solve_camera_model` — even 2–3 rallies makes the eval
   axes 3D-sensitive, which they have never been.
2. Conditioning and court-volume plausibility gates in the arc fit, and a
   sigma-gated wall snap, so a contact cannot be labelled a wall hit 26 ft from
   the wall.
3. Then re-run the A/B in `RESULTS-3d-contact.md` and beat `gb_model` on missed
   contacts with no regressions.

## What stayed behind

`court_model.solve_camera_model` is **not** archived — it still backs the
calibration health check (`/api/camera-check`, `/api/camera-model`) and the
live camera-solve strip in the wizard. Only the pipeline's use of it went.
