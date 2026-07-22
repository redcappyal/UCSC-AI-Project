# 3D Contact Detection — A/B Eval Results (Task 11)

**Date:** 2026-07-21
**Branch:** feature/3d-contact-detection (A/B run at commit 1e68572)
**Verdict: GATE NOT PASSED — `fusion_3d` default stays OFF (opt-in flag only).**

## Setup

Both arms replayed with `rerun_detection.py` over `ui_runs` (no audio in either arm),
eval sets rebuilt per arm with `build_eval_set.py`, scored with `eval_line_calls.py`.
3 labeled runs replayed, 114 skipped (no labels or no tracking CSV).

Camera solve (after the Task-11 solver rework — DLT init, free principal point,
median gate): `1784583924415` solves (median 3.0 px, rms 5.8 px, pp (946, 667),
det(R) = −1, analyzed self-consistent); `bayclub-fusion-1` and `corr-ui-test` are
v1 calibrations (no floor plane) → 2D in both arms, by design.

## Quantitative result: identical on every axis

| Axis | 2D arm | 3D arm |
|---|---|---|
| IN/OUT accuracy | 2/2 | 2/2 |
| Hit-type accuracy | 2/4 | 2/4 |
| Ball-position error (mean) | 9.6 px | 9.6 px |
| Bounce timing confirmed | 3/3 | 3/3 |
| Missed bounces | 10/14 | 10/14 |

**Why identical:** every scorable label lives where 3D cannot engage. All 14
ground-truth events are on `bayclub-fusion-1` (v1 calibration → both arms 2D).
The single 3D-capable run (`1784583924415`) carries only 2 correction cases whose
scoring replays label-time snapshots. **The current eval corpus cannot measure the
3D delta at all.**

## Qualitative result: the 3D arm is not trustworthy yet

Hit-list comparison on `1784583924415` (11 hits 2D → 9 hits 3D) shows the ballistic
contacts' 3D positions clustered at y ≈ 24.7–30 ft, z ≈ 4–5.6 ft — mid/back court,
near the solved camera position — including two grammar-forced **"wall" labels at
y ≈ 26–28 ft (a front-wall contact must be y ≈ 0)**, which then got snapped wall
impact pixels and `impact_height_ft` fabricated from bad geometry. Failure chain:

1. **Depth degeneracy in the ballistic fit.** On short/noisy real arcs the
   least-squares system is full-rank but ill-conditioned along the viewing ray;
   the solution collapses depth toward the camera. The rank gate (Task 5 fix)
   catches exact degeneracy only — a conditioning gate (singular-value ratio)
   and/or a court-volume plausibility check on the contact point is needed.
2. **The grammar can force a surface label onto an interior contact.** Viterbi's
   sequence pressure (racket → wall → …) overrides the plane-distance far-penalty.
   The wall snap in the hit-assembly should require plane distance within
   k·sigma before synthesizing impact fields; otherwise the label should not carry
   metric impact data.
3. **Calibration quality amplifies both.** The one solvable calibration has 4/7
   landmarks, one un-refined outlier tap (12.9 px), and a shifted principal point.

## Decisions

- `fusion_3d` default remains **off** (`job.get("fusion_3d")`, opt-in). No
  regression risk to shipped behavior: default path is byte-identical 2D.
- **Task 12 (weight grid search) skipped as unmeasurable**: the tuning objective
  (type-axis accuracy) is blind to 3D with this corpus; tuning would fit noise on
  4 correction cases. Revisit after the corpus gap closes.

## What unblocks the gate (in order of leverage)

1. **Labeled, v2-calibrated footage**: record with the agreed mount (back wall,
   centered, ultrawide, 4K60), run the full 7-landmark wizard carefully (refined
   taps, not raw), and label ground truth. Even 2–3 rallies would make every eval
   axis 3D-sensitive.
2. **Conditioning + plausibility gates in `ballistic.py`** (fix #1 above) and the
   sigma-gated wall snap (fix #2) — small, well-scoped follow-ups.
3. Re-run this A/B (`rerun_detection.py` both arms) and flip the default only when
   missed-contact and type axes improve with no regressions, per the plan.

## Addendum, 2026-07-21: cheirality fix and `1784583924415`'s post-fix status

`court_model._init_camera_dlt` normalized its projection-matrix sign before RQ
decomposition (forcing det(rotation) = +1, a proper rotation, deterministically
instead of leaving it to the SVD's arbitrary null-vector sign) and now returns
`None` rather than flipping to a mirror (det = -1) when positive depth and a
proper rotation can't both hold. `solve_camera_model` also gained a
belt-and-braces `det_rotation < 0 -> "implausible_geometry"` gate.

Re-running `solve_camera_model` on `ui_runs/1784583924415/calibration.json`
after the fix:

```
Before: {'rms_px': 5.79, 'median_px': 3.01, 'det_rotation': -1.0, 'status': 'ok', ...}
After:  {'rms_px': 139.6, 'median_px': 50.3, 'det_rotation': 1.0,
         'status': 'implausible_geometry', 'reason': 'camera_center_below_floor',
         'init': 'homography', ...}
```

The DLT init now hits the mirror-only case (returns `None`) on this
calibration's sparse/noisy correspondences (4/7 floor landmarks, one 12.9 px
un-refined outlier tap), so `solve_camera_model` falls back to the
homography-based initializer — which also fails to converge to a physically
plausible pose (camera center lands ~250-2500 ft away, below the floor) and is
rejected outright. **This calibration no longer solves at all** (previously it
solved "ok" with a mirror camera, det = -1). That is the intended, correct
outcome per the fix brief: this calibration is unhealthy, and a mirror-fit
"ok" status was masking that rather than reporting it honestly.

**A/B conclusions are unchanged.** `1784583924415` was never a scoring input
for either arm's axis numbers — the eval corpus's 14 ground-truth events all
live on `bayclub-fusion-1`, a v1 calibration with no floor plane, so that run
is 2D in both arms by construction (row 2 of the Setup section above). This
calibration only fed the qualitative failure-chain analysis (the two
grammar-forced "wall" labels at y ~ 26-28 ft) and the "solves self-consistently"
note in Setup — neither of which was a quantitative A/B input. The 2D-arm and
3D-arm numbers in the Quantitative-result table above are unaffected and remain
identical, for the same reason they were identical before this fix (the two
arms differ only where a camera model exists and 3D evidence can be computed at
all, and that never happened where labels exist).
