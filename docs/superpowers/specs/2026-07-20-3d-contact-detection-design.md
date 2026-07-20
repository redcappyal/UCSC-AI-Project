# 3D Contact Detection — Design

**Date:** 2026-07-20
**Status:** Approved design, pre-implementation
**Replaces:** image-space contact heuristics in `event_engine.py` (2D mode retained as fallback)

## Problem

The current bounce/contact detection is the load-bearing layer of the app — every
downstream feature (line calls, floor zones, review UI, eval) fails when it fails — and it
is not good enough on two axes at once:

1. **Missed contacts** (recall): contacts happen with no detected event. Root cause: the
   trajectory methods reason in image pixels, and a front-wall bounce reverses the ball's
   *depth* velocity, which barely bends the image path. The eval build measured 13/17
   labeled bounces unmatched (overstated by label-only runs — see Pre-work).
2. **Wrong contact type**: floor/racket/front-wall/side-wall confusion. Root cause: the
   emission scores are image-space proxies (wall band, apparent-size depth proxy,
   image-y flip) for what are really 3D questions.

## Decision

Reconstruct the ball's flight in metric court coordinates and make contact detection and
classification 3D-geometric. Physics first; ML assists (corrections-v2 flywheel tunes
weights). Assumption per product direction: **the calibrated fixed back-wall camera is the
norm**; uncalibrated clips gracefully keep today's 2D behavior.

The fusion architecture (audio evidence + skip-state Viterbi + squash grammar) is kept
unchanged; only the trajectory event source and the emission evidence are upgraded.

## Section 1: Camera model & pose solver

Lives in `court_model.py`. Pure numpy.

- **Inputs (no new wizard taps):** calibration-v2 floor landmarks (≤8 points, court
  `(x, y, 0)`) + v1 front-wall lines (out-line and tin endpoints = 4 points at
  `(0|21 ft, 0, h)`, h = 15 ft out line / 19 in tin top). 11–12 correspondences
  (`half_court_back` is optional) across two orthogonal planes.
- **Unknowns (7):** focal (fx = fy), rotation (3), camera position (3). Principal point
  fixed at image center. Existing division-model distortion seam stays in the loop
  (identity now).
- **Solve:** initialize from floor-homography decomposition; refine by Levenberg–Marquardt
  on pixel reprojection error over all points (~40 lines numpy, no scipy).
- **Output:** `CameraModel` dataclass — `project(court_xyz) → px`,
  `ray(px) → (origin, direction)`, `to_dict()/from_dict()` (serialized into the run's
  calibration payload for reproducibility).
- **Validation:** per-point residuals reported; RMS above threshold (~4 px @1080p, tuned
  on the golden run) → `calibration_warning` + 2D mode. Never fails a run.

## Section 2: 3D piecewise-ballistic trajectory segmentation

New module `ballistic.py`, consumed by `event_engine.py`. Reuses `split_into_tracks`
(gap/jump limits) as-is; replaces only per-track arc logic (2D `segment_into_arcs` remains
for 2D mode).

- **Flight model:** `X(t) = X₀ + V₀·Δt + ½·(0,0,−g)·Δt²`; 6 unknowns per arc; no drag
  term in v1 (robust threshold absorbs mild drag/spin; per-segment drag scalar is a later
  refinement if residuals warrant).
- **Fit is linear:** with a calibrated camera each detection gives 2 linear constraints on
  the ball's 3D position (cross-product form of the projection equation), and X(t) is
  linear in (X₀, V₀) — so each arc fit is linear least squares in 6 unknowns. Optional
  single LM polish converts algebraic residual to true pixel reprojection error.
- **Segmentation:** greedy maximal arcs (same shape as today): grow while reprojection RMS
  ≤ threshold (3D analogue of `arc_rms_px`, tuned on golden run); min ~5 points per arc
  (~150 ms @30 fps). Depth reversals (wall hits) violently break the ballistic fit — this
  is what recovers the recall the 2D arcs miss.
- **Sub-frame impact refinement:** at each arc boundary, extrapolate arc A forward and
  arc B backward; time of closest approach gives sub-frame impact time + 3D impact point +
  3D in/out velocities. Generalizes and eventually replaces `detect_bounce_two_stage`.
- **Degenerate cases:** arcs too short to fit still emit a breakpoint event without
  refined 3D (enters fusion with weaker, 2D-style evidence). Detection gaps keep today's
  track-splitting behavior. The 2D derivative method and audio windows remain independent
  parallel event sources — recall stays additive.

## Section 3: Classification & fusion

`_emission_scores` gains a 3D evidence mode; audio matching, skip-state Viterbi, grammar,
`none` state, and `ALLOWED_TRANSITIONS` unchanged. Mode selected per run by presence of a
valid `CameraModel`.

- **Surface scores = plane distances:** distance of the 3D impact point to floor (z≈0),
  front wall (y≈0), side walls (x≈0, x≈21), converted to scores by a smooth falloff whose
  width is the event's positional uncertainty propagated from pixel noise through the ray
  geometry (near-front-wall events get proportionally wider tolerance; ~1 in/px depth
  there).
- **Racket = the "nowhere" hypothesis:** impact point in the court-volume interior, not
  near any surface. Replaces the apparent-size depth proxy as primary racket evidence.
- **Velocity-reflection physics check:** per surface hypothesis, reflect V_in about the
  surface normal, compare with V_out: direction agreement + restitution ∈ (0, 1] supports;
  energy gain vetoes passive surfaces and votes racket (speed-gain rule in real units).
  Distinguishes floor vs front-wall (perpendicular normals) that image space conflates.
- **Retained evidence:** audio bonuses/penalties unchanged; size ratio kept at reduced
  weight as corroboration. Back-wall contacts out of scope (camera sits on the back wall).
- **Downstream wins:** front-wall contact height vs tin/out-line feeds `judge_call` in
  metric space (the line call falls out of the same 3D impact); floor contacts already
  carry court coordinates for `floor_zone` (no separate homography pass).
- **ML-assist hook:** all evidence lands in a per-event feature dict; `FUSION_DEFAULTS`
  weights become the tunable — grid search against the eval set first, small logistic
  scorer over the same features later. Deterministic features, learned weights.

## Section 4: Error handling & eval

**Degradation ladder (worst case = today's behavior, never broken):**

1. Full calibration + good camera fit → 3D evidence mode.
2. Camera fit fails validation → `calibration_warning`, run in 2D mode (floor-plane
   pattern).
3. In a 3D run, an event whose arcs were too short for refined 3D → 2D evidence for that
   event only (per-event fallback).
4. No calibration → 2D mode, unchanged.

**Debuggability:** every hit's `signals` dict grows plane distances, reflection agreement,
restitution, positional sigma, and the evidence mode; flows to run payload → review UI and
eval can explain any label from the artifact alone.

**Eval:**

- **Pre-work — rebuild the eval baseline:** current 13/17 FN figure includes label-only
  runs with no `detected_hits.json`; rebuild `eval_set/cases.jsonl` from runs with real
  detection output before measuring anything.
- **Synthetic ground-truth tests** (in `tests/`): render synthetic ballistic trajectories
  through a known camera; assert pose/focal recovery within tolerance, breakpoints found
  at injected contacts, correct surface labels, sub-frame impact time within tolerance.
- **Regression gates per phase:** missed-contact and type-agreement axes must improve, no
  other axis regresses. 2D path byte-stable on golden run `ui_runs/1784236711057`. Bay
  Club GT frames (135: racket→side→front-wall; 220: far-court racket) become named eval
  cases.
- **Perf:** 6-unknown linear solves are negligible; verify with `benchmark_tracking.py`.

## Build order (each phase independently shippable)

0. Rebuild eval baseline.
1. Camera solver in `court_model.py` + synthetic tests + payload serialization.
2. `ballistic.py` segmentation + synthetic tests; wired behind a config flag as an
   additional event source; eval.
3. 3D emissions in `event_engine.py`; default flipped only when eval beats 2D.
4. Emission-weight tuning from corrections-v2 labels (grid search).

## Out of scope

Back-wall contact classification; drag/spin modeling (v2 refinement); new wizard
landmarks (side-wall taps only if residuals prove the 12-point fit under-conditioned);
ML-primary detector (revisit when the flywheel has ~10× more labels); front-wall metric
homography as a separate feature (subsumed by the full camera model).
