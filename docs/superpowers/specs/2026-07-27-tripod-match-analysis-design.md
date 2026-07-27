# Tripod Match Analysis — Design Spec

**Date:** 2026-07-27
**Status:** Approved design; implementation plan at `docs/superpowers/plans/2026-07-27-tripod-match-analysis-mvp.md`
**Audience:** an autonomous implementation session with full codebase access and zero conversation context. Everything it needs is in this spec, the plan, `CLAUDE.md`, and the code.

## 1. Goal

Let a player put one phone on a simple tripod (or shelf/gallery ledge), record a match in
landscape, and get a useful match analysis — rally timeline, per-player movement stats,
and (when footage allows) shot-level detail — with **zero calibration effort** and
**no dependence on the footage being our own locked 4K60 capture**.

This grows the existing line-calling pipeline into an analysis product. Line calling
itself is *not* the product here: it remains a quality-gated bonus tier and the live
two-phone product keeps owning refereeing.

## 2. Design principles

1. **The ball is never load-bearing.** Every analysis is placed on a robustness ladder;
   the bottom rungs (rally structure from audio+motion, player movement from person
   tracking + floor homography) must work on arbitrary camera-roll footage where
   per-frame ball detection is hopeless. Today the pipeline is inverted: everything
   hangs downstream of ball tracking.
2. **Calibration is invisible.** Squash courts are dimensionally standardized; a
   back-court frame shows enough known lines to solve the camera automatically. The
   12-13-tap wizard becomes a fallback/nudge surface, never a prerequisite.
3. **Honest capability cards.** Analyses that can't run on a given recording are
   *disabled with a stated reason* ("ball tracking off: 30 fps + motion blur"), never
   silently degraded and never reported as empty success. A run always reports coverage
   (e.g. % of frames with a ball detection) so "found nothing" and "couldn't look" are
   distinguishable.
4. **One pipeline, cloud-first.** The Python pipeline stays the single source of truth;
   nothing in this MVP adds on-device inference or new Swift logic twins.
5. **Grounded increments.** The line-calling eval baseline must not regress; every
   refactor that touches judged outputs proves equivalence via `/eval`. New tiers get
   their own eval axes before being trusted.

## 3. The analysis ladder (target architecture)

| Tier | Signal | Outputs | Works on |
|---|---|---|---|
| 1. Rally structure | audio transients + frame-motion energy | rally boundaries, rally count/lengths, tempo, work:rest | any footage with audio |
| 2. Player movement | person detection → 2 tracks → feet through floor homography | per-player heatmap, distance, speed, T-time, front/back balance | almost any footage (needs solved court) |
| 3. Ball detail | existing YOLOX ball detector + trackers | shots per rally, wall-impact heights, target zones, speeds | sharp ≥50 fps footage (our capture) |
| 4. Line calls | existing judge | IN/OUT on demand | unchanged; not part of this MVP |

Tier N never requires tier N+1. The report renders whatever tiers are enabled, with
capability cards explaining the rest.

## 4. Current-state gaps this design corrects

Verified against the codebase on 2026-07-27 (file:line refs were checked):

- **Inverted dependency:** default engine emits `hits=[]` with `status="complete"` when
  the ball is never found (`job_runner.py:1479-1484`); rally segmentation exists but
  consumes *front-wall hits* (`job_runner.py:863`), so rally structure inherits ball
  fragility. Audio is subordinate by design (`audio_events.py:1-7`).
- **Fictional player attribution:** `player_number` is serve-alternation arithmetic
  (`job_runner.py:927-931`, env-var first server at `:84`), not observation; the whole
  coach report builds on it (`app.py:446-482`). There is zero person detection in the
  repo.
- **Calibration is a hard gate:** empty calibration 400s at `app.py:1136-1139`
  ("effectively required today" per the comment at `app.py:1186-1188`); the wizard
  needs ~12-13 taps; there is no automatic court-line detection anywhere;
  `/api/calibration/latest` serves the most-recently-touched calibration from any
  court (`app.py:974`). Separately, the 3D `solve_camera_model` rejects every real
  stored calibration (chirality defect, tracked outside this MVP) — which is why this
  design gates everything on the 2D floor homography instead.
- **Capture-coupled constants:** frame-window and pixel thresholds are tuned to 4K60 in
  native pixel units (`detect_wall_hits.py:12-41`, `tracking_common.py:12-21`) and break
  at 30 fps / 1080p; the front-wall ball at 1080p sits at or below the detector's
  7-24 px operating range (`ios/MODEL.md`) — reduced recall at 1080p60 is the standing
  eval headline (71/109 missed). No quality probe exists: width/height are computed
  and discarded (`app.py:955-963`).
- **Upload model:** whole-file multipart POST with a 2 GB cap (`app.py:55`,
  `deploy/Caddyfile`) ≈ 5 minutes of our own 4K60 — a 40-minute match cannot be
  ingested at all.
- **No report surface:** Matches/Coach tabs are placeholders (`index.html:1064`,
  `:1083`); no run-listing endpoint; no annotated output; per-run summaries never roll
  up to a match.
- **Open-loop flywheel & eval blind spots:** corrections land only in the eval set;
  eval axes are line-call-only (missed bounces 71/109 is the standing headline weakness,
  `eval_set/BASELINE-2026-07-23.md`); rally boundaries, auto-solve accuracy, and
  movement stats have no eval.

## 5. MVP definition

A user can, on the **web app** (iOS untouched this cycle):

1. **Import** any landscape video (file picker already exists) — including 1080p30
   camera-roll footage — or use existing in-app recordings. Uploads work past 2 GB via
   chunked upload.
2. Ingest **probes** the footage (fps, resolution, sharpness, audio) and computes a
   **capability set** with reasons. Time-domain pipeline constants are fps-normalized
   (identity at the 60 fps reference, so the line-call eval must show zero drift);
   pixel-domain thresholds stay at reference deliberately — the ball-tier gate only
   admits near-reference footage, and tiers 1-2 don't consume them.
3. The court is solved **automatically** from detected lines against the known court
   geometry, accepted by a **floor-homography residual gate plus wall-line ordering**
   (2D only — chirality-safe; the full 3D camera solve stays off the critical path);
   the user sees a skeleton overlay with Accept / Adjust / Skip (Adjust drops into the
   existing wizard pre-seeded; Skip analyzes without court lines). No calibration →
   run still proceeds with tiers that don't need it.
4. **Rally timeline** is computed from audio+motion, upstream of and independent of ball
   tracking; reconciled with hit-based rallies when the ball tier is on.
5. **Player movement stats** (heatmap, distance, speed, T-time, front/back split) come
   from a two-player tracker behind a swappable person-detector seam. A classical
   motion-blob fallback makes the tier work end-to-end today; provisioning real detector
   weights is a HUMAN GATE and the capability card says which backend produced the stats.
6. A **match report** (`report-v1` JSON + `GET /api/runs/<id>/report`) renders in a real
   Matches tab: capability cards, rally timeline, movement stats + heatmaps, and the
   existing shot/coach data when tier 3 is on. Runs get a listing endpoint.
7. Every new analysis has an **eval axis** and a committed baseline: rally-boundary F1
   vs labels, auto-solve px-delta vs the stored wizard calibrations in `ui_runs/`, and
   the untouched line-call baseline proving no regression.

## 6. Explicitly out of scope (MVP)

- iOS changes of any kind (import, proxy upload, native report). The web report is
  reachable from the iOS webview tabs as-is.
- Event-windowed/proxy upload optimization (chunked whole-file is enough for MVP).
- Score entry/score inference, highlights export, shot-type taxonomy beyond what tier 3
  already produces, accounts/identity, Android.
- Any change to the stereo/peer/live stack or to line-call judging behavior.
- Rally *winner* inference is retained as-is in tier 3 outputs but must be labeled in
  the report UI as inferred ("est."), not asserted.

## 7. Human gates (the intended stopping line)

The implementation session stops when everything is done **except**:

1. **Person-detector weights provision.** The seam ships with the classical fallback and
   a documented weights contract (`models/person-<name>/` mirroring the ball-model
   convention); dropping in real weights (e.g. COCO-pretrained YOLOX-Nano person class,
   Apache-2.0, matching existing YOLOX infra) is a human step. No training is strictly
   required for MVP.
2. **Optional ball-detector retrain** for recall (existing known lever; not required for
   MVP since tier 3 is quality-gated).
3. **Label verification:** silver rally-boundary labels seeded from existing good runs
   should be human-spot-checked; the eval runs regardless.

## 8. Success criteria

- Full pytest suite green (≥ current count) at every commit; line-call eval shows zero
  drift after the constant-normalization refactor and no regression at MVP end.
- Auto-solve: accepted (`status=="ok"`) solutions on ≥70% of the *frame-recoverable*
  stored calibrations (source video still on disk — ~38 of 46), with median landmark
  delta ≤ 15 px vs wizard `refined_px` on the v2 subset that has floor landmarks
  (7 calibrations) and a wall-line y-delta metric reported for the v1/unversioned
  rest (report committed as `eval_set/BASELINE-AUTOSOLVE-<date>.md`). Falling short
  is reportable, not fatal — the wizard fallback keeps the product whole; the gate
  protects correctness.
- Rally boundaries: F1 ≥ 0.8 at ±1.5 s tolerance against the seeded silver label set
  (`eval_set/BASELINE-RALLY-<date>.md`); the 1080p30 end-to-end proxy clip must still
  produce a nonempty rally timeline.
- End-to-end: the repo's real reference clip `SquashAnalytics.mp4` (1080p60 — passes
  the ball-tier gate) AND its 1080p30 PyAV proxy both produce a rendered report — the
  former with tiers 1-3, the latter with tiers 1-2 enabled and tier 3 visibly gated
  with a reason. Verified in both themes at a phone viewport via `/verify`. (No 4K
  asset exists in the repo; 1080p60 vs 1080p30 exercises the same gate honestly.)
- `docs/HANDOFF-tripod-mvp.md` exists, stating baselines and the exact human steps.

## 9. Risks / notes

- Auto court-line assignment is the riskiest CV piece; it is validated synthetically
  (render known poses through `court_model` and recover them) before touching real
  frames, then scored against every stored wizard calibration. If the accept-gate pass
  rate is low on real frames, the wizard fallback keeps the product functional — the
  gate protects correctness, not availability.
- The motion-blob fallback will under-perform crossing/occluded players; identity
  confidence is surfaced and the capability card names the backend, so stats are honest
  until real weights land.
- All thresholds introduced by this design (probe gates, rally gaps, T radius) are
  named constants with an eval that exercises them — "tuned by eval", never silently.
