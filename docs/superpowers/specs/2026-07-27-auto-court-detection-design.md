# Auto Court Detection — Design Spec

**Date:** 2026-07-27
**Branch:** `feat/auto-court-detection`
**Status:** Approved design; implementation plan to follow.
**Audience:** an autonomous implementation session with full codebase access and zero
conversation context. Everything it needs is in this spec, `CLAUDE.md`, `DESIGN.md`, and
the code.

## 1. Goal

Replace the judge-a-clip calibration wizard's **~12–19 precise taps across 7 screens**
with **one screen and at most one tap**: the app detects the court from the picked frame,
fills in every calibration artifact, and shows a single confirm screen where the user
either accepts or drags a bad anchor.

This is the first delivery of principle 2 in
`2026-07-27-tripod-match-analysis-design.md` — *"Calibration is invisible… the wizard
becomes a fallback/nudge surface, never a prerequisite."* It is also the first half of the
capture goal in `CLAUDE.md`: the same detector, later run on a live frame instead of a
saved one, is what makes calibration-during-recording possible.

**The manual wizard is not replaced and not deleted.** It stays wired, reachable, and
unchanged, as the escape hatch and the path for courts detection cannot handle.

## 2. What calibration returns today (the thing being automated)

Three artifacts, produced by `buildJson()` (`index.html:2593`) under schema
`squash-calibration-v2`:

| Artifact | Produced by | Consumed by |
|---|---|---|
| `lines[]` — out / service / tin, as image-space lines | 3 flood-fill tap screens | `judge_call.load_calibration_lines` (`judge_call.py:144`), `load_service_line` (`:155`), target-zone *y* normalization, `pixels_per_foot` velocity scale, `detect_wall_hits.py:447` |
| `planes.wall.corners` — 4 taps | `tap_wall` screen | `judge_call.load_wall_corners` (`judge_call.py:163`) → lateral x-gate; `court_model._camera_correspondences` (`:628`) |
| `planes.floor.landmarks` — 4–7 taps | `tap_floor` screen | `court_model.load_floor_calibration` (`:375`) → `job_runner.py:825` `court_position_ft` + `floor_zone` |

Current phase order (`index.html`, `setPhase`): `frame → tap_out → tap_tin → tap_service
→ review → tap_wall → tap_floor → clip`.

**The detector's output contract is: produce exactly these same structures.** Nothing
downstream of `buildJson()` changes.

## 3. Why this is tractable — the geometric argument

Everything the wizard collects lies on **two planes of one rigid, dimensionally
standardized court**:

- **Front wall plane** — the out line, service line, tin, and all four wall corners are
  one **8-DOF homography**.
- **Floor plane** — another **8-DOF homography**.
- The two planes **share the front-wall/floor seam**, so two of the four wall corners
  *are* two of the floor landmarks. The existing wizard already exploits this
  (`seedFrontSeamFromWall`, `index.html:3087`).

So every landmark the wizard asks a human to tap is an **intersection of two long
lines** — and a line intersection from hundreds of fitted pixels is far more precise
than a fingertip:

| Wizard landmark | Intersection of |
|---|---|
| wall `top_left` / `top_right` | front-wall out line × left/right side-wall out line |
| wall `bottom_left` / `bottom_right` (= floor `front_seam_left` / `_right`) | front seam × left/right side-wall floor seam |
| floor `short_line_left` / `_right` | short line × left/right side-wall floor seam |
| floor `t_point` | short line × half-court line |
| `out_line_lower_edge`, `service_line_top_edge`, `tin_top_edge` | detected directly |

Both fits are plain 2D DLTs (`court_model.fit_homography`, `:237`). **No 3D pose is
solved anywhere in this feature**, which keeps it clear of the `solve_camera_model`
chirality defect that rejects every real stored calibration (tracked separately; see
`2026-07-27-tripod-match-analysis-design.md` §4).

### 3.1 Feasibility evidence

A throwaway spike (colour-agnostic: temporal median → LAB black-hat → `HoughLinesP`, zero
tuning) run on the real fin-mount footage `~/Desktop/Training Data/Bay Club Squash 1
Rally+audio.mov` found, cleanly and at full span: **front-wall out line, service line,
front wall/floor seam, both side-wall floor seams, and the floor short line**. Both
players disappeared into the temporal median. One false positive (the ceiling junction).

That is essentially every structure this design needs.

## 4. Two corrections the spike forced

1. **Never key on hue.** Bay Club's court lines are **navy**; SquashAnalytics' are
   **red**. The invariant that actually holds across both is *"a thin stripe darker and
   more chromatic than its local surround"*. A red-hue detector works on one court and
   fails on the next.
2. **Detect the out line; do not extrapolate to it.** Deriving the out line (15 ft) by
   projecting up from a tin+service baseline (1.58 → 5.84 ft) amplifies endpoint error
   ~3.5× and lands it on the most call-critical artifact in the pipeline. The out line is
   the highest-contrast structure in frame — it spans the front wall *and* runs up both
   side walls. Extrapolation is the **fallback** for mounts that clip it out of frame,
   and is deferred out of this slice (§9).

A third finding: naive HSV floor segmentation is brittle (0.3% coverage on Bay Club,
31.5% on SquashAnalytics — same thresholds). **This design does not use colour
segmentation for the floor.** The floor's extent is bounded by the seam lines the line
detector already returns.

## 5. Datum rules — get these wrong and every number is silently biased

The codebase uses **two different datums**, deliberately, and the detector must honour
both.

**Front-wall lines use painted-band EDGES**, per `CFG` (`index.html:1172`) and
`extractEdge` (`index.html:2183`; mode `'min'` = topmost mask row per column, `'max'` = lowest):

| Line name | Edge | Height (ft) |
|---|---|---|
| `out_line_lower_edge` | `max` — **lower** edge of the band | `OUT_LINE_HEIGHT_FT` = 15.0 |
| `service_line_top_edge` | `min` — **top** edge | `SERVICE_LINE_HEIGHT_FT` ≈ 5.8399 |
| `tin_top_edge` | `min` — **top** edge | `TIN_TOP_HEIGHT_FT` = 19/12 ≈ 1.5833 |

**Floor landmarks use painted-line CENTERLINES**, per the `FLOOR_LANDMARKS` comment block
(`court_model.py:49-62`): e.g. `short_line_left` sits at
`[0.0, SHORT_LINE_CENTER_Y_FT]` ≈ `[0.0, 17.918]`, which is the short line's *middle*,
not its WSF datum edge at 18.0.

So: for the three wall lines, fit the specified **boundary** of the detected stripe; for
the short line and every other floor line, fit its **centerline**. A 50 mm line is
~0.164 ft; getting this backwards is a systematic half-line-width bias, well above the
px-level residual targets the rest of the system is built to.

## 6. Architecture

| File | Change |
|---|---|
| `court_detect.py` | **new** — pure detection functions (cv2 + numpy). No Flask import. |
| `tests/test_court_detect.py` | **new** — paired test; the PostToolUse hook auto-runs it on every edit to `court_detect.py`, so it must stay fast. |
| `app.py` | **new** `POST /api/detect-court` |
| `index.html` | detect trigger on the frame step; new `confirm` phase; drag handles |
| `DESIGN.md` | document the confirm screen in §8 |

**Detection runs server-side in Python/cv2**, not in browser JS: cv2 supplies the
morphology and Hough tooling the spike proved out (`cv2 4.10.0`, `HoughLinesP`, LSD, and
`ximgproc` all confirmed present in `.venv`); `court_model.py` is already the
authoritative fitter; it is unit-testable under pytest; and one endpoint serves both the
web app and the iOS app.

**The client posts 5 JPEG frames**, sampled across the clip using the seek machinery
already in `setupFrameTimeline`. This avoids uploading the video (which today happens
only at track time, `index.html:4380`) and gives the server enough frames for a temporal
median. ~1.5 MB total over localhost.

## 7. The detector

1. **Median** the 5 posted frames. Compute inter-frame difference first; if it indicates
   a moving camera, fall back to the single picked frame and return a warning. (Required:
   `SquashAnalytics.mp4` pans, and a whole-video median smears it into a ghost. The
   product's fin mount is static, so the median is the normal path.)
2. **Line-ness map** — `MORPH_BLACKHAT` over LAB luminance (responds to structures darker
   and thinner than the kernel), maxed with local chroma deviation (`ab` channels minus
   their local median). Colour-agnostic; this is what caught navy *and* red.
3. **Segments** — Otsu threshold, `HoughLinesP`, merge collinear segments into full lines,
   keep those longer than ~25% of frame width.
4. **Assign by geometry** (thin-slice heuristics):
   - near-horizontal family, ordered top-to-bottom by image *y*;
   - the two long diagonals on opposite sides of the frame are the side-wall floor seams;
   - the **front seam** is the horizontal line both diagonals terminate at;
   - the **out line** is the topmost horizontal spanning the full wall width;
   - the **short line** is the horizontal below the seam crossing both floor seams.
5. **Extract the correct datum per §5**, then intersect → 4 wall corners, 2 short-line
   ends. Feed `court_model.fit_homography` twice → wall homography, floor homography.
6. **Self-verify** — project the *unused* structures (tin, service line, half-court line,
   service box lines, the T) through the homographies and measure the distance from each
   prediction to the nearest detected paint. These residuals drive the confirm screen's
   per-anchor state and the overall confidence verdict.

## 8. The confirm screen (new `confirm` phase)

Registered in `STEP_META` (`index.html:1229`) and in the `setPhase` section list. Reuses
the existing stage canvas and `render()`.

**Trigger:** the frame step's existing primary, **USE FRAME** (`#useFrame`), runs
detection instead of jumping straight to `tap_out`. There is no separate "detect" button —
detection is the default path, and the manual flow is reached from the confirm screen.

**Back-chevron:** `hdrBack` from `confirm` returns to `frame` and clears detection state,
matching how `tap_out` already returns to `frame` (`index.html:1890`).

**On-site recording calibration inherits half of this for free.** The record phase runs
the same wizard behind `S.rec.calibrating`, and `finishCalWizard()` (`index.html:1878`)
already branches on it, so *accepting* a detected calibration works identically from
either entry point. The *trigger* does not carry over: `recCalibrateBtn` freezes a live
camera frame and enters `tap_out` directly, and frame grabbing here seeks the `<video>`
element, which the record path is not using. Automatic detection during recording needs a
second frame source off the live `camVid` stream and is out of scope (§9); the record
phase keeps the manual wizard.

**Draws:** the fitted lines in their existing reserved colours (`CFG.out` `#35e0ff`,
`CFG.service` `#ff9f43`, `CFG.tin` `#b4ff3a`), the floor wireframe via the existing
`COURT.wireframe` projection, and 6 draggable anchor handles at the 4 wall corners and 2
short-line ends.

**Anchor state colours** follow the sanctioned floor-wizard palette in DESIGN.md §8.10 —
dim → `#3ddc84` done → `#f5c518` warned. **Never red**: DESIGN.md §0.3 reserves red for
OUT verdicts. Per the same rule, colour is never the sole carrier — a named text list
calls out any anchor that is off.

**Controls:**
- Primary (`.proxied`, per DESIGN.md §3.4): **USE THIS CALIBRATION** → `finishCalWizard()`
- Secondary: **TAP IT MANUALLY** → `setPhase('tap_out')`, the existing flow, unchanged

### 8.1 What a drag does — and what it must not do

This is the one place the two artifact families can disagree, so the rule is explicit:

- **Anchors are authoritative for `planes`.** Dragging one updates that anchor's pixel
  position and re-fits the wall and/or floor homography live.
- **Detected line fits stay authoritative for `lines[]`.** A drag never re-derives a line
  from the two dragged corners. The detected fit runs through hundreds of edge pixels;
  regressing it to a 2-point fit through fingertip taps would *lower* the precision of the
  most call-critical artifact in the pipeline — the exact failure this whole design is
  built to avoid (§4.2).
- **Divergence is surfaced, not silently absorbed.** If a dragged corner ends up further
  than a threshold from the fitted line it belongs to, the screen names the disagreement
  and offers the manual re-tap for that one line rather than guessing which is right.

**States:** the "Detecting court…" wait uses the sanctioned analyzing-scrim pattern
(DESIGN.md §8.12). The status line has reserved height (§0.9, no layout shift). On
detection failure the screen is skipped entirely and the user lands on `tap_out` with an
explanatory status.

Verify in both themes at 390 × 844 with the `/verify` skill (DESIGN.md §0.12).

## 9. Explicitly out of scope for this slice

Deferred to hardening, each with a named reason:

- **Vanishing-point line grouping.** The thin slice treats a detected line's endpoints as
  the wall junction. That worked in the spike because the junction is a real slope
  discontinuity, but VP-based intersection of the front-wall and side-wall lines is the
  correct method.
- **Tin+service → out-line extrapolation fallback** for mounts that clip the out line.
- **The scoring harness** against all 46 stored calibrations (§10).
- **Recording-time detection** (the `record` phase / iOS capture).
- **Lens distortion.** `distortion` stays `null`, as today.

## 10. Validation assets (already on disk)

- **46 stored calibrations** in `ui_runs/*/calibration.json`, each carrying `frame_time_s`
  so the exact source frame can be re-extracted. Breakdown, measured 2026-07-27:
  **all 46 have line fits; 7 have floor landmarks (all 4-point, `fit_rms_px` 0); 0 have
  wall corners.**
- **Source videos:** `SquashAnalytics.mp4` (in repo, 1920×1080@60, panning);
  `Bay Club Squash 1 Rally+audio.mov` and `Bay Club Squash 5min+audio.mov` in
  `~/Desktop/Training Data/` (1920×1080@60, static fin mount — the target geometry).

So line-fit ground truth is plentiful, floor ground truth is thin, and **wall-corner
ground truth does not exist**. Early validation of the wall corners therefore leans on
self-consistency residuals (§7.6) plus agreement with the 46 line-triples.

## 11. Testing

- **Synthetic courts are the backbone.** Project a known camera onto a blank image, draw
  the court's lines at their true positions, add noise and glare, and assert the detector
  recovers both homographies within tolerance. Deterministic, fast, no video assets — and
  it is the only way to test correctness of the §5 datum handling exactly.
- **One real-frame smoke test** against `SquashAnalytics.mp4` (in repo). Bay Club assets
  live outside the repo, so any test touching them must skip cleanly when absent.
- Full suite stays green: `.venv/bin/python -m pytest tests/ -q` → currently
  "318 passed, 1 deselected".

## 12. Risks

- **Only two courts of footage.** Colour-agnostic detection is the hedge, but a third
  court is a genuine unknown. This is the main reason §1 keeps the manual wizard wired.
- **Detected-endpoint-as-wall-junction** is a deliberate shortcut (§9).
- **No wall-corner ground truth** (§10).
- **Glare and ball-marked walls** produce false line candidates; length, straightness, and
  span filters are the first defence, VP consistency the second (§9).

## 13. Success criteria

1. On both Bay Club clips, "Use this frame" produces a confirm screen whose overlaid
   wireframe visibly hugs the real court, with no anchor dragging required.
2. Accepting that calibration produces a `calibration.json` that `judge_call` and
   `court_model.load_floor_calibration` parse with no code changes, and a run completes
   end to end.
3. The manual wizard still works, reached from the confirm screen, unchanged.
4. `pytest tests/ -q` green; UI verified in both themes at 390 × 844.
