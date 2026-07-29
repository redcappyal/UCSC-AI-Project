# Outside-Glass Court Detection — Design Spec

**Date:** 2026-07-29
**Branch:** `claude/auto-homography-frame-72-test-23ca74`
**Status:** Approved direction; implemented in the same session.
**Extends:** `2026-07-27-auto-court-detection-design.md`. Everything in that spec still
holds — output contract, datum rules (§5), "never key on hue", "never solve 3-D pose" —
except its §7.4 assignment heuristics, which this spec replaces.

## 1. Goal

`detect_court` currently works from the fin-mount geometry: camera inside the court,
against the back wall. Real users bring a tripod and film **through the back glass**,
from wherever the gallery lets them stand. On such footage today the detector refuses
every frame (correct fail-safe, wrong capability): tested 2026-07-29 on
"2026 Squash Zone Gold Loss vs Faraaz Khan (Match 2)", every probe returned
`insufficient_lines`, and the partial assignments were wrong (front-wall service line
taken as the floor short line; side-wall out stripes taken as `service`/`front_seam`).

Make the same single detection path work for both geometries, and make it robust to
**partial line occlusion**: any straight court line whose fragments are detected must be
usable at full span by extending its fit, never limited to the span occlusion left
visible.

## 2. What outside-glass footage actually breaks (measured)

From the Squash Zone median frame (players erased, camera steady):

1. **The side floor seams are invisible.** Fogged/ball-marked side glass hides the
   floor–side-wall junctions. The current assigner *requires* both: every floor anchor
   is an intersection with a side seam. This is the load-bearing failure.
2. **Glass furniture wins the current heuristics.** Pane-edge verticals (full-height,
   very strong) get picked as "seams"; the decorative frit band at the bottom of the
   glass produces long near-horizontal lines that get picked as the short line.
3. **Vertical ordering above/below one seam guess is not viewpoint-stable.** With the
   court occupying only the middle of the frame, "topmost horizontal" is a ceiling or a
   side-wall stripe, not the out line.
4. **Lines fragment.** Lights wash out parts of the out line; pane boundaries introduce
   small refraction offsets; players occlude mid-line. Fragments merge (find_lines
   already does this) but detected *extent* is not trustworthy anywhere.

What outside-glass footage still offers, same frame: the full front-wall stack (out,
service, tin), the front seam's left portion, **both side-wall out stripes**, the short
line's left and right portions, a half-court-line piece, and right-box paint.

## 3. The geometry that replaces the heuristics

All 2-D projective constructs; no pose, no hue, no learned model.

- **Cross-ratio pins the front-wall stack.** Out (15.0 ft), service (5.8399 ft), tin
  top (1.5833 ft) and the seam (0 ft) are four parallel lines of one plane, so the
  cross-ratio of the four points they cut on ANY image transversal is a viewpoint-free
  constant. That constant both *selects* the true stack among horizontal candidates and
  *derives* any one missing member (e.g. a seam hidden behind boards: solve the 1-D
  projective equation per column, two columns give the line).
- **The stack's pencil gives VP_x** (vanishing point of court-x). Every other court-x
  line — the short line, box back lines — must pass through it.
- **The side-wall out stripes give VP_y.** They run along court-y, as do the side
  seams and the half-court line. Their intersection is VP_y.
- **Occluded side seams are reconstructed, not found**: seam corner = front seam
  evaluated at the wall's edge column; side seam = the line joining that corner to
  VP_y. When real steep seam lines ARE detected (fin mount), they are used directly and
  VP_y is taken from them instead.
- **Wall edge columns** come from the stack fits' shared x-extent (the paint physically
  ends at the corners), refined by corner verticals and side-out × front-out
  intersections when present.
- **The short line is chosen by hypothesis, not by length.** Candidates = horizontals
  below the seam, VP_x-consistent, with paint inside the reconstructed seams (this
  kills the frit band, whose paint extends outside the court). Each surviving candidate
  fits a floor homography and is scored by the existing self-verification checks; the
  best verified candidate wins. Ties without independent verification stay
  low-confidence, as today.

## 4. Occlusion contract (the user-facing promise)

A court line is usable if its detected fragments pin its direction — in practice: the
fragments span enough columns for a stable fit (`MIN_DATUM_COLUMNS` already encodes
this). Anchors are then intersections of *infinite* fitted lines, so mid-line occlusion
and occluded far endpoints cost nothing. Detected extent is never load-bearing except
at the wall edge columns, where three independent stacked lines vote and corner
verticals/side-out intersections override.

## 5. Changes

| File | Change |
|---|---|
| `court_detect.py` | new geometry helpers (cross-ratio, VP estimation, line-through-point construction); `assign_lines` rewritten around §3; anchor derivation gains the reconstruction fallbacks; datum refit, homography fit, checks, payload untouched |
| `tests/synthetic_court.py` | render side-wall out stripes (planes x=0/x=21), optional occluders, optional glass-frit clutter band, optional fog that hides the side seams; new outside-glass camera preset |
| `tests/test_court_detect.py` | outside-glass synthetic cases (fog, mid-line occluders, frit); real-frame regression on the Squash Zone median (`tests/data/`, skip-if-absent); existing fin-mount tests must pass unchanged |

`app.py`, `index.html`, schema: no changes. One detection path; evidence-graded
fallbacks inside it, no engine fork.

## 6. Alternatives considered

- **Global template fit (hypothesize-and-verify over full assignments):** strictly more
  robust in theory, but a rewrite with real local-optimum risk on foggy glass, much
  slower (many homography fits per frame), and it abandons the measured, explainable
  per-line failure reasons the wizard fallback depends on. The §3 design keeps
  hypothesis testing only where evidence is genuinely ambiguous (short line).
- **Learned court detector:** no labeled data for a third court exists; classical
  projective structure covers the observed failures without a training pipeline.

## 7. Success criteria

1. On the Squash Zone median frame: `status="ok"`, all seven entities correct by
   overlay inspection, floor anchors within a few px of the visible paint
   intersections, and the wireframe hugging the real court.
2. Frame-72-style single frames (player occluding mid-lines) either succeed with
   correct geometry or refuse — never a confidently wrong calibration.
3. Every existing fin-mount and synthetic test passes unchanged.
4. `pytest tests/ -q` fully green.

## 8. Out of scope

- Refraction modelling at pane boundaries (observed ~px-scale jumps; absorbed by the
  robust fits and reported honestly by check residuals).
- Wall-mounted courts with no visible side-wall out stripes AND no visible side seams
  (no VP_y source → refuse, as today).
- UI changes; recording-time detection; lens distortion.
