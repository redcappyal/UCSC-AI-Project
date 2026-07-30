# Confirm screen: give the frame the space it needs

**Date:** 2026-07-30
**Screen:** `#p-confirm` — the auto-detected court review, reached from `useFrame` →
`runCourtDetection` → `applyDetection` → `setPhase('confirm')`.

## The problem, measured

Driven at 390×844 with a real detection (`tests/data/crosscourt-demo-median-frame.png`
→ `/api/detect-court` → `status: ok`, `confidence: high`, all six anchors derived), the
video frame renders at **185 × 104 CSS px**. The 844 px viewport is spent like this:

| Region | px |
| --- | --- |
| `<header>` | 60 |
| `#instr` — "Check the overlay hugs the real court. Drag any anchor that sits off the line." | 54 |
| **`#stage`** | **104** |
| Court anchors card (`.targetZones`: head + help paragraph + six chips) | 300 |
| `#confirmUndoBtn` / `#confirmManualBtn` row | 44 |
| `#confirmSummary` + `#confirmWarn` | 78 |
| `#confirmDrift` — reserved, empty in the common case | 60 |
| `main` padding (12 top, 92 nav-dock clearance) | 104 |

This screen exists for exactly one judgement: does the overlay sit on the real paint? At
185 × 104 that judgement cannot be made. Pinch-zoom does not rescue it either — zooming
into a 104 px-tall stage yields a 104 px slit.

The single largest consumer is the **Court anchors card at 300 px — 57% of the control
area** — and it is a lookup table restating what the six numbered pucks already draw on
the frame. Its help paragraph ("Drag each numbered circle to the matching court point.")
duplicates `#instr`, and `#confirmSummary` duplicates `#instr`'s other half.

## The ceiling

The canvas is `max-width:100%` inside a full-bleed stage, so a 16:9 frame caps at
**390 × 219** at zoom 1. Freeing ~126 px reaches that cap; anything beyond becomes black
well. That surplus is not waste — it is the pinch-zoom headroom this screen currently
lacks, and it gives `#zoomCtrl` somewhere to sit that is not on top of the court.

## Changes

### 1. Delete the Court anchors card (−300 px)

Remove the `.targetZones` wrapper, `#confirmCount`, `#confirmAnchorHelp`, and
`#confirmAnchorList` from `#p-confirm`, and the `guide` field from
`confirmAnchorPoints()` that fed it.

The numerals stay on the pucks: they give the six anchors stable identity during a drag,
and `number` still belongs to the court landmark rather than the array position, so a
missing derived anchor never renumbers the rest. `#instr` carries the affordance, and
`confirmDivergence()` already names drifted corners by their human `WALL_CORNERS` label
rather than by number, so no message depends on the deleted lookup.

`confirmAnchorPoints()` itself stays — `drawConfirmOverlay`, `confirmDragHitTest`,
`confirmAnchorAt` and the `confirmIsTrusted` anchor count all read it.

### 2. Pin `#confirmDrift` to the stage (−60 px)

`#confirmDrift` reserves 60 px of permanent air to satisfy §0.9: it is the only text a
drag can change, and a height change mid-drag reflows the flex-grow stage, re-centers the
canvas, and corrupts the pointer math (a steady drag jumped ~50 px the first time that
sentence wrapped).

Move it inside `#stage` as an absolutely positioned banner along the bottom edge. It then
costs **zero** layout height and *cannot* reflow the stage — strictly stronger than the
reserve it replaces, and it lands in the black margin the larger frame creates. It gets
stage-overlay treatment (white on a dark scrim, §8.12's language) rather than `.status`,
whose `var(--text)` would be dark-on-dark in the light theme, and it is hidden whenever it
has nothing to say or the phase is not `confirm`.

### 3. Cut the duplicated sentences (−46 px)

`#instr` and `#confirmSummary` say the same thing twice. Split them by job:

- `#instr` — the action: "Drag any anchor that sits off the line." (one line)
- `#confirmSummary` — the verdict only: "Court detected." / "Court detected, but the fit
  is unconfirmed." (one line)

The reasons stay in `#confirmWarn` verbatim, unchanged. Green stays earned: the `.status
ok` line still requires a loaded detection, `status: ok`, `confidence: high`,
`checks_verified > 0`, no `off` check, and all six anchors derived.

### 4. Let the secondary buttons size to their text

`UNDO DRAG` and `TAP IT MANUALLY` sit in a `.row` whose `>*{flex:1}` stretches each to
half the screen. Two heavy full-width outlines for secondary actions is more weight than
they have earned. They keep their 44 px touch height and their order; they just stop
claiming the full width.

## Result

`main` 626 → ~227, `#instr` 54 → 27, so `#stage` goes **104 → ~530** and the frame renders
at its **390 × 219** ceiling — 2.1× wider, **4.4× the area** — with ~150 px of black well
above and below as zoom headroom. On a 375 × 667 phone the stage lands near 353, still
comfortably past the 219 the frame needs.

## Not changing

- The 92 px nav-dock bottom clearance on `main` and its safe-area term (§3.2 shell
  invariant), even though `#navPill` is hidden on this phase.
- `#confirmWarn`'s content, including the drag caveat — it is the greyscale carrier for
  the ok/warn signal (§8.19) and every sentence in it is fixed while the screen is up.
- Puck size, the 44 CSS px drag target, the grab offset, undo behaviour, the fitted-line
  hues, or the wireframe's two-colour ok/warn palette.

## Verification

- Playwright at 390×844 and 375×667, both themes, through the real detector: confirm the
  measured stage/canvas sizes above.
- Drag an anchor far enough to trigger `#confirmDrift` and confirm the banner appears over
  the stage while the canvas does **not** move — the §0.9 regression this change is meant
  to make impossible.
- `tests/test_calibration_ui_contract.py::test_auto_calibration_drag_handles_have_stable_numbers_and_guide`
  pins the deleted guide text and list template; it is rewritten to pin what now matters —
  stable puck numerals, no guide list in the section, and the stage-pinned drift banner —
  so the reclaimed space cannot silently regress.
- Full suite (`pytest tests/ -q`).

## DESIGN.md

§8.19 is updated in the same change: the numbered-guide bullet, the quoted green sentence,
and the three-status-paragraph bullet all describe the layout above.
