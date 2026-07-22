# Roadmap placeholder UI — design spec (2026-07-22)

Frontend-only scaffolding for the six future roadmap pillars so the team can see the
navigation hierarchy before any backend exists. No backend calls; every new page is a
placeholder.

## Problem

The roadmap (data flywheel, live match mode, auto video editor, stats + AI coaching,
coaching platform, shot-selection bot) has no home in the UI. DESIGN.md forbids new nav
chrome (§3.3, §8.3, §18), so the hierarchy must grow out of the existing
phase/section grammar.

## Decision: Home hub + drill-down placeholder phases

Approaches considered:

1. **Nav-pill tabs per pillar** — rejected: §8.3 caps the pill at 3 items, modes only,
   and 6 pillars can't fit.
2. **One combined "Roadmap" phase** — rejected: violates "one thing per screen" (§1) and
   gives the team no per-feature page to grow into.
3. **Feature cards on the start screen → one phase per pillar** (CHOSEN): `p-load` is
   already the app's home (only screen with the nav pill). Add a stack of tappable
   feature cards below the load button. Each card opens a placeholder phase via the
   existing `setPhase()` + back-chevron grammar. Zero new nav chrome; scales.

## Information architecture

```
p-load (home)
├─ Load clip …existing judge flow (p-frame → … → p-target)   [unchanged]
├─ nav pill: Judge | Label                                     [unchanged]
└─ UP NEXT (feature cards, in roadmap order)
   ├─ Live match        → p-live       (phase 2 · pillar 1)
   ├─ Clip library      → p-clips      (phase 3 · pillar 2)
   ├─ Stats + coaching  → p-coach-ai   (phase 4 · pillar 3)
   ├─ Coaching platform → p-platform   (phase 5 · pillar 4)
   ├─ Shot selection    → p-shot-bot   (phase 6 · pillar 5)
   └─ Model improvement → p-flywheel   (data flywheel — ongoing)
```

- Back chevron from any placeholder returns to `load`. No deeper navigation yet.
- The nav pill stays Judge/Label modes only and stays hidden off the home screen.
- `p-target` (existing stats screen) is untouched; when Stats + coaching ships it will
  absorb/replace the placeholder, not the other way around.
- Note: `p-coach-ai` / `p-shot-bot` names avoid colliding with existing `coach*` element
  ids (`coachMetrics` etc. in `p-target`).

## New components (to be codified in DESIGN.md in this same change)

### Feature card (`.featureCard`) — tappable card variant

A `<button>` styled on the §8.9 card recipe: `--surface` fill, `1px --line` border,
radius 8, full width, text-align left, normal case (cards are content, not controls —
the uppercase rule applies to their STATUS TAG only). Layout: 24 px line icon (§9 style)
· title 16/700 + one-line 13 px `--dim` description · right-aligned 12/600 uppercase
`--dim` tag (`PHASE 2`, `ONGOING`). Min-height 56 px, padding 12px 14px, pressed state =
`--line` fill (0 ms, §10). The whole card is one ≥44 pt target.

A 12/600 uppercase `--dim` section heading (`UP NEXT`) separates the cards from the load
button; heading + cards live inside `p-load` below `#loadStatus`.

### Placeholder page pattern (`.placeholderHero` + capability list)

Each placeholder phase section contains, top to bottom (§7 stack, gap 10):

1. Instruction line (sentence case, `--dim`): one line saying what this screen will do.
2. **Placeholder hero**: the §13 `.blank` treatment scaled to a page — dashed 1px
   `--line` border, radius 8, min-height 180 px, centered column: the feature's 24×24
   line icon at 40 px, dim, then `COMING SOON!` (16/700, uppercase, tracking .05em,
   `--dim`). This is the "placeholder image" — drawn inline (SVG icon + text), no
   image files, no network (§0.5).
3. **Planned card** (§8.9 card): header "Planned" + meta right (`PHASE N · PILLAR M`),
   body = capability rows (15 px, sentence case) each with a leading chip-style tag:
   `CORE` (`.corePill` — `--line` fill, `--text`) or `LATER` (`.laterPill` — transparent,
   `1px dashed --line`, `--dim`). Chips are informational (not interactive) so they may
   be < 44 px.
4. No primary action — header shows back chevron + step label only (like `p-label`).

Copy exception: the roadmap placeholder hero is the **only** sanctioned exclamation mark
in the app ("COMING SOON!"), per product owner request. DESIGN.md §14 gains this
exception in the same change.

### Per-page content (from the roadmap board)

| Phase | Step label | Core capabilities | Later |
|---|---|---|---|
| `p-live` | Live match | In-app match recording · Point-by-point score log | Contested-call review · Streaming inference |
| `p-clips` | Clip library | Per-rally clip library | Highlight auto-reels · Dead-time removal · Annotated shot overlays |
| `p-coach-ai` | Stats + coaching | Shot-type classifier · Per-shot rally dataset | Heatmaps + trend stats · LLM coach reports |
| `p-platform` | Coaching platform | Accounts + cloud storage | Match sharing · Clip comments + review · Drill assignments |
| `p-shot-bot` | Shot selection | Win-probability model | Shot decision grading · What-if rally replay · Opponent profiles |
| `p-flywheel` | Model improvement | One-tap call corrections · Corrections → eval set | Scheduled retraining · Model versioning |

Instruction lines (one sentence, referee voice, §14):
- live: "Record a match and get calls as they happen."
- clips: "Every rally, cut and saved automatically."
- coach-ai: "Shot types, patterns, and coaching from your matches."
- platform: "Share matches and clips with your coach."
- shot-bot: "See the shot the numbers would have played."
- flywheel: "Your corrections make the model better."

## Implementation notes

- All changes live in `index.html` (markup + `<style>` + `STEP_META` + `setPhase()`
  phase list + card click wiring) and `DESIGN.md` (new §8 subsections, §14 exception,
  §16 blueprint rows). No backend changes.
- `setPhase()` hides sections from a hard-coded id list — add the six new ids there and
  in the phase-specific `if` blocks (`stage` stays hidden like `load`/`target`:
  add the new phases to the `stage.classList.toggle` condition and give each a
  `document.body` class only if needed — reuse `phase-load` centering only for `p-load`).
- STEP_META entries: no `action` key (no primary), instr = lines above.
- Icons: inline SVG, §9 grammar (24×24, stroke 2, round caps, ≤2 shapes). One icon per
  pillar, reused card + hero.
- Verify both themes at 390×844 (§0.12) with the `/verify` skill or manual browser pass.

## Out of scope

Any real feature behavior, routes, backend endpoints, nav-pill changes, changes to the
judge/label flows, model versioning UI beyond the placeholder list.
