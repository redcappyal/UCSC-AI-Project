# Production home IA — design spec v2 (2026-07-22)

Supersedes `2026-07-22-roadmap-placeholder-ui-design.md`. Product-owner feedback on v1:
the flat six-card list is not a hierarchy; Load clip / Debug targets clash visually with
the cards; a real app has no "Model improvement" button. This revision is a **prototype
of the production UI** — organized the way the shipped app will be.

## Information architecture (three sections + drill-down)

```
Nav pill (tab bar on section roots only): Play · Matches · Coach

Play (home, p-load)                          ← default section
├─ HERO: Judge a clip  (accent card, WORKS TODAY → existing load/judge flow)
├─ HERO: Live match    (surface card, SOON tag → p-live placeholder)
└─ DEV row (11/600 dim "DEV" micro-label + small buttons):
   Debug targets (existing) · Label mode (toggle, replaces nav-pill Label)

Matches (p-matches)   placeholder — the auto video editor / library pillar
Coach (p-coach)       hub page — feature cards drilling into:
├─ Stats + trends     → p-stats     placeholder (phase 4)
├─ Shot selection     → p-shot-bot  placeholder (phase 6)
└─ Your coach         → p-sharing   placeholder (phase 5, coaching platform)
```

Removed from the UI: **Model improvement / data flywheel** (one-tap corrections already
live in the call screen's Challenge pane; retraining, eval sets, and model versioning
are backend concerns a player never sees) and **per-shot rally dataset** (internal).

Rationale:
- **Play / Matches / Coach** is the player's mental model: do the thing → see what
  happened → get better. Three sections fit the 3-item nav-pill cap exactly.
- The nav pill becomes a true **section switcher (tab bar)**: visible on the three
  section roots only, hidden inside flows and sub-pages. Back chevron is hidden on
  section roots (they are siblings — the pill switches between them) and shown on
  sub-pages/flows. This is a deliberate DESIGN.md §8.3 amendment: items are top-level
  sections now, not judge/label modes.
- **Label** is an internal data-collection tool, not a player feature → demoted to a
  dev-row toggle on Play. Judge stays the default mode; toggling Label mode routes the
  next loaded clip into the labeling flow exactly as before.
- **Live match** will one day be the primary action; until it works, the working action
  (Judge a clip) keeps the accent and Live sits beside it with a SOON tag.

## New/changed components

### Hero action cards (`.heroCard`) — DESIGN.md §8.15

The two Play actions, replacing the old `label.filebtn` + secondary button stack:
radius 8 (containers, not pills), full width, min-height 72px, padding 14px,
flex row gap 12: 28px icon · column of title 16/700 + desc 13/400.
- **Accent variant** (Judge a clip): `--accent-bg` fill, all ink `--accent-text`;
  it IS the file input (`label.filebtn` recast — keeps hidden `<input type=file>`).
- **Surface variant** (Live match): `--surface` fill, `1px --line` border, `--text`
  title, `--dim` icon/desc, right-aligned `.fcTag` "SOON".
- `:active` = instant darken via `--line` fill (surface) / no change needed (accent
  uses default press). One ≥48px target each (these are primaries).
- Section heading above: 12/600 uppercase `--dim` "PLAY".

### Dev row

Bottom of Play, after a `1px --line` hairline: micro-label "DEV" (11/600 uppercase
`--dim`) + row of `button.small`: "Debug targets" (existing behavior) and "Label mode"
— a toggle showing active state like correction chips (`.active` = accent fill 700).
Replaces the nav pill's Label item; wire to the existing S.mode judge/label logic.

### Nav pill v2

Same liquid-glass container. Items: Play (play-triangle icon) · Matches (filmstrip
icon) · Coach (whistle/clipboard-style §9 icon, ≤2 shapes). Play → setPhase('load');
Matches → setPhase('matches'); Coach → setPhase('coach'). Active state follows the
current section. Visible only on p-load / p-matches / p-coach.

### Placeholder pages (§8.14 pattern, unchanged)

`p-live`, `p-matches`, `p-stats`, `p-shot-bot`, `p-sharing` keep the dashed
"COMING SOON!" hero + "Planned" card with CORE/LATER chips. `p-coach` is a **hub**,
not a placeholder: instruction line + three `.featureCard`s (no hero of its own).

Planned lists (user-facing capabilities only):
| Page | Meta | CORE | LATER |
|---|---|---|---|
| p-live | PHASE 2 | In-app match recording · Point-by-point score log | Contested-call review · Streaming inference |
| p-matches | PHASE 3 | Per-rally clip library | Highlight auto-reels · Dead-time removal · Annotated shot overlays |
| p-stats | PHASE 4 | Shot-type breakdown · Heatmaps + trend stats | AI coach reports |
| p-sharing | PHASE 5 | Accounts + cloud storage | Match sharing · Clip comments + review · Drill assignments |
| p-shot-bot | PHASE 6 | Win-probability model | Shot decision grading · What-if rally replay · Opponent profiles |

STEP_META (no action keys): matches {label:'Matches', instr:'Your matches, cut into
rallies automatically.'}; coach {label:'Coach', instr:'Pick where to improve.'};
live {label:'Live match', instr:'Record a match and get calls as they happen.'};
stats {label:'Stats + trends', instr:'Shot types, patterns, and progress over time.'};
shot_bot {label:'Shot selection', instr:'See the shot the numbers would have played.'};
sharing {label:'Your coach', instr:'Share matches and clips with your coach.'}

Coach hub cards: Stats + trends ("Shot types, patterns, progress" · PHASE 4),
Shot selection ("The shot the numbers would play" · PHASE 6),
Your coach ("Share matches, get drills" · PHASE 5).

## Navigation rules

- Back chevron: hidden on p-load/p-matches/p-coach (section roots); on p-live → load;
  on p-stats/p-shot-bot/p-sharing → coach. Existing flow behavior untouched.
- Stage stays hidden on all new phases.
- Old ids p-clips/p-coach-ai/p-platform/p-flywheel and STEP_META keys clips/coach_ai/
  platform/flywheel are REMOVED (replaced by the ids above; flywheel gone entirely).

## DESIGN.md changes (same commit)

§8.3 nav pill = section tab bar (Play/Matches/Coach, roots only); §8.13 feature cards
now "hub cards" (Coach hub); new §8.15 hero action cards; §16 blueprint rows updated
(remove flywheel row, rename pages, note dev row + back-chevron rule); §3.3 phase list
updated. "COMING SOON!" §14 exception unchanged.

## Out of scope

Real behavior for any placeholder; changes to judge/label flows beyond the mode-toggle
relocation; recent-matches previews on home (comes with the library backend).
