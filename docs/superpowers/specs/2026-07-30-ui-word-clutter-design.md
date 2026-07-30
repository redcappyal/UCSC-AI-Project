# Reducing word clutter in the UI

**Date:** 2026-07-30
**Status:** design approved, ready for implementation planning

## Problem

The app tells the user the same thing several times, in several registers, on the same
screen. Four distinct kinds of clutter, found by touring every root and leaf at a 375 px
viewport:

1. **Every section root titles itself twice.** `#stepLabel` in the header and a `<h2>` in
   the section's `.viewhead` render the same word, ~50 px apart.
2. **Internal vocabulary is on the user's screen.** Run ids, `live pipeline`, detector
   backend names, roadmap phase numbers, a capability inventory shown even when every
   tier ran.
3. **The Dashboard prints the same coach sentences twice**, stacked.
4. **Provenance and confidence hedges are body copy.** The Coaching screen opens with
   three stacked caveat lines before any advice.

The information in (4) is real and load-bearing — §8.20 is explicit that a motion-blob
fallback and real weights must not be presented as equally trustworthy. This design
reduces its *visual weight* without removing it.

## Decisions taken

- Drill prose on the Coaching screen is **out of scope** and unchanged.
- Hedges are **demoted behind a tap**, not deleted or condensed.
- Section roots become **header-title-only**, matching how `p-load` already works.
- The Dashboard hero note is **dropped** for analyzed sessions; Coach Notes owns the prose.

## Approach: inline disclosure, not a popover

A new `.whybtn` control (a small circled-`i`) sits at the end of the heading its
provenance belongs to. Tapping it toggles a `.whynote` block visible directly beneath the
heading; tapping again collapses it.

Rejected: an anchored popover. It needs a new `z-index ≥ 50` tier per the §3.2 ladder,
outside-tap dismissal, and edge-collision handling at 375 px. Inline disclosure is also
the established house pattern — §8.16's delete control arms inline rather than opening a
modal, and §8.20 records that WKWebView renders no JS dialogs inside the iOS shell.

### Component contract

- `.whybtn` — `<button type="button">`, 24 px hit box inside a ≥ 44 px touch target,
  `--dim` stroke, circled-`i` glyph, `aria-expanded` + `aria-controls` pointing at its
  `.whynote`. Label: `aria-label="Why this number"`.
- `.whynote` — hidden by default; when open, the existing `.metaline` type ramp
  (12 / 400–600 / `--dim`), full width, directly under the heading row.
- One `.whybtn` per heading, never per tile. Multiple provenance facts for one section
  concatenate into a single `.whynote`.
- Reduced-motion: the expand is a height/opacity transition gated on
  `prefers-reduced-motion`, per §10.

## Edits

### 1. Titles (`index.html`)

| Site | Change |
|---|---|
| `index.html:885` | `#stepLabel` `<div>` → `<h1>`. Id selector, so styling is unaffected; gives each phase exactly one heading. |
| `index.html:1461` (`p-matches`) | Drop `<h2>Analysis</h2>`; keep `#clipMeta`. |
| `index.html:1473` (`p-match`) | Drop `<h2 id="matchTitle">`; keep `#matchMeta` (duration is new information). Remove the now-dead `matchTitle` write. |
| `index.html:1484` (`p-coach`) | Drop `<h2>Training</h2>`, leaving an empty `.viewhead` — remove the wrapper too. |
| `index.html:1582` (`p-progress`) | Same as `p-coach`. |

`stepLabelFor()` and `matchDetailLabel()` are unchanged — the header already carries the
right string for every one of these phases.

CSS to retire once the last `.viewhead h2` is gone: `index.html:802` and `:852`.

### 2. Dashboard repeats (`index.html`)

- `index.html:8066` — remove `if(note) $('heroNote').textContent = note;`, and hide
  `#heroNote` whenever `LIVE.runs.length` is non-zero.
- `index.html:984` — `#heroNote`'s markup default is the **empty state** and stays, but
  loses its internal vocabulary: "…and the pipeline turns it into…" → wording that does
  not name the pipeline.
- `renderCoachNotes()` is unchanged and becomes the only renderer of coach prose on the
  Dashboard.

### 3. Internal vocabulary

| Site | Before | After |
|---|---|---|
| `index.html:8184` | `${n} runs · live pipeline` | `${n} session(s)` |
| `index.html:8302` | `run <id>` `.metaline` | same text, wrapped in `.devOnly` |
| `index.html:1509, 1516, 1523` | `.fcTag` `Phase 4/5/6` | `Soon` |
| `index.html:8300` | section title `What this clip could measure` | `Not measured` |

**Capability card gating.** The `What this clip could measure` block currently renders one
row per tier with an `ON`/`OFF` chip, always. Change it to render **only when at least one
tier is disabled**, listing only the disabled tiers with their `reason`. §8.20 states the
card exists so that absence is rendered as its reason rather than as an empty result —
with no absence, there is nothing for it to explain, and four `ON` chips are noise. When
every tier ran the section is omitted entirely.

Capability *reasons* stay visible body text. They are not demoted behind a `.whybtn`;
doing so would breach §8.20.

### 4. Hedges behind `.whybtn`

| Screen | Heading carrying the `i` | Text moved into `.whynote` |
|---|---|---|
| `p-match` Rallies | `Rallies` | `index.html:8158` — signal source, plus the disagreement note when `agrees_with_hits === false`. Reword `hit-derived rallies`, which is pipeline jargon. |
| `p-match` Movement | `Movement` | `index.html:8174` — `Detector: <backend> · <n>% of rally time observed` |
| `p-coach-advice` | `#adviceHeadline` ("4 things to work on") | All three of: `#adviceMeta` (`index.html:8062`), `#adviceCaveat` / `pooling_note` (`app.py:1725`), and `player.advice.note` (`coaching_advice.py:446`) |
| Movement heatmap | — | `index.html:5881` splits: the legend sentence ("Where <player> stood during rallies — stronger color means more time") stays visible; the trailing `Detector: <backend>.` moves into the section's `.whynote`. |

On `p-coach-advice` the `#adviceProvenance` card is kept as the container: it holds
`#adviceHeadline` with its `.whybtn`, and the collapsed `.whynote` replaces the
`#adviceMeta` / `#adviceCaveat` paragraph pair. Collapsed, the card is one line.

The pooling caveat carries a code comment marking it load-bearing ("attribution is
per-clip, so 'Player 1' is a slot, not a person"). Its `.whybtn` sits on the advice
headline, immediately under the Player 1 / Player 2 segment it qualifies — the point of
use, not a footer.

`index.html:5854` (`<n>% of rally time observed`) is a separate call site from `:8174`;
both feed the same `.whynote` copy and change together.

### 5. Backend strings

- `app.py:1725–1731` — `pooling_note` keeps its meaning but is rewritten for a disclosure
  panel rather than a paragraph: shorter, no "Pooled across your last N sessions"
  preamble restating what `#adviceMeta` already says.
- `coaching_advice.py:446–449` — the `total < 20` note merges into the same disclosure
  text instead of rendering as a standalone `.adviceEmpty` line. The `not items` branch
  ("No clear weakness stood out…") is a **result**, not a hedge, and stays as body copy.

Both are consumed by the iOS webviews as well as the web UI; neither has a native-side
copy to keep in sync.

### 6. DESIGN.md

Updated in the same commit, per the CLAUDE.md no-silent-drift rule:

- **§8 (new subsection)** — `.whybtn` / `.whynote` component contract.
- **§8.20** — run-card and Progress blueprints: metalines that moved behind the `i`; the
  capability card's new "only when a tier was gated off" rule; `.devOnly` run id.
- **§8.13** — `.fcTag` wording.
- **§14** — a voice rule: provenance and confidence belong behind a disclosure, not in
  body copy; the app states the fact and offers the caveat.
- **§16** — blueprint rows for `matches`, `match`, `coach`, `progress`: header carries the
  title, sections carry no `<h2>`.

## Out of scope

- Coaching drill prose (`coaching_advice.py` progression text) — unchanged.
- Capability reasons for tiers that did not run — stay as visible body copy.
- The `p-load` Dev row, calibration wizard copy, and error-banner strings.
- Any change to which metrics are shown; the 2026-07-29 metric review stands.

## Verification

- `/verify` recipe: launch, drive at 375 px, check **both themes** per CLAUDE.md.
- Screens to re-shoot: Dashboard, Analysis list, match detail, Training, Coaching,
  Progress.
- Both disclosure states (collapsed and expanded) on every screen that gains a `.whybtn`.
- Capability card in both branches: a run with every tier `ON` (section absent) and a run
  with a tier gated off (section present, reason shown).
- `.venv/bin/python -m pytest tests/ -q` — must stay green, with the one skip and one
  deselect CLAUDE.md documents as expected. Record the pass count from the pre-change run
  and match it; no test should be added or lost by a copy change.
- No pipeline numbers change, so `/eval` is not required: no edit touches `judge_call.py`,
  calibration, impact estimation, or detection.

## Accessibility (§15)

- `.whybtn` has an `aria-label`, `aria-expanded`, and `aria-controls`; its glyph is
  `aria-hidden`.
- Touch target ≥ 44 px despite the 24 px visual.
- `#stepLabel` as `<h1>` gives every phase exactly one heading, which the removed `<h2>`s
  previously duplicated.
- Expansion respects `prefers-reduced-motion`.
- Expanding a `.whynote` must not shift the content below it in a way that loses the
  user's scroll position — the disclosure opens below its heading, above the tiles.
