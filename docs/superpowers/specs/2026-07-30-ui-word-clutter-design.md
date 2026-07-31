# Reducing word clutter in the UI

**Date:** 2026-07-30
**Status:** design approved, ready for implementation planning

## Problem

The app tells the user the same thing several times, in several registers, on the same
screen. Four distinct kinds of clutter, found by touring every root and leaf at a 375 px
viewport:

1. **Every section root titles itself twice.** `#stepLabel` in the header and a `<h2>` in
   the section's `.viewhead` render the same word, ~50 px apart. On `p-stats` the two
   copies do not even agree: the header says "Stats + trends" and the page says
   "Your stats".
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

Sites are given as **content anchors, not line numbers** — `index.html` moved ~30 lines
mid-design when `bf15df4` landed, and will move again.

| Anchor | Change |
|---|---|
| `<div id="stepLabel">` | → `<h1 id="stepLabel">`. Id selector, so styling is unaffected; gives each phase exactly one heading. |
| `p-matches` `<h2>Analysis</h2>` | Drop; keep the `#clipMeta` span. |
| `p-match` `<h2 id="matchTitle">Match</h2>` | Drop; keep `#matchMeta` (duration is new information). Remove the now-dead `matchTitle` write. |
| `p-coach` `<div class="viewhead"><h2>Training</h2></div>` | Drop the whole wrapper — nothing else is in it. |
| `p-progress` `<div class="viewhead"><h2>Progress</h2></div>` | Same. |
| `p-stats` `<h2>Your stats</h2>` | Drop; keep the `#trainingStatsMeta` span. |

**`p-stats` is a name conflict, not just a duplicate.** `bf15df4` turned it from a roadmap
placeholder into a live page titled "Your stats", but left `STEP_META.stats.label` as
`'Stats + trends'`. The header and the page therefore call the same screen two different
names, stacked. Fix `STEP_META.stats.label` to `'Your stats'` so the surviving header
title is the right one. The Training hub card that opens it says "Your stats" too, so
three surfaces agree afterwards.

`stepLabelFor()` and `matchDetailLabel()` are otherwise unchanged — the header already
carries the right string for every one of these phases.

CSS to retire once the last `.viewhead h2` is gone: the `#p-matches .viewhead h2` and
`.viewhead h2` rules. `.viewhead` itself stays — `p-matches`, `p-match` and `p-stats`
still use it to lay out their meta span.

### 2. Dashboard repeats (`index.html`)

- In `renderDashboardLive()` — remove `if(note) $('heroNote').textContent = note;`, and
  hide `#heroNote` whenever `LIVE.runs.length` is non-zero.
- `#heroNote`'s markup default is the **empty state** and stays, but loses its internal
  vocabulary: "…and the pipeline turns it into calls, stats, and coaching" → wording that
  does not name the pipeline.
- `renderCoachNotes()` is unchanged and becomes the only renderer of coach prose on the
  Dashboard.

### 3. Internal vocabulary

| Anchor | Before | After |
|---|---|---|
| `$('clipMeta').textContent` | `${n} runs · live pipeline` | `${n} session(s)` |
| run-id `.metaline` in the match report | `run <id>` | same text, wrapped in `.devOnly` |
| `.fcTag` on the Training hub | `Phase 5`, `Phase 6` | `Soon` |
| capability card title | `What this clip could measure` | `Not measured` |
| `p-stats` feature-card subtitle | `Your shots and patterns across identified sessions` | shortened until it stops ellipsizing at 375 px |

The `.fcTag` change is now only two cards: `bf15df4` already replaced the Phase 4 tag with
`Live` when it shipped "Your stats", and a `Soon` tag exists elsewhere in the file, so this
follows precedent rather than inventing a word.

The `p-stats` subtitle currently truncates mid-word on the Training hub
("Your shots and patterns across ident…"), which is word clutter causing real information
loss — the one case in this pass where cutting words *adds* information.

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
| `p-match` Rallies | `Rallies` | The `rallySection` `.metaline` — signal source, plus the disagreement note when `agrees_with_hits === false`. Reword `hit-derived rallies`, which is pipeline jargon. |
| `p-match` Movement | `Movement` | The movement `.metaline` — `Detector: <backend> · <n>% of rally time observed` |
| `p-coach-advice` | `#adviceHeadline` ("4 things to work on") | All three of: `#adviceMeta`, `#adviceCaveat` (fed by `me_pooling_note`), and `player.advice.note` from `coaching_advice.py` |
| Movement heatmap | — | The `renderPlayerMovement` note splits: the legend sentence ("Where <player> stood during rallies — stronger color means more time") stays visible; the trailing `Detector: <backend>.` moves into the section's `.whynote`. |

**`#adviceHeadline` has four states**, and the `.whybtn` belongs to only one of them.
`renderAdvice` sets it to `'Coaching unavailable'` (error), `'Working on your coaching'`
(loading), `'Tell us which player is you'` (no identified session), and only then
`'N things to work on'`. The button renders in the last state alone; the other three have
no provenance to offer and must not grow a dead control.

On `p-coach-advice` the `#adviceProvenance` card is kept as the container: it holds
`#adviceHeadline` with its `.whybtn`, and the collapsed `.whynote` replaces the
`#adviceMeta` / `#adviceCaveat` paragraph pair. Collapsed, the card is one line.

The pooling caveat is load-bearing — a code comment in `app.py` marks it so. Its `.whybtn`
sits on the advice headline, immediately under the Player 1 / Player 2 segment it
qualifies — the point of use, not a footer.

There are **two** call sites rendering `<n>% of rally time observed` (one in
`renderPlayerMovement`, one in the match report's movement section). Both feed the same
`.whynote` copy and change together.

### 5. Backend strings

- `coaching_advice.py` — the `total < 20` note ("Based on N front-wall shots, so treat
  this as a pointer rather than a verdict") merges into the disclosure text instead of
  rendering as a standalone `.adviceEmpty` line. The `not items` branch ("No clear
  weakness stood out…") is a **result**, not a hedge, and stays as body copy.
- `app.py` `me_pooling_note` — already short after `bf15df4` ("Built from N identified
  sessions. Only matches where you selected your player are included."). It moves behind
  the disclosure unchanged; no rewrite needed.

**`app.py`'s older `pooling_note` field is left alone.** `bf15df4` switched the UI to
`me_pooling_note`, so nothing renders `pooling_note` any more, but it is still returned by
`/api/coach/advice` and pinned by `tests/test_pooled_coach_advice.py` (`assert "served
first" in body["pooling_note"]`). Removing it is a separate decision about a public API
field, not part of a copy pass.

These are consumed by the iOS webviews as well as the web UI; neither has a native-side
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
