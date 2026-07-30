# UI Word-Clutter Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut duplicated titles, internal vocabulary, and repeated prose out of the web UI, and move provenance hedges behind an inline tap disclosure instead of rendering them as body copy.

**Architecture:** All UI lives in the single-file `index.html` (inline HTML/CSS/JS). A new `.whybtn` / `.whynote` disclosure pair is added once and reused by four call sites. Two Python files supply server copy consumed by that disclosure. Every change is copy or markup — no pipeline code is touched.

**Tech Stack:** Single-file HTML/CSS/vanilla JS front end; Flask (`app.py`) + `coaching_advice.py` for server copy; pytest text-contract tests that read `index.html` as a string.

**Spec:** `docs/superpowers/specs/2026-07-30-ui-word-clutter-design.md`

## Global Constraints

- **Read `DESIGN.md` before touching any UI.** It is the source of truth for tokens, components, and the "never do" list. Any change that deviates from it must update it **in the same commit** — never drift silently (CLAUDE.md).
- **Voice (DESIGN.md §14):** referee's voice — calm, terse, factual. Buttons verb-first, ≤ 3 words. No exclamation marks, no praise, no anthropomorphism. Domain terms exactly: *out line, tin, service line, front/side wall, floor, rally, bounce*. The single sanctioned exclamation is the "Coming soon!" placeholder hero.
- **Escape server text at every innerHTML sink** (DESIGN.md §8.20). Coach feedback, capability reasons, and detector names are server-supplied.
- **Accessibility (DESIGN.md §15):** text ≥ 4.5:1 and UI shapes ≥ 3:1 **in both themes**; icon-only buttons carry `aria-label`; decorative SVG carries `aria-hidden="true"`; touch targets ≥ 44 px.
- **Absence stays visible (DESIGN.md §8.20):** a tier that could not run must still render its `reason` as body text. Only provenance *about a number that is shown* may move behind the disclosure.
- **Sites are content anchors, not line numbers.** `index.html` moved ~30 lines mid-design when `bf15df4` landed. Locate every edit by searching for the quoted string.
- **Environment:** `.venv/bin/python` for everything. System `python3` has no flask or cv2.
- **Test command:** `.venv/bin/python -m pytest tests/ -q`. A green run has exactly one skip and one deselect (both expected — see CLAUDE.md). Record the pass count before you start and match it at the end.
- **Editing a `*.py` with a paired `tests/test_*.py` auto-runs that file via a PostToolUse hook.** A failure comes back as a *blocked edit*, not a warning.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `index.html` | All markup, CSS, and render JS | 1–6 |
| `app.py` | `/api/coach/advice` response copy | 6 |
| `coaching_advice.py` | Low-sample note text | 6 |
| `DESIGN.md` | Component contract, blueprints, voice rule | 1–6 |
| `tests/test_word_clutter_ui_contract.py` | **New.** Contract tests for this pass | 1–6 |
| `tests/test_analysis_typography_contract.py` | **Modify.** Pins a CSS rule Task 2 deletes | 2 |
| `tests/test_coach_hero_ui_contract.py` | **Modify.** Pins the advice-caveat sink Task 6 changes | 6 |

---

### Task 1: The `.whybtn` / `.whynote` disclosure component

Builds the component every later task consumes. Nothing user-visible changes yet.

**Files:**
- Modify: `index.html` (CSS block, JS helpers near `escapeHtml`)
- Modify: `DESIGN.md` (new §8.23)
- Test: `tests/test_word_clutter_ui_contract.py` (create)

**Interfaces:**
- Consumes: `escapeHtml(value)` — existing, returns an HTML-escaped string.
- Produces:
  - `whyDisclosure(text)` → `{btn: string, note: string}`. Returns `{btn:'', note:''}` for falsy `text`. Escapes `text` internally; callers pass **plain strings, not markup**.
  - `sectionHead(label, why)` → `string`. Renders a `.cardtitle.whyhead` row with `label`, an optional `.whybtn`, and the `.whynote` immediately after. `label` is a trusted literal and is **not** escaped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_word_clutter_ui_contract.py`:

```python
"""Contract tests for the 2026-07-30 word-clutter pass.

The UI is one file of inline markup, so these read index.html as text. They
exist to stop the cut copy creeping back in: every assertion here is a line
someone deliberately removed, or a component that replaced one.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")
DESIGN_MD = (ROOT / "DESIGN.md").read_text(encoding="utf-8")


def test_the_disclosure_builder_escapes_server_text():
    # Detector names and capability reasons reach this builder from the
    # server, so it must escape rather than trusting its caller.
    assert "function whyDisclosure(text){" in INDEX_HTML
    assert "escapeHtml(text)" in INDEX_HTML


def test_the_disclosure_starts_collapsed_and_is_announced():
    assert 'class="whybtn"' in INDEX_HTML
    assert 'aria-expanded="false"' in INDEX_HTML
    assert 'aria-controls=' in INDEX_HTML
    assert 'aria-label="Why this number"' in INDEX_HTML
    assert 'class="whynote hidden"' in INDEX_HTML


def test_the_toggle_is_delegated_because_reports_rebuild_their_markup():
    # Every .whybtn is rendered into an innerHTML sink that is replaced on
    # each report load, so a per-button listener would be dropped.
    assert "e.target.closest('.whybtn')" in INDEX_HTML


def test_design_md_documents_the_disclosure():
    assert "### 8.23 Provenance disclosure" in DESIGN_MD
    assert ".whybtn" in DESIGN_MD
    assert ".whynote" in DESIGN_MD
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 4 failures, all `AssertionError` — nothing named `whyDisclosure` exists yet.

- [ ] **Step 3: Add the CSS**

In `index.html`, immediately after the `.metaline` rules (search for `.metaline{display:flex`):

```css
  /* Provenance disclosure (§8.23): the number is the headline, the caveat is
     one tap away. Plain show/hide — .hidden is display:none, so there is no
     height to animate and nothing for prefers-reduced-motion to suppress. */
  .whyhead{display:flex; align-items:center; gap:2px}
  .whybtn{flex:0 0 auto; width:44px; height:44px; margin:-14px 0 -14px -4px;
    display:inline-flex; align-items:center; justify-content:center;
    background:none; border:0; padding:0; color:var(--dim); cursor:pointer}
  .whybtn svg{width:15px; height:15px}
  .whybtn[aria-expanded="true"]{color:var(--text)}
  .whynote{font-size:12px; color:var(--dim); line-height:1.45; margin:0 0 2px}
```

The negative vertical margin keeps the 44 px touch target from inflating the
heading row. Verify visually in Step 8 — the heading must sit at the same
height it does today.

- [ ] **Step 4: Add the builders**

In `index.html`, directly after `function escapeHtml(value){…}`:

```js
/* Provenance disclosure (DESIGN.md §8.23). Provenance about a number that IS
   shown goes behind the tap; the reason a tier could NOT run stays as body
   text (§8.20) — do not route capability reasons through here. */
let WHY_SEQ = 0;
function whyDisclosure(text){
  if(!text) return {btn:'', note:''};
  const id = `why${++WHY_SEQ}`;
  return {
    btn: `<button type="button" class="whybtn" aria-label="Why this number"` +
         ` aria-expanded="false" aria-controls="${id}">` +
         `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">` +
         `<circle cx="12" cy="12" r="9"/><path d="M12 11v5.2"/>` +
         `<circle cx="12" cy="7.6" r=".95" fill="currentColor" stroke="none"/></svg></button>`,
    note: `<div class="whynote hidden" id="${id}">${escapeHtml(text)}</div>`,
  };
}
/* Section heading + its optional disclosure. `label` is a trusted literal. */
function sectionHead(label, why){
  const d = whyDisclosure(why);
  return `<div class="cardtitle whyhead" style="font-size:14px">${label}${d.btn}</div>${d.note}`;
}
document.addEventListener('click', (e) => {
  const btn = e.target.closest && e.target.closest('.whybtn');
  if(!btn) return;
  const note = document.getElementById(btn.getAttribute('aria-controls'));
  if(!note) return;
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  note.classList.toggle('hidden', open);
});
```

- [ ] **Step 5: Document it in DESIGN.md**

Insert a new `### 8.23 Provenance disclosure (`.whybtn` / `.whynote`)` subsection after §8.22:

```markdown
### 8.23 Provenance disclosure (`.whybtn` / `.whynote`)

Provenance and confidence about **a number that is shown** ride behind a tap, not
in body copy. A `.whybtn` — 15 px circled-`i`, `--dim`, inside a 44 px target
whose negative margins keep the heading row its original height — sits at the end
of the heading the caveat belongs to. Tapping toggles a `.whynote` (12 / `--dim`)
directly beneath. Built by `whyDisclosure(text)` / `sectionHead(label, why)`.

- Collapsed by default. `aria-expanded` + `aria-controls`; glyph `aria-hidden`.
  `aria-label="Why this number"`.
- Plain show/hide via `.hidden`. No animation, so §10's reduced-motion rule has
  nothing to suppress.
- `whyDisclosure` escapes its own text — callers pass plain strings.
- The click handler is **delegated** from `document`, because every call site
  renders into an innerHTML sink that is rebuilt on each report load.
- **One button per heading, never per tile.** Several facts about one section
  concatenate into a single note.
- **Never route a capability `reason` through this.** §8.20 requires that the
  reason a tier could not run stay visible; hiding it re-creates the "we looked
  and found nothing" reading that card exists to prevent.
```

- [ ] **Step 6: Run the test**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 4 passed.

- [ ] **Step 7: Run the whole suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: green, one skip and one deselect. Record the pass count.

- [ ] **Step 8: Verify nothing moved on screen**

Start the app and confirm the component is inert so far (no `.whybtn` has a call
site yet), and that no layout shifted:

```bash
.venv/bin/python app.py
```

Open `http://127.0.0.1:5188` at a 375 px viewport. The match detail page must look
byte-for-byte as before. Check the browser console is free of new errors.

- [ ] **Step 9: Commit**

```bash
git add index.html DESIGN.md tests/test_word_clutter_ui_contract.py
git commit -m "Add the .whybtn/.whynote provenance disclosure"
```

---

### Task 2: One title per screen

**Files:**
- Modify: `index.html` (five `.viewhead` blocks, `#stepLabel`, `STEP_META.stats`, two CSS rules)
- Modify: `DESIGN.md` (§16 blueprint rows)
- Modify: `tests/test_analysis_typography_contract.py`
- Test: `tests/test_word_clutter_ui_contract.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_word_clutter_ui_contract.py`:

```python
def test_section_roots_do_not_repeat_the_header_title():
    # #stepLabel already names every one of these screens; the <h2> under it
    # said the same word ~50px lower.
    assert "<h2>Analysis</h2>" not in INDEX_HTML
    assert '<div class="viewhead"><h2>Training</h2></div>' not in INDEX_HTML
    assert '<div class="viewhead"><h2>Progress</h2></div>' not in INDEX_HTML
    assert '<h2 id="matchTitle">Match</h2>' not in INDEX_HTML
    assert "<h2>Your stats</h2>" not in INDEX_HTML


def test_the_header_label_is_the_pages_one_heading():
    assert '<h1 id="stepLabel">' in INDEX_HTML
    assert '<div id="stepLabel">' not in INDEX_HTML


def test_the_stats_screen_has_exactly_one_name():
    # bf15df4 made p-stats a live page titled "Your stats" but left the header
    # label as "Stats + trends", so the two chrome layers disagreed.
    assert "stats:   {label:'Your stats', instr:''}," in INDEX_HTML
    assert "Stats + trends" not in INDEX_HTML


def test_the_meta_spans_survive_because_they_carry_new_information():
    assert 'id="clipMeta"' in INDEX_HTML
    assert 'id="matchMeta"' in INDEX_HTML
    assert 'id="trainingStatsMeta"' in INDEX_HTML
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 3 new failures (`test_the_meta_spans_survive…` already passes).

- [ ] **Step 3: Drop the duplicate headings**

Four edits in `index.html`.

`p-matches` — delete the `<h2>` line only:

```html
    <div class="viewhead">
      <span class="dim" id="clipMeta"></span>
    </div>
```

`p-match` — delete the `<h2>` line only:

```html
    <div class="viewhead">
      <span class="dim" id="matchMeta"></span>
    </div>
```

`p-stats` — delete the `<h2>` line only:

```html
    <div class="viewhead">
      <span class="dim" id="trainingStatsMeta"></span>
    </div>
```

`p-coach` and `p-progress` hold nothing but the heading, so delete the whole
wrapper. Remove this line from each (it is identical apart from the word):

```html
    <div class="viewhead"><h2>Training</h2></div>
```

```html
    <div class="viewhead"><h2>Progress</h2></div>
```

- [ ] **Step 4: Remove the dead `matchTitle` write**

`buildMatchView()` still assigns to the element Step 3 deleted. Delete this line:

```js
  $('matchTitle').textContent = r ? fmtRunDate(r.created) : 'Match';
```

The header already shows the same date through `stepLabelFor('match')` →
`matchDetailLabel()`, so nothing replaces it.

- [ ] **Step 5: Promote the header label to the page's heading**

```html
  <h1 id="stepLabel">Squash Line Calling</h1>
```

`#stepLabel` is styled by id, so the existing 17/700 rule still applies. Confirm
there is no global `h1` rule that would override it — `index.html` currently
styles only `h2`.

- [ ] **Step 6: Fix the stats label**

In `STEP_META`:

```js
  stats:   {label:'Your stats', instr:''},
```

- [ ] **Step 7: Retire the orphaned CSS**

Delete both rules — no `.viewhead h2` survives:

```css
  #p-matches .viewhead h2{font-size:26px}
```

```css
  .viewhead h2{text-align:left}
```

Keep `.viewhead` itself: `p-matches`, `p-match` and `p-stats` still use it to lay
out their meta span.

- [ ] **Step 8: Update the test that pinned the deleted rule**

In `tests/test_analysis_typography_contract.py`, delete this assertion:

```python
    assert "#p-matches .viewhead h2{font-size:26px}" in INDEX_HTML
```

Leave the other five assertions in that test untouched — `.viewhead .dim`,
`.cliptop .name`, `.cliptop .metaline`, `.statechip` and `.emptycard` all survive.

- [ ] **Step 9: Update DESIGN.md §16**

In the screen-blueprint table, amend the `matches`, `match`, `coach`, `progress`
and `stats` rows so the Body column no longer begins with a `.viewhead h2`, and
add this sentence under the table:

```markdown
Section roots and their leaves carry **no in-page `<h2>`** — `#stepLabel` (an
`<h1>`) is the screen's only title, as `p-load` has always done. A `.viewhead`
survives only where it lays out a dim meta span carrying something the header
does not say (run count, clip duration).
```

- [ ] **Step 10: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py tests/test_analysis_typography_contract.py -q
```

Expected: all pass.

- [ ] **Step 11: Verify in the browser, both themes**

Launch, then at 375 px walk Dashboard → Analysis → a match → Training → Your
stats → Progress. Each screen shows its name exactly once, in the header. Flip
the theme with the in-app sun/moon toggle (emulating `prefers-color-scheme` after
load does nothing — the theme is pinned from `localStorage('slc-theme')` before
first paint) and walk them again.

- [ ] **Step 12: Commit**

```bash
git add index.html DESIGN.md tests/
git commit -m "Show each screen's title once, in the header"
```

---

### Task 3: Stop the Dashboard saying it twice

`renderDashboardLive()` writes the first two sentences of the coach feedback into
`#heroNote`; `renderCoachNotes()` then renders the first three sentences of the
same string directly below.

**Files:**
- Modify: `index.html` (`renderDashboardLive`, `#heroNote` markup)
- Modify: `DESIGN.md` (§8.15)
- Test: `tests/test_word_clutter_ui_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_hero_does_not_repeat_the_coach_notes_below_it():
    # renderCoachNotes renders the same sentences from the same string.
    assert "if(note) $('heroNote').textContent = note;" not in INDEX_HTML
    assert "$('heroNote').classList.toggle('hidden', true);" in INDEX_HTML


def test_the_hero_note_survives_as_the_empty_state():
    # With no runs it is the only thing on the page that says what to do.
    assert 'id="heroNote"' in INDEX_HTML
    assert "the pipeline turns it into" not in INDEX_HTML
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 2 new failures.

- [ ] **Step 3: Hide the hero note once runs exist**

In `renderDashboardLive()`, replace:

```js
  const note = splitSentences((LIVE.coach[latest.run_id]||{}).feedback).slice(0,2).join(' ');
  if(note) $('heroNote').textContent = note;
```

with:

```js
  // The prose belongs to Coach Notes below, which renders the same sentences
  // from the same string. The hero keeps the ring and its two tiles.
  $('heroNote').classList.toggle('hidden', true);
```

`splitSentences` is still used by `renderCoachNotes()` — leave it alone.

- [ ] **Step 4: Drop the pipeline vocabulary from the empty state**

```html
      <div class="heronote" id="heroNote">Record a rally on court, then upload it here for calls, stats, and coaching.</div>
```

- [ ] **Step 5: Update DESIGN.md §8.15**

Amend the dashboard blueprint so the hero note is described as the empty state
only, and add:

```markdown
The hero's `#heroNote` is the **empty state and nothing else**. Coach Notes owns
coach prose on this page; both once rendered the same sentences from the same
feedback string, stacked.
```

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: all pass.

- [ ] **Step 7: Verify both states in the browser**

With analyzed runs present, the Dashboard hero shows the ring, Sessions and
Unforced errors — no paragraph — and Coach Notes below is unchanged. Then check
the empty state by opening the app with no runs (or temporarily returning `[]`
from the runs fetch in the console) and confirm the note reappears with the new
wording. Check both themes.

- [ ] **Step 8: Commit**

```bash
git add index.html DESIGN.md tests/test_word_clutter_ui_contract.py
git commit -m "Let Coach Notes own the Dashboard prose"
```

---

### Task 4: Internal vocabulary off the user's screen

**Files:**
- Modify: `index.html` (`clipMeta`, run-id metaline, two `.fcTag`s, `p-stats` card subtitle)
- Modify: `DESIGN.md` (§8.13, §8.20)
- Test: `tests/test_word_clutter_ui_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_analysis_list_counts_sessions_not_pipeline_runs():
    assert "runs · live pipeline" not in INDEX_HTML
    assert "session${LIVE.runs.length===1?'':'s'}`" in INDEX_HTML


def test_the_run_id_is_dev_only():
    # A 13-digit epoch id is for whoever is sitting at the Mac.
    assert '<div class="metaline devOnly">run ${escapeHtml(id)}</div>' in INDEX_HTML


def test_roadmap_cards_do_not_expose_internal_phase_numbers():
    assert "Phase 5" not in INDEX_HTML
    assert "Phase 6" not in INDEX_HTML


def test_the_stats_card_subtitle_fits_on_one_line():
    # "Your shots and patterns across identified matches" truncated mid-word
    # at 375px, which loses information rather than saving space.
    assert "Your shots and patterns across identified matches" not in INDEX_HTML
    assert "<strong>Your stats</strong><span>Shots and patterns</span>" in INDEX_HTML
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 4 new failures.

- [ ] **Step 3: Count sessions, not runs**

Replace:

```js
  $('clipMeta').textContent = `${LIVE.runs.length} runs · live pipeline`;
```

with:

```js
  $('clipMeta').textContent = `${LIVE.runs.length} session${LIVE.runs.length===1?'':'s'}`;
```

This also fixes "1 runs", which the old template produced for a single session.

- [ ] **Step 4: Make the run id dev-only**

In the match-report template, replace:

```js
    <div class="metaline">run ${escapeHtml(id)}</div>
```

with:

```js
    <div class="metaline devOnly">run ${escapeHtml(id)}</div>
```

`body.shell-embed .devOnly{display:none !important}` already exists, so this also
hides it inside the iOS shell.

- [ ] **Step 5: Retire the phase numbers**

Both Training-hub cards, `sharing` and `shot_bot`:

```html
      <span class="fcTag">Soon</span>
```

`Soon` is already in use on the Live match card, so this follows the existing
vocabulary rather than inventing a word.

- [ ] **Step 6: Shorten the stats card subtitle**

```html
      <span class="fcBody"><strong>Your stats</strong><span>Shots and patterns</span></span>
```

- [ ] **Step 7: Update DESIGN.md**

In §8.13 (feature cards), record that `.fcTag` carries a readiness word — `Live`
or `Soon` — never an internal phase number. In §8.20, note that the run id
metaline is `.devOnly`, and that the Analysis meta span counts sessions.

- [ ] **Step 8: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: all pass.

- [ ] **Step 9: Verify in the browser**

Analysis header reads "2 sessions". The match page shows no run id (it is
`.devOnly`, so it stays visible in a plain browser — confirm it disappears with
`?shell=1`). Training shows `Soon` on the last two cards and the "Your stats"
subtitle no longer ellipsizes at 375 px. Both themes.

- [ ] **Step 10: Commit**

```bash
git add index.html DESIGN.md tests/test_word_clutter_ui_contract.py
git commit -m "Keep pipeline vocabulary off the user's screens"
```

---

### Task 5: Show the capability card only when something was gated off

Four `ON` chips under "What this clip could measure" are noise. §8.20 says the
card exists so absence is rendered as its reason — with no absence it has no job.

**Files:**
- Modify: `index.html` (`capabilityRows`, the match-report template)
- Modify: `DESIGN.md` (§8.20)
- Test: `tests/test_word_clutter_ui_contract.py`

**Interfaces:**
- Consumes: `TIER_LABELS` — existing map of tier key → display label.
- Produces: `capabilityRows(rep)` returns `''` when every tier ran and the run is not legacy.

- [ ] **Step 1: Write the failing test**

```python
def test_the_capability_card_is_titled_for_what_it_reports():
    assert "What this clip could measure" not in INDEX_HTML
    assert "Not measured" in INDEX_HTML


def test_the_capability_card_lists_only_the_tiers_that_did_not_run():
    assert "Object.keys(TIER_LABELS).filter(k => caps[k] && !caps[k].enabled)" in INDEX_HTML


def test_the_capability_card_vanishes_when_everything_ran():
    # Its whole purpose is explaining absence; with none there is nothing to say.
    assert "const capRows = capabilityRows(rep);" in INDEX_HTML
    assert "capRows ? `<div class=\"cardtitle\" style=\"font-size:14px\">Not measured</div>${capRows}` : ''" in INDEX_HTML


def test_capability_reasons_stay_visible_body_text():
    # They must never move behind the §8.23 disclosure.
    assert 'escapeHtml(t.reason||\'\')' in INDEX_HTML
    assert "whyDisclosure(t.reason" not in INDEX_HTML
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 3 new failures (`test_capability_reasons_stay_visible_body_text` already passes).

- [ ] **Step 3: Filter to the disabled tiers**

Replace the body of `capabilityRows` after the legacy branch:

```js
  const caps = rep.capabilities || {};
  return Object.keys(TIER_LABELS).filter(k => caps[k] && !caps[k].enabled).map(k => {
    const t = caps[k];
    const why = `<span class="s" style="color:var(--dim)">${escapeHtml(t.reason||'')}</span>`;
    return `<div class="fbrow">
      <div class="info"><span class="s" style="color:var(--text)">${TIER_LABELS[k]}</span>${why}</div>
      <span class="statechip">OFF</span>
    </div>`;
  }).join('');
```

The `t.enabled` ternaries go with it — every surviving row is a disabled one.
The reason stays as visible body text, per §8.20.

- [ ] **Step 4: Drop the heading when there are no rows**

In the match-report template, replace:

```js
    <div class="cardtitle" style="font-size:14px">What this clip could measure</div>
    ${capabilityRows(rep)}
```

with a reference to a value computed just above the `return`:

```js
  const capRows = capabilityRows(rep);
```

```js
    ${capRows ? `<div class="cardtitle" style="font-size:14px">Not measured</div>${capRows}` : ''}
```

A legacy run still returns its `legacy_reason` metaline from the first branch of
`capabilityRows`, so it keeps the heading — which is correct: a legacy run is
exactly the case where the reader needs telling.

- [ ] **Step 5: Update DESIGN.md §8.20**

Replace the `"What this clip could measure"` bullet with:

```markdown
  **"Not measured"** — rendered **only when a tier was gated off** (or the run is
  legacy). One `.fbrow` per disabled tier, label left, `OFF` `.statechip` right,
  and the `reason` as dim `.s` text under the label. When every tier ran the whole
  section is omitted: the card exists so absence is rendered as its reason, and
  four `ON` chips assert nothing a reader needs. The reason stays **visible body
  text** — never behind the §8.23 disclosure ·
```

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: all pass.

- [ ] **Step 7: Verify both branches in the browser**

This needs two runs. Open a match whose tiers all ran — the "Not measured"
section must be absent entirely, and "Open full review" must follow the ball
tier directly with no gap. Then open one with a tier gated off (or temporarily
flip `enabled` to `false` for one tier in the `/api/runs/<id>/report` response via
the console) and confirm the heading returns with that tier's reason readable.
Both themes.

- [ ] **Step 8: Commit**

```bash
git add index.html DESIGN.md tests/test_word_clutter_ui_contract.py
git commit -m "Show the capability card only when a tier was gated off"
```

---

### Task 6: Provenance behind the tap

The last task, and the one that consumes Task 1.

**Files:**
- Modify: `index.html` (`rallySection`, `movementSection`, the match-report template, `renderPlayerMovement`, `renderAdvice`, `#adviceProvenance` markup)
- Modify: `app.py` (docstring only)
- Modify: `coaching_advice.py` (low-sample note)
- Modify: `DESIGN.md` (§8.20, §14)
- Modify: `tests/test_coach_hero_ui_contract.py`
- Test: `tests/test_word_clutter_ui_contract.py`

**Interfaces:**
- Consumes: `whyDisclosure(text)` and `sectionHead(label, why)` from Task 1.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
def test_the_rally_and_movement_provenance_moved_behind_the_tap():
    assert '<div class="metaline">From impact sounds and frame motion' not in INDEX_HTML
    assert '<div class="metaline">Detector: ' not in INDEX_HTML
    assert "sectionHead('Rallies', " in INDEX_HTML
    assert "sectionHead('Movement', " in INDEX_HTML


def test_the_rally_provenance_drops_the_pipeline_jargon():
    assert "hit-derived rallies" not in INDEX_HTML
    assert "disagrees with the rallies counted from ball contacts" in INDEX_HTML


def test_the_heatmap_keeps_its_legend_and_hides_only_the_detector():
    assert "stronger color means more time" in INDEX_HTML
    assert "stronger color means more time. Detector:" not in INDEX_HTML


def test_the_coaching_screen_collapses_its_three_caveats_into_one():
    assert 'id="adviceMeta"' not in INDEX_HTML
    assert 'id="adviceCaveat"' not in INDEX_HTML
    assert 'id="adviceWhy"' in INDEX_HTML


def test_the_coaching_disclosure_only_exists_in_the_loaded_state():
    # renderAdvice also has error / loading / no-identified-player states,
    # none of which have provenance to offer.
    assert "$('adviceWhy').innerHTML = '';" in INDEX_HTML


def test_the_low_sample_note_is_provenance_not_body_copy():
    advice = (ROOT / "coaching_advice.py").read_text(encoding="utf-8")
    assert "treat this as a pointer" not in advice
    assert "low_sample_note" in advice
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py -q
```

Expected: 6 new failures.

- [ ] **Step 3: Move the rally provenance**

In `rallySection`, replace the trailing metaline. The function returns the tiles
only; the caller supplies the heading. Change the `return` to:

```js
  const why = (tl.audio_available ? 'From impact sounds and frame motion' : 'From frame motion only — no audio track') +
    (tl.agrees_with_hits === false ? ' · disagrees with the rallies counted from ball contacts' : '');
  return {why, html: `<div class="scorecols" style="grid-template-columns:1fr">
      <div class="scol"><span class="sl">Longest rally${where}</span><span class="sv num">${longest ? longest.dur.toFixed(1) : '—'}<span class="ss">s</span></span></div>
    </div>`};
```

Both early returns in that function must match the new shape:

```js
  if(!tl) return {why:'', html:'<div class="metaline">Rally structure was not measured for this run.</div>'};
```

```js
    return {why:'', html:`<div class="metaline">No rallies found${tl.audio_available ? '' : ' (clip has no audio track, so this used frame motion alone)'}.</div>`};
```

Those two are **absence reasons**, so they stay as visible body text.

- [ ] **Step 4: Move the movement provenance**

Same shape in `movementSection`:

```js
  const cov = Math.round(100*((pv.player_a||{}).sample_coverage||0));
  return {
    why: `Detector: ${pv.backend||'unknown'} · ${cov}% of rally time observed`,
    html: row('P1', pv.player_a||{}) + row('P2', pv.player_b||{}),
  };
```

`whyDisclosure` escapes, so drop the `escapeHtml` around `pv.backend` here — but
keep it on the `!pv` early return, which still emits raw markup:

```js
  if(!pv){
```

leave that branch's body exactly as it is, wrapped to the new shape:

```js
    return {why:'', html:`<div class="metaline">${escapeHtml(why ? 'Movement not measured: ' + why : 'Movement was not measured for this run.')}</div>`};
```

Note the local `why` inside that branch shadows nothing — it is the existing
reason variable. Rename the outer key usage if the linter complains; the
returned object property is what matters.

- [ ] **Step 5: Rebuild the match-report template**

Replace the two heading lines and their calls:

```js
  const rally = rallySection(rep);
  const movement = movementSection(rep);
  const capRows = capabilityRows(rep);
```

```js
    ${sectionHead('Rallies', rally.why)}
    ${rally.html}
    ${sectionHead('Movement', movement.why)}
    ${movement.html}
```

- [ ] **Step 6: Split the heatmap note**

In `renderPlayerMovement`, the note currently concatenates a legend and a
detector name. Keep the legend on screen:

```js
    `Where ${playerDisplayName(playerNumber)} stood during rallies — stronger color means more time.`;
```

The detector is already disclosed on the Movement section of the match report, so
it is not re-added here.

- [ ] **Step 7: Collapse the three coaching caveats**

Markup — replace the two paragraphs in `#adviceProvenance`:

```html
    <div class="card" id="adviceProvenance">
      <div class="cardtitle whyhead" id="adviceHeadline">Working on your coaching</div>
      <div id="adviceWhy"></div>
    </div>
```

`renderAdvice` — in each of the error, loading and no-identified-player branches,
replace the `$('adviceMeta')` / `$('adviceCaveat')` writes with:

```js
    $('adviceWhy').innerHTML = '';
```

For the error and no-player branches the message currently written to
`#adviceMeta` is a **result**, not provenance, so keep it visible by appending it
to the headline element's sibling instead — write it as body text:

```js
    $('adviceWhy').innerHTML = `<div class="whynote">${escapeHtml(ADVICE.error)}</div>`;
```

(the `.whynote` class without `.hidden` renders it as plain dim body copy.)

In the loaded branch, build one disclosure from all three facts:

```js
  $('adviceHeadline').textContent = items.length
    ? `${items.length} thing${items.length===1?'':'s'} to work on`
    : 'Nothing stood out';
  const why = [
    `${shots} front-wall shot${shots===1?'':'s'} across ${data.session_count} session${data.session_count===1?'':'s'}.`,
    ADVICE.data.me_pooling_note || '',
    (player.advice || {}).low_sample_note || '',
  ].filter(Boolean).join(' ');
  const d = whyDisclosure(why);
  $('adviceHeadline').innerHTML = escapeHtml($('adviceHeadline').textContent) + d.btn;
  $('adviceWhy').innerHTML = d.note;
```

Then delete the block that appended `player.advice.note` as an `.adviceEmpty`
line — the low-sample half of it now rides in the disclosure. The
"No clear weakness stood out…" half is a **result** and moves to the headline
path, which already reads `'Nothing stood out'`.

- [ ] **Step 8: Split the server note in two**

In `coaching_advice.py`, `player_advice` returns a single `note` that mixes two
different things: a **result** ("no clear weakness stood out", "too few to draw a
pattern from") and a **hedge** ("treat this as a pointer rather than a verdict").
Only the hedge belongs behind the disclosure.

`player_advice` has **two** return statements. The early one, for
`total < MIN_HITS_FOR_ADVICE`, is a result and keeps its `note` untouched — it
only gains the new key so both returns have the same shape:

```python
    if total < MIN_HITS_FOR_ADVICE:
        return {
            "items": [],
            "note": (
                f"Only {total} front-wall "
                f"{'shot was' if total == 1 else 'shots were'} analyzed — too "
                "few to draw a pattern from. Track a longer clip for coaching "
                "advice."
            ),
            "low_sample_note": None,
        }
```

The tail return splits. Keep `note`'s `None` default — the existing contract is
`str|None`, and `renderAdvice` tests it for truthiness:

```python
    note = None
    low_sample_note = None
    if not items:
        note = (
            "No clear weakness stood out in this clip — shot height, width and "
            "pace were all inside the expected range."
        )
    elif total < 20:
        low_sample_note = f"Based on {total} front-wall shots, so read it as a pointer."
    return {"items": items, "note": note, "low_sample_note": low_sample_note}
```

Update the docstring to match:

```python
    """-> {'items': [...], 'note': str|None, 'low_sample_note': str|None} for one player's report."""
```

- [ ] **Step 8a: Re-point the test that pinned the hedge**

`tests/test_coaching_advice.py` has a test asserting the hedge lands in `note`.
It must now assert the new field. Replace:

```python
def test_small_but_usable_sample_is_flagged_as_a_pointer():
    result = player_advice(player(total_wall_hits=8, average_wall_height_ft=13.0))

    assert result["items"]
    assert "pointer" in result["note"]
```

with:

```python
def test_small_but_usable_sample_is_flagged_as_a_pointer():
    # The hedge is provenance, so it rides in the disclosure field, not in the
    # note the page renders as body copy.
    result = player_advice(player(total_wall_hits=8, average_wall_height_ft=13.0))

    assert result["items"]
    assert result["note"] is None
    assert "pointer" in result["low_sample_note"]
```

The three other `note` assertions in that file — `"No clear weakness"`,
`"too few"`, and `"N front-wall shot was analyzed"` — are all results and must
keep passing unchanged. Do not touch them.

Editing `coaching_advice.py` auto-runs `tests/test_coaching_advice.py` via the
PostToolUse hook, so make this test edit **before** the source edit or the source
edit will come back blocked.

- [ ] **Step 9: Correct the stale app.py docstring**

`app.py` has a comment pointing at `pooling_note` as the field the caller must
surface. The UI reads `me_pooling_note` now. Update the sentence to name the
field that is actually rendered. Do **not** remove `pooling_note` itself — it is
still returned by `/api/coach/advice` and pinned by
`tests/test_pooled_coach_advice.py`.

- [ ] **Step 10: Update the coach-hero test**

In `tests/test_coach_hero_ui_contract.py`, replace:

```python
    assert "$('adviceCaveat').textContent = ADVICE.data.me_pooling_note || ''" in INDEX_HTML
```

with:

```python
    # The pooling caveat now rides in the §8.23 disclosure on the headline.
    assert "ADVICE.data.me_pooling_note || ''," in INDEX_HTML
```

- [ ] **Step 11: Update DESIGN.md**

In §8.20, amend the Rallies and Movement bullets so the provenance is described as
a §8.23 disclosure on the section heading rather than a `.metaline` beneath it.

In §14, add a voice rule:

```markdown
- **Provenance and confidence sit behind a disclosure, not in body copy** (§8.23).
  State the fact; offer the caveat. The exception is *absence* — the reason a tier
  could not run, or that no rallies were found, stays visible (§8.20).
```

- [ ] **Step 12: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_word_clutter_ui_contract.py tests/test_coach_hero_ui_contract.py tests/test_coaching_advice.py tests/test_pooled_coach_advice.py -q
```

Expected: all pass. `test_pooled_coach_advice.py` asserts `"served first" in
body["pooling_note"]` — that field is deliberately untouched (Step 9), so this
test must still pass without modification. If it fails, `pooling_note` was
removed by mistake; restore it.

- [ ] **Step 13: Run the whole suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: green, one skip and one deselect, pass count matching Task 1 Step 7
plus the tests added here.

- [ ] **Step 14: Verify in the browser**

On a match page: Rallies and Movement each show a circled-`i` after the heading;
the caveat text is absent until tapped; tapping reveals it below the heading and
tapping again hides it. The heading row must be the same height it was before the
button existed. Confirm the ARIA state flips with `aria-expanded`. Repeat in the
other theme, then check the coaching screen the same way. Confirm nothing in the
console errors when a report loads twice (the delegated handler must survive the
innerHTML rebuild).

- [ ] **Step 15: Commit**

```bash
git add index.html app.py coaching_advice.py DESIGN.md tests/
git commit -m "Move provenance behind a tap instead of into body copy"
```

---

### Task 7: Whole-app verification

**Files:** none modified unless a defect is found.

- [ ] **Step 1: Full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: green with one skip and one deselect.

- [ ] **Step 2: Walk every screen in both themes**

Launch and walk Dashboard → Analysis → match → Training → Coaching → Your stats →
Progress at 375 px, in dark and light. Flip themes with the in-app toggle, not by
emulating `prefers-color-scheme`.

- [ ] **Step 3: Check the disclosure contrast**

`.whybtn` at `--dim` must clear 3:1 against its surface in **both** themes (§15).
If dark theme fails, the fix is a token change in DESIGN.md §4.1, not a one-off
color.

- [ ] **Step 4: Confirm no `/eval` run is needed**

No task touched `judge_call.py`, calibration, impact estimation, or detection, so
the line-calling numbers cannot have moved. Confirm with:

```bash
git diff --stat main -- judge_call.py inference_engine.py job_runner.py court_model.py
```

Expected: empty output. If it is not empty, run `/eval` before merging.

- [ ] **Step 5: Screenshot the before/after set**

Capture Dashboard, Analysis, match detail, Training and Coaching in both themes
for the PR description.

---

## Self-Review

**Spec coverage.** Spec §1 Titles → Task 2. §2 Dashboard repeats → Task 3. §3
Internal vocabulary → Task 4, with the capability card split into Task 5 because
it is a logic change rather than a copy change and deserves its own review gate.
§4 Hedges → Task 6. §5 Backend strings → Task 6 Steps 8–9. §6 DESIGN.md → folded
into each task. Verification and accessibility sections → Task 7.

**Interfaces.** `whyDisclosure` / `sectionHead` are defined in Task 1 and used
under those exact names in Task 6. `capabilityRows` keeps its name and gains a
documented empty-string return in Task 5, consumed by the same task.

**Known collateral, already handled:**
`tests/test_analysis_typography_contract.py` pins a CSS rule Task 2 deletes
(Task 2 Step 8). `tests/test_coach_hero_ui_contract.py` pins the advice-caveat
sink Task 6 rewrites (Task 6 Step 10). `tests/test_coaching_advice.py` pins the
low-sample hedge in `note`, which Task 6 moves to `low_sample_note` (Task 6
Step 8a — and it must be edited *before* the source, because the PostToolUse hook
auto-runs it on any `coaching_advice.py` edit).

**Deliberately not collateral:** `tests/test_pooled_coach_advice.py` asserts on
`pooling_note`, which no task touches.

**Ordering.** Task 1 must run first — Task 6 imports its helpers. Tasks 2–5 are
independent of each other and of Task 1, so they can be reordered or parallelised
if worked in separate worktrees. Task 6 must run last. Task 7 gates the merge.
