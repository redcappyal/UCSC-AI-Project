# Match review page — design

Date: 2026-07-29
Status: approved, implementing

## Problem

The post-analysis experience is scattered across three phases (`track`, `player1_report`,
`player2_report`) plus a fourth, `target`, that is unreachable dead code. Back from the
Call page exits to the Dashboard, not to Analysis. Reaching a past match's review takes a
card expand plus an "Open full review" button. And after analysis completes the user can
still walk back through the calibration wizard for a run that is already analyzed.

Separately: the per-player reports render player attribution as fact, while the one signal
that could flag that attribution as wrong is computed and discarded.

## Goals

1. One full-screen review page covering Call + Player 1 + Player 2 + Players naming.
2. Back arrow returns to the Analysis tab, always.
3. Tapping a past match in Analysis opens its review page directly.
4. Analysis completion routes to the review page; calibration for that run becomes
   unreachable.
5. Surface per-rally attribution provenance, and repair it where the evidence supports it.

## 1. Page structure — one page, three panes

**Implementation deviation (decided during build).** The design said one phase with a
`S.reviewPane` sub-state. What shipped keeps the three existing phases (`track`,
`player1_report`, `player2_report`) and groups them with a `REVIEW_PHASES` constant.
About 25 call sites — transport, strip rendering, the restore stash, the resize handler —
key off `S.phase === 'track'`; collapsing the phases would have rewritten all of them for
no user-visible difference. The page behaves exactly as specified: one switcher, one
header, Back exits to Analysis from any pane.

A `.seg` three-way switcher (existing component, `index.html` §`.seg`) sits at the top of
`<main>`: `Call | <Player 1 name> | <Player 2 name>`. Pane labels use
`playerDisplayName(n)` so saved names appear.

The existing sections `p-track`, `p-player1-report`, `p-player2-report` are reused verbatim
as pane bodies — the timelines, judge controls, court charts and coaching panels are not
re-authored. Pane switching drives:

| Pane | `#stage` | body class | section shown |
|---|---|---|---|
| `call` | visible | `phase-track` | `p-track` |
| `p1` | hidden | `phase-target` | `p-player1-report` |
| `p2` | hidden | `phase-target` | `p-player2-report` |

Each pane therefore keeps the internal scroller it already relies on (DESIGN.md §3.1 —
a stage-less page without one silently loses its lower content).

The Players naming card (`#playersCard`: name A/B, serve crop prompt, save) moves out of
`p-target` into the Call pane, below the judge controls. `p-target` and `buildTargetView`
are deleted, along with the orphaned `#targetBtn` ("Player maps") that pointed into the
old flow.

## 2. Header

- Back chevron → Analysis, unconditionally.
- Home button hidden on `review`.
- `#stepLabel` shows the match date (`fmtRunDate(run.created)`), not a step number.
- `stepSequence()` drops `track`, `player1_report`, `player2_report` — step numbering is
  meaningless once calibration is unreachable.
- The Call pane keeps its proxied "Judge frame" primary (DESIGN.md §3.4 is binding for
  every phase). Report panes have no primary.

## 3. Routing

- Analysis completion (`runTrackBtn` handler) → `setPhase('review')`, and tears down
  `S.work`, `S.lines`, `S.wall`, `S.floor`, `S.base`, `#fileIn.value`.
- Back from `review` → `setPhase('matches')` + `refreshLiveViews()` so a just-finished run
  appears in the list.
- `openRunReview(id)` and the iOS `#run=` deep link keep the stash-and-reload rehydration
  path (proven, and shared with native); the restore branch in `routeLoadedVideo` lands on
  `review` instead of `track`.
- Analysis tab: tapping a `.clipcard` opens the review directly. The expand/collapse
  summary and its "Open full review" button are removed — the review page supersedes them.
  "Watch source video" moves onto the review page.
- Calibration remains reachable only by starting a new analysis from the Dashboard.

## 4. Attribution repair (pipeline)

In `assign_front_wall_hit_players` (`job_runner.py`), the winner back-fill gains a repair.

**Rule.** When rally N's `winner_crosscheck_agrees` is `False` **and** rally N's own
`server_source` is `"propagated"`, flip that rally's parity: swap `server_player_number`
and rewrite `player_number` on every hit in the rally.

**Why propagated-only.** A whole-rally parity flip is equivalent to flipping
`rally_server`, because hit 1's `player_number` *is* the server. When the rally's serve was
`propagated` the flip replaces an alternation guess with an inference grounded in vision
evidence — a strict upgrade. When it was `observed`, the flip would discard a direct
observation of who served, and if the true cause was a missed *mid-rally* hit it would
merely move the error from the end of the rally to the start. Those stay flagged, unrepaired.

**No cascade.** The back-fill only fires when rally N+1's serve is `observed`, so rally
N+1's server does not depend on rally N's winner. A repair is therefore local to its own
rally; no re-propagation pass is needed.

*Refined during implementation.* There is one path where rally N+1's observed server does
trace back to rally N's est winner: when rally N+1 is the **anchor** rally, `track_player_map`
is built from the propagated `server`, which is rally N's est winner. That case can never
trigger a repair — at the anchor, `observed_winner == track_player_map[resolved] == server
== est_winner`, so the cross-check is always `True` (or `None` when rally N's winner was
unjudged). Repairs therefore only ever fire strictly after the anchor, where the map is
already fixed. The conclusion holds; the argument needed this extra step.

**Field contract** (added to each rally in `player_assignment.rallies[]` and mirrored
through `build_players_v1` into `players_v1.rallies[]`):

- `attribution_state`: `"observed" | "repaired" | "assumed" | "conflict"`
  - `observed` — serve observed; cross-check agreed, or no next observed serve to check
  - `repaired` — serve was propagated, cross-check disagreed, parity flipped
  - `assumed` — serve propagated; no repair applied
  - `conflict` — serve observed but cross-check disagreed; left untouched
- `parity_repaired`: `bool`

`winner_crosscheck_agrees` keeps its **pre-repair** value, so the diagnostic that triggered
the repair survives in the payload. `server_source` is not rewritten by a repair —
`attribution_state` carries that nuance — so `attributionAnchor()` in `index.html`, which
keys on `server_source === 'observed'`, is unaffected.

Fields kept consistent after a flip: `server_player_number`, `last_player_number`,
`winner_reason`, and every hit's `player_number`.

## 5. Rally trust in the UI

The rally ribbon segment for each rally carries its `attribution_state` as a modifier
class, with a legend that appears only when two or more states are present (a clean run
should not be made to look uncertain by a key full of warnings it never triggered):

- `observed` — solid accent (today's look)
- `repaired` — solid accent, `↻` marker
- `assumed` — dashed border, dim
- `conflict` — `--tint-poor` fill, dashed border, `!` marker

**Color family.** Attribution provenance is a *data-quality* split, so per DESIGN.md §5.2
it uses the quality-tint family plus the §13 solid-vs-dashed grammar. It must never borrow
the verdict family (`--in`/`--outcall`) or the calibration hues (`--out`/`--service`/`--tin`).
A first draft used `--out` for `conflict` — that reads as "the out line" and crossed a
stream the doc calls load-bearing; it was corrected to `--tint-poor`.

Each player pane gains a one-line provenance count above the coaching copy, e.g.
`Shot attribution across 4 rallies: 1 assumed · 1 corrected from the next serve ·
1 conflicting.` It takes `.warn` (700 `--text`, no red, per §13) when any rally conflicts.

## 5b. Stage-height clamp (regression found in verification)

`preserveTrackStageHeight()` sized `main` from the Call section's natural height. Moving
the Players card onto that pane pushed the measurement to 820px against an 812px viewport,
which collapsed `#stage` to zero and pushed the page tail off a body that never scrolls.
It now clamps against real available space, reserving `TRACK_MIN_STAGE_PX = 180` for the
video and a `TRACK_MIN_MAIN_PX = 240` floor for the controls, letting the surplus scroll
inside `main` — which is what its `overflow-y:auto` is for.

## 6. Verification — results

Run on the Windows box, which has no `.venv`; used the training venv's interpreter
(`C:\Users\alann\Code\ball-detector-train\.venv`) with `flask`/`joblib`/`sklearn`
installed to a scratch `--target` dir on `PYTHONPATH`. That venv was not modified.

- `pytest tests/ -q` → **2 failed, 479 passed, 1 skipped, 1 deselected**. Baseline before
  any edit was 2 failed / 472 passed: the same two failures, both in
  `tests/test_court_detect.py::test_detect_court_fails_when_a_front_wall_line_is_missing`,
  an OpenCV-version difference on this machine. Neither `court_detect.py` nor its test file
  is touched by this change (`git status` confirms), so they cannot be caused by it.
  Note the suite is 475 tests here, not the 283 CLAUDE.md advertises.
- `tests/test_pipeline.py:649` and `:660` **passed unchanged** — verified, not assumed.
  Both rallies there have `server_source == "observed"`, so under the propagated-only rule
  they become `conflict` and stay unrepaired.
- `eval_line_calls.py --eval-set eval_set/cases.jsonl` → drift 0/3, hit-type 2/4,
  missed 71/109, matched-pair 38/38 — matches `eval_set/BASELINE-2026-07-23.md` exactly.
  Zero drift; judging and detection unaffected.
- Browser verification at 375×812 in both themes: all four provenance states render, Back
  exits to Analysis from every pane, calibration state is cleared on commit, the body never
  scrolls, and no console errors across all seven reachable phases.

**What is still not claimed.** `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md` was scored
against template labels and carries a "NOT A REAL ACCURACY" banner; the human labeling gate
is still pending. This change is *better-founded* — it replaces a guess with vision-derived
inference — but is **not** measurably more accurate. Per CLAUDE.md, `eval_attribution.py`
against human-verified labels is the only path to that claim.

**What cannot be claimed.** `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md` was scored
against template labels and carries a "NOT A REAL ACCURACY" banner; the human labeling
gate is still pending. This change may therefore be described as *better-founded* — it
replaces a guess with vision-derived inference — but **not** as measurably more accurate.
Per CLAUDE.md, `eval_attribution.py` against human-verified labels is the only path to
that claim.

## Out of scope

- Repairing observed-vs-observed conflicts (needs per-shot evidence the pipeline lacks).
- Cross-session comparison on the review page.
- iOS native changes beyond the existing `#run=` deep link, which keeps working.
