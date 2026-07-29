# Archived: the Challenge pane and the Review | Challenge dock

**Archived 2026-07-29.** Last commit with it live: `6566c60`. Tag that commit
`archive/challenge-ui-v1` when this change merges, so the restore point matches
the convention the other two archives use.

Nothing here is served, imported, compiled, collected, or linted. Unlike
[`archive/stereo/`](../stereo/README.md) and
[`archive/fusion-engine/`](../fusion-engine/README.md), these files did **not**
arrive by `git mv` — the code lived inside `index.html`, a single file, so it
was excised and copied here verbatim. `git log --follow` will not reach its
history; `git show 6566c60:index.html` will.

Same rules as the other two: **do not extend this, and do not import from it.**
If it comes back, it comes back through the gate below.

---

## What it was

The front door of the data flywheel: a way for a human to disagree with the
pipeline's reading of a specific frame, one judged hit at a time.

| Piece | What it did |
|---|---|
| **`.callTabs` dock** | A floating pill at the bottom of the call page (`p-track`) switching between the **Review** and **Challenge** panes. Shared its entire shell with `#navPill` (DESIGN.md §8.3) and differed only in keeping text labels. |
| **Challenge pane** (`#challengeTab`) | Per-hit correction controls: a **type** dropdown (front wall In/Out, side wall, floor, racket, not-a-hit) supervising the event engine and its false positives, and a **Bounce / Not bounce** toggle supervising impact timing — with "Not bounce" entering a scrub-and-confirm mode to name the true bounce frame. |
| **Ball-dot overlay** | A draggable dot on the track canvas, seeded from the impact fit or the nearest detected ball center, correcting the ball's pixel position — the RF-DETR retraining signal. |
| **Correction transport** | `loadCorrections` / `sendCorrection` / `postDraft` against `/api/runs/<id>/corrections`, saving every edit immediately and reflecting the saved state back into the panel. |

Files here: `challenge-ui.html` (dock + pane markup, and the CSS that left with
them), `challenge-ui.js` (all the behavior, plus a "call sites that were
deleted" note listing every line removed from a surviving function).

## What did NOT go with it

**The server side is untouched and still works.** `GET`/`POST
/api/runs/<id>/corrections`, `CORRECTION_SCHEMA_VERSION = "corrections-v2"`,
the validator, `correction_agreement`, and their tests all remain in `app.py`
and `tests/`. Any run that already has a `corrections.json` keeps it, and the
endpoint will still accept a POST — there is simply nothing in the app that
sends one. That is deliberate: restoring the UI should not also mean rebuilding
the contract underneath it.

`hitAtFrame()` also stayed in `index.html`. It was defined inside the
corrections block but the judge flow calls it independently.

## What this costs — read before restoring or replacing it

**This was one of the two label streams feeding the eval set.**
`build_eval_set.py` distills `<run_dir>/corrections.json` *and*
`<run_dir>/ground_truth.json` into `eval_set/cases.jsonl`, which
`eval_line_calls.py` replays and which the `/eval` skill scores every judge,
calibration, and detector change against. Archiving this UI removes the only
in-app way to produce **new** correction labels.

What that does and does not break:

- **Already-distilled labels are safe.** `eval_set/cases.jsonl` and its manifest
  are git-tracked artifacts; every `corr:` case already in them keeps scoring.
  Existing `corrections.json` files in gitignored `ui_runs/` are untouched, so
  re-running `build_eval_set.py` still finds them.
- **The other stream is still live.** Labeling mode (`p-label`, reachable from
  the Dashboard dev row) writes `ground_truth.json` and is unaffected — it also
  captures bounces the detector never saw, which corrections structurally
  cannot.
- **What is gone is the per-hit signal `p-label` cannot express:** the corrected
  *ball pixel position* (the RF-DETR retraining signal) and the IN/OUT verdict
  on a front-wall call. If an eval axis needs those to grow, this is the gap.

## Why it went

Two reasons, in order of weight:

1. **Direction.** Challenging a line call is a *refereeing* interaction, and the
   product is [a training feedback loop, not a referee](../../CLAUDE.md) —
   IN/OUT is one statistic among many. A correction UI that only supervises the
   call is aimed at the thing the product stopped being about.
2. **Chrome.** It was the last floating dock in the app, and it sat on a page
   that had just lost the other one. Two docks with one shell, one of them on a
   destination screen whose only other exit is a back chevron, is more
   navigation furniture than the screen earns.

## The gate

Bring it back when **the eval set needs correction cases it is not getting**,
and someone is committed to sitting down and producing them. Concretely, when
either:

- an eval axis stalls for want of the signal only this captured — corrected ball
  pixel position, or a human IN/OUT on a front-wall call — and
  `eval_set/cases.jsonl` has stopped growing `corr:` cases; **or**
- the ball detector needs position supervision at a volume `p-label` cannot
  supply, since labeling mode records event type and frame but has no dot.

Check first whether labeling mode (`p-label`) can carry the task instead: it is
live, it already feeds the same eval set through `ground_truth.json`, and it
catches bounces the detector missed entirely — which corrections, being
per-detected-hit, never could.

If it does return, reconsider the dock rather than restoring it by reflex. With
the Challenge pane inline under Review the page scrolls further but needs no
switcher at all, and that was the alternative on the table when this was cut.
