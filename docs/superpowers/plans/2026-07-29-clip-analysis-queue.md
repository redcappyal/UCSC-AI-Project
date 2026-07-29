# Clip Analysis — autonomous work queue

**Created:** 2026-07-29, from a full codebase review against the approved design at
`docs/superpowers/specs/2026-07-27-tripod-match-analysis-design.md` and its implementation
plan `docs/superpowers/plans/2026-07-27-tripod-match-analysis-mvp.md` (referred to below as
**the MVP plan**; task numbers are its task numbers).

**This file is the loop's memory.** Each iteration reads it, does exactly one `TODO` task,
and marks it `DONE` with a one-line result. Nothing else in the repo records where the
loop got to.

---

## Why this queue, in one paragraph

`job_runner.py:657` segments rallies from **front-wall hits**. Detection recall is ~35%
(`eval_set/BASELINE-2026-07-23.md`: 71 of 109 labeled bounces missed, 65 of them wall
hits). So every rally count, rally length, tempo and work:rest figure the product can
show today is computed from roughly a third of the events that happened — and, unlike a
missing IN/OUT call, a wrong number *looks fine*. CLAUDE.md names this exact failure mode:
"Statistics built on missed bounces are wrong *quietly*, which is worse in a coaching
product." The MVP plan's Principle 1 is the fix: **the ball is never load-bearing.** Rally
structure must come from audio + frame motion, which are available on any clip, and the
ball tier must be gated with a stated reason when the footage can't support it. That is
what this queue builds, bottom of the ladder up.

## The analysis ladder (target)

| Tier | Signal | Outputs | Status |
|---|---|---|---|
| 1. Rally structure | audio transients + frame-motion energy | rally boundaries, count, lengths, tempo, work:rest | **queue tasks 4-6** |
| 2. Player movement | person detection → 2 tracks → feet via floor homography | heatmap, distance, speed, T-time, front/back split | seam landed; **queue task 7** |
| 3. Ball detail | YOLOX ball detector + trackers | shots/rally, wall-impact heights, target zones | exists; gated by **queue task 2** |
| 4. Line calls | `judge_call.py` | IN/OUT | done, unchanged — do not touch |

Tier N never requires tier N+1.

---

## Standing rules for every task

1. **TDD, no exceptions.** Write the failing test, run it, watch it fail *for the stated
   reason*, then implement. Use the `superpowers:test-driven-development` skill.
2. **Full suite green before every commit:**
   `.venv/Scripts/python.exe -m pytest tests/ -q` (Windows box; `.venv/bin/python` on
   macOS). Baseline as of 2026-07-29 is **473 passed, 2 skipped, 1 deselected**. The
   deselection (`requires_model`) is expected. The count must only ever go up.
3. **Never regress the judge.** Tasks that touch anything upstream of `judge_call.py` must
   show the line-call eval is unmoved:
   `.venv/Scripts/python.exe eval_line_calls.py --eval-set eval_set/cases.jsonl`
   against `eval_set/BASELINE-2026-07-23.md`. Zero drift is the pass condition.
4. **One commit per task**, message given in the task. Conventional-commit prefix, body
   explaining *why*, `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` last.
5. **Honest reporting.** If a task cannot be completed, mark it `BLOCKED` with the reason
   and move to the next one. Never mark `DONE` what is not done and verified.
6. **UI work follows `DESIGN.md`.** Tokens only, both themes, phone viewport, `/verify`.
7. **Do not touch** `archive/`, the stereo path, or the `fusion_3d` flag. Do not reintroduce
   a selectable bounce engine.
8. **Windows note:** never pass a non-ASCII path to `cv2.imread`/`imwrite` (CLAUDE.md).
   `cv2.VideoCapture`/`VideoWriter` are fine.

---

## Already landed — do not redo

Verified against the working tree on 2026-07-29. The MVP plan predates these and still
lists them as open; it is stale in these respects, not wrong about the rest.

- **MVP Task 7/8 (court line detection + auto-solve)** — landed as `court_detect.py`
  (`find_lines`, `assign_lines`, `detect_court`) behind `POST /api/detect-court`
  (`app.py:1073`), with a confirm screen in `index.html` (`S.detect`, `app.py` detect flow).
- **MVP Task 10 (calibration optional)** — `POST /api/track` no longer 400s on an empty
  calibration; `validate_floor_calibration` warns and records `calibration_warning` on the
  job (`app.py:1180-1195`).
- **MVP Task 12 (person-detector seam)** — `person_model.py`, wired at `job_runner.py:144`.
- **MVP Task 13 (two-player tracker)** — `player_tracker.py`, plus `player_attribution.py`
  for serve attribution with its own eval axis (`eval_attribution.py`,
  `eval_set/BASELINE-ATTRIBUTION-2026-07-27.md`).

Still genuinely missing: `media_probe.py`, `capabilities.py`, `rally_segmenter.py`,
`movement_stats.py`, `match_report.py`, chunked upload, and any `BASELINE-AUTOSOLVE-*` /
`BASELINE-RALLY-*` doc.

---

## Queue

### Task 1 — `media_probe.py` — TODO

Probe a clip for fps, resolution, duration, per-frame sharpness and audio presence, so
later tasks can gate on footage quality instead of assuming our own 4K60 capture.

Follow **MVP plan Task 1** for the contract. Key points: `cv2.VideoCapture` for
fps/dims/count; sample frames at `np.linspace` indices; variance-of-Laplacian for
sharpness; guard zero/absent fps to 30.0 the way `app.py:91-110` already does; audio
presence via PyAV (`av` is in requirements.txt), returning `None` — not `False` — when the
probe itself could not run, because "no audio track" and "could not look" are different
answers and Principle 3 forbids conflating them.

Test with synthetic videos written by `cv2.VideoWriter` in `tmp_path`, following the
existing pattern in `tests/test_ball_track_offline.py`.

Commit: `feat: media probe (fps/size/sharpness/audio) for capability gating`

---

### Task 2 — `capabilities.py` + pipeline threading — TODO

Turn a probe into a capability set with *stated reasons*, and make the ball stages skip
themselves on footage that cannot support them.

Follow **MVP plan Task 2**. The rule that matters: a disabled tier carries a
human-readable reason ("ball tracking off: 30 fps + motion blur"), and a run always
reports coverage so "found nothing" and "couldn't look" stay distinguishable. Thread the
probe into `job["probe"]` at track start and the capability set into `job["capabilities"]`.

Gate values must be named constants with a comment saying what evidence set them.

Commit: `feat: capability gating with honest reasons; ball stages skipped on unqualified footage`

---

### Task 3 — fps-normalize the time-domain constants — TODO

`detect_wall_hits.py:12-41` and `tracking_common.py:12-21` hold frame-window constants
tuned to 60 fps in native frame units. At 30 fps every window covers twice the wall-clock
it was tuned for, which is why 30 fps camera-roll footage behaves differently for reasons
nobody chose.

Follow **MVP plan Task 3**. This is an **equivalence refactor**: at the 60 fps reference
the scaled values must be *identical* to today's, so the line-call eval moves by exactly
zero. Pixel-domain thresholds stay at reference deliberately — the ball tier only admits
near-reference footage anyway, and tiers 1-2 never consume them.

The real gate for this task is not the unit tests: replay a real run at 60 fps and diff
the judged output against the pre-refactor output. Identical, or it is not done.

Commit: `refactor: fps-normalize time-domain hit/track windows (identity at 60fps, replay-verified)`

---

### Task 4 — `rally_segmenter.py` (pure core) — TODO

The keystone. Rally boundaries from audio transients + frame-motion energy, with **no
reference to ball detections anywhere in the module**.

Follow **MVP plan Task 4**. Keep it a pure function over two time series so it is testable
without video: deterministic synthetic series in, boundaries out. Gap/duration thresholds
are named constants with an eval that exercises them (task 6) — "tuned by eval", never
silently.

Commit: `feat: audio+motion rally segmenter (pure, ball-independent)`

---

### Task 5 — motion hook + pipeline integration — TODO

Compute frame-motion energy during the existing decode pass (no second full decode), feed
it plus `audio_events.extract_audio_candidates` into the segmenter, and emit the timeline
in the completion payload **whether or not the ball tier ran**.

Follow **MVP plan Task 5**. When the ball tier is on, reconcile with hit-derived rallies
rather than replacing them — record both and which one the report should trust.

Commit: `feat: rally timeline from audio+motion, emitted independent of ball tier`

---

### Task 6 — rally-boundary eval axis + baseline — TODO

An analysis with no eval axis is an opinion. Score the segmenter against silver labels
seeded from existing good runs, spot-checkable by a human later.

Follow **MVP plan Task 6**. Ship `rally_labels.jsonl` (text, small, committed) and write
`eval_set/BASELINE-RALLY-2026-07-29.md`. Target from the design spec §8: **F1 ≥ 0.8 at
±1.5 s tolerance**. Falling short is *reportable, not fatal* — write the real number.

Commit: `feat: rally-boundary eval axis + silver labels + baseline`

---

### Task 7 — `movement_stats.py` + pipeline integration — TODO

Tier 2 output: per-player distance, mean/peak speed, T-occupancy, front/back split and a
floor heatmap, from `player_tracker` tracks projected through `court_model.FloorMap`.

Follow **MVP plan Task 14**. Gate on the floor homography being present; when it is not,
the tier is disabled *with a reason*, never silently empty. The capability card names
which detector backend produced the stats, because the motion-blob fallback and real
weights are not equally trustworthy and the report must not pretend otherwise.

Commit: `feat: per-player movement stats behind capability gate`

---

### Task 8 — `match_report.py` + endpoints — TODO

Assemble `report-v1` from whatever tiers ran, and serve it:
`GET /api/runs/<run_id>/report`. Must be **legacy-tolerant** — runs recorded before any of
this exist on disk and must still render, with their absent tiers shown as unavailable.

Follow **MVP plan Task 15**. The report carries capability cards, the rally timeline,
movement stats, and the existing shot/coach data when tier 3 is on. Rally *winner* stays
labeled "est." — it is inferred, not observed (design spec §6).

Commit: `feat: report-v1 assembly + runs index/report endpoints (legacy-tolerant)`

---

### Task 9 — the Matches tab becomes the report surface — TODO

`index.html` `p-matches` currently renders live run cards (`renderClipCards`). Make it the
real report surface: run list → `report-v1` render.

Follow **MVP plan Task 16** and **`DESIGN.md`** (§8 components, tokens only, both themes,
phone viewport). Verify with the `/verify` skill and attach a screenshot to the commit
description of what was checked.

Commit: `feat(web): real Matches tab — run list + report-v1 surface`

---

### Task 10 — chunked upload — TODO

`POST /api/upload` is whole-file multipart capped at 2 GB (`app.py:55`,
`deploy/Caddyfile`) ≈ 5 minutes of 4K60. A 40-minute match cannot be ingested at all,
which makes "record a session and analyze it" false for real sessions.

Follow **MVP plan Task 17**: chunked upload endpoints, byte-identical reassembly proven by
test, and a `uploadFileChunked(file, progressLabel)` client with progress.

Commit: `feat: chunked upload so a full match can be ingested`

---

## When the queue is empty

Do **not** invent new scope. Write `docs/HANDOFF-clip-analysis.md` recording, for each
task: what shipped, the numbers it actually scored, and the exact human steps left
(person-detector weights provision, silver-label spot check, ball-detector retrain). Then
output the completion promise.
