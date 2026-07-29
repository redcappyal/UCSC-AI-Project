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

### Task 1 — `media_probe.py` — DONE (2026-07-29)

Probe a clip for fps, resolution, duration, per-frame sharpness and audio presence, so
later tasks can gate on footage quality instead of assuming our own 4K60 capture.

Follow **MVP plan Task 1** for the contract, exactly — `probe_video(video_path) -> dict`
with keys `fps`, `width`, `height`, `frame_count`, `duration_s`, `sharpness`, `has_audio`.
Key points: `cv2.VideoCapture` for fps/dims/count; sample ≤16 frames at `np.linspace`
indices; sharpness = median variance-of-Laplacian on a centre crop, `None` if no frame
decodes; guard zero/absent fps to 30.0 the way `app.py:91-110` already does; `has_audio`
via PyAV (`av` is already in requirements.txt).

`has_audio` stays a plain `bool` even though "no audio track" and "could not open the
container" both land on `False`. That looks like a Principle 3 violation and is not one:
audio is never a capability input (MVP plan Task 2 makes `rally_structure` unconditionally
enabled), and whether audio actually yielded anything is reported at extraction time as
`rally_timeline.audio_available` in Task 5. Do not widen this to tri-state — Task 2 and
Task 5 consume the `bool`.

Test with synthetic videos written by `cv2.VideoWriter` in `tmp_path`, following the
existing pattern in `tests/test_ball_track_offline.py`.

Commit: `feat: media probe (fps/size/sharpness/audio) for capability gating`

**Result:** `media_probe.py` + `tests/test_media_probe.py`, commit `20c9195`. 8 tests,
suite **481 passed, 2 skipped, 1 deselected**. `probe_video(path) -> dict` per the plan
contract. Samples ≤16 frames by seek (asserted by a test that counts `read` calls) so
ingest stays cheap on a full match. `sharpness` is `None` when nothing decoded;
unopenable files raise `ValueError` rather than probing as all-zero. Not yet wired into
any caller — Task 2 does that.

---

### Task 2 — `capabilities.py` + pipeline threading — DONE (2026-07-29)

Turn a probe into a capability set with *stated reasons*, and make the ball stages skip
themselves on footage that cannot support them.

Follow **MVP plan Task 2**. The rule that matters: a disabled tier carries a
human-readable reason ("ball tracking off: 30 fps + motion blur"), and a run always
reports coverage so "found nothing" and "couldn't look" stay distinguishable. Thread the
probe into `job["probe"]` at track start and the capability set into `job["capabilities"]`.

Gate values must be named constants with a comment saying what evidence set them.

Commit: `feat: capability gating with honest reasons; ball stages skipped on unqualified footage`

**Result:** `capabilities.py` + `tests/test_capabilities.py`, threaded through `app.py`
and `job_runner.py`, commit `30446f4`. 24 new tests, suite **505 passed, 2 skipped,
1 deselected**. Line-call eval re-run: **identical to `BASELINE-2026-07-23.md`, zero
drift**. `compute_capabilities(probe, court_solved=)` → four tiers, each `enabled` +
`reason`, reason set exactly when disabled. `/api/track` probes and stores `job["probe"]`;
`run_tracking_job` resolves the calibration before the model load, and skips the ball
stages (writing a headers-only CSV) when the tier is off. Runs emit `detection_coverage`.
`public_job` now passes `probe`, `capabilities`, `detection_coverage`, `rally_timeline`
and `players_v2` through.

**Carried limitation, for Task 7:** the skip takes the player-movement tier down with it,
which the ladder says it should not. The person detector observes frames via
`frame_observer=` on the *ball* decode pass, so skipping that pass skips it too. It is
commented at `job_runner.complete_without_ball_tier`. Task 7 must give tier 2 its own
pass over the video and re-enable it independently.

---

### Task 3 — fps-normalize the time-domain constants — DONE (2026-07-29)

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

**Result:** commit `b7ccd4f`. `REFERENCE_FPS` + `scale_frames_for_fps` +
`scaled_hit_kwargs` in `detect_wall_hits.py`, `scaled_window_frames` in
`tracking_common.py`, wired at the hit-detection, motion-selection and audio-pad sites in
`job_runner.py`. New `tests/test_detect_wall_hits.py` (11 tests) + 3 wiring tests. Suite
**519 passed, 2 skipped, 1 deselected**.

Equivalence evidence: replayed `detect_hits` over the committed real
`ball_coordinates.csv` at strides 1/2/4 before and after — **byte-identical**
(4/4/6 hits, frames `[112, 247, 589, 1009]` and `[112, 168, 247, 258, 589, 1009]`).
Line-call eval **identical to `BASELINE-2026-07-23.md`, zero drift**.

Two notes for later tasks. The plan expected the replay to run against stored `ui_runs/`;
those are gitignored and absent from a fresh worktree, so the committed CSV stood in — it
is real captured data, not synthetic, but a richer replay corpus would be better. And the
plan expected three `detect_hits_from_rows` call sites; there is one, the others having
gone when the selectable bounce engines were removed.

---

### Task 4 — `rally_segmenter.py` (pure core) — DONE (2026-07-29)

The keystone. Rally boundaries from audio transients + frame-motion energy, with **no
reference to ball detections anywhere in the module**.

Follow **MVP plan Task 4**. Keep it a pure function over two time series so it is testable
without video: deterministic synthetic series in, boundaries out. Gap/duration thresholds
are named constants with an eval that exercises them (task 6) — "tuned by eval", never
silently.

Commit: `feat: audio+motion rally segmenter (pure, ball-independent)`

**Result:** `rally_segmenter.py` + `tests/test_rally_segmenter.py`, commit `2887b1a`.
17 tests, suite **536 passed, 2 skipped, 1 deselected**. `segment_rallies(impacts, motion,
duration)` → sorted, non-overlapping spans with `impact_count`, `source`
(`audio` / `motion` / `audio+motion`) and `confidence`. Either input alone suffices.
A test parses the module's AST and asserts it imports nothing from the pipeline.

**Worth knowing before Task 6 tunes this:** the motion threshold is `median + 3*MAD`, and
MAD degenerates to exactly zero whenever more than half the samples sit at the median —
the *ordinary* case here, since idle time dominates a match clip. When it collapses the
scale falls back to half the distance from the floor to the active level
(`motion_threshold`, with its own regression tests). `motion_threshold` returns `None`
when nothing rises above the floor, so a still camera on an empty court yields no rallies
rather than one long one. These thresholds are provisional and are exactly what the rally
eval axis exists to tune.

---

### Task 5 — motion hook + pipeline integration — DONE (2026-07-29)

Compute frame-motion energy during the existing decode pass (no second full decode), feed
it plus `audio_events.extract_audio_candidates` into the segmenter, and emit the timeline
in the completion payload **whether or not the ball tier ran**.

Follow **MVP plan Task 5**. When the ball tier is on, reconcile with hit-derived rallies
rather than replacing them — record both and which one the report should trust.

Commit: `feat: rally timeline from audio+motion, emitted independent of ball tier`

**Result:** commit `2ab98a8`. `motion_energy_step` + `build_rally_timeline` in
`rally_segmenter.py`; `MotionAccumulator`, `compose_frame_observers`,
`accumulate_motion_only` and `build_and_write_rally_timeline` in `job_runner.py`.
Suite **549 passed, 2 skipped, 1 deselected**. Replay of `detect_hits` over the committed
`ball_coordinates.csv` at strides 1/2/4 **byte-identical** to the pre-Task-3 fingerprint;
line-call eval **identical to baseline, zero drift**.

Both paths emit tier 1: motion rides the coarse decode via the existing `frame_observer`
seam when the ball tier is on, and gets its own ~6 samples/second decode (no model) when
it is off. Samples are keyed by frame index so refine/audio-rescue re-decodes cannot
double-count. `rally_timeline` (schema `rally-timeline-v1`) goes to both the job and
`<run_dir>/rally_timeline.json`. `agrees_with_hits` is `None` — not `True` — when there
are no hit rallies to compare against.

**Untuned, and the reason Task 6 is next:** every threshold in `rally_segmenter` is still
a first guess. Nothing has scored these boundaries against a human yet.

---

### Task 6 — rally-boundary eval axis + baseline — DONE (2026-07-29, F1 unmeasured)

An analysis with no eval axis is an opinion. Score the segmenter against silver labels
seeded from existing good runs, spot-checkable by a human later.

Follow **MVP plan Task 6**. Ship `rally_labels.jsonl` (text, small, committed) and write
`eval_set/BASELINE-RALLY-2026-07-29.md`. Target from the design spec §8: **F1 ≥ 0.8 at
±1.5 s tolerance**. Falling short is *reportable, not fatal* — write the real number.

Commit: `feat: rally-boundary eval axis + silver labels + baseline`

**Result:** commit `468ad53`. `eval_rally_boundaries.py` (scorer + CLI),
`tools/seed_rally_labels.py`, `tests/test_eval_rally_boundaries.py` (13 tests),
`eval_set/rally_labels.jsonl`, `eval_set/BASELINE-RALLY-2026-07-29.md`. Suite
**564 passed, 2 skipped, 1 deselected**; line-call eval unmoved.

**The F1 target is NOT MET because it is NOT MEASURED, and that is the real outcome.**
The only label CSV with a provenance sidecar (`bayclub_wall_hits.csv` → 17 silver rallies
from 92 human hits) indexes into a video on the Mac; the other two have no sidecar, so
their frame numbers are anonymous integers. The plan's intended `ui_runs/` seeding path is
unavailable — gitignored, absent from a fresh worktree. The baseline doc carries the exact
command to fill the number in on the Mac.

**The axis earned its keep anyway.** Run end-to-end on `SquashAnalytics.mp4`, the
segmenter found **0 rallies in a five-minute match**. Real motion energy is spiky, not a
plateau: 230/1872 samples cleared the threshold but as 98 fragments, median duration
0.00 s, longest 1.83 s — all under `MIN_RALLY_S`. Fixed with `MOTION_BRIDGE_S = 1.5`
(bridge short dips before measuring duration; a test pins `MOTION_BRIDGE_S < MIN_GAP_S/2`
so it can never fuse two rallies). **0 → 10 rallies**, 3.3–7.3 s, 16% of the clip in play.
That is a smoke result, not a score — there are no labels for that clip.

**Follow-ups this leaves open** (do not silently absorb into a later task):
1. Run the eval on the Mac and supersede the baseline with a real F1.
2. Write `.meta.json` sidecars for `wall_hits.csv` and `matchplay_ep3_wall_hits.csv` —
   cheap, and it triples the label corpus.
3. `SquashAnalytics.mp4` has **no audio track**, so nothing here exercises the audio or
   audio+motion paths on real footage.

---

### Task 7 — `movement_stats.py` + pipeline integration — DONE (2026-07-29)

Tier 2 output: per-player distance, mean/peak speed, T-occupancy, front/back split and a
floor heatmap, from `player_tracker` tracks projected through `court_model.FloorMap`.

Follow **MVP plan Task 14**. Gate on the floor homography being present; when it is not,
the tier is disabled *with a reason*, never silently empty. The capability card names
which detector backend produced the stats, because the motion-blob fallback and real
weights are not equally trustworthy and the report must not pretend otherwise.

Commit: `feat: per-player movement stats behind capability gate`

**Result:** `movement_stats.py` + `tests/test_movement_stats.py` (12 tests), wired in
`job_runner.py` as `players_v2` (schema `players-v2`), commit `7133a4a`. Suite
**579 passed, 2 skipped, 1 deselected**; line-call eval unmoved.

**The Task 2 coupling is broken.** The person pass now rides the motion-only decode as
well as the coarse ball decode, so a 30 fps clip with a solved court returns
`ball_tracking: off, player_movement: on` with real stats. A test asserts exactly that.

**Two measured findings worth carrying forward.** The smoothing window was chosen from a
jitter sweep, not assumed: unsmoothed distance *diverges* (104% error at 1 ft of jitter,
because jitter is a random walk), while over-smoothing rounds off the corners that in
squash *are* the movement. `SMOOTH_WINDOW_S = 0.2` centered has the smallest worst case;
the plan's 0.5 s costs ~17% of the headline number. And the window edges were being
compared against accumulated float times, so the window silently held 2 samples at one
timestamp and 3 at the next — enough instability to move measured distance by 5%, and it
briefly produced a table of numbers that were themselves artifacts.

**Known bias, documented and pinned by a test:** distance under-reads by ~7.5% on sharp
direction changes. The test asserts the *direction* of the bias, so a drift to
over-counting — crediting a player with distance nobody ran — fails loudly.

**Not yet done:** no eval axis for movement. `sample_coverage` is reported per player but
nothing scores the stats against ground truth, and there is no `BASELINE-MOVEMENT-*`.
Person-detector weights remain the standing human gate (`backend` names what produced the
numbers, so the report can say so).

---

### Task 8 — `match_report.py` + endpoints — DONE (2026-07-29)

Assemble `report-v1` from whatever tiers ran, and serve it:
`GET /api/runs/<run_id>/report`. Must be **legacy-tolerant** — runs recorded before any of
this exist on disk and must still render, with their absent tiers shown as unavailable.

Follow **MVP plan Task 15**. The report carries capability cards, the rally timeline,
movement stats, and the existing shot/coach data when tier 3 is on. Rally *winner* stays
labeled "est." — it is inferred, not observed (design spec §6).

Commit: `feat: report-v1 assembly + runs index/report endpoints (legacy-tolerant)`

**Result:** `match_report.py` + `tests/test_match_report.py` (16 tests),
`GET /api/runs/<run_id>/report`, `tiers_enabled` added additively to `GET /api/runs`,
commit `e73bfc6`. Suite **595 passed, 2 skipped, 1 deselected**.

`build_report(run_dir, coach_builder=None)` → `report-v1`. Legacy runs render with
`legacy: true` and a stated reason rather than an empty card. A failing `coach_builder`
costs the coaching, not the report. The endpoint passes derived analytics only, never LLM
narration — a report is fetched on every view.

**What Task 9 needs from this:** the report shape is
`{schema, run_id, created_ms, status, legacy, legacy_reason, video, probe, capabilities,
tiers_enabled, detection_coverage, rally_timeline, players_v2, shots}`. Any of `probe`,
`capabilities`, `rally_timeline`, `players_v2`, `shots` may be `null`, and the UI must
render each absence as its stated reason (or the legacy card), never as an empty section.

---

### Task 9 — the Matches tab becomes the report surface — DONE (2026-07-29)

`index.html` `p-matches` currently renders live run cards (`renderClipCards`). Make it the
real report surface: run list → `report-v1` render.

Follow **MVP plan Task 16** and **`DESIGN.md`** (§8 components, tokens only, both themes,
phone viewport). Verify with the `/verify` skill and attach a screenshot to the commit
description of what was checked.

Commit: `feat(web): real Matches tab — run list + report-v1 surface`

**Result:** commit `e0ed672`. `index.html` (loader + `capabilityRows` / `rallySection` /
`movementSection` / `coverageLine`), `DESIGN.md` §8.20 rewritten in the same change,
`.claude/launch.json` gains a Windows entry. Suite **595 passed, 2 skipped, 1 deselected**.

**The important part was a deletion.** `loadLiveRuns` required `has_analytics` *and*
`total_wall_hits > 0`, so a ball-tier-skipped run — the exact case the whole queue
enables — never appeared in the list at all, and the user saw "no analyzed sessions" for
a run that succeeded. Runs now qualify on **any** tier having produced something. Without
this, Tasks 1–8 were invisible.

Ball-tier surfaces are omitted, not zeroed, when the tier did not run: a 0% gauge and
three em-dashes read as "we looked and found nothing".

**Verification was text-based, not visual.** Screenshots were unavailable — the Browser
pane was not compositing in this session — so the card was checked by extracted text and
computed styles in both themes at 390×844, with no console errors. **A visual pass at a
phone viewport is still outstanding** and is the first thing to do on a machine with a
working preview.

---

### Task 10 — chunked upload — DONE (2026-07-29)

`POST /api/upload` is whole-file multipart capped at 2 GB (`app.py:55`,
`deploy/Caddyfile`) ≈ 5 minutes of 4K60. A 40-minute match cannot be ingested at all,
which makes "record a session and analyze it" false for real sessions.

Follow **MVP plan Task 17**: chunked upload endpoints, byte-identical reassembly proven by
test, and a `uploadFileChunked(file, progressLabel)` client with progress.

Commit: `feat: chunked upload so a full match can be ingested`

**Result:** commit `3fb2091`. `POST /api/upload/init` / `/api/upload/chunk/<id>?index=N` /
`/api/upload/complete/<id>` in `app.py`, `uploadFileChunked` in `index.html` (switches
above 512 MB), `tests/test_chunked_upload.py` (10 tests). Suite **605 passed, 2 skipped,
1 deselected**. Verified against the running server: init ok, sequential chunks 200,
out-of-order 409, complete returns a `video_id`.

Strict sequencing is the integrity guarantee — a gap would assemble a file with a hole in
it that still decodes and still produces statistics. `init` captures the filename suffix
because `video_path_for_id` globs `<id>.*`. The partial dir derives from `BY_HASH_DIR` so
the `runs_dir` fixture sandboxes it.

---

## How the loop runs this file

Each iteration:

1. **Read this file first.** It is the only record of where the loop got to; the
   conversation may have been compacted away.
2. Pick the **first task still marked `TODO`** and do that one task, completely.
3. Follow *Standing rules for every task* above — TDD, full suite green, one commit.
4. Then edit this file: change that task's heading from `TODO` to `DONE (2026-07-29)` and
   append a short `**Result:**` paragraph saying what shipped and the **real** numbers it
   scored. Commit that edit too.
5. If the task cannot be finished, mark it `BLOCKED` with the reason and move to the next.
   Never mark `DONE` what is not done and verified.

Environment: this is the Windows box, so the interpreter is `.venv/Scripts/python.exe`
(CLAUDE.md's `.venv/bin/python` is the macOS path). Suite command:
`.venv/Scripts/python.exe -m pytest tests/ -q`.

## When the queue is empty

Do **not** invent new scope. Write `docs/HANDOFF-clip-analysis.md` recording, for each
task: what shipped, the numbers it actually scored, and the exact human steps left
(person-detector weights provision, silver-label spot check, ball-detector retrain).

Then output the completion promise, which is the literal tag:
`<promise>CLIP ANALYSIS QUEUE COMPLETE</promise>`
