# Handoff — clip analysis

Work done 2026-07-29 on `claude/codebase-review-clip-analysis-e2c254`, against the queue
at `docs/superpowers/plans/2026-07-29-clip-analysis-queue.md`. Ten tasks, each TDD, each
committed green.

**Suite: 473 → 605 passing** (2 skipped, 1 deselected). The `requires_model` deselection
is expected. **The line-call eval was re-run after every task that touched the pipeline
and is identical to `eval_set/BASELINE-2026-07-23.md` throughout — zero drift.**

---

## What changed, in one paragraph

The pipeline used to hang everything off ball detection. Rallies were segmented from
front-wall hits while detection recall sits near 35%, so every rally count, length and
tempo figure was computed from about a third of the events — and a wrong number looks
exactly like a right one. There is now an **analysis ladder**: rally structure from audio
and frame motion (needs nothing), player movement in court feet (needs a solved court),
ball detail (needs sharp fast frames *and* a court), line calls (needs ball detail). Each
rung stands on the one below, a clip that cannot support a tier says so with a stated
reason instead of returning an empty success, and the Analysis tab renders all of it.

---

## Shipped, with the numbers it actually scored

| # | What | Commit | Real result |
|---|---|---|---|
| — | Line-merge fix in `court_detect` | `12fd498` | Two red tests fixed at their root cause; see below |
| 1 | `media_probe.py` | `20c9195` | 8 tests; ≤16 seeks per probe |
| 2 | `capabilities.py` + gating | `30446f4` | 24 tests; eval zero drift |
| 3 | fps-normalized windows | `b7ccd4f` | **Replay byte-identical** at strides 1/2/4 |
| 4 | `rally_segmenter.py` | `2887b1a` | 17 tests; imports no pipeline module (AST-asserted) |
| 5 | Rally timeline in the pipeline | `2ab98a8` | Emitted on both paths; replay still identical |
| 6 | Rally eval axis + baseline | `468ad53` | **F1 NOT MEASURED** — see below |
| 7 | `movement_stats.py` | `7133a4a` | 12 tests; smoothing window chosen by measurement |
| 8 | `match_report.py` + endpoints | `e73bfc6` | 16 tests; legacy runs render |
| 9 | Analysis tab renders the ladder | `e0ed672` | Verified both themes at 390×844 (text, not visual) |
| 10 | Chunked upload | `3fb2091` | 10 tests; verified against the running server |

### Three findings that were not on the plan

**A silently-wrong calibration (`12fd498`).** `find_lines` grouped Hough fragments by
perpendicular offset *from the image origin*, which levers each fragment's sub-degree
slope error by its distance from the origin. Two pieces of one painted line measured 555
and 564 — a 9 px gap that split one court line into two. `assign_lines` then handed the
same physical stripe to both `service` and `tin`, and `detect_court` returned
`status: "ok"` for a calibration whose tin sat on the service line. Occlusion reproduces
this on real footage every time a player splits a line. The OpenCV pin at 4.10 was masking
it; the two tests now pass on cv2 5.0.0 for the stated reason rather than by version luck.

**Zero rallies on a real match (`468ad53`).** Run end-to-end on `SquashAnalytics.mp4`, the
segmenter found nothing in five minutes of play. Real motion energy is spiky, not a
plateau: 230 of 1872 samples cleared the threshold, but as 98 fragments with a **median
duration of 0.00 s** and a longest of 1.83 s — all under the 2 s minimum. Fixed by
bridging short dips (`MOTION_BRIDGE_S = 1.5`, pinned below `MIN_GAP_S/2` by a test so it
can never fuse two rallies). **0 → 10 rallies**, 3.3–7.3 s, 16% of the clip in play.

**A 17% error in the headline movement number (`7133a4a`).** The plan specified a 0.5 s
smoothing window. Measured against a 40 ft shuttle over 20 seeds per cell:

| jitter (ft) | none | 0.2 s | 0.4 s |
|---|---|---|---|
| 0.00 | 0.0% | 7.5% | 17.0% |
| 0.25 | 9.7% | **5.8%** | 15.9% |
| 0.50 | 35.5% | **3.0%** | 14.1% |
| 1.00 | 104.3% | **10.4%** | 8.4% |

Unsmoothed distance diverges (jitter is a random walk); over-smoothing rounds off the
corners that in squash *are* the movement. 0.2 s centered has the smallest worst case. A
float-comparison bug at the window edge was separately moving the measurement by 5%, and
briefly produced a table of numbers that were themselves artifacts.

---

## What is NOT true yet — read this before trusting anything

1. **The rally segmenter has never been scored against a human.** `BASELINE-RALLY-2026-07-29.md`
   records `Aggregate: NOT MEASURED`. The one label set with a provenance sidecar
   (`bayclub_wall_hits.csv`, 17 silver rallies from 92 human hits) indexes a video that
   lives on the Mac. The 10 rallies found on `SquashAnalytics.mp4` are a **smoke result,
   not a score** — nothing says they are the right 10.
2. **Every threshold in `rally_segmenter` and `movement_stats` is provisional.** They are
   named constants with their reasoning attached, which makes them tunable — not tuned.
3. **There is no movement eval axis at all.** `players_v2` reports `sample_coverage` per
   player, but nothing scores distance, T-time or the heatmap against truth.
4. **Detection recall is unchanged.** Still 71/109 missed. Nothing here touched it; the
   ball tier is now *gated* rather than improved.
5. **Task 9 was verified by extracted text and computed styles, not by eye.** The Browser
   pane was not compositing, so no screenshot exists. A visual pass at a phone viewport in
   both themes is outstanding.
6. **The silver labels are unverified** (`"verified": false` on every row). They encode a
   human's *hits* clustered on a fixed gap, not a human's judgement of rally boundaries.

---

## Exact human steps left

**On the Mac, holding the labeled footage — highest value, unblocks item 1 above:**

```bash
.venv/bin/python tools/seed_rally_labels.py --video-root "/Users/Ian2/Desktop/Training Data"
.venv/bin/python eval_rally_boundaries.py --labels eval_set/rally_labels.jsonl
```

Then supersede `eval_set/BASELINE-RALLY-2026-07-29.md` with the real F1. Target is
**≥ 0.8 at ±1.5 s**. If it falls short, tune the provisional constants and log every
iteration in the baseline doc — the design spec requires these be tuned by eval, never
silently.

**Cheap and it triples the label corpus:** write `.meta.json` sidecars for `wall_hits.csv`
and `matchplay_ep3_wall_hits.csv`. Without fps and a video sha their frame numbers are
anonymous integers and the seeder skips them by design.

**Visual pass on Task 9:** load the Analysis tab at 390×844 in both themes with at least
one real run. Fixture runs can be regenerated — two fabricated `ui_runs/<epoch-ms>/`
directories with `job.json` holding `capabilities` / `rally_timeline` / `players_v2` are
enough; the ones used during development were deleted so they could not be mistaken for
real analysis.

**Person-detector weights** remain the standing gate from the original design. The seam
ships with the motion fallback and `players_v2.backend` names whichever produced the
numbers, so the report is honest either way — but the motion backend under-performs on
crossing and occluded players.

**Optional ball-detector retrain** for recall. Unchanged, still the known lever, and now
strictly optional: the ball tier gates itself off rather than reporting an empty success.

---

## Where things live

| Path | What |
|---|---|
| `media_probe.py` | fps / size / sharpness / audio, ≤16 seeks |
| `capabilities.py` | the four tiers, each `enabled` + `reason` |
| `rally_segmenter.py` | tier 1 — pure, imports no pipeline module |
| `movement_stats.py` | tier 2 — court feet, pure over `CourtSample` |
| `match_report.py` | `report-v1` assembly, legacy-tolerant |
| `eval_rally_boundaries.py` | the rally scorer + CLI |
| `tools/seed_rally_labels.py` | human hit CSVs → silver rally labels |
| `eval_set/BASELINE-RALLY-2026-07-29.md` | what was and was not measured |
| `docs/superpowers/plans/2026-07-29-clip-analysis-queue.md` | per-task results |

`DESIGN.md` §8.20 was rewritten in the same change as the UI, per CLAUDE.md.
