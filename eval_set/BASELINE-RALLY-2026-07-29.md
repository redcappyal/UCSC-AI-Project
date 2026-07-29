# Rally-Boundary Baseline — 2026-07-29

First baseline for the rally-structure tier (analysis ladder tier 1). Taken on branch
`claude/codebase-review-clip-analysis-e2c254`, segmenter at commit `2887b1a` plus the
span-bridging fix recorded below.

**Headline: the aggregate F1 is NOT MEASURED on this machine, and that is the honest
result, not a formality.** The one label set with a usable provenance sidecar indexes into
a video that lives on the Mac. The axis, the scorer and the labels all exist and run; the
number is waiting on the file. The exact command to produce it is at the bottom.

Reproduce with:

```
.venv/Scripts/python.exe eval_rally_boundaries.py --labels eval_set/rally_labels.jsonl
```

## What the labels are

`eval_set/rally_labels.jsonl`, seeded by `tools/seed_rally_labels.py`:

| Source | Rallies | From | Video on this machine? |
|---|---|---|---|
| `bayclub_wall_hits.csv` | 17 | 92 human-labeled hit frames @ 60 fps | **No** — `/Users/Ian2/Desktop/Training Data/Bay Club Squash 5min+audio.mov` |

Skipped, and why:

- `wall_hits.csv` — no `.meta.json` sidecar. Without fps and a video sha the frame numbers
  are anonymous integers, exactly as the README says of the eval set generally.
- `matchplay_ep3_wall_hits.csv` — same.

**These labels are silver and unverified.** They are human hit labels clustered into
rallies on a fixed 5 s gap, so they encode a human's *hits* under a clustering rule, not a
human's judgement of where the rallies were. Every row carries `"verified": false`. They
are seeded from human labels rather than from `detected_hits.json` deliberately: scoring
against detector output would measure the segmenter against the very recall problem it was
built to route around, and would flatter it precisely where it matters least.

The plan's intended seeding path — `ui_runs/*/` with recomputed hit rallies — is
unavailable here. `ui_runs/` is gitignored and absent from a fresh worktree. The seeder
supports that path; nothing on this machine can exercise it.

## Full report

```
Rally-boundary eval: 1 labeled video(s), tolerance +/-1.5s

Aggregate: NOT MEASURED - no labeled video was scoreable.
  skipped Bay Club Squash 5min+audio.mov: source video not on this machine
```

The CLI counts and names every skip rather than dropping it. A label set that scores 1.0
because most of it was skipped is worse than no number at all.

## What the axis did catch

Running the segmenter end-to-end on the one real video the repo does carry —
`SquashAnalytics.mp4`, 1920x1080, 60 fps, 311.9 s, sharpness 92.1, **no audio track** —
found a real defect that unit tests had not:

**Before: 0 rallies on a five-minute match clip.**

The motion series is spiky, not a plateau. 230 of 1872 samples (12.3%) cleared the
`median + 3*MAD` threshold, but as **98 fragments with a median duration of 0.00 s and a
longest of 1.83 s** — every one under the 2 s minimum. `_active_motion_spans` required
strictly *contiguous* above-threshold samples, so a rally interrupted by a player pausing,
the ball leaving frame, or the exposure settling shattered into sub-second pieces and none
survived.

Fix: `MOTION_BRIDGE_S = 1.5` — active spans separated by less than that are one span. A
rally is *sustained* activity, meaning active most of the time, not at every sample. The
value sits at MIN_GAP_S/2.67, well under the shortest genuine between-points pause, so
bridging cannot fuse two rallies; a test asserts `MOTION_BRIDGE_S < MIN_GAP_S / 2`.

**After: 10 rallies**, durations 3.3–7.3 s (median 4.4 s), 51 s of 312 s in play (16%).

This is a **smoke result, not a score.** There are no rally labels for this clip, so
nothing here says those 10 rallies are the right 10. Two things are worth flagging to
whoever picks this up:

- 16% work ratio is low for a real match (40–60% would be typical), which is consistent
  with either a practice/knock-up clip or with the segmenter still under-detecting.
- The clip has **no audio track**, so every rally came back `source: "motion"` at
  `confidence: 0.30`. That is the design working — motion-only structure is reported and
  flagged as unverified rather than hidden — but it also means this clip cannot exercise
  the audio or audio+motion paths at all.

## Segmenter constants at this baseline

| Constant | Value | Set by |
|---|---|---|
| `MIN_RALLY_S` | 2.0 | design spec |
| `DEFAULT_GAP_S` | 5.0 | mirrors `job_runner.DEFAULT_RALLY_GAP_SECONDS` |
| `MIN_GAP_S` | 4.0 | mirrors `job_runner.RALLY_GAP_MIN_SPLIT_SECONDS` |
| `GAP_RATIO_SPLIT` | 2.0 | mirrors `job_runner.RALLY_GAP_RATIO_SPLIT` |
| `MOTION_MAD_MULTIPLIER` | 3.0 | provisional |
| `MOTION_BRIDGE_S` | 1.5 | **set by this baseline** — see above |
| `RALLY_PAD_S` | 0.5 | provisional |
| `MOTION_ONLY_CONFIDENCE` | 0.3 | design spec (Principle 3) |
| `CONFIDENT_IMPACT_COUNT` | 4 | provisional |

Everything marked provisional is a first guess that no evidence has yet moved. The design
spec's target is **F1 >= 0.8 at +/-1.5 s**; whether these constants reach it is unknown.

## To fill in the number

On the machine holding the labeled footage:

```bash
.venv/bin/python tools/seed_rally_labels.py --video-root "/Users/Ian2/Desktop/Training Data"
.venv/bin/python eval_rally_boundaries.py --labels eval_set/rally_labels.jsonl
```

Then replace the "Full report" block above with the real output and supersede this file.
If the aggregate falls under 0.8, tune the provisional constants and log each iteration
here — the design spec requires these be "tuned by eval", never silently.

Seeding more labels is cheap and is the highest-value follow-up: one video with 17 silver
rallies is not a corpus, and both other label CSVs would join it for the cost of writing
their `.meta.json` sidecars.
