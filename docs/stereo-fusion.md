# Stereo fusion (spec Phase 5) — what it does and what it does not claim

Two phones record the same rally. Each uploads and tracks independently, exactly
as a single-camera run does. When both runs of a `session_id` complete, a third
job fuses them into one court-frame 3D result.

## Using it

Two extra form fields on `/api/track` turn a run into half of a pair:

| Field | Required | Meaning |
|---|---|---|
| `session_id` | with `camera_role` | Groups the two runs. Any non-empty string. |
| `camera_role` | with `session_id` | `a` or `b`. The fused result is expressed on **role a's** clock and fps. |
| `peer_video_id` | no | The peer's `/api/upload` sha256. Recorded, not yet consumed. |
| `sync_manifest_json` | no | JSON object; seeds the offset refiner. |

`session_id` and `camera_role` are all-or-nothing: a role with no session can
never pair, and a session whose two runs share a role cannot be told apart at
fuse time. Both fail at 400 rather than producing a run that silently never
fuses.

Omit all of them and the pipeline is byte-identical to before Phase 5 — that is
the property the tests lead with, not an afterthought.

The manifest is read tolerantly and every unrecognised key ignored. Only two are
consumed today:

```json
{"clap_anchor_s": 0.0071, "offset_series": [0.004, 0.009, 0.005]}
```

`clap_anchor_s` wins when present — it is an acoustic fix on one shared instant
and beats the network estimator. Otherwise the median of `offset_series`.
Neither is required: the refiner starts from zero and searches.

## What the fuse job produces

In `ui_runs/stereo-<session_id>/`:

- **`stereo_track.jsonl`** — one JSON object per line: `t_s`, `point_ft`,
  `gap_ft`, `view_count`. The court-frame 3D track.
- **`detected_hits.json`** — the fused calls, in the same vocabulary a
  single-camera run writes, so every existing consumer reads it unchanged.
  Additive per hit: `view_count`, `method`, and a `stereo` block holding the
  native values (`surface`, `point_ft`, `margin_ft`, `confidence`,
  `snap_disagreement_ft`).
- **`sync_report.json`** — schema `stereo-sync-v1`: the seed, the refined
  offset, the cost at both, an accept/reject verdict with its reason, the
  search parameters, and `pair_agreement` between the two calibrations.

All three are downloadable through the existing
`GET /api/runs/<run_id>/<filename>` route with no new code.

## The offset refiner

The correct clock offset is the one that makes both pixel tracks agree on where
the ball was in 3D. The cost is the mean **clipped** ray gap over a timeline held
**fixed** across every candidate offset:

- *Clipped*, because one frame where a camera lost the ball produces a gap large
  enough to dominate a raw mean.
- *Fixed timeline*, because a grid that moves with the offset scores different
  candidates on different numbers of points, and the minimum becomes an artefact
  of the grid rather than a fact about the ball.

Measured against a 3 ft fin pair at 60 fps:

| Detection noise | Offset error | Mean ray gap |
|---|---|---|
| 0 px | 0.00000 s | 0.0038 ft |
| 1 px | 0.00010 s | 0.0084 ft |
| 3 px | 0.00030 s | 0.0212 ft |
| 12 px | 0.00070 s | 0.0818 ft |

So it beats the live 2 ms budget by roughly 3× even at an absurd 12 px. Those
numbers are a test, not a memory.

It rejects rather than guesses. No improvement over the seed, a minimum sitting
on the edge of the search range, or a best cost still above
`MAX_AGREEING_GAP_FT` all keep the seed and record why in `sync_report.json`. A
confident wrong offset is worse than an honest miss.

## Decisions this implementation makes that the spec left open

- **Input is the CSVs, not the videos.** Both runs already paid for detection.
- **The dense timeline, not literally "the full recording."** The refiner
  evaluates the overlap of the two streams, not every frame of both clips.
- **`method: "estimated"` is declared but unreachable.** Nothing in
  `detect_impacts` produces a point without either a plane snap or a
  triangulation. It stays in the vocabulary for a future ballistic path.
- **`"down"` maps to `"OUT"`.** A tin hit ends the rally exactly as an out ball
  does, and the mono vocabulary has no third word.
- **`margin_px` is `None`** on a fused hit. A stereo call has no pixel margin.
- **No `racket` events.** Stereo has no racket surface; the mono classifier does.
  This is a real coverage gap, not an oversight.
- **Per-device readout time is carried in the manifest and consumed by nothing.**
  Rolling-shutter correction (`t_effective = PTS + (row/H)·readout`) is out of
  scope for a first cut rather than half-implemented. Until it exists, the sync
  budget is optimistic for fast cross-frame motion.
- **The fuse run deliberately has no `video_path`.** `build_eval_set` keeps one
  run per `video_sha`; a fuse run carrying role a's video could displace the
  mono run it was derived from and silently rebaseline the eval set.

## What this is NOT

**The fused path is unevaluated. It must not be described as an improvement over
the monocular baseline.**

The spec is explicit that *"the fused line-call path gets an eval pass against
labeled clips before it may be called an improvement over the monocular
baseline"*, and that gate cannot be satisfied from this repository:

- No paired two-camera footage exists here. The only two `.mp4`s in the tree are
  single-camera outputs.
- The one labeled clip (`bayclub_wall_hits.csv`) is single-camera, and its source
  video lives on another machine.
- Every test above runs on synthetic trajectories through synthetic cameras. That
  proves the geometry and the plumbing. It cannot prove the detector finds the
  ball in *both* views often enough on a real court — occlusion by players is the
  dominant real failure mode, and a synthetic detector is perfect by
  construction.

### What a paired eval clip needs

1. Two phones on the mounts recording the same rallies, **both in the same
   orientation** (mixed orientations produce transposed pixel spaces that
   triangulate to plausible-looking nonsense).
2. A 4K calibration per phone.
3. Wall-hit frames labeled on the **role a** clip, so the existing `video_sha`
   join works.
4. Then: `stereo_fuse` output through the eval harness against the monocular
   baseline — and the `--fail-under` gate extended past IN/OUT accuracy, which
   today is computed from a handful of corrections and skipped entirely when
   accuracy is `None`. It cannot detect a missed-bounce or position regression on
   the fused path.

Until all four exist, "unevaluated" is the honest word.
