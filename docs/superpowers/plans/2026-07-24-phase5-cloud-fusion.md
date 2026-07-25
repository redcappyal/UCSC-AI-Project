# Phase 5: Cloud Fusion (paired uploads, stereo_fuse, offset refinement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn two independently-uploaded single-camera runs of the same rally into one
fused, court-frame 3D result — with the whole path testable on a dev machine, with no
video, no model, and no second phone.

**Architecture:** Phase 5 is purely additive server work. `stereo_engine.py` is finished
and untouched by this plan: the numerics stay byte-identical so `tests/stereo_goldens.json`
and its iOS twin do not move. What is new is (a) four optional `/api/track` fields, (b) a
session-discovery trigger, (c) an offline clock-offset refiner — the only genuinely new
algorithm here — and (d) a fuse job that writes three artifacts.

**Tech Stack:** Python 3, Flask, numpy. No scipy, no new dependencies. Tests run on
`requirements-test.txt` alone (no torch, no `inference`, no cv2 video decode).

## Global Constraints

- **The single-camera flow must be byte-compatible.** A request without the new fields
  must produce a `job.json`, a `detected_hits.json`, a `ball_coordinates.csv` and an
  `/api/track` response identical to today's. This is provable and Task 7 pins it.
- **New fields are additive only.** `public_job`'s allowlist (app.py:127-140) is the gate:
  a new job key is invisible to clients until its name is added there. Only set a key when
  the client actually sent the corresponding field — use the existing `extra_job_fields`
  idiom (app.py:847-849), never `session_id=None`.
- **The goldens do not move.** `method` and `view_count` are derived *outside*
  `stereo_engine` from the existing `confidence` string. If a later change does touch
  those numerics, regenerate both copies in one commit via
  `python tests/generate_stereo_goldens.py` and bump the schema on both sides.
- **`run_id` is `str(int(time.time() * 1000))`** (app.py:813) — a wall-clock stamp with no
  uniqueness guarantee. Session pairing keys on `session_id`, never on run_id adjacency or
  ordering.
- Repo test conventions: **no `conftest.py`, no pytest config** — adding the first would
  change collection for the whole suite. Each test file does its own
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`. Tests must use
  `tmp_path`, never the real `ui_runs/` tree, and must pop any `run_id` they add to
  `job_runner.JOBS` under `JOBS_LOCK` in a `finally`.
- **Never run `build_eval_set.py --replace` on a machine with an empty `ui_runs/`.**
  `merge_with_existing` is the only reason the committed cases survive; `--replace` would
  destroy them.
- Baseline to hold: **Python 271 passing**. Test command:
  `.venv/bin/python -m pytest tests/ -q`

## The honest-state gate (read before claiming anything)

The spec makes an eval pass a precondition: *"the fused line-call path gets an eval pass
against labeled clips before it may be called an improvement over the monocular
baseline."*

**That gate cannot be satisfied in this repository.** Verified: `session_id`,
`camera_role` and `peer_video_id` appear nowhere in any `.py` or `.html`; the only two
`.mp4`s in the tree are single-camera outputs; `ui_runs/` is empty and gitignored; the one
labeled clip (`bayclub_wall_hits.csv`, 95 wall-hit frames) is single-camera and its source
video lives on another machine.

The blocking artifact is **a two-phone synchronized recording of the same rallies, with
wall-hit frames labeled on the role-a clip** so the existing `video_sha` join works. Until
that exists this plan ships the machinery and its tests, and the fused path is described
as **unevaluated** — never as an improvement.

## File Structure

- `app.py` (Task 1) — four optional `/api/track` fields + five `public_job` names.
- `job_runner.py` (Tasks 2, 5, 6) — `session_runs`, `maybe_start_stereo_fuse`,
  `run_stereo_fuse_job`, the fused-hits writer.
- `stereo_sync.py` (Tasks 3, 4) — **new**: CSV→`TrackSample` adapter and the offset refiner.
- `tests/test_stereo_sync.py` (Tasks 3, 4) — **new**.
- `tests/test_stereo_fuse.py` (Tasks 2, 5, 6) — **new**.
- `tests/test_pipeline.py` (Tasks 1, 7) — extended.

---

### Task 1: `/api/track` optional session inputs

**Files:** `app.py`. Test: `tests/test_pipeline.py`.

**Interfaces:**
- `session_id` — non-empty string, `""` → absent.
- `camera_role` — one of `{"a", "b"}` (matches `fuse_clips`' side naming); anything else
  is a 400.
- `peer_video_id` — non-empty string; the peer's `/api/upload` sha256.
- `sync_manifest_json` — JSON object. **Must short-circuit before `json.loads`** when
  absent: `json.loads("")` raises, which is exactly why `calibration_json` is effectively
  required today.

Copy the `event_engine`/`fusion_3d` idiom (app.py:803-811): `request.form.get(name,
"").strip()`, membership test, 400 otherwise, `"" → None` at the `create_job` call.
Persist via `extra_job_fields`. Write the manifest to `run_dir/sync_manifest.json`.

- [ ] **Step 1: Write the failing tests** — four byte-compat proofs (a request with none of
the fields yields a `job.json` whose key set is unchanged, and a `public_job` response
whose key set is unchanged); one 400 per new guard; one positive test asserting all four
land in `job.json` and the manifest file is written.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests** — the existing `tests/test_pipeline.py` track tests must pass
unmodified.
- [ ] **Step 5: Commit.**

---

### Task 2: Session discovery and the fuse trigger

**Files:** `job_runner.py`. Test: `tests/test_stereo_fuse.py`.

**Interfaces:**
```python
def session_runs(session_id):
    """Every completed run dir carrying this session_id, newest first per role.

    Globs ui_runs/*/job.json like /api/calibration/latest does. Skips stereo-*
    dirs so a fuse run can never be mistaken for a camera run.
    """

def maybe_start_stereo_fuse(session_id):
    """Start the fuse job iff both roles have completed and no one else claimed it."""
```

The claim must be atomic: `(_RUNS_DIR / f"stereo-{session_id}").mkdir(parents=True,
exist_ok=False)` — whoever wins the `FileExistsError` race starts the job, everyone else
returns. Two runs completing concurrently is the normal case, not an edge case.

Trigger site: immediately after the terminal `update_job(run_id, status="complete", ...)`
(job_runner.py:1081-1094). It must be wrapped so a fuse failure can never flip a completed
camera run to `failed`.

- [ ] **Step 1: Write the failing tests** — only one completes → no fuse dir and
`/api/track/status/stereo-<id>` 404s; both complete → exactly one fuse job; concurrent
triggers → one winner; peer failed → no fuse; duplicate roles → newest wins; unreadable
`job.json` tolerated; `stereo-*` dirs excluded; the trigger cannot fail a completed run.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement**, with `run_stereo_fuse_job` a stub that completes immediately.
- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit.**

---

### Task 3: `track_samples_from_csv`

**Files:** `stereo_sync.py`. Test: `tests/test_stereo_sync.py`.

```python
def track_samples_from_csv(csv_path, fps, offset_s=0.0):
    """ball_coordinates.csv -> [stereo_engine.TrackSample], time-sorted.

    Rows with no detection are skipped. offset_s is ADDED to every t_s.
    """
```

- [ ] **Step 1: Write the failing tests** — round-trip a synthetic CSV; assert sort order,
that undetected rows are skipped, that `offset_s` is applied, and that an all-undetected
CSV yields `[]` rather than raising.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit.**

---

### Task 4: The offset refiner — the heart of this phase

**Files:** `stereo_sync.py`. Test: `tests/test_stereo_sync.py`.

This is the only genuinely new algorithm in Phase 5, and it is **100% testable without
footage**. The spec asks for "offline offset refinement (reprojection-error minimization
over the full recording — expected to beat the live 2 ms budget)".

```python
def refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0,
                  coarse_range_s=0.05, coarse_step_s=0.002, fine_step_s=0.0001):
    """Search the offset that best explains both pixel tracks as one 3D ball.

    Cost: mean CLIPPED ray gap (feet) from stereo_engine.triangulate over a
    fixed timeline. Clipping bounds the influence of frames where one view lost
    the ball, so the cost stays a smooth function of the offset instead of a
    cliff. Returns a dict with the seed, the refined offset, the cost at both,
    an accept/reject verdict, and a half-width uncertainty proxy.
    """
```

Design points that make it testable and honest:
- **Fixed timeline.** The evaluation timeline is computed once from the seed overlap and
  held constant across every candidate offset — otherwise the cost compares different
  sample counts at different offsets and the minimum is an artefact.
- **Clipped cost, not raw mean.** A single frame where one camera lost the ball produces a
  huge gap that dominates a raw mean.
- **Two-stage grid, not gradient descent.** Deterministic, dependency-free, and its
  evaluation count is reportable. No scipy.
- **Reject rather than trust.** If the refined cost does not improve on the seed, or the
  argmin sits on the edge of the search range (`argmin_interior == False`), keep the seed
  and say so in `sync_report.json`.
- **`half_width_s`** = the δ range where `cost ≤ 1.1 × cost_min`. An uncertainty proxy with
  no distributional assumptions.

- [ ] **Step 1: Write the failing tests.** Synthesise a known 3D trajectory, project it
through a `make_fin_pair` camera pair, shift one stream by a known offset, and assert:
recovery to within `fine_step_s`; bit-for-bit determinism across repeated runs; rejection
when the tracks are unrelated noise; no bias at zero offset; graceful monotone degradation
as pixel noise grows; `argmin_interior` false when the true offset is outside the range;
and that an empty overlap returns a rejected verdict rather than raising.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit.**

---

### Task 5: The fuse job body, `stereo_track.jsonl`, `sync_report.json`

**Files:** `job_runner.py`. Test: `tests/test_stereo_fuse.py`.

Solve both cameras from their calibrations, normalise frame space, seed the offset from
the manifest, refine (Task 4), build the dense timeline, call `build_track3d` **once** and
pass the result into `detect_impacts(track=...)` so the track written to disk is provably
the track the impacts came from.

`stereo_track.jsonl` — one object per line, `json.dumps(record, sort_keys=True)`:
```json
{"gap_ft": 0.0412, "point_ft": [12.31, 4.02, 6.88], "t_s": 3.14166, "view_count": 2}
```

`sync_report.json` — schema `stereo-sync-v1`, carrying `seed`, `refined`, `cost`,
`search`, `timeline`, `half_width_s`, `sample_counts`, `pair_agreement` (reusing
`stereo_engine.pair_agreement`, the same non-tautological cross-check
`/api/camera-pair-check` runs) and both camera models.

**Offset convention, stated once and carried in the file:** *seconds added to camera_role
`b`'s timestamps to place both tracks on camera_role `a`'s clock.*

- [ ] **Step 1: Write the failing tests** — end-to-end fuse over synthetic CSVs; a camera
solve failure fails the fuse job with a named status and leaves both camera runs
untouched; mismatched frame sizes are scaled; an unknown frame size fails by name; the
JSONL is well-formed; the written track equals the track fed to `detect_impacts`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit.**

---

### Task 6: Fused `detected_hits.json`

**Files:** `job_runner.py`. Test: `tests/test_stereo_fuse.py`.

**The fused file speaks the `judge_hits` vocabulary**, with stereo values alongside as
additive fields. This is the sharpest decision in the phase: `build_eval_set` filters with
`isinstance(h.get("frame"), int)` (build_eval_set.py:163, :195), so a float `frame` yields
*zero* matches and the missed-bounce axis silently reads as total detector failure.

| Stereo | Hit entry | Rule |
|---|---|---|
| `t_s` | `timestamp_seconds` | verbatim, role-a clock |
| `t_s` | `frame` | `int(round(t_s * fps_a))` — role-a's fps, so the frame joins role-a's CSV and labels |
| `surface="front_wall"` | `event_type="wall"` | |
| `surface="left/right/back_wall"` | `event_type="side_wall"` | |
| `surface="floor"` | `event_type="floor"` | |
| `call="in"/"out"` | `call="IN"/"OUT"` | |
| `call="down"` | `call="OUT"` | tin hit; the mono vocabulary has no `DOWN`. **Deliberate.** |
| `call="bounce"` | `call=None` | floor bounces are unjudged in the mono schema too |
| — | `margin_px` | **`None`** — a stereo call has no pixel margin, and the schema permits it |

Additive: `view_count` (1 or 2), `method`, and a nested `stereo` block preserving the
native vocabulary losslessly.

`method`/`view_count` derive from the existing `confidence` string, so **`stereo_engine`
needs no change**:

| `confidence` | `method` | `view_count` |
|---|---|---|
| `"high"` | `plane_snap` | 2 |
| `"one_view"` | `plane_snap` | 1 |
| `"no_call"` | `triangulated` | 2 |

**`method: "estimated"` is declared in the spec's enum but unreachable in this
implementation** — nothing in `detect_impacts` produces a point without either a snap or a
triangulation. Say so; do not manufacture a third bucket.

**Known coverage gap:** stereo has no racket surface, so a fused run never produces
`event_type="racket"`. The mono classifier does.

Keeping the coaching analytics alive: compute `wall_diagram` and `target_zone` from the
front-wall 3D point (`x/21`, `(15 − z)/(15 − 19/12)`, matching job_runner.py:765) and reuse
`assign_front_wall_hit_players`, `build_target_zone_summary` and
`build_player_target_zone_summaries` verbatim.

- [ ] **Step 1: Write the failing tests** — every row of the translation table;
hand-computed `wall_diagram` geometry for a known 3D point; the fused file round-tripped
through `front_wall_hits_from_payload`, `build_coaching_analytics` and
`find_hit_impact_near_frame`; and a pin that the fuse `job.json` has **no `video_path`**, so
`index_runs_by_video_sha` cannot displace a mono run in the eval set.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.**
- [ ] **Step 5: Commit.**

---

### Task 7: Single-camera regression seal

**Files:** `tests/test_pipeline.py`.

- [ ] **Step 1:** A `job.json` key-set freeze, a `public_job` response-shape freeze, and a
`ball_coordinates.csv` column freeze — each asserting the no-stereo path is unchanged.
- [ ] **Step 2:** Update `requirements-test.txt`'s stale "verified, 117 passing" comment to
the real count.
- [ ] **Step 3: Commit.**

---

### Task 8: Documentation and the honest-state gate

**Files:** `docs/`, `README.md`.

- [ ] **Step 1:** Record the decisions this plan makes that the spec left open: CSV-not-video
as the fuse input; the dense-window timeline instead of literally "the full recording";
`method: "estimated"` declared but unreachable; `"down" → "OUT"`; no stereo `racket`
events; the offset sign convention; and that per-device readout time is carried in the
manifest but **consumed by nothing** — rolling-shutter correction is out of scope for a
first cut rather than half-implemented.
- [ ] **Step 2:** State plainly that no paired labeled footage exists, that the eval pass is
therefore not run, and that the fused path must not be described as an improvement until
it is. Include the recording + labeling checklist such a clip would need.
- [ ] **Step 3: Commit.**

---

## User-owned checklist (needs hardware)

- A two-phone synchronized recording of the same rallies, both phones mounted in the same
  orientation, each with its own 4K calibration.
- Wall-hit frames labeled on the **role-a** clip, so the existing `video_sha` join works.
- Then, and only then: `stereo_fuse` output through the eval harness vs. the monocular
  baseline, and the `--fail-under` gate extended past IN/OUT accuracy (today it is computed
  from n=2 corrections and skipped entirely when accuracy is `None` — it cannot detect a
  missed-bounce or position regression on the fused path).

## Self-review notes

- **Spec coverage:** optional `/api/track` fields (Task 1), `stereo_fuse` triggered on both
  runs completing (Task 2), offline offset refinement (Task 4), `stereo_track.jsonl` +
  fused `detected_hits.json` with `view_count`/`method` + `sync_report.json` (Tasks 5-6),
  additive-only fields with existing consumers proven (Tasks 6-7). The eval pass is the one
  spec item deliberately **not** delivered, for the reason above.
- **Biggest risk:** the fused `detected_hits.json` displacing a mono run in the eval set.
  `index_runs_by_video_sha` keeps one run per `video_sha` by `(len(hits), run_dir.name)`, so
  a fuse run sharing role-a's sha with enough hits would silently rebaseline the label-CSV
  axis. Mitigation: the fuse `job.json` deliberately omits `video_path`, so
  `video_sha_from_path` yields `None` and it cannot join. Task 6 pins that.
- **Deliberately deferred:** rolling-shutter correction, any UI surface for a fuse run
  (none is specified, and any would have to go through DESIGN.md), and re-tracking at
  stride 1 if real footage proves the CSV too sparse.
