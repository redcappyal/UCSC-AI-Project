# Player Detection & Identification — Design Spec

**Date:** 2026-07-27
**Status:** Approved design (brainstormed with Ian 2026-07-27); implementation plan to follow
**Audience:** an autonomous implementation session with full codebase access and zero
conversation context. Everything it needs is in this spec, `CLAUDE.md`, and the code.

## 1. Goal

Add real, observed player attribution to the analysis pipeline:

1. **Detection** — find both players in the video (person boxes + pose keypoints).
2. **Identification** — know which hits, rallies, and (later) movement belong to which
   *named* player, so a user can filter every statistic to their own actions — e.g. the
   front-wall impact positions of only the shots *they* hit.

This replaces today's fictional attribution: `player_number` is currently
serve-alternation arithmetic seeded from an env var
(`PLAYER_ASSIGNMENT_FIRST_SERVER`, `job_runner.py:578-584`), with rally servers
propagated by judge-dependent winner inference (`job_runner.py:606-620`). The whole
coach report builds on it (`app.py:427-467`). Nothing in the repo observes a player.

## 2. Relationship to the tripod match-analysis plan

`docs/superpowers/specs/2026-07-27-tripod-match-analysis-design.md` (Phase 5, Tasks
12-14 of its plan) designed a person-detector seam with a motion-blob fallback and
movement stats. **This spec supersedes that Phase 5 design** as a standalone vertical
slice built against what works today (existing rally segmentation from front-wall
hits; no dependence on the tripod plan's Tasks 1-11). Differences from that Phase 5:

- Real detector from day one (RF-DETR Keypoint) instead of a motion-blob fallback;
  the "weights are a human gate" framing is dropped.
- Identification (names, serve anchoring, winner chain) is in scope; movement stats
  are **out** of scope here and become the natural next slice on top of this one.
- When the tripod plan is executed later, its Tasks 12-14 must be reconciled against
  what this slice ships (the detector seam and tracker land here first).

## 3. Decisions already made (do not relitigate)

| Decision | Choice | Why |
|---|---|---|
| Detector | **RF-DETR Keypoint** (`rfdetr` pip package, pinned; `RFDETRKeypointPreview`-class COCO-pretrained checkpoint, 17 body keypoints) | Apache-2.0 core (fits the repo's permissive-runtime policy — Ultralytics YOLO26-pose was rejected: AGPL-3.0 network-use clause would encumber the hosted Flask service); beats YOLO26-pose on COCO keypoints; same Roboflow family as existing tooling |
| Execution | **Full-video pass** riding the existing coarse decode (option A; windowed-post-pass B and per-hit-verify C were considered and deferred) | Future-proofs movement stats (next slice needs full-video tracks); coarse pass already decodes every `frame_stride`-th frame |
| Identity maintenance | **Per-rally re-anchoring** via serve observation + within-rally alternation | Within a rally, hits strictly alternate (squash rule) — per-hit identity is unnecessary; identity only has to survive rally boundaries |
| Name carry across rallies | **Track continuity through breaks, zero human input** (v1) | Ian's explicit call: no confirmation prompts in the first iteration; confidence data is collected silently for later debugging |
| Winner inference | **Winner of rally N = observed server of rally N+1** (squash rule: winner serves next) | Inverts the dependency — serve observation *produces* the winner chain instead of judge-based winner inference producing identity; score progression falls out for free |
| Naming | **Post-hoc labeling on the results page** — pipeline is name-agnostic (tracks A/B); user types names once and matches them to a crop of the rally-1 server | Analysis never blocks on human input; relabeling never re-runs analysis |
| v1 outputs | Hit attribution + winner chain/score progression | Movement stats and pose-derived stats explicitly deferred |

## 4. Architecture

```
coarse decode (frame_stride, existing)          [job_runner.track_segments :183]
        │
        ├── ball inference (existing, unchanged)
        └── person inference every PERSON_DETECT_EVERY coarse frames   [person_model.py]
                    │ PersonDetection(x, y, w, h, confidence, keypoints)
                    ▼
        two-player tracker: anonymous tracks A / B                     [player_tracker.py]
                    │ TrackSample(t_s, frame_idx, foot_px, bbox, confidence)
                    ▼
hits → judge → rally segmentation (existing)    [job_runner.assign_front_wall_hit_players :560]
                    │
                    ▼
        serve attribution + alternation + winner chain                 [player_attribution.py]
                    │  per-rally server track, per-hit player_track,
                    │  winner chain, score progression, identity confidence
                    ▼
        payload: hits gain player_track; players_v1 block; serve crop saved
                    │
                    ▼
        results page: type names → "who is this?" (rally-1 server crop)
        → player_names stored as run metadata → per-player stat filtering
```

### 4.1 `person_model.py` — detector seam

- `rfdetr` is a **new pinned dependency in `requirements.txt`, lazy-imported** inside
  the loader (pattern: `yolo_model_eval.py` / `train_yolo_ball.py` lazy `ultralytics`
  imports). The test suite must never import it; CI stays light.
- Public surface (final signatures are the plan's to pin, this is the shape):

  ```python
  PERSON_SCHEMA_VERSION = "person-model-v1"

  @dataclass(frozen=True)
  class PersonDetection:
      x: float; y: float; width: float; height: float
      confidence: float
      keypoints: tuple  # 17 COCO keypoints as (x, y, confidence), kept from day one

  def available_backend() -> str        # "rfdetr" iff import+weights succeed, else "none"
  def load_person_detector(...)         # .detect(frame_bgr) -> list[PersonDetection], .backend
  ```

- Keypoints are stored even though v1 consumes only boxes: ankles are the true foot
  point for the movement-stats slice, wrists the swing-arm signal for later
  hitter verification. Do not strip them to "simplify".
- Weights/checkpoint provisioning, version pinning, and offline behavior are
  documented in `docs/PERSON_MODEL.md` (what downloads on first use, where it is
  cached, how to pre-provision on the server).
- **Fallback honesty:** if the backend is `"none"` (rfdetr not installed, weights
  unavailable), the pipeline runs exactly as today — env-var alternation — and the
  payload marks `attribution_backend: "assumed"`. Never silently.

### 4.2 Person pass placement

- Rides the **coarse pass** consumer loop (`track_segments`, `job_runner.py:183-231`):
  the producer already decodes full-resolution BGR frames at `frame_stride`
  (default 4 → 15 Hz at 60 fps, `app.py:1095`). `track_segments` gains an optional
  frame observer seam (`on_frame` currently receives only `frame_idx`,
  `job_runner.py:213`; the observer needs the frame itself).
- `PERSON_DETECT_EVERY` (named module-level constant): run person inference every
  N-th *coarse* frame, chosen so detection cadence lands ≥ ~4 Hz of video time at
  default settings. Person inference must not run in refine/rescue passes (those
  re-visit windows; the observer is coarse-pass-only).
- Perf note: RF-DETR adds real per-frame cost on MPS. The coarse pass is the
  throughput-critical path, so the plan must include a measured before/after wall-clock
  number on the golden clip (`SquashAnalytics.mp4`) and `PERSON_DETECT_EVERY` is the
  tuning knob. No hard perf gate for v1 — it's an offline batch job — but the number
  gets reported honestly in the PR.

### 4.3 `player_tracker.py` — two anonymous tracks

- Pure logic, no cv2/torch imports. Consumes timestamped `PersonDetection` lists,
  maintains **exactly two tracks, "A" and "B"** (top-2 detections by confidence/area
  when more people are visible — spectators through glass are real).
- Greedy min-cost assignment on pixel distance (foot point = bbox bottom-center for
  tracking; keypoint ankles are carried on samples but not load-bearing in v1).
  Unmatched tracks coast at last position up to `COAST_MAX_S`, then stop emitting
  (no fabrication).
- **Ambiguity accounting:** an assignment is ambiguous when the two possible pairings'
  costs are within a named ratio threshold. Ambiguous events are timestamped so
  attribution can compute per-rally identity confidence. Collected silently — v1
  surfaces no prompt and no UI warning (Ian's call: debug later).

### 4.4 `player_attribution.py` — serve + alternation + winner chain

Pure logic. Consumes: judged hits with rally segmentation (existing
`segment_front_wall_hits_into_rallies`, `job_runner.py:529`), track samples, ball
coordinates (existing CSV rows). Produces per-hit and per-rally attribution.

- **Serve attribution:** for each rally, the server is the track nearest the ball in
  a small window just before the rally's first front-wall hit (`SERVE_LOOKBACK_S`,
  named constant; ball positions come from the existing track rows around
  `hit_frame`). Raw pixels — **no homography, no calibration dependency anywhere in
  this slice.**
- **Within-rally attribution:** strict alternation from the serve (squash rule —
  a rule, not an inference). Every front-wall hit gets `player_track: "A" | "B"`.
- **Winner chain:** winner of rally N := observed server of rally N+1 (squash rule:
  winner serves next), `winner_source: "next_serve"`. The final rally has no
  successor: it falls back to the existing last-hitter logic labeled
  `winner_source: "est"`, or `winner: null` if that logic can't decide. For all
  other rallies the existing winner inference (`job_runner.py:606-620`) is retained
  as a **silent cross-check**: disagreement between observed and inferred winners is
  recorded per rally (debug signal), never shown as truth.
- **Score progression:** running tally from the winner chain (PAR semantics: rally
  winner scores). Emitted per rally.
- **Integration point:** `assign_front_wall_hit_players` (`job_runner.py:560`,
  called at `:1085`) is re-seeded from observation when the backend is `"rfdetr"`:
  per-rally `server_player_number` comes from the observed serve (track A ↦ player 1,
  B ↦ player 2 by convention: **player 1 := the rally-1 server**), alternation
  within rallies unchanged. Existing payload keys keep their shapes and meanings;
  observed mode changes *values* (that is the point), and `attribution_backend`
  says which mode produced them.

### 4.5 Payload + persistence

- Hit rows gain `player_track` (additive; existing keys untouched in shape).
- New top-level `players_v1` block in the job payload (additive):

  ```json
  {
    "attribution_backend": "observed" | "assumed",
    "detector_backend": "rfdetr" | "none",
    "rallies": [
      {"rally_number": 1, "server_track": "A", "winner_track": "B",
       "winner_source": "next_serve" | "est" | null,
       "score_after": {"A": 0, "B": 1},
       "identity_confidence": 0.94,
       "winner_crosscheck_agrees": true}
    ],
    "serve_crop": "players/serve_rally1.jpg",
    "player_names": {"A": null, "B": null}
  }
  ```

- **Serve crop:** after attribution, re-open the video (single seek), grab the
  rally-1 serve frame, save the server's bbox crop (padded) under the run dir.
  ASCII filename (CLAUDE.md rule); served from the run dir by a small GET route,
  same access model as existing run artifacts.
- **Naming:** `POST /api/runs/<run_id>/players` stores
  `{"A": "<name>", "B": "<name>"}` in the run dir (route style `@app.post`,
  `app.py` convention). Pure metadata: re-POSTing relabels; analysis never re-runs.
  `GET` of the run payload returns names merged into `players_v1`.

### 4.6 Results-page UX (web only; iOS untouched)

On the existing clip results page in `index.html` (DESIGN.md is binding; both themes
at a phone viewport via `/verify`):

1. A "Players" card appears when `players_v1` is present with backend `"observed"`:
   two name inputs (Player A / Player B placeholder), the rally-1 serve crop, and the
   question "Who is this?" answered by tapping one of the two entered names.
2. On save, per-player stat surfaces that already exist
   (`target_zones_by_player`, coach report Player 1/Player 2 sections) show real
   names instead of "Player 1"/"Player 2", and hit-level views can filter to one
   player's shots (front-wall impact positions of only your hits — the headline
   user-facing win of this slice).
3. Backend `"assumed"` → the card explains attribution is assumed alternation
   (honest capability card pattern from the tripod spec), names still enterable.

## 5. Testing & eval

- Every new module gets its paired `tests/test_<module>.py` (PostToolUse hook runs
  the pair on edit; failures block).
  - `person_model`: backend reports `"none"` without rfdetr; manifest/contract
    validation; **no rfdetr import at test time** (stub/monkeypatch).
  - `player_tracker`: scripted detections — separated walkers keep identity;
    dropout → coast → reacquire; crossing → ambiguity recorded.
  - `player_attribution`: synthetic rallies — serve attribution from scripted ball +
    track positions; alternation; winner chain incl. final-rally `null`; score
    progression; disagreement-with-est cross-check recorded.
- Integration: extend `tests/test_job_runner.py` with a stubbed detector
  (monkeypatch `load_person_detector`) proving `players_v1` lands in the completion
  payload with correct backend labels, and that no-detector runs are byte-identical
  to today's behavior apart from `attribution_backend: "assumed"`.
- `/eval` (line-call judge): this slice must not move judged outputs — run the eval
  once and state zero drift in the PR.
- **New eval axis — serve attribution:** hand-label the true server of each rally in
  `SquashAnalytics.mp4` (1080p60 golden clip), score observed serve attribution
  against it, commit `eval_set/BASELINE-ATTRIBUTION-<date>.md`. This is the number
  that gates any future claim that identification "improved". Target for v1:
  stated honestly, whatever it measures — no minimum gate; the baseline is the
  deliverable.
- Full suite green (`.venv/bin/python -m pytest tests/ -q`, currently
  "283 passed, 1 deselected") at every commit; rfdetr must not be needed to run it.

## 6. Explicitly out of scope (v1)

- Movement stats, heatmaps, distance/speed/T-time (next slice; this slice
  deliberately builds the tracks it will need).
- Pose-derived stats (lunge depth, racket side) — keypoints are stored, not consumed.
- Per-hit verification windows (architecture option C) — named follow-up, adds a
  per-rally confidence upgrade path.
- Appearance re-ID / kit-color models; any human confirmation prompts mid-analysis.
- Floor homography / calibration dependencies of any kind.
- iOS changes; any change to judged line-call outputs; touching archived code.
- More-than-two-player scenarios (doubles) — top-2 tracks by design.

## 7. Risks

- **RF-DETR Keypoint is "Preview"-branded.** Pin the `rfdetr` version; document the
  checkpoint hash in `docs/PERSON_MODEL.md`. If the preview API shifts, the seam
  isolates the change to `person_model.py`.
- **Track continuity through breaks is the identity carrier** and breaks include
  players crossing to towel/door. v1 accepts this (Ian's explicit call — zero human
  input, debug later); ambiguity events + winner cross-check are recorded from day
  one so the failure mode is measurable when we return to it.
- **MPS throughput.** Person inference rides the throughput-critical coarse loop;
  `PERSON_DETECT_EVERY` is the knob and the golden-clip wall-clock delta gets
  reported in the PR.
- **Serve-lookback attribution** assumes the ball track exists shortly before the
  first front-wall hit of a rally. Detection recall (~35% at rally scale) means some
  rallies will have no usable pre-hit ball positions → those rallies get
  `server_track: null` and are excluded from the winner chain (honest gap, surfaced
  in `players_v1`), not guessed.
