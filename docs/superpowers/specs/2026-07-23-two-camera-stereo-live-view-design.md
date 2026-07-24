# Two-Camera Stereo Live View — Design

**Date:** 2026-07-23
**Status:** Approved in conversation (Ian), pending written-spec review
**Team split:** Ian builds this (two-phone stereo live view). Alvin continues the
single-camera cloud pipeline; every server-side change here is additive so his path keeps
working untouched.

## Decision log (rulings that shaped this design)

1. **True stereo is the core, non-negotiable** (Ian). The justification for a second
   phone is a continuous, measured 3D model of the match — shot types, shot quality,
   pattern analysis — not just line calls. Monocular-plus-gravity fitting stays Alvin's
   lane; this path gets 3D by triangulation, not by model assumptions.
2. **Bluetooth is an acceptable transport candidate** (Ian). Phones mount ~7 ft up on the
   two glass fins flanking the back-wall door; the phone-to-phone line of sight runs
   above head height along the back wall and players never cross it mid-rally. Remaining
   BLE constraints are latency-shaped (iOS ~15 ms min connection interval + delivery
   batching, ~1 ms clock-sync floor), not range-shaped. Transport goes behind an
   abstraction; BLE and peer-to-peer Wi-Fi are spiked side by side and measurements
   decide.
3. **Main-wide lens (1×)** (Ian). Favors stereo depth accuracy (~2× finer pixels than
   ultrawide) at a quantified coverage cost near the back wall (§ Accuracy & coverage).
   This supersedes, for two-camera sessions only, the single-camera capture norm from the
   3D-contact-detection spec (ultrawide 4K60 on the center back-wall mount).
4. **Share detections, not video.** The peer link carries ball observations and control
   messages (a few KB/s), never live video. Each phone records locally and uploads
   afterward exactly as today. (A low-res cross-preview stream is a deferred nice-to-have
   and is the only feature that would ever require the Wi-Fi path.)
5. **No MultipeerConnectivity.** Live iOS 26 connection regression; Apple's own engineers
   steer developers to Network.framework / Wi-Fi Aware. We use NWConnection/NWListener
   (`includePeerToPeer`) for the Wi-Fi candidate and CoreBluetooth for the BLE candidate.

## Problem

One fixed camera cannot measure depth; the current pipeline recovers it offline with a
gravity-constrained monocular fit (ballistic.py) — cloud-only, assumption-laden, and
unavailable in real time. Meanwhile a squash match already has two players with two
phones, and every court has two fins flanking the door — a natural stereo mount. The
product goal: pair the two phones on-court, reconstruct the ball's flight in 3D live,
mark calls in real time, and upload both clips as a paired session for deeper cloud
analysis (shot classification, quality, patterns). Secondary product effect: the second
player must install the app to play — built-in distribution.

## Goals

- Pair two phones on-court with no Wi-Fi infrastructure and no account dependency.
- Continuous live 3D ball track in the existing court frame (feet, z-up, origin at the
  front-left floor corner — same frame `court_model.py` solves poses in).
- Real-time line calls: plane-snapped impact estimates, call + confidence, rendered on
  both phones (optionally announced audibly — open product choice).
- Post-rally 3D mini-court replay on both phones (the "wow" moment that justifies
  phone #2 to users).
- Both phones record and upload as today; a paired session links the two runs plus sync
  metadata so the cloud can re-fuse offline at higher fidelity.

## Non-goals

- Player 3D tracking (cloud-later; not in the live path or this spec's implementation).
- Live video streaming between phones (deferred preview feature only).
- Shot-classification models themselves (this spec delivers the 3D track they consume;
  classifiers are a follow-on).
- One-phone-two-lens capture (AVCaptureMultiCamSession wide+ultrawide) — noted as a
  future coverage option, out of scope.
- Back-wall contact calls (both cameras are on the back wall; unchanged gap).

## Architecture overview

```
  Phone A (primary)                       Phone B (secondary)
  ┌─────────────────────────┐             ┌─────────────────────────┐
  │ CameraController 1080p60 │             │ CameraController 1080p60 │
  │  └─ CoreMLBallDetector   │             │  └─ CoreMLBallDetector   │
  │  └─ AVAssetWriter (.mp4) │             │  └─ AVAssetWriter (.mp4) │
  │ BallTracker ──┐          │  detections │          ┌── BallTracker │
  │               ▼          │◄────────────┤          ▼               │
  │        StereoEngine      │  (x,y,t,c)  │   (local trail only)     │
  │  interpolate → triangulate│            │                          │
  │  → 3D track → impacts    │   events    │                          │
  │  → plane-snap → calls ───┼────────────►│  mirror UI               │
  │ PeerSession + ClockSync  │◄──sync─────►│ PeerSession + ClockSync  │
  └─────────────────────────┘             └─────────────────────────┘
         │ upload run A + session manifest        │ upload run B
         ▼                                        ▼
  Server: two ordinary runs + pairing manifest → stereo_fuse job (offline
  offset refinement, refined 3D track, fused calls) → analytics consumers
```

One stereo brain: the **primary** phone (pairing initiator) runs StereoEngine on its own
detections plus the secondary's stream, and pushes rendered events back for the
secondary's mirror UI. The secondary never fuses. With no peer connected, the app is
exactly today's single-camera app — pairing strictly adds capability (graceful
degradation is a hard requirement).

## Component 1 — PeerSession + transport abstraction

New: `ios/Sources/Peer/PeerSession.swift`, `PeerTransport.swift` (protocol),
`BLETransport.swift`, `WiFiP2PTransport.swift`, `PeerMessages.swift` (codec).

- `PeerTransport` protocol: `send(control:)` (reliable, ordered), `send(datagram:)`
  (lossy, latest-wins), connection state publisher, measured RTT statistics surface (the
  spike instrumentation is a first-class API, not scaffolding).
- **BLE candidate:** CoreBluetooth, primary as central. Detection tuples batched — at a
  15 ms connection interval, 2–4 tuples per notification (≤ 128 B) sustains 60–120 Hz
  production comfortably; adds ~15–30 ms delivery latency, acceptable for marking UX.
- **Wi-Fi candidate:** NWListener/NWBrowser + Bonjour, `includePeerToPeer = true`, UDP
  datagrams + one TCP control connection. Requires `NSLocalNetworkUsageDescription`,
  `NSBonjourServices`, and the `com.apple.developer.networking.multicast` entitlement —
  **manually approved by Apple; the request is filed at project start** (it gates only
  this candidate, not BLE). Known AWDL behavior: 50–200 ms duty-cycle stalls every
  1–12 s; the stream design buffers through them (sequence numbers, no retransmit of
  stale tuples).
- **Selection gate (end of Phase 1):** run both transports on the actual mounts; record
  median/p95/max one-way latency, loss, and sync-handshake RTT symmetry over ≥ 10 min
  with players on court. Pick the primary transport on data; keep the other as fallback
  behind the same protocol.
- Pairing UX: "Pair second camera" on both phones → discovery → 4-digit code confirm →
  roles fixed (initiator = primary). State machine modeled on `RunSubmission.Phase`
  (`idle → discovering → confirming → syncing → ready → live → degraded → ended`).
  `degraded` (link lost mid-rally) keeps recording locally and shows a visible banner —
  never silently downgrade; recording never depends on the link.

Wire protocol (versioned, little-endian binary for datagrams, JSON for control):

- `DetectionTuple` datagram: `seq:u32, pts_ns:u64` (sender's host clock),
  `x:f32, y:f32` (pixels in sender's 1920×1080 frame; y doubles as the rolling-shutter
  row), `conf:f16, bbox_h:f16`. 24 B; batched.
- Control messages: `hello{proto_version, app_version, device_model}`,
  `role{primary|secondary}`, `calibration{profile_id, calibration_json}` (secondary →
  primary at session start), `sync_ping/sync_pong{t1,t2,t3}`, `record{start|stop, pts_ns}`,
  `event{rally_id, impact}` (primary → secondary, mirrors calls), `heartbeat`,
  `session_manifest{session_id, video_ids}` (exchanged at stop for upload pairing).
- Version skew: `hello` mismatch on `proto_version` → explicit "update app" state, no
  best-effort parsing.

## Component 2 — ClockSync

New: `ios/Sources/Peer/ClockSync.swift`. Output: an affine mapping
`t_secondary → t_primary` with a live uncertainty estimate, consumed by StereoEngine and
written into the session manifest for the cloud.

1. **Network offset:** NTP-style 4-timestamp exchange over the active transport, 20–50
   rounds at pairing, min-RTT filtered (discards AWDL-stall / BLE-batching outliers);
   light re-sync every 10–30 s tracks drift (10–20 ppm ⇒ < 0.1 ms between re-syncs).
   Monotonic clocks only (`mach_absolute_time` domain — the same clock that stamps
   `CMSampleBuffer` PTS).
2. **Acoustic anchor (bias correction):** at pairing, the user claps once from the T.
   The fins are symmetric about the court centerline, so the sound path to the two mics
   is equal by construction and the speed-of-sound asymmetry term cancels; each phone
   localizes the clap on its own audio track (audio and video share the session host
   clock — Apple QA1643) to ~0.02 ms resolution. This nulls the network estimator's
   asymmetric-path bias to well under 1 ms. (A flash event was rejected: the cameras
   face away from each other and share no view of a torch.)
3. **Continuous self-check:** ball impacts are sharp, co-observed events; StereoEngine
   periodically re-solves the residual offset by minimizing cross-camera reprojection
   error over recent rally segments (motion-based self-calibration). Live it acts as a
   drift alarm; in the cloud it becomes the refinement step.
4. **Per-detection timing:** `t_effective = PTS + (row / 1080) × readout_time`.
   `readout_time` measured once per device model at the capture mode (flicker-LED bench
   test, Phase 2 task; iPhone 15 Pro rear ≈ 5 ms at 25 fps is the starting estimate; no
   published 60 fps figure exists). Whether PTS marks exposure start or midpoint is
   undocumented — pinned down empirically in the same bench task.

**Error budget (target, live):** ≤ 2 ms effective cross-camera timing error (RSS of
residual offset ~1 ms, drift ~0.1 ms, PTS jitter ~0.3–0.5 ms, rolling-shutter residual
~0.5 ms, interpolation fit ~0.5–1 ms). At 30–60 m/s that is ~6–12 cm of along-track
error in the live 3D model — invisible for shot analytics, and removed at impact points
by plane-snap. The budget is asserted by the Phase 2 bench (clap-anchor vs. network
estimate cross-check), not assumed.

## Component 3 — Calibration pairing

Each phone is calibrated once per fin with the existing wizard (fixed mounts make
profiles durable). Changes are additive to calibration-v2:

- New optional fields: `camera_id` (e.g. `"<court>-left-fin"`), `session_id`,
  `baseline_partner_id`. `court_model.solve_camera_model()` is untouched — it already
  yields the full pose + `projection_matrix()` per camera in the shared court frame,
  which is the entire geometric enabler of this project.
- `/api/calibration/latest` grows a `?camera_id=` filter (unfiltered behavior unchanged
  for Alvin's path and the current iOS fetch).
- **Cross-camera agreement gate** (extends the calibration health check). *Amended
  2026-07-24 during Plan B1: the original formulation (triangulate synthetic projections
  of known landmarks) is tautological — `project` and `ray` are exact per-model
  inverses, so it cannot detect a biased solve.* The implemented gate triangulates each
  camera's ray through its own **observed** calibration pixels for landmarks shared by
  both calibrations, measures 3D error vs. the known court positions (gate ≤ 0.1 ft
  median), and independently checks a pose-plausibility envelope (baseline 1.5–10 ft,
  camera heights 3–12 ft, both mounts near the back wall). Known limitation, by
  design: a self-consistent but physically-wrong tap set is unobservable at
  calibration time; the runtime triangulation gap on live ball detections (StereoEngine
  reports it per track point) is the catch for that class. Baseline distance between
  solved camera centers is reported (measured, not assumed).
- iOS: the wizard stays web-based (unchanged); each phone stores its own fin's profile
  locally and sends it to the primary in the `calibration` control message at pairing.

## Component 4 — StereoEngine

New: `ios/Sources/Stereo/StereoEngine.swift` (+ pure-math `StereoMath.swift`), with a
line-for-line Python mirror `stereo_engine.py` used by the cloud fuse job and by tests
(golden vectors shared between both, same pattern as existing geometry parity tests).

Runs on the primary, fed by both detection streams. Per rally:

1. **Track association:** gate raw detections per camera (confidence, motion continuity)
   into per-camera tracklets — reusing the coarse gating logic of `BallTracker`, which
   stays UI-facing and untouched; StereoEngine subscribes via the existing
   `BallTracker.subscribe` seam.
2. **Temporal interpolation (mandatory — never nearest-frame):** fit a short local
   trajectory (quadratic, gravity-informed) to each camera's tracklet and evaluate both
   at a common 120 Hz timeline using the ClockSync mapping. Nearest-frame pairing at
   60 fps would alone contribute up to ±8.3 ms ⇒ ±25–50 cm; interpolation reduces this
   to the fit residual (< 1 ms equivalent).
3. **Triangulation:** two-ray closest-approach (equivalently DLT) through the two
   projection matrices → continuous 3D track + per-point covariance + ray-gap residual.
   A large closest-approach gap flags association errors (two different objects) —
   drop, don't fuse.
4. **Impact detection:** velocity-discontinuity detection on the 3D track, classified by
   proximity to the court surfaces (floor z=0, front wall y=0, side walls x=0|21 ft,
   tin/out lines from `court_model.py` constants). Replaces image-space heuristics with
   the 3D-geometric test the 3D-contact spec wanted — here it's measured, not fitted.
5. **Plane-snap for calls:** at an impact frame, re-estimate the impact point by
   intersecting each camera's ray with the identified surface plane and inverse-variance
   fusing the two — this collapses the weak stereo depth axis and yields the call-grade
   estimate (§ Accuracy). Call = signed distance to the relevant line; confidence folds
   in ray residual, per-camera reprojection health, sync uncertainty × ball speed, and
   view count.
6. **Degradation policy:** one-view impacts (occlusion, blind zone) fall back to
   single-ray plane-snap with a lowered confidence tier; no-view impacts near a line
   produce an explicit **no-call** state, never a guess. Occlusion is structural here —
   both cameras sit at the back wall, so a mid-court player often blocks both on exactly
   the low targets that decide rallies (tin, floor). The honest UI states are
   `called (high) / called (one-view) / no-call (obstructed)`.
7. **Blind-zone handling (main-wide consequence):** deep-court floor bounces (see
   coverage numbers below) are reconstructed by ballistic extrapolation of the visible
   flight into the floor plane and flagged `estimated`; they feed length analytics, not
   line calls (no floor lines live back there except deep service-box/half-court lines —
   serve-depth calls are explicitly `estimated` in v1).

## Component 5 — Live experience (UI)

New phases per DESIGN.md §17's extension process (`p-pair`, and `p-live` graduates from
roadmap placeholder to real screen). The DESIGN.md additions land in the same change as
the UI implementation, per CLAUDE.md. Constraints honored: single-section phase shell,
one accent, no new nav chrome, 44 pt targets; no dual-video layout is needed anywhere
(detections travel, video doesn't), so no DESIGN.md never-do is challenged.

- `p-pair`: discovery list → code confirm → clap-sync prompt ("stand at the T, clap
  once") → agreement-gate result → ready.
- `p-live`: existing preview + trail, plus stereo status chip (link, sync ms, fused fps)
  and call banners with the three confidence states.
- Post-rally: 3D mini-court replay (SceneKit or Canvas-projected court wireframe +
  trajectory ribbon) on both phones — primary computes, secondary mirrors.
- Capture changes: landscape orientation for mounted sessions (portrait lock and
  `OverlayView`'s hardcoded 1080×1920 content size become parameterized), 1080p60
  (`activeVideoMinFrameDuration` set explicitly; today's code never sets fps), main-wide
  lens as today. Thermal/battery at 60 fps + ANE + radio for a full match is a Phase 1
  bench measurement with a go/no-go on 60 vs 30 fps.

## Component 6 — Cloud paired runs

Additive server changes (Alvin's single-camera flow byte-compatible throughout):

- `/api/track` accepts optional `session_id`, `camera_role`, `peer_video_id`,
  `sync_manifest_json` (offset series, clap anchor, per-device readout time). Absent →
  exactly today's behavior.
- New `stereo_fuse` job (job_runner.py): triggered when both runs of a `session_id`
  complete. Inputs: both runs' server-side detection tracks + both calibrations + sync
  manifest. Steps: offline offset refinement (reprojection-error minimization over the
  full recording — expected to beat the live 2 ms budget), `stereo_engine.py` fusion,
  outputs `stereo_track.jsonl` (court-frame 3D track), fused `detected_hits.json`
  (additive fields: `view_count`, `method: triangulated|plane_snap|estimated`), and a
  `sync_report.json`. Existing consumers (index.html rendering, eval scripts) read the
  unchanged fields; new fields are additive.
- Per repo convention, the fused line-call path gets an eval pass (eval skill) against
  labeled clips before it may be called an improvement over the monocular baseline.

## Accuracy & coverage (main-wide, measured baseline ~1.0–2.0 m, mounts ~7 ft)

Numbers from the closed-form error model (rerunnable: `two_cam_error.py`, session
scratchpad; 2 px detection noise, 1σ; calibration residual inflates by 1.06–1.25×):

| Regime | Where | Error |
|---|---|---|
| Free triangulation (3D model) | floor bounces, mid-court | 2.4–3.1 cm depth |
| Free triangulation (3D model) | front-wall region | ≈13 cm depth at B=1.5 m (scales ~1/B: ~19 cm at 1.0 m, ~9.5 cm at 2.0 m) |
| Plane-snap (calls) | tin / front-wall out / service line | ~1.0 cm fused, 1.4 cm one-view |
| Plane-snap (calls) | side-wall out mid-court | 0.6–0.8 cm |
| Plane-snap (calls) | floor bounces (visible zone) | 0.6–1.2 cm |
| Timing (live, ≤2 ms budget) | along-track, 30–60 m/s | 6–12 cm (removed at snap) |

Plane-snap accuracy is nearly baseline-independent — exact fin spacing is a reporting
item, not a design constraint. Coverage limits of main-wide (VFOV 42.2°) from the fins
with the front-wall out-line kept in frame (max ~7° down-tilt): floor visible only
≥ ~13 ft (~3.95 m) from the back wall — the rear ~40% of the floor is below the frustum;
side-wall out-lines visible only from ~12 ft (~3.6 m) in front of the back wall,
forward. Ball in flight above that
frustum edge remains visible everywhere. Consequences accepted with the main-wide
ruling: deep floor bounces are `estimated` (ballistic + floor plane), serve-depth calls
are `estimated`, all front-court and mid-court calls (tin, front out, side out, service
line, short line) are measured. If deep-court coverage later matters, the recorded
option is a per-phone second-lens capture (MultiCam wide+ultrawide) — not v1.

## Testing

- **Pure math:** golden-vector tests for triangulation, interpolation, plane-snap, and
  ClockSync's estimator (simulated RTT distributions incl. stall outliers) — Swift
  (`ios/Tests/StereoMathTests.swift`) and Python mirror share the vectors.
- **Protocol:** codec round-trip + version-skew tests (`PeerMessagesTests`).
- **Bench (on hardware, gates phases):** transport latency table (Phase 1), sync error
  vs. clap anchor + readout-time / PTS-semantics measurement (Phase 2), thermal at 60 fps
  (Phase 1).
- **End-to-end:** two-phone court session against labeled footage; `stereo_fuse` output
  through the existing eval harness (eval skill) vs. monocular baseline before any
  improvement claim.
- **Degradation:** kill-the-link and single-phone tests asserting recording continuity
  and honest UI states.

## Phasing

1. **Transport spike + PeerSession** (BLE vs Wi-Fi P2P on the mounts; selection gate;
   thermal bench; multicast entitlement filed day one).
2. **ClockSync** (network estimator + clap anchor + bench: readout time, PTS semantics;
   ≤ 2 ms assertion).
3. **Stereo core** (StereoMath + StereoEngine + Python mirror + calibration pairing +
   agreement gate).
4. **Live experience** (p-pair, p-live, mini-court replay, landscape/60 fps capture,
   DESIGN.md additions).
5. **Cloud fusion** (paired uploads, `stereo_fuse`, offline refinement, eval pass).
6. **Later:** Wi-Fi Aware `.realtime` fast path (iOS 26+/iPhone 12+), low-res
   cross-preview stream, shot-classification consumers, player 3D (cloud).

Phases 1–2 carry the project risk and produce numbers, not features; they are
deliberately first. The live product exists at the end of Phase 4; Phase 5 makes it
analytically credible.

## Risks & open questions

- **PTS exposure semantics + 60 fps readout time are unmeasured** — Phase 2 bench; the
  sync budget is provisional until then.
- **Thermal/battery** for 60 fps capture + ANE inference + radio over a 45-min match —
  Phase 1 bench; fallback is 30 fps capture (halves interpolation quality, still within
  analytics budget).
- **Multicast entitlement lead time** (Wi-Fi candidate only) — filed day one; BLE
  candidate is not gated on it.
- **Occlusion is not solved by phone #2** (shared back-wall vantage). The no-call
  policy is the mitigation; measure real occlusion rates from paired footage in Phase 5.
- **Core ML artifact absent from the repo** (`ios/Model/` is empty; export is
  user-owned per ios/MODEL.md) — live-path work needs a real model dropped in early.
- **Detection quality is the historical dominant error** (prior art: In/Out tennis) and
  current rally-scale recall is ~35%; the RF-DETR/YOLO retrain lever remains the
  highest-leverage companion work — stereo cannot fuse detections that don't exist.
- Open: exact fin spacing + clamp orientation for landscape (site visit); audible call
  announcements default on/off; whether serve-depth `estimated` calls render as calls or
  analytics-only in v1.
- Deferred decision: iOS floor stays 17 (BLE + AWDL both work there); Wi-Fi Aware is a
  Phase 6 fast path, never a requirement.

## Alternatives considered

- **Dual-monocular event fusion (no continuous stereo):** rejected by ruling — fails the
  actual product goal (3D match model).
- **Sync-only pairing, cloud-only fusion:** no real-time marking; kept only as the
  implicit degraded mode when the link dies.
- **MultipeerConnectivity:** live iOS 26 regression, Apple steering away.
- **WebRTC for the peer link:** heavyweight dependency solving NAT problems we don't
  have; hand-rolled framing over BLE/NWConnection is smaller and prioritizable.
- **Ultrawide lens:** full-court coverage minus a 1.2–2 m back strip, but ~2× coarser
  pixels and unmodeled edge distortion; rejected by ruling in favor of stereo accuracy.
- **Personal Hotspot / infrastructure Wi-Fi:** manual toggle each session, single point
  of failure; not a tier we design for.
