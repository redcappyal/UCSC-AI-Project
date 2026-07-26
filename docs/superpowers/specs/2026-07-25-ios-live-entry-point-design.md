# iOS Live-Match Entry Point — Design

**Date:** 2026-07-25
**Status:** Approved in conversation (Ian), pending written-spec review
**Predecessor:** [2026-07-23 Two-Camera Stereo Live View](2026-07-23-two-camera-stereo-live-view-design.md)
— this spec is the composition layer that spec's Phases 3–5 left unbuilt.

## Problem

Every component of the live two-camera path exists and is unit-tested. None of it is
reachable. Grepping the iOS tree for production callers finds none for `PairingView`,
`MiniCourtView`, `CallAnnouncer`, `RecordModel.attachPeer`, `RecordModel.attachStereo`,
or `APIClient.fetchSolvedCameraModel` — each is referenced only by its own file, by a
test, or by the DEBUG bench. `RootTabView`'s Play tab is `RecordView` directly, so there
is no route to `p-pair` at all.

This is the shape a TDD-driven build leaves behind: every unit is tested in isolation, so
integration is the only thing with no test forcing it into existence. `StereoWiringTests`
calls `attachPeer` and `sendCalibration` by hand — which is exactly why the seams work
and nothing invokes them.

Three things beyond routing are missing outright:

1. **Role selection.** The predecessor spec fixes `initiator = primary`, and BLE forces
   the asymmetry (`startInitiator()` opens a `CBCentralManager`, `startResponder()` a
   `CBPeripheralManager`). `PeerBenchView` asks the human with a picker; `PairingView`
   has no such control, so nothing decides.
2. **Calibration exchange.** `attachStereo` arms `peer.onCalibration`, but no production
   code calls `sendCalibration`, and none fetches this phone's own solved model. Without
   both, the primary never builds a `StereoEngine` and no call can land.
3. **Recording during a live rally.** DESIGN.md §16 promises "recording never depends on
   the link", but nothing starts a recording when a rally goes live.

## Goals

- The Play tab routes to `p-pair` and `p-live`, matching DESIGN.md §16.
- Two phones on court can pair, exchange calibration, and produce live calls.
- Both clips upload as a paired session so the (already-built, currently unreachable)
  server-side stereo fusion runs.
- Plain single-camera recording is untouched, structurally — not by convention.

## Non-goals

- Live video preview between phones (predecessor non-goal, unchanged).
- Player tracking and shot classification (predecessor non-goals, unchanged).
- A UI for choosing landscape capture. `CameraController.orientation` stays settable in
  code and defaults to `.portrait`; nothing in this change exposes it.
- Mini-court replay on the secondary phone — see Deliberate v1 narrowing.

## Decision log

1. **Full live path, not just navigation** (Ian). Routing alone leaves a screen that
   cannot complete a session; the change is worth making only if two phones can finish a
   rally and get calls.
2. **Explicit role picker on `p-pair`** (Ian). Over "first to tap PAIR wins" and a
   deterministic UUID tie-break. The tie-break would need symmetric discovery, which BLE
   does not offer; the first-tap race has a fallback-timing failure mode on a noisy
   court. A picker is honest and debuggable, and costs one DESIGN.md addition.
3. **No default role.** If both phones defaulted to primary, both would open a
   `CBCentralManager` and pairing would hang with nothing honest to say about why. PAIR
   stays disabled until a role is picked. This makes the most likely on-court failure
   impossible rather than merely diagnosable.
4. **Calibration fetched on entry, gating PAIR** (Ian). Over failing during sync or at
   START RALLY. Fails before two people have walked to opposite corners of the court, and
   reuses §16's existing Failed row rather than inventing a state.
5. **Paired upload with server-side auto-fuse** (Ian). `/api/track` already accepts
   `session_id`, `camera_role`, `peer_video_id`, and `sync_manifest_json`, and
   `job_runner.maybe_start_stereo_fuse` already starts a `stereo-<session_id>` run once
   both camera runs complete. That path is built and idle; this change feeds it.
6. **Fix the orientation guard here** (Ian). PEER.md documents it as broken. It is latent
   only because live mode is unreachable, and this change is what makes it reachable.
7. **A `LiveSessionModel` coordinator** over extending `RecordModel` or wiring in view
   code. Matches the codebase's existing pattern: `PairingModel` was extracted so §16's
   state table is assertable without a view, and `RunSubmission` is separate from
   `ResultsView` for the same reason.
8. **The primary drives the rally; START RALLY is gated on the engine existing.** Raised
   during spec self-review, not in conversation. An earlier draft let either phone start
   a rally, which meant a lost calibration message produced a recorded rally that nothing
   could ever call, with nothing on screen explaining why. Gating on the engine turns
   that into an honest waiting state, and making the start primary-only removes the
   double-start race as a side effect. The cost is two extra rows in §16's state table.

## Architecture

### Ownership

New `ios/Sources/Live/LiveSessionModel.swift`, an `@MainActor ObservableObject` owning:

- the `PeerSession` and its `PairingModel`
- the operator's role choice, and the DEBUG transport choice
- the calibration fetch, through an injected `APIClientProtocol`
- the session identity and the post-rally `RunSubmission`
- driving `RecordModel.attachPeer` / `attachStereo`, and record start/stop

`RecordModel` keeps exactly its current responsibilities — camera, inference queue,
stereo engine, live presentation — and takes on none of the above.

**`PlayRootView` owns `RecordModel` as a `@StateObject`; `LiveSessionModel` receives it
through an idempotent `bind(record:)`.** The inverse (the live model owning the record
model) is rejected deliberately: DESIGN.md §16 and `PairingModel`'s own type doc make
"pairing adds capability, never gates it" a hard requirement, and if the live layer owned
the camera model, a defect in pairing could break plain recording. Owning `RecordModel`
above the live layer makes that guarantee structural.

`RecordView` changes from `@StateObject private var model` to an injected
`@ObservedObject`. `RecordModel.startCamera()` gains an idempotence guard, because the
record stage and the live stage both call it from `.task`. `PlayRootView` never calls it,
so opening the Play tab still triggers no camera permission prompt.

### Navigation

The Play tab becomes a `NavigationStack` rooted at a new `PlayRootView` (§16's `p-load`):

```
PlayRootView ──"Record a clip"──> RecordView          (p-record)
     └────────"Live match"──────> PairingView         (p-pair)
                                       └─START RALLY─> LiveStageView (p-live)
```

Back routes follow §16: `p-record` → Play, `p-pair` → Play, `p-live` → `p-pair`, using
`NavigationStack`'s own back button (the proxied-primary pattern of §3.4 is a web-shell
mechanism and does not apply natively).

The server-settings gear and the DEBUG peer-bench button move from the record stage to
`PlayRootView`, which is the section root a settings affordance belongs on. The DEBUG
stereo-demo cube stays on the record stage — it needs `RecordModel`.

### DESIGN.md changes (§17.1 requires these land in the same change)

1. **New §8 subsection — two-way segment (`.corrSeg`).** `.corrSeg` is referenced exactly
   once in DESIGN.md today, inside §16's `p-track` blueprint row, and has no component
   entry. §17.1 requires a subsection before a component is reused on a new screen, so
   this change documents the existing control (both segments equal width, ≥ 44 px per
   §0.6, uppercase labels per §0.7, selected segment carries the accent) and then the
   role picker reuses it rather than inventing a second segment grammar.
2. **§16 `p-load`:** the native shell shows two hero cards, not three. "Judge a clip" is
   a web file input with no native equivalent, so "Record a clip" takes the accent slot
   on iOS.
3. **§16 `p-pair` body:** add the role segment, visible only in the idle state. Copy:
   "This phone calls" / "This phone assists".
4. **§16 `p-pair` state table:** add two rows —

   | State | `.link-status` text | Primary | Notes |
   |---|---|---|---|
   | No role chosen | "Pick this phone's job to start." | PAIR (disabled) | §7: the primary never advertises a tap that cannot fire |
   | Ready, awaiting peer calibration (primary only) | "Paired · waiting for the other phone's calibration" | START RALLY (disabled) | the engine does not exist yet; starting would record a rally nothing can call |

   and extend the Idle row's note to cover the calibration fetch:
   "Checking this phone's court calibration…" with PAIR disabled until it lands.
5. **§16 `p-pair` Ready row:** START RALLY is the primary phone's action. On the secondary
   the row reads "Paired · the other phone starts the rally" with no enabled primary.
6. **§16 `p-live`:** note that the mini-court replay is primary-only in v1, and that
   while the link is degraded the secondary exposes its own STOP (below).
7. **§3.2 shell anatomy:** record that the native shell substitutes `NavigationStack` and
   its system back button for the web shell's header chevron and proxied primary (§3.4),
   which are web mechanisms. The phase inventory and blueprints in §16 are unchanged;
   only the chrome that moves between them differs per client.

## Data flow

### Entering `p-pair` (both phones, independently)

1. `LiveSessionModel.prepare()` → `api.latestCalibration()` →
   `api.fetchSolvedCameraModel(calibrationJSON:)` → parse via `CameraModel.fromJSON`
   **and** validate `adoptedForCapture()`. Validating adoption at the gate is the point:
   today an unusable calibration only throws inside `attachStereo`, after both operators
   have walked to opposite corners.
2. The operator picks a role. PAIR enables only once role *and* calibration are ready.

### PAIR

3. Build the transport (BLE default; DEBUG picker offers Wi-Fi P2P), then
   `PeerSession(transport:isInitiator: role == .primary)`.
4. `record.attachPeer(session)` → `record.attachStereo(localModelJSON:)` →
   `session.start()`. **This order is load-bearing:** `attachStereo` installs
   `peer?.onCalibration`, so it must run after `attachPeer` sets `self.peer` and before
   any calibration message can arrive. `attachStereo` is called on both phones; its
   handler already guards `role == .primary` internally, so the secondary installing it
   is harmless.
5. `PairingModel` drives §16's table: searching → confirming → syncing → ready.

### Reaching `.ready`

6. The secondary sends `.calibration(profileID:payloadJSON:)` carrying its solved model.
   `.ready` is also the first phase where `PeerSession.sendCalibration` is not dropped by
   its own guard, so this is the earliest correct moment.
7. The primary mints `sessionID = UUID().uuidString` and sends `.sessionManifest`.

### START RALLY (primary only)

8. **Gated on the primary holding a live `StereoEngine`.** The engine is built when the
   secondary's calibration message arrives; if that message is lost, an ungated START
   RALLY would record a rally that nothing can ever call, with nothing on screen saying
   so. Until the engine exists the primary reads START RALLY disabled against
   "Paired · waiting for the other phone's calibration".
9. `session.goLive()`, start local recording, send `.record("start", ptsNs)`.
10. The secondary's `onRecord` starts its own recording and advances to `p-live`. Only
    the primary can start, so there is no double-start race; the handler still ignores a
    start while already recording. `sendDetections` accepts both `.ready` and `.live`, so
    the two phases need not be synchronized.
11. Live calls flow as already built: the primary's engine → `livePresentation` + flash;
    the primary relays `.event`; the secondary mirrors it.

### STOP

12. The primary stops; `.record("stop", ptsNs)` stops the secondary.
13. **While the link is degraded, the secondary exposes its own STOP.** Otherwise a
    dropped link leaves it recording 4K60 indefinitely with no way to end the rally —
    the honest counterpart to "Link lost — still recording" actually being survivable.
14. Each phone submits independently — primary `camera_role: "a"`, secondary `"b"`.
    Neither attaches a `sync_manifest_json`: the parameter is plumbed the whole way
    through (`RunSubmission.submit` → `APIClient.startTrack`) but the value is `nil`, so
    `stereo_sync.seed_offset_from_manifest` returns `(0.0, "none")` and the server's own
    ±50 ms search finds the offset. See Deliberate v1 narrowing for why sending the
    `ClockSync` estimate was worse than sending nothing.
15. After its own upload, each phone sends `.sessionManifest` with its real `videoID`;
    the peer passes it as `peer_video_id` if it has not uploaded yet. **Best-effort
    only** — fusion pairs on `session_id` + `camera_role`, so a late or lost manifest
    never blocks it.
16. Server-side: the second run completing triggers `try_start_stereo_fuse`, producing a
    `stereo-<sessionID>` run. No server change is required.

## Seam changes

### `PeerSession`

Replace `case .record, .sessionManifest: break` with dispatch to two new callbacks, and
add the matching senders:

- `sendRecord(action:ptsNs:)` / `var onRecord: ((String, UInt64) -> Void)?`
- `sendSessionManifest(sessionID:videoID:)` / `var onSessionManifest: ((String, String) -> Void)?`

Both follow the existing convention (`onRemoteDetections`, `onCalibration`, `onEvent`):
plain `var` closures fired on the transport's delivery queue, with the consumer
responsible for hopping to main. The wire format is unchanged — `ControlMessage` already
encodes and decodes both cases.

### Orientation guard (decision 6)

`PeerSession` currently builds its own hello from, and compares the peer's hello against,
the static portrait constants `CaptureSettings.frameWidth/frameHeight`. The advertised
value therefore does not depend on the session's orientation and the comparison can never
fail. Fix:

- `PeerSession` takes the session's `CaptureSettings.CaptureOrientation` at init and uses
  `CaptureSettings.frameSize(for:)` for both its hello and its guard.
- `RecordModel.peerFrameW/peerFrameH` read `CaptureSettings.frameSize(for: camera.orientation)`
  instead of the statics, so detection tuples are labelled in the space they were
  actually captured in.

Both phones still default to `.portrait`, so this changes no behavior today; it makes the
guard capable of firing when a landscape mount is eventually exposed. Update PEER.md's
"Known limits" to remove the ⚠️ entry.

### Stereo track for the mini-court

`StereoTrack.detectImpacts` builds the 3D track and discards it, and `StereoEvent` only
carries `.impact(StereoImpact)`, so `MiniCourtView` cannot be fed.

- Add `StereoTrack.analyze(...) -> (track: [TrackPoint3D], impacts: [StereoImpact])`;
  `detectImpacts` delegates to it and keeps its signature. No call site changes and no
  arithmetic changes.
- `StereoEvent.impact` carries the track alongside the impact. `TrackPoint3D` gains
  `Equatable` (trivial — `Double` plus `SIMD3<Double>`).
- `RecordModel` publishes `liveTrack` beside `livePresentation`.

Because no math changes, `tests/stereo_goldens.json` and
`ios/Tests/Fixtures/stereo_goldens.json` are untouched and
`tests/generate_stereo_goldens.py` does not need rerunning. The Swift stereo suite
passing unchanged is the evidence for that claim.

### API

`APIClientProtocol.startTrack` and `RunSubmission.submit` gain `sessionID`, `cameraRole`,
`peerVideoID`, and `syncManifestJSON`, **all defaulted `nil`** so the existing
single-camera path is byte-identical — which is what `app.py`'s Phase 5 block already
assumes ("absent must mean byte-identical behavior"). The server enforces
both-or-neither on session/role; the client never sends one without the other.

## Error handling

| Condition | Behavior |
|---|---|
| No calibration, or server unreachable | Reason verbatim in `.link-status`; PAIR stays **enabled** as the retry and re-runs the fetch (§16's "Calibration fetch failed" row). Enabled with or without a role picked — the retry concerns this phone's calibration, not its job |
| Calibration present but not adoptable | Same, with the adoption error verbatim — caught at the gate, not inside `attachStereo` |
| Orientation mismatch | The existing `PeerSession` guard's message, verbatim (now able to fire) |
| Peer calibration never arrives | Primary holds START RALLY disabled against "Paired · waiting for the other phone's calibration" — never a rally nothing can call |
| Link lost mid-rally | `.degraded` → "Link lost — still recording"; recording continues; the secondary gains its own STOP so the rally can still be ended; STOP still uploads with session and role, so the server fuses if the peer's run also lands |
| Upload failure | Per-phone and independent — one phone failing never blocks the other. The reason appears verbatim in `.link-status` once the rally ends (§16's "Rally ended badly" row), and the clip is **kept**: `liveClip` is cleared only on a successful upload, so the recording survives the failure. A "Try again" affordance mirroring `ResultsView` is a follow-up, not v1 — §7 allows one primary per phase and that primary is START RALLY |
| Tracking never finishes (worker died; HTTP layer healthy, so nothing throws) | `RunSubmission` caps the poll at 20 min and fails with the reason, so the rally routes through `finishRally` and releases the peer session instead of stranding at `.submitting` forever. The server-side run and its auto-fuse are unaffected — `try_start_stereo_fuse` is triggered by the run completing, never by this poll |
| Stereo demo while paired | Unchanged — `startStereoDemo` already refuses when `peer != nil` |

## Testing

New `ios/Tests/LiveSessionModelTests.swift`, over `LoopbackTransport` and a mock
`APIClientProtocol`:

- calibration in flight, and calibration failure, both keep PAIR disabled
- PAIR stays disabled until a role is chosen
- the secondary sends its calibration on reaching `.ready`
- the primary mints and broadcasts a session ID
- **START RALLY stays disabled on the primary until the peer's calibration builds the
  engine** — the silent-failure case this design exists to close
- START RALLY is never enabled on the secondary
- record start propagates to the peer, and is ignored while already recording
- **the secondary gains a local STOP when the link degrades, and loses it on recovery**
- submission carries `camera_role` "a" on the primary and "b" on the secondary
- a degraded link does not stop recording

Additions elsewhere:

- `PeerSessionTests`: `.record` and `.sessionManifest` round-trip through the new
  callbacks.
- `CaptureOrientationTests`: the guard fails a genuinely mismatched pair (it cannot today).
- `StereoEngineTests`: `analyze`'s impacts equal `detectImpacts`'s; `.impact` carries a
  non-empty track.
- `RunSubmissionTests`: paired params reach `startTrack`; all nil on the unpaired path.

Full suite (348 Python + the Swift suite) must pass. `/verify` covers `p-load`, `p-pair`,
and `p-live` at a 390 × 844 phone viewport in both themes per §0.12.

## Deliberate v1 narrowing

**No `sync_manifest_json` is sent.** An earlier shape of this change had the primary
attach `{"offset_series": [ClockSync.estimate.offset]}`. That is not a weaker seed than
the server wants — it is a different quantity, and it broke every fusion run:

- `ClockSync`'s offset is `remote − local` on each device's **capture host clock**
  (`CMClockGetHostTimeClock()`, seconds since that phone booted). Two phones differ by
  their uptime difference — routinely hours.
- The sign is inverted. `stereo_sync.py` defines the offset as "seconds ADDED to
  `camera_role` `b`'s timestamps to place both tracks on `a`'s clock" and `job_runner`
  applies exactly that, so the server wants `t_a − t_b` while the primary measures
  `t_b − t_a`.
- Fatally, the server's timestamps are **clip-relative**: `track_samples_from_csv` reads
  `timestamp_seconds`, which `tracking_common.py` writes as `source_frame / source_fps`,
  so both clips start near t=0. The seed the search wants is the **recording-start skew**
  between the phones — tens of milliseconds, which is why `refine_offset`'s
  `coarse_range_s` is `0.05`. A clock offset does not appear in it at all.

The result was silent: `_timeline` found no overlap at a seed of tens of thousands of
seconds, `refine_offset` bailed with `refined.offset_s` still at the seed, and the fuse
completed with an empty track and zero impacts — on every rally. Sending nothing is
strictly better, because then the seed is `0.0` and the ±50 ms search works.

The client does not currently know either phone's recording-start instant in a shared
frame, so it has no seed worth sending and `seed_source: "none"` is the correct outcome.
The follow-up that closes it: the `.record` message already carries `ptsNs`, which
`PeerSession` hands to `onRecord` and `LiveSessionModel` currently discards. Exchanging
each phone's recording-start host time, mapping the peer's through
`ClockSync.remoteToLocal`, and sending `t_a_start − t_b_start` is a seed in the right
quantity, on the right clock, with the right sign. The plumbing and the wire protocol are
unchanged, so that is a value change when it lands, not a redesign.

The predecessor spec lists "Post-rally 3D mini-court replay on both phones" as a goal.
This change delivers it on the **primary only**: the secondary's view of a call arrives
as relayed `.event` JSON, which carries no 3D track. Relaying a decimated track (≤ 60
points, ~2 KB, once per rally on the control channel rather than the hot loop) is the
follow-up that closes the gap. Called out here so the narrowing is a recorded decision
rather than an omission.

## Risks

- **Biggest:** `LiveSessionModel` coordinates four independent lifecycles (transport,
  pairing, camera, upload) whose failures interleave. Mitigation is that each is already
  a tested unit with its own state machine, and the coordinator's own logic is tested
  against `LoopbackTransport` with no radio.
- The role picker is a new control on a screen whose state table was previously complete;
  a stale role choice across a session restart would silently swap primary and secondary.
  `PairingModel.onSessionEnded` already exists as the reset seam — the role must reset
  through it.
- Nothing here can be validated on court without the hardware items PEER.md already
  lists: a real `BallDetector.mlpackage`, the transport-selection bench, and the ≤ 2 ms
  sync assertion. This change makes the path reachable; it does not make it proven.
