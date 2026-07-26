# iOS Live-Match Entry Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live two-camera path reachable and completable on iOS — pair two phones from the Play tab, produce live calls, and upload both clips as a paired session.

**Architecture:** A new `LiveSessionModel` coordinator owns the live lifecycle (pairing, calibration, session identity, rally, upload) while `RecordModel` keeps owning the camera. The Play tab becomes a `NavigationStack` rooted at a hero-card section root that routes to `p-pair` and `p-live`. Three existing seams get filled in: two dropped `ControlMessage` cases, the non-firing orientation guard, and the discarded 3D track.

**Tech Stack:** Swift 5.9 / SwiftUI, iOS 17 deployment target, XCTest, XcodeGen, CoreBluetooth + Network.framework transports, Flask backend (unchanged).

**Spec:** [docs/superpowers/specs/2026-07-25-ios-live-entry-point-design.md](../specs/2026-07-25-ios-live-entry-point-design.md)

## Global Constraints

- **No Swift toolchain exists on the Windows dev box.** `xcodebuild`, `xcodegen`, `swift`, and `swiftc` are all absent (verified 2026-07-25). Every task here is Swift except Task 5 and Task 11's doc steps. On Windows the code can be written but **not compiled and not tested** — the "run the test, watch it fail" steps cannot be honored, and any such task must report its test status as `UNVERIFIED — no Swift toolchain` rather than claiming a pass. Run this plan on a Mac with Xcode if you want its TDD cycle to mean anything.
- **Swift test command:** `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`. Add `-only-testing:SquashLineCallingTests/<ClassName>` to scope it.
- `ios/SquashLineCalling.xcodeproj` is generated and gitignored. Never edit it; edit `ios/project.yml`. `project.yml` globs `Sources` and `Tests`, so **new files need no project.yml change**.
- All UI work follows DESIGN.md. Deviations must be written into DESIGN.md in the same change (§17.1), never left silent.
- Verdict colors (green/red) are reserved for IN/OUT and must never be used for link or status state (§0.3).
- No layout shift: anything that appears or disappears reserves its footprint (§0.9).
- Touch targets ≥ 44 pt; primary actions 48 pt (§0.6).
- Uppercase control labels with `letter-spacing: .05em`; sentence-case status text in `--dim` (§0.7).
- Callback convention on `PeerSession`: plain `var` closures fired on the transport's delivery queue. **Consumers hop to main**, the session never does it for them.
- No stereo math changes anywhere in this plan. `tests/stereo_goldens.json` and `ios/Tests/Fixtures/stereo_goldens.json` must remain byte-identical; `tests/generate_stereo_goldens.py` must not be run.

---

### Task 1: `PeerSession` record and session-manifest seams

`ControlMessage` already encodes and decodes `.record` and `.sessionManifest`, but `handleControl` drops both and there are no senders. Fill in the reserved slots.

**Files:**
- Modify: `ios/Sources/Peer/PeerSession.swift`
- Test: `ios/Tests/StereoWiringTests.swift`

**Interfaces:**
- Consumes: `ControlMessage.record(action:ptsNs:)`, `ControlMessage.sessionManifest(sessionID:videoID:)` (both already defined).
- Produces: `PeerSession.onRecord: ((String, UInt64) -> Void)?`, `PeerSession.onSessionManifest: ((String, String) -> Void)?`, `PeerSession.sendRecord(action:ptsNs:)`, `PeerSession.sendSessionManifest(sessionID:videoID:)`.

- [ ] **Step 1: Write the failing tests**

Append to `ios/Tests/StereoWiringTests.swift`:

```swift
    func testRecordMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (String, UInt64)?
        secondary.onRecord = { received = ($0, $1) }
        primary.sendRecord(action: "start", ptsNs: 1_234_567_890)
        XCTAssertEqual(received?.0, "start")
        XCTAssertEqual(received?.1, 1_234_567_890)
    }

    func testSessionManifestRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (String, String)?
        secondary.onSessionManifest = { received = ($0, $1) }
        primary.sendSessionManifest(sessionID: "S-1", videoID: "V-9")
        XCTAssertEqual(received?.0, "S-1")
        XCTAssertEqual(received?.1, "V-9")
    }

    func testSendRecordGatedOnPhase() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        var fired = false
        primary.onRecord = { _, _ in fired = true }
        primary.sendRecord(action: "start", ptsNs: 1)
        XCTAssertFalse(fired)
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/StereoWiringTests
```

Expected: compile failure — `value of type 'PeerSession' has no member 'onRecord'`.

- [ ] **Step 3: Add the callbacks**

In `ios/Sources/Peer/PeerSession.swift`, beside the existing `onCalibration` / `onEvent` declarations:

```swift
    /// Phase 4: synchronized rally recording. `action` is "start" or "stop";
    /// `ptsNs` is the sender's host clock, for logging — actual alignment
    /// comes from ClockSync, never from this field.
    var onRecord: ((String, UInt64) -> Void)?
    /// Phase 5: paired-upload identity. `videoID` is empty on the initial
    /// announce and carries the real ID once that phone has uploaded.
    var onSessionManifest: ((String, String) -> Void)?
```

- [ ] **Step 4: Add the senders**

In the `// MARK: outbound` section, after `sendEvent`:

```swift
    func sendRecord(action: String, ptsNs: UInt64) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        sendControl(.record(action: action, ptsNs: ptsNs))
    }

    func sendSessionManifest(sessionID: String, videoID: String) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        sendControl(.sessionManifest(sessionID: sessionID, videoID: videoID))
    }
```

- [ ] **Step 5: Dispatch on receipt**

Replace the drop in `handleControl`:

```swift
        case .record, .sessionManifest:
            break   // consumed by Phase 4/5 code; parsing is already validated
```

with:

```swift
        case .record(let action, let ptsNs):
            onRecord?(action, ptsNs)
        case .sessionManifest(let sessionID, let videoID):
            onSessionManifest?(sessionID, videoID)
```

Note these fire with `stateLock` held, exactly as `onCalibration` and `onEvent` already do — consumers must not call back into the session synchronously.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/StereoWiringTests
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/Peer/PeerSession.swift ios/Tests/StereoWiringTests.swift
git commit -m "feat(peer): record and session-manifest seams"
```

---

### Task 2: Make the orientation guard able to fire

`PeerSession` builds its hello from, and compares the peer's hello against, the *static portrait* constants — so the advertised size never depends on the session's orientation and the guard can never fail. PEER.md documents this as a known hazard.

**Files:**
- Modify: `ios/Sources/Peer/PeerSession.swift`
- Modify: `ios/Sources/Record/RecordModel.swift`
- Modify: `ios/PEER.md`
- Test: `ios/Tests/CaptureOrientationTests.swift`

**Interfaces:**
- Consumes: `CaptureSettings.frameSize(for:) -> (width: Int, height: Int)`, `CaptureSettings.CaptureOrientation` (both exist).
- Produces: `PeerSession.init(transport:isInitiator:orientation:now:heartbeatTimeout:)` with `orientation` defaulted to `.portrait`, so no existing call site changes.

- [ ] **Step 1: Write the failing test**

Append to `ios/Tests/CaptureOrientationTests.swift`:

```swift
    func testMismatchedOrientationsFailTheHandshake() {
        let pair = LoopbackTransport.pair()
        let portrait = PeerSession(transport: pair.0, isInitiator: true,
                                   orientation: .portrait, now: { 0 })
        let landscape = PeerSession(transport: pair.1, isInitiator: false,
                                    orientation: .landscapeRight, now: { 0 })
        landscape.start(); portrait.start()

        // Transposed pixel spaces triangulate to confident, wrong line calls.
        // Refusing is the only safe outcome.
        guard case .failed(let reason) = portrait.phase else {
            return XCTFail("expected the guard to refuse, got \(portrait.phase)")
        }
        XCTAssertTrue(reason.contains("orientation"))
    }

    func testMatchedOrientationsHandshakeNormally() {
        let pair = LoopbackTransport.pair()
        let a = PeerSession(transport: pair.0, isInitiator: true,
                            orientation: .landscapeRight, now: { 0 })
        let b = PeerSession(transport: pair.1, isInitiator: false,
                            orientation: .landscapeRight, now: { 0 })
        b.start(); a.start()
        guard case .confirming = a.phase else {
            return XCTFail("expected .confirming, got \(a.phase)")
        }
    }
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/CaptureOrientationTests
```

Expected: compile failure — no `orientation:` parameter.

- [ ] **Step 3: Thread orientation through `PeerSession`**

Change the initializer signature and hello construction:

```swift
    init(transport: PeerTransport, isInitiator: Bool,
         orientation: CaptureSettings.CaptureOrientation = .portrait,
         now: @escaping () -> TimeInterval = ClockSync.hostNow,
         heartbeatTimeout: TimeInterval = 3.0) {
        self.transport = transport
        self.isInitiator = isInitiator
        self.orientation = orientation
        self.now = now
        self.heartbeatTimeout = heartbeatTimeout
        let frame = CaptureSettings.frameSize(for: orientation)
        self.myHello = Hello(protoVersion: peerProtoVersion,
                             appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                             deviceModel: ProcessInfo.processInfo.hostName,
                             nonce: UInt32.random(in: .min ... .max),
                             frameW: frame.width,
                             frameH: frame.height)
        transport.onControl = { [weak self] in self?.handleControl($0) }
        transport.onDatagram = { [weak self] in self?.handleDatagram($0) }
        transport.onStateChange = { [weak self] in self?.handleTransportState($0) }
    }
```

Add the stored property beside `isInitiator`:

```swift
    private let orientation: CaptureSettings.CaptureOrientation
```

- [ ] **Step 4: Make the guard compare against the session's own frame**

In `handleControl`'s `.hello` case, replace:

```swift
            guard theirs.frameW == CaptureSettings.frameWidth,
                  theirs.frameH == CaptureSettings.frameHeight else {
```

with:

```swift
            let mine = CaptureSettings.frameSize(for: orientation)
            guard theirs.frameW == mine.width, theirs.frameH == mine.height else {
```

Leave the comment above it and the failure message unchanged.

- [ ] **Step 5: Label detection tuples in the space they were captured in**

In `ios/Sources/Record/RecordModel.swift`, replace:

```swift
    private let peerFrameW = CaptureSettings.frameWidth
    private let peerFrameH = CaptureSettings.frameHeight
```

with computed properties that follow the camera:

```swift
    // Must match the Hello this device advertises (PeerSession), which is
    // built from the session's own orientation — a mismatch skews stereo.
    private var peerFrameW: Int { CaptureSettings.frameSize(for: camera.orientation).width }
    private var peerFrameH: Int { CaptureSettings.frameSize(for: camera.orientation).height }
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/CaptureOrientationTests
```

Expected: PASS. Both phones still default to `.portrait`, so no runtime behavior changes today — the guard simply becomes capable of firing.

- [ ] **Step 7: Update PEER.md**

In "Known limits (by design, Plan A)", delete the whole ⚠️ bullet beginning "**The frame-space guard does not currently catch that.**" and replace the tail of the preceding bullet's last sentence with:

```markdown
  runs the same orientation** — mixed orientations give transposed pixel
  spaces that triangulate to plausible-looking nonsense. `PeerSession` now
  advertises and checks `CaptureSettings.frameSize(for:)`, so a mismatched
  pair fails the handshake instead of producing confident, wrong calls.
```

- [ ] **Step 8: Commit**

```bash
git add ios/Sources/Peer/PeerSession.swift ios/Sources/Record/RecordModel.swift ios/Tests/CaptureOrientationTests.swift ios/PEER.md
git commit -m "fix(peer): make the orientation guard able to fire"
```

---

### Task 3: Carry the 3D track on the impact event

`StereoTrack.detectImpacts` builds the track and throws it away, so `MiniCourtView` has nothing to draw. Surface it without recomputing and without touching any arithmetic.

**Files:**
- Modify: `ios/Sources/Stereo/StereoTrack.swift`
- Modify: `ios/Sources/Stereo/StereoEngine.swift`
- Modify: `ios/Sources/Record/RecordModel.swift`
- Test: `ios/Tests/StereoEngineTests.swift`, `ios/Tests/LiveWiringTests.swift`

**Interfaces:**
- Consumes: `StereoTrack.buildTrack3D`, `TrackPoint3D` (both exist).
- Produces: `StereoTrack.analyze(_:_:_:_:timelineS:) -> (track: [TrackPoint3D], impacts: [StereoImpact])`; `StereoEvent.impact(StereoImpact, track: [TrackPoint3D])`; `TrackPoint3D: Equatable`; `RecordModel.liveTrack: [TrackPoint3D]`.

- [ ] **Step 1: Write the failing test**

Append to `ios/Tests/StereoEngineTests.swift`:

```swift
    func testAnalyzeReturnsTheTrackThatProducedTheImpacts() {
        // Reuses this class's existing `left` / `right` golden camera models and
        // the same sample construction `testCleanTrajectoryEmitsGoldenImpact`
        // uses. `analyze` must agree with `detectImpacts` exactly — it is the
        // same computation, surfaced, not a second one.
        let samplesLeft = Self.goldenSamples(for: left)
        let samplesRight = Self.goldenSamples(for: right)
        let timeline = stride(from: 0.0, to: 1.0, by: 1.0 / 240.0).map { $0 }

        let viaDetect = StereoTrack.detectImpacts(left, samplesLeft,
                                                  right, samplesRight,
                                                  timelineS: timeline)
        let analyzed = StereoTrack.analyze(left, samplesLeft,
                                           right, samplesRight,
                                           timelineS: timeline)
        XCTAssertEqual(analyzed.impacts, viaDetect)
        XCTAssertFalse(analyzed.track.isEmpty)
    }
```

`left` and `right` are this class's existing `CameraModel!` properties, decoded from the goldens in its setup. Add `goldenSamples(for:)` as a private static helper that projects the same court-feet trajectory `testCleanTrajectoryEmitsGoldenImpact` already builds — lift that construction into the helper and have both tests call it rather than duplicating it.

- [ ] **Step 2: Run to verify failure**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/StereoEngineTests
```

Expected: compile failure — no member `analyze`.

- [ ] **Step 3: Extract `analyze`, keep `detectImpacts` as a delegate**

In `ios/Sources/Stereo/StereoTrack.swift`, rename the body of `detectImpacts` into `analyze`, returning the track it already built, and leave `detectImpacts` as a one-line delegate:

```swift
    /// The track and the impacts from one pass. `detectImpacts` built this
    /// track and discarded it; the post-rally replay (§8.10) needs it, and
    /// recomputing would be the same triangulation twice.
    static func analyze(_ a: CameraModel, _ samplesA: [TrackSample],
                        _ b: CameraModel, _ samplesB: [TrackSample],
                        timelineS: [Double]) -> (track: [TrackPoint3D], impacts: [StereoImpact]) {
        let track = buildTrack3D(a, samplesA, b, samplesB, timelineS: timelineS)
        guard track.count >= 3 else { return (track, []) }
        var impacts: [StereoImpact] = []
        // ... the entire existing body of detectImpacts from its
        // `for surface in StereoMath.surfaces {` loop onward, unchanged ...
        return (track, impacts)
    }

    static func detectImpacts(_ a: CameraModel, _ samplesA: [TrackSample],
                              _ b: CameraModel, _ samplesB: [TrackSample],
                              timelineS: [Double]) -> [StereoImpact] {
        analyze(a, samplesA, b, samplesB, timelineS: timelineS).impacts
    }
```

Do not alter a single arithmetic line while moving it. The existing golden tests passing unchanged is the proof.

- [ ] **Step 4: Make `TrackPoint3D` equatable and put the track on the event**

In `ios/Sources/Stereo/StereoTrack.swift`:

```swift
struct TrackPoint3D: Equatable { let tS: Double; let pointFt: SIMD3<Double>; let gapFt: Double }
```

In `ios/Sources/Stereo/StereoEngine.swift`:

```swift
enum StereoEvent: Equatable {
    case impact(StereoImpact, track: [TrackPoint3D])
}
```

and in `process()`, switch to `analyze` and emit the track alongside:

```swift
        let analyzed = StereoTrack.analyze(localModel, localSamples,
                                           remoteModel, remoteSamples,
                                           timelineS: timeline)
        for impact in analyzed.impacts {
            let isDuplicate = emitted.contains {
                $0.surface == impact.surface && abs($0.tS - impact.tS) < Self.emitDedupeS
            }
            if isDuplicate { continue }
            emitted.append((impact.tS, impact.surface))
            onEvent?(.impact(impact, track: analyzed.track))
        }
```

- [ ] **Step 5: Update every `.impact` pattern match**

Exactly five sites exist. Find them with:

```bash
cd ios && grep -rn "case .impact" Sources Tests
```

Update `ios/Sources/Record/RecordModel.swift:216` and `:293` and the two test sites to `guard case .impact(let impact, let track) = event else { return }`. In `RecordModel`'s `attachStereo` handler, publish the track on the same main hop that already sets `livePresentation`:

```swift
    /// The 3D track behind the most recent call, for the §8.10 post-rally
    /// replay. Never derived from `stereoEvents` — that list is relayed JSON
    /// with no geometry in it.
    @Published private(set) var liveTrack: [TrackPoint3D] = []
    /// The impact that produced `livePresentation`. `MiniCourtView` colours its
    /// marker from this, so the replay always agrees with the call that was
    /// made — the same reason `CallPresentation.color` is the one mapping.
    @Published private(set) var liveImpact: StereoImpact?
```

```swift
                    DispatchQueue.main.async {
                        self?.appendStereoEvent(json)
                        self?.livePresentation = presentation
                        self?.liveTrack = track
                        self?.liveImpact = impact
                        self?.showFlash(presentation)
                    }
```

Do the same in the DEBUG `startStereoDemo` handler at `:293` (it has no `appendStereoEvent` call, but sets the same three published values).

- [ ] **Step 6: Run the full Swift suite**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS, including `StereoGoldenTests` unchanged. If a golden test fails, the extraction changed arithmetic — revert and redo Step 3 as a pure move.

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/Stereo/StereoTrack.swift ios/Sources/Stereo/StereoEngine.swift ios/Sources/Record/RecordModel.swift ios/Tests/StereoEngineTests.swift ios/Tests/LiveWiringTests.swift
git commit -m "feat(stereo): carry the 3D track on the impact event"
```

---

### Task 4: Paired-session parameters on the upload path

**Files:**
- Modify: `ios/Sources/API/APIClient.swift`
- Modify: `ios/Sources/Results/RunSubmission.swift`
- Test: `ios/Tests/RunSubmissionTests.swift`

**Interfaces:**
- Consumes: `/api/track`'s Phase 5 fields (`session_id`, `camera_role`, `peer_video_id`, `sync_manifest_json`) — server-side, already implemented.
- Produces: `startTrack(videoID:calibrationJSON:duration:sessionID:cameraRole:peerVideoID:syncManifestJSON:)` and `RunSubmission.submit(videoURL:duration:sessionID:cameraRole:peerVideoID:syncManifestJSON:)`, all new parameters defaulted `nil`.

- [ ] **Step 1: Write the failing tests**

In `ios/Tests/RunSubmissionTests.swift`, extend the existing mock API client to record what `startTrack` received, then add:

```swift
    func testUnpairedSubmissionSendsNoPairedFields() async {
        let api = MockAPIClient()
        let submission = await RunSubmission(api: api, pollInterval: .milliseconds(1))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/a.mp4"), duration: 3)
        XCTAssertNil(api.lastSessionID)
        XCTAssertNil(api.lastCameraRole)
        XCTAssertNil(api.lastSyncManifestJSON)
    }

    func testPairedSubmissionCarriesSessionAndRole() async {
        let api = MockAPIClient()
        let submission = await RunSubmission(api: api, pollInterval: .milliseconds(1))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/a.mp4"), duration: 3,
                                sessionID: "S-1", cameraRole: "b",
                                peerVideoID: "V-7", syncManifestJSON: "{\"clap_anchor_s\":0.01}")
        XCTAssertEqual(api.lastSessionID, "S-1")
        XCTAssertEqual(api.lastCameraRole, "b")
        XCTAssertEqual(api.lastPeerVideoID, "V-7")
        XCTAssertEqual(api.lastSyncManifestJSON, "{\"clap_anchor_s\":0.01}")
    }
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/RunSubmissionTests
```

Expected: compile failure — extra arguments in call.

- [ ] **Step 3: Widen the protocol and client**

In `ios/Sources/API/APIClient.swift`, update the protocol requirement and the implementation:

```swift
    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String?, cameraRole: String?,
                    peerVideoID: String?, syncManifestJSON: String?) async throws -> JobStatus
```

```swift
    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String? = nil, cameraRole: String? = nil,
                    peerVideoID: String? = nil,
                    syncManifestJSON: String? = nil) async throws -> JobStatus {
        var request = URLRequest(url: baseURL.appending(path: "api/track"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded",
                         forHTTPHeaderField: "Content-Type")
        var fields: [(String, String)] = [
            ("video_id", videoID),
            ("calibration_json", calibrationJSON),
            ("start_time", "0"),
            ("end_time", String(duration)),
            ("frame_stride", "4"),
            ("inference_width", "960"),
            ("event_engine", ""),
            ("fusion_3d", ""),
        ]
        // The server treats these as strictly optional and requires session and
        // role together — omitting them entirely must stay byte-identical to
        // the single-camera path, so append only what is actually set.
        if let sessionID, let cameraRole {
            fields.append(("session_id", sessionID))
            fields.append(("camera_role", cameraRole))
        }
        if let peerVideoID { fields.append(("peer_video_id", peerVideoID)) }
        if let syncManifestJSON { fields.append(("sync_manifest_json", syncManifestJSON)) }
        request.httpBody = Data(Multipart.formURLEncoded(fields).utf8)
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }
```

Protocol requirements cannot carry default values — add a protocol extension so existing three-argument call sites keep compiling:

```swift
extension APIClientProtocol {
    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        try await startTrack(videoID: videoID, calibrationJSON: calibrationJSON,
                             duration: duration, sessionID: nil, cameraRole: nil,
                             peerVideoID: nil, syncManifestJSON: nil)
    }
}
```

- [ ] **Step 4: Thread them through `RunSubmission`**

```swift
    func submit(videoURL: URL, duration: Double,
                sessionID: String? = nil, cameraRole: String? = nil,
                peerVideoID: String? = nil, syncManifestJSON: String? = nil) async {
```

and in the body, replace the `startTrack` call with:

```swift
            var job = try await api.startTrack(
                videoID: upload.videoID,
                calibrationJSON: calibration.calibrationJSON,
                duration: clipDuration,
                sessionID: sessionID, cameraRole: cameraRole,
                peerVideoID: peerVideoID, syncManifestJSON: syncManifestJSON)
```

Everything else in `submit` is unchanged.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/RunSubmissionTests
```

Expected: PASS, including the pre-existing single-camera tests untouched.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/API/APIClient.swift ios/Sources/Results/RunSubmission.swift ios/Tests/RunSubmissionTests.swift
git commit -m "feat(api): paired-session parameters on the upload path"
```

---

### Task 5: DESIGN.md — segment component, p-pair rows, native shell note

Doc-only, and it must land **before** Tasks 8–10: §17.1 requires a §8 subsection to exist before a component is reused on a new screen, and `.corrSeg` has none today.

**Files:**
- Modify: `DESIGN.md`

**Interfaces:**
- Consumes: nothing.
- Produces: §8.22 (two-way segment), the two new §16 `p-pair` state rows, and the §3.2 native-shell note that Tasks 8–10 implement against.

- [ ] **Step 1: Add §8.22, the two-way segment**

After §8.21 (coaching report panel), add a subsection documenting the control that `.corrSeg` already is, so the role picker reuses it rather than minting a second segment grammar:

```markdown
### 8.22 Two-way segment (`.corrSeg`)

Two equal-width options in one capsule; exactly one selected. Used by the call
page's Bounce / Not-bounce toggle (§16, `p-track`) and `p-pair`'s role picker
(§16, `p-pair`).

- Capsule (`border-radius:999px`), `--surface` fill, 1 px `--line` border.
- Each half ≥ 44 px tall and ≥ 44 px wide (§0.6); labels uppercase with
  `letter-spacing:.05em` (§0.7).
- The selected half takes `--accent-bg` with `--accent-text`; the unselected
  half stays `--dim` on `--surface`. Never green/red — selection is not a
  verdict (§0.3).
- Both halves are always present and always the same size, so selection never
  reflows anything (§0.9).
- No third state. A segment that needs one is a different component.
```

- [ ] **Step 2: Add the two new `p-pair` state rows**

In §16's `p-pair` state table, insert above the Idle row:

```markdown
| No role chosen | "Pick this phone's job to start." | PAIR (disabled) | §7: the primary never advertises a tap that cannot fire. There is no default role — two phones both defaulting to primary would both open a central and hang with nothing honest to say |
```

and after the Ready row:

```markdown
| Ready, awaiting peer calibration (primary) | "Paired · waiting for the other phone's calibration" | START RALLY (disabled) | the StereoEngine does not exist until the secondary's calibration arrives; starting would record a rally nothing can call |
| Ready (secondary) | "Paired · the other phone starts the rally" | — | the engine lives on the primary, so only the primary can honestly know a rally is callable |
```

Extend the existing Idle row's note with: `entering p-pair fetches this phone's solved camera model; while that is in flight the row reads "Checking this phone's court calibration…" and PAIR is disabled`.

- [ ] **Step 3: Add the `p-pair` body and `p-live` notes**

In §16's blueprint table, extend the `p-pair` Body cell with `· role segment (§8.22), idle state only`, and the `p-live` Body cell with `· mini-court replay is primary-only in v1 (the secondary receives relayed event JSON, which carries no 3D track) · while the link is degraded the secondary shows its own STOP, so a dropped link cannot leave it recording indefinitely`.

- [ ] **Step 4: Add the native-shell note to §3.2**

```markdown
**Native shell (iOS).** The native client substitutes `NavigationStack` and the
system back button for the header chevron and the proxied primary (§3.4), which
are web-shell mechanisms. The phase inventory and the §16 blueprints are shared;
only the chrome that moves between phases differs per client. The native Play
root shows two hero cards, not three — "Judge a clip" is a web file input with no
native equivalent, so "Record a clip" takes the accent slot there.
```

- [ ] **Step 5: Commit**

```bash
git add DESIGN.md
git commit -m "docs(design): two-way segment component, p-pair gates, native shell note"
```

---

### Task 6: `LiveSessionModel` — calibration gate, role, pairing

**Files:**
- Create: `ios/Sources/Live/LiveSessionModel.swift`
- Modify: `ios/Sources/Live/PairingModel.swift`
- Modify: `ios/Sources/Record/RecordModel.swift`
- Test: `ios/Tests/LiveSessionModelTests.swift`

**Interfaces:**
- Consumes: `PairingModel`, `PeerSession`, `RecordModel.attachPeer/attachStereo`, `APIClientProtocol.latestCalibration/fetchSolvedCameraModel`, `CameraModel.fromJSON/adoptedForCapture`.
- Produces: `LiveSessionModel.init(api:makeTransport:)`, `bind(record:)`, `prepare() async`, `primaryTapped()`, `role: PeerRole?`, and the published view state `calibration`, `linkStatus`, `primaryTitle`, `primaryEnabled`. Also `PairingModel.refresh()` and `RecordModel.onStereoReady`.

- [ ] **Step 1: Write the failing tests**

Create `ios/Tests/LiveSessionModelTests.swift`:

```swift
// ios/Tests/LiveSessionModelTests.swift
import XCTest
@testable import SquashLineCalling

@MainActor
final class LiveSessionModelTests: XCTestCase {
    /// A model with no camera bound and a stub API — enough to assert every
    /// gate in §16's p-pair table without a radio or a camera.
    private func makeModel(calibration: Result<String, Error> = .success("{}"))
        -> (LiveSessionModel, StubAPI) {
        let api = StubAPI(cameraModel: calibration)
        let model = LiveSessionModel(api: api, makeTransport: { _ in LoopbackTransport.pair().0 })
        return (model, api)
    }

    func testCalibrationInFlightKeepsPairDisabled() {
        let (model, _) = makeModel()
        XCTAssertEqual(model.calibration, .loading)
        XCTAssertEqual(model.linkStatus, "Checking this phone's court calibration…")
        XCTAssertFalse(model.primaryEnabled)
    }

    func testCalibrationFailureShowsTheReasonVerbatimAndBlocksPair() async {
        let (model, _) = makeModel(calibration: .failure(APIError.http(404, "No calibration on this phone.")))
        await model.prepare()
        XCTAssertEqual(model.calibration, .failed("No calibration on this phone."))
        XCTAssertEqual(model.linkStatus, "No calibration on this phone.")
        XCTAssertFalse(model.primaryEnabled)
    }

    func testPairStaysDisabledUntilARoleIsChosen() async {
        let (model, _) = makeModel()
        await model.prepare()
        XCTAssertEqual(model.calibration, .ready)
        XCTAssertNil(model.role)
        XCTAssertEqual(model.linkStatus, "Pick this phone's job to start.")
        XCTAssertFalse(model.primaryEnabled)

        model.role = .primary
        XCTAssertTrue(model.primaryEnabled)
        XCTAssertEqual(model.primaryTitle, "PAIR")
    }
}
```

Add a `StubAPI` conforming to `APIClientProtocol` in the same file: `latestCalibration()` returns a fixture, `fetchSolvedCameraModel` returns the injected result, and the remaining requirements throw `APIError.badResponse`. Reuse the camera-model JSON fixture that `StereoDemo.localModelJSON` already provides so `adoptedForCapture()` has something real to validate.

- [ ] **Step 2: Run to verify failure**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/LiveSessionModelTests
```

Expected: compile failure — no type `LiveSessionModel`.

- [ ] **Step 3: Add the two seams this model needs**

In `ios/Sources/Live/PairingModel.swift`, expose the existing private republish so an owner that is already pumping the session elsewhere can refresh without double-ticking:

```swift
    /// Republish without advancing the session's timers. `RecordModel.attachPeer`
    /// already ticks the peer at 20 Hz; a second `pump` here would tick it twice.
    func refresh() { syncFromSession() }
```

In `ios/Sources/Record/RecordModel.swift`, announce when the engine actually exists — the gate in decision 8 depends on it:

```swift
    /// Fired once the primary's StereoEngine is built from both camera models.
    /// Until then the primary cannot make a call, and START RALLY must say so.
    var onStereoReady: (() -> Void)?
```

and in `attachStereo`'s `onCalibration` handler, immediately after `self.stereoEngine = engine`:

```swift
                self.onStereoReady?()
```

- [ ] **Step 4: Write `LiveSessionModel`**

Create `ios/Sources/Live/LiveSessionModel.swift`:

```swift
// ios/Sources/Live/LiveSessionModel.swift
import Foundation

/// Owns the live two-camera lifecycle: the calibration gate, the role choice,
/// the PeerSession, and (Task 7) the rally and its paired upload.
///
/// Deliberately NOT part of `RecordModel`. DESIGN.md §16 makes "pairing adds
/// capability, never gates it" a hard requirement, so the camera model is owned
/// above this one and merely lent here — a defect in this file cannot reach
/// plain single-camera recording.
///
/// Publishes §16's p-pair table as flat view state so the whole table is
/// assertable without standing up a view, the same reason `PairingModel` exists.
@MainActor
final class LiveSessionModel: ObservableObject {
    enum Calibration: Equatable { case loading, ready, failed(String) }

    @Published private(set) var calibration: Calibration = .loading
    @Published private(set) var linkStatus = "Checking this phone's court calibration…"
    @Published private(set) var primaryTitle = "PAIR"
    @Published private(set) var primaryEnabled = false
    @Published private(set) var pairing = PairingModel(session: nil)
    /// Set once the primary's engine exists. The gate that stops START RALLY
    /// from recording a rally nothing can ever call.
    @Published private(set) var engineReady = false

    /// No default: two phones both defaulting to primary would both open a
    /// CBCentralManager and hang with nothing honest to show for it.
    @Published var role: PeerRole? { didSet { republish() } }

    #if DEBUG
    @Published var transportName = "ble"
    #endif

    private let api: APIClientProtocol
    private let makeTransport: (String) -> PeerTransport
    private weak var record: RecordModel?
    private var session: PeerSession?
    private var localModelJSON: String?
    private var pumpTimer: Timer?

    init(api: APIClientProtocol = APIClient(),
         makeTransport: @escaping (String) -> PeerTransport = LiveSessionModel.defaultTransport) {
        self.api = api
        self.makeTransport = makeTransport
    }

    static func defaultTransport(_ name: String) -> PeerTransport {
        name == "wifi-p2p" ? WiFiP2PTransport() : BLETransport()
    }

    /// Idempotent. The camera model is owned by `PlayRootView`, not by this
    /// object — see the type doc.
    func bind(record: RecordModel) {
        guard self.record == nil else { return }
        self.record = record
        record.onStereoReady = { [weak self] in
            Task { @MainActor in
                self?.engineReady = true
                self?.republish()
            }
        }
    }

    // MARK: - Calibration gate

    /// Fetch and validate this phone's solved camera model. Validating
    /// adoption here rather than inside `attachStereo` is the point: an
    /// unusable calibration must fail before two people walk to opposite
    /// corners of the court.
    func prepare() async {
        guard calibration != .ready else { return }
        calibration = .loading
        republish()
        do {
            let latest = try await api.latestCalibration()
            let json = try await api.fetchSolvedCameraModel(calibrationJSON: latest.calibrationJSON)
            guard let data = json.data(using: .utf8) else { throw APIError.badResponse }
            _ = try CameraModel.fromJSON(data).adoptedForCapture()
            localModelJSON = json
            calibration = .ready
        } catch {
            calibration = .failed(Self.describe(error))
        }
        republish()
    }

    /// §16 shows a failure reason verbatim, so prefer the server's own words.
    static func describe(_ error: Error) -> String {
        if let apiError = error as? APIError,
           case .http(_, let message) = apiError, let message {
            return message
        }
        return error.localizedDescription
    }

    // MARK: - The one primary (§7)

    func primaryTapped() {
        guard session != nil else { return beginPairing() }
        switch pairing.step {
        case .idle, .failed: beginPairing()
        case .confirm:       pairing.confirm()
        case .ready:         startRally()          // Task 7
        default:             break
        }
    }

    private func beginPairing() {
        guard case .ready = calibration, let role, let localModelJSON,
              let record else { return }
        #if DEBUG
        let transport = makeTransport(transportName)
        #else
        let transport = makeTransport("ble")
        #endif
        let session = PeerSession(transport: transport,
                                  isInitiator: role == .primary,
                                  orientation: record.camera.orientation)
        self.session = session
        // Order is load-bearing: attachStereo installs peer.onCalibration, so
        // attachPeer's `self.peer = peer` has to happen first, and both must
        // precede start() or a fast radio can deliver before we are listening.
        record.attachPeer(session)
        record.attachStereo(localModelJSON: localModelJSON)
        let pairing = PairingModel(session: session)
        pairing.onSessionEnded = { [weak self] in self?.endSession() }
        self.pairing = pairing
        pairing.start()
        startPump()
        republish()
    }

    func endSession() {
        pumpTimer?.invalidate(); pumpTimer = nil
        session = nil
        engineReady = false
        pairing = PairingModel(session: nil)
        republish()
    }

    /// `RecordModel.attachPeer` already ticks the peer at 20 Hz, so this pump
    /// only republishes — ticking here too would double-drive the timers.
    private func startPump() {
        pumpTimer?.invalidate()
        pumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pairing.refresh()
                self?.republish()
            }
        }
    }

    // MARK: - §16's p-pair table, as flat state

    private func republish() {
        let status = computedLinkStatus
        let title = computedPrimaryTitle
        let enabled = computedPrimaryEnabled
        if status != linkStatus { linkStatus = status }
        if title != primaryTitle { primaryTitle = title }
        if enabled != primaryEnabled { primaryEnabled = enabled }
    }

    private var computedLinkStatus: String {
        switch calibration {
        case .loading:              return "Checking this phone's court calibration…"
        case .failed(let reason):   return reason
        case .ready:                break
        }
        if session == nil && role == nil { return "Pick this phone's job to start." }
        if case .ready = pairing.step {
            if role == .primary {
                return engineReady ? pairing.statusLine
                                   : "Paired · waiting for the other phone's calibration"
            }
            return "Paired · the other phone starts the rally"
        }
        return pairing.statusLine
    }

    private var computedPrimaryTitle: String {
        guard session != nil else { return "PAIR" }
        switch pairing.step {
        case .idle, .searching, .failed: return "PAIR"
        case .confirm, .syncing:         return "CONFIRM"
        case .ready, .degraded:          return "START RALLY"
        case .live:                      return "RALLY LIVE"
        }
    }

    private var computedPrimaryEnabled: Bool {
        guard case .ready = calibration, role != nil else { return false }
        guard session != nil else { return true }
        switch pairing.step {
        case .searching, .syncing, .live, .degraded: return false
        case .idle, .failed:                         return pairing.canPair
        case .confirm:                               return pairing.canConfirm
        // Only the primary can honestly know a rally is callable, and only
        // once its engine exists.
        case .ready:                                 return role == .primary && engineReady
        }
    }

    deinit { pumpTimer?.invalidate() }
}
```

`startRally()` is added in Task 7; until then stub it as `private func startRally() {}` so this task compiles on its own.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/LiveSessionModelTests
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Live/LiveSessionModel.swift ios/Sources/Live/PairingModel.swift ios/Sources/Record/RecordModel.swift ios/Tests/LiveSessionModelTests.swift
git commit -m "feat(live): session coordinator with the calibration and role gates"
```

---

### Task 7: `LiveSessionModel` — rally lifecycle and paired upload

**Files:**
- Modify: `ios/Sources/Live/LiveSessionModel.swift`
- Test: `ios/Tests/LiveSessionModelTests.swift`

**Interfaces:**
- Consumes: `PeerSession.sendRecord/onRecord/sendSessionManifest/onSessionManifest` (Task 1), `RunSubmission.submit(...)` with paired params (Task 4), `RecordModel.toggleRecording()`, `ClockSync.estimate`.
- Produces: `startRally()`, `stopRally()`, `rally: RallyState`, `showsLocalStop: Bool`, `sessionID: String?`.

- [ ] **Step 1: Write the failing tests**

Append to `ios/Tests/LiveSessionModelTests.swift`:

```swift
    func testPrimaryBroadcastsRecordStartAndTheSecondaryFollows() async {
        let rig = await makePairedRig()          // two bound models over loopback
        rig.primary.startRally()
        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertEqual(rig.secondary.rally, .recording)
    }

    func testASecondStartWhileRecordingIsIgnored() async {
        let rig = await makePairedRig()
        rig.primary.startRally()
        rig.primary.startRally()
        XCTAssertEqual(rig.secondary.rally, .recording)
    }

    func testSecondaryGainsALocalStopOnlyWhileDegraded() async {
        let rig = await makePairedRig()
        rig.primary.startRally()
        XCTAssertFalse(rig.secondary.showsLocalStop)
        rig.secondary.handleDegraded(true)
        // Otherwise a dropped link leaves it recording 4K60 with no way out.
        XCTAssertTrue(rig.secondary.showsLocalStop)
        rig.secondary.handleDegraded(false)
        XCTAssertFalse(rig.secondary.showsLocalStop)
    }

    func testRolesMapToTheServersCameraRoles() async {
        let rig = await makePairedRig()
        XCTAssertEqual(rig.primary.cameraRole, "a")
        XCTAssertEqual(rig.secondary.cameraRole, "b")
    }

    func testPrimaryMintsAndBroadcastsASessionID() async {
        let rig = await makePairedRig()
        XCTAssertNotNil(rig.primary.sessionID)
        XCTAssertEqual(rig.secondary.sessionID, rig.primary.sessionID)
    }
```

`makePairedRig()` builds two `LiveSessionModel`s over one `LoopbackTransport.pair()`, each bound to a `RecordModel(detector: SyntheticBallDetector())`, drives them to `.ready` with the same tick loop `StereoWiringTests` uses, and returns both. Write it as a private helper in this file.

- [ ] **Step 2: Run to verify failure**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/LiveSessionModelTests
```

Expected: compile failure — no member `rally`.

- [ ] **Step 3: Add rally state and the session identity**

Add to `LiveSessionModel`:

```swift
    enum RallyState: Equatable { case idle, recording, submitting, submitted, failed(String) }

    @Published private(set) var rally: RallyState = .idle
    /// Only shown while the link is degraded: without it a dropped link
    /// leaves the secondary recording 4K60 with no way to end the rally.
    @Published private(set) var showsLocalStop = false
    @Published private(set) var sessionID: String?

    private var peerVideoID: String?
    private var submission = RunSubmission()

    /// The server's paired-run role. Fixed by the pairing role: initiator = a.
    var cameraRole: String? { role.map { $0 == .primary ? "a" : "b" } }
```

- [ ] **Step 4: Mint and exchange the session identity**

In `beginPairing()`, after `pairing.start()`, wire the two new callbacks and mint the ID on the primary:

```swift
        session.onRecord = { [weak self] action, _ in
            Task { @MainActor in self?.handleRemoteRecord(action) }
        }
        session.onSessionManifest = { [weak self] sessionID, videoID in
            Task { @MainActor in
                self?.sessionID = sessionID
                if !videoID.isEmpty { self?.peerVideoID = videoID }
            }
        }
```

and in `republish()`, once the primary first reaches `.ready`, announce it. Add to `republish()`:

```swift
        if role == .primary, case .ready = pairing.step, sessionID == nil {
            let minted = UUID().uuidString
            sessionID = minted
            session?.sendSessionManifest(sessionID: minted, videoID: "")
        }
        showsLocalStop = rally == .recording && {
            if case .degraded = pairing.step { return true }
            return false
        }()
```

- [ ] **Step 5: Start, stop, and submit**

```swift
    /// Primary-only, and only once the engine exists — both enforced by
    /// `computedPrimaryEnabled`. Recording is started locally first so a
    /// dropped message can never leave this phone not recording.
    func startRally() {
        guard rally != .recording else { return }
        rally = .recording
        Task { await record?.toggleRecording() }
        session?.goLive()
        session?.sendRecord(action: "start", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        republish()
    }

    func stopRally() {
        guard rally == .recording else { return }
        session?.sendRecord(action: "stop", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        finishRecordingAndSubmit()
    }

    private func handleRemoteRecord(_ action: String) {
        switch action {
        case "start": if rally != .recording { startLocalRecording() }
        case "stop":  if rally == .recording { finishRecordingAndSubmit() }
        default:      break
        }
    }

    /// The secondary's half of a remote start — no goLive broadcast, no
    /// re-send, or the two phones would ping-pong record messages forever.
    private func startLocalRecording() {
        rally = .recording
        Task { await record?.toggleRecording() }
        republish()
    }

    private func finishRecordingAndSubmit() {
        rally = .submitting
        republish()
        Task { [weak self] in
            guard let self, let record = self.record else { return }
            await record.toggleRecording()
            guard let clip = record.finishedClip else {
                self.rally = .failed("The rally produced no clip.")
                return self.republish()
            }
            await self.submission.submit(videoURL: clip.url, duration: clip.duration,
                                         sessionID: self.sessionID,
                                         cameraRole: self.cameraRole,
                                         peerVideoID: self.peerVideoID,
                                         syncManifestJSON: self.syncManifestJSON())
            switch self.submission.phase {
            case .complete:
                self.rally = .submitted
                // Best-effort enrichment only: fusion pairs on session_id +
                // camera_role, so a lost manifest never blocks it.
                if let sessionID = self.sessionID, let videoID = self.submission.completedRunID {
                    self.session?.sendSessionManifest(sessionID: sessionID, videoID: videoID)
                }
            case .failed(let message): self.rally = .failed(message)
            default:                   self.rally = .failed("Upload did not finish.")
            }
            self.republish()
        }
    }

    /// Seeds the server's offset refinement with what the phones measured.
    /// Primary-only — `job_runner` reads the manifest from whichever run
    /// carries it, so sending it twice would be redundant.
    ///
    /// Always `offset_series`, never `clap_anchor_s`: `ClockSync.anchor` is
    /// private, so the client cannot tell an anchored estimate from a network
    /// one. It costs nothing — when the anchor is applied, `estimate.offset`
    /// *is* the anchor value, and the server takes the median of a one-element
    /// series, which returns it exactly. Only the report's `seed.source` label
    /// differs.
    private func syncManifestJSON() -> String? {
        guard role == .primary, let estimate = session?.clockSync.estimate else { return nil }
        let payload: [String: Any] = ["offset_series": [estimate.offset]]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    /// Test seam for the degraded-link path, which needs no radio to assert.
    func handleDegraded(_ degraded: Bool) {
        showsLocalStop = degraded && rally == .recording
    }
```

Check `ClockSync.Estimate`'s actual property name before writing `estimate.offset` — `PairingModel` reads `estimate?.uncertainty`, so the offset field is adjacent; use whatever it is actually called.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/LiveSessionModelTests
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/Live/LiveSessionModel.swift ios/Tests/LiveSessionModelTests.swift
git commit -m "feat(live): rally lifecycle and paired upload"
```

---

### Task 8: `PairingView` renders the coordinator, plus the role segment

**Files:**
- Modify: `ios/Sources/Live/PairingView.swift`
- Test: none new — §16's table is asserted in `LiveSessionModelTests`, which is why the view can stay dumb.

**Interfaces:**
- Consumes: `LiveSessionModel.linkStatus/primaryTitle/primaryEnabled/role/pairing/primaryTapped()`.
- Produces: `PairingView(model: LiveSessionModel, onGoLive: () -> Void)`.

- [ ] **Step 1: Swap the observed object**

Change the view's stored properties:

```swift
struct PairingView: View {
    @ObservedObject var model: LiveSessionModel
    /// Tapping START RALLY hands off to `p-live` — the host owns that
    /// transition, the same way `p-analyze` auto-advances.
    var onGoLive: () -> Void = {}
```

- [ ] **Step 2: Read the flat state instead of recomputing it**

Delete `primaryTitle`, `primaryEnabled`, and `primaryAction` from the view and read the model's published values:

```swift
    private var primary: some View {
        Button(action: {
            model.primaryTapped()
            if model.primaryTitle == "START RALLY" { onGoLive() }
        }) {
            Text(model.primaryTitle)
                .font(.system(.headline).weight(.bold))
                .tracking(0.05 * 17)
                .foregroundStyle(Theme.accentText)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(Theme.accentBg, in: Capsule())
                .opacity(model.primaryEnabled ? 1 : 0.4)
        }
        .disabled(!model.primaryEnabled)
    }
```

Point `linkStatus` at `model.linkStatus` and `displayedCode` at `model.pairing.step`, and change `rejectCode`'s button to `model.pairing.rejectCode()`.

- [ ] **Step 3: Add the §8.22 role segment**

Shown only before a session exists, per §16:

```swift
    /// §8.22 two-way segment. No default selection: two phones both defaulting
    /// to primary would both open a central and hang.
    private var roleSegment: some View {
        HStack(spacing: 0) {
            segmentHalf("THIS PHONE CALLS", role: .primary)
            segmentHalf("THIS PHONE ASSISTS", role: .secondary)
        }
        .background(Theme.surface, in: Capsule())
        .overlay(Capsule().strokeBorder(Theme.line, lineWidth: 1))
        .opacity(model.pairing.step == .idle ? 1 : 0)
    }

    private func segmentHalf(_ title: String, role: PeerRole) -> some View {
        Button { model.role = role } label: {
            Text(title)
                .font(.system(.subheadline).weight(.semibold))
                .tracking(0.05 * 15)
                .foregroundStyle(model.role == role ? Theme.accentText : Theme.dim)
                .frame(maxWidth: .infinity, minHeight: 44)
                .background(model.role == role ? Theme.accentBg : Color.clear, in: Capsule())
        }
        .accessibilityLabel(title)
    }
```

Place it in `body`'s `VStack` between `linkStatus` and `pairCode`. Use `.opacity`, not a conditional, so the layout never shifts when it hides (§0.9).

- [ ] **Step 4: Fetch calibration on entry**

Add to `body`:

```swift
        .task { await model.prepare() }
```

- [ ] **Step 5: Build and run the full suite**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Live/PairingView.swift
git commit -m "feat(live): p-pair renders the coordinator and gains the role segment"
```

---

### Task 9: Play section root and navigation

**Files:**
- Create: `ios/Sources/Live/PlayRootView.swift`
- Modify: `ios/Sources/RootTabView.swift`
- Modify: `ios/Sources/Record/RecordView.swift`
- Modify: `ios/Sources/Record/RecordModel.swift`

**Interfaces:**
- Consumes: `RecordView(model:)`, `PairingView(model:onGoLive:)`, `LiveStageView(record:live:)` (Task 10).
- Produces: `PlayRootView`, and `RecordView(model: RecordModel)` taking an injected model.

- [ ] **Step 1: Make `startCamera` idempotent**

Both stage screens will call it from `.task`. In `ios/Sources/Record/RecordModel.swift`:

```swift
    private var cameraStarted = false

    func startCamera() async {
        guard !cameraStarted else { return }
        cameraStarted = true
        do {
            try await camera.configure()
            camera.start()
            exposureNote = CaptureSettings.summary(for: try await camera.lockForCourt())
        } catch {
            cameraStarted = false      // let a retry re-enter
            errorText = error.localizedDescription
        }
    }
```

- [ ] **Step 2: Inject the model into `RecordView`**

Replace `@StateObject private var model = RecordModel()` with:

```swift
    @ObservedObject var model: RecordModel
```

Nothing else in `RecordView` changes — it already reads only `model`.

- [ ] **Step 3: Write `PlayRootView`**

Create `ios/Sources/Live/PlayRootView.swift`:

```swift
// ios/Sources/Live/PlayRootView.swift
import SwiftUI

/// §16 `p-load` — the Play section root.
///
/// Owns `RecordModel`, which the live layer only borrows: DESIGN.md §16 makes
/// "pairing adds capability, never gates it" a hard requirement, so a defect in
/// `LiveSessionModel` must not be able to reach plain recording.
///
/// Two hero cards, not the web's three — "Judge a clip" is a file input with no
/// native equivalent, so "Record a clip" takes the accent slot (§3.2).
struct PlayRootView: View {
    @StateObject private var record = RecordModel()
    @StateObject private var live = LiveSessionModel()
    @State private var showServerSettings = false
    #if DEBUG
    @State private var showPeerBench = false
    #endif

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.bg.ignoresSafeArea()
                VStack(spacing: 12) {
                    NavigationLink { RecordView(model: record) } label: {
                        heroCard("Record a clip", "Film a rally with this phone's camera",
                                 systemImage: "video", accent: true)
                    }
                    NavigationLink {
                        PairingView(model: live)
                    } label: {
                        heroCard("Live match", "Record and call in real time",
                                 systemImage: "dot.radiowaves.left.and.right", accent: false)
                    }
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.top, 18)
            }
            .navigationTitle("Play")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showServerSettings = true } label: {
                        Image(systemName: "gearshape").foregroundStyle(Theme.dim)
                    }
                    .accessibilityLabel("Server settings")
                }
                #if DEBUG
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showPeerBench = true } label: {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                            .foregroundStyle(Theme.dim)
                    }
                }
                #endif
            }
        }
        .task { live.bind(record: record) }
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        #if DEBUG
        .sheet(isPresented: $showPeerBench) { NavigationStack { PeerBenchView() } }
        #endif
    }

    private func heroCard(_ title: String, _ subtitle: String,
                          systemImage: String, accent: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(accent ? Theme.accentText : Theme.text)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(.headline).weight(.bold))
                    .foregroundStyle(accent ? Theme.accentText : Theme.text)
                Text(subtitle).font(.footnote)
                    .foregroundStyle(accent ? Theme.accentText.opacity(0.7) : Theme.dim)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .frame(minHeight: 72)
        .background(accent ? Theme.accentBg : Theme.surface,
                    in: RoundedRectangle(cornerRadius: 14))
    }
}
```

The `PairingView` destination gains its `onGoLive` push in Task 10; leave it as shown here for now.

- [ ] **Step 4: Point the Play tab at the new root**

In `ios/Sources/RootTabView.swift`, replace the whole `RecordView()` entry and its overlays/sheets with:

```swift
            PlayRootView()
                .tabItem { Label("Play", systemImage: "record.circle") }
```

Delete the now-unused `showServerSettings` / `showPeerBench` state from `RootTabView` — they moved to `PlayRootView`.

- [ ] **Step 5: Build and run the full suite**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Live/PlayRootView.swift ios/Sources/RootTabView.swift ios/Sources/Record/RecordView.swift ios/Sources/Record/RecordModel.swift
git commit -m "feat(ios): Play section root routing to record and live"
```

---

### Task 10: `LiveStageView` — the `p-live` screen

**Files:**
- Create: `ios/Sources/Live/LiveStageView.swift`
- Modify: `ios/Sources/Live/PlayRootView.swift`

**Interfaces:**
- Consumes: `RecordModel.livePresentation/flashPresentation/liveTrack`, `LiveSessionModel.rally/showsLocalStop/stopRally()`, `CallFlashView`, `CallBannerView`, `MiniCourtView`.
- Produces: `LiveStageView(record:live:)`.

- [ ] **Step 1: Write the screen**

Create `ios/Sources/Live/LiveStageView.swift`:

```swift
// ios/Sources/Live/LiveStageView.swift
import SwiftUI

/// §16 `p-live` — the record stage plus the call flash, the honest-state
/// banner, and the post-rally mini-court.
///
/// The mini-court is primary-only in v1: the secondary sees relayed event JSON,
/// which carries no 3D track (spec, "Deliberate v1 narrowing"). Its footprint is
/// reserved either way so nothing shifts when a call lands (§0.9).
struct LiveStageView: View {
    @ObservedObject var record: RecordModel
    @ObservedObject var live: LiveSessionModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: record.camera.session).ignoresSafeArea()
            OverlayView(trail: record.trail).ignoresSafeArea()
            CallFlashView(presentation: record.flashPresentation)
                .allowsHitTesting(false)
                .ignoresSafeArea()

            VStack(spacing: 10) {
                Text(live.linkStatus)
                    .font(.system(.subheadline).monospacedDigit())
                    .foregroundStyle(Theme.dim)
                    .frame(minHeight: 40)
                Spacer(minLength: 0)
                miniCourt
                CallBannerView(presentation: record.livePresentation)
                    .padding(.horizontal, 14)
                if live.showsLocalStop {
                    Button("STOP") { live.stopRally() }
                        .font(.system(.headline).weight(.bold))
                        .tracking(0.05 * 17)
                        .foregroundStyle(Theme.accentText)
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(Theme.accentBg, in: Capsule())
                        .padding(.horizontal, 14)
                } else {
                    Color.clear.frame(height: 48).padding(.horizontal, 14)
                }
            }
            .padding(.bottom, 24)
        }
        .task { await record.startCamera() }
    }

    /// Never visible before a call (§16), and its footprint is reserved either
    /// way so appearing shifts nothing.
    @ViewBuilder private var miniCourt: some View {
        if record.livePresentation != nil, record.flashPresentation == nil,
           !record.liveTrack.isEmpty {
            MiniCourtView(track: record.liveTrack, impact: record.liveImpact)
                .frame(height: 180)
                .padding(.horizontal, 14)
        } else {
            Color.clear.frame(height: 180)
        }
    }
}
```

- [ ] **Step 2: Push it from `p-pair`**

In `PlayRootView`, replace the `PairingView` destination with one that pushes the live stage on START RALLY:

```swift
                    NavigationLink {
                        PairingView(model: live) { showLive = true }
                            .navigationDestination(isPresented: $showLive) {
                                LiveStageView(record: record, live: live)
                            }
                    } label: {
```

adding `@State private var showLive = false` to `PlayRootView`.

- [ ] **Step 3: Build and run the full suite**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ios/Sources/Live/LiveStageView.swift ios/Sources/Live/PlayRootView.swift
git commit -m "feat(live): the p-live stage with flash, banner, and mini-court"
```

---

### Task 11: Verification pass

**Files:**
- Modify: `ios/PEER.md`
- Modify: `CLAUDE.md` (test count only, if it changed)

- [ ] **Step 1: Run both suites and record the numbers**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 348 passed, unchanged — no Python file is touched by this plan.

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS, with the new `LiveSessionModelTests` included.

- [ ] **Step 2: Confirm the goldens are untouched**

```bash
git status --porcelain tests/stereo_goldens.json ios/Tests/Fixtures/stereo_goldens.json
```

Expected: empty output. Any diff means Task 3 changed arithmetic — stop and fix it.

- [ ] **Step 3: Verify the screens**

Use the `/verify` skill on `p-load`, `p-pair`, and `p-live` at 390 × 844 in **both** themes (§0.12). Confirm: the role segment reserves its footprint when hidden, the calibration status line does not reflow the pair code, and the mini-court appearing shifts nothing.

- [ ] **Step 4: Drive the DEBUG stereo demo end to end**

From the record stage's cube button, confirm it still resolves to `IN · high confidence` after Task 3's event-shape change, and that the flash and banner still render.

- [ ] **Step 5: Re-read §0 and §18 against the new screens**

Check the twelve binding rules and the never-do list against `p-load`, `p-pair`, and `p-live`. Fix drift, or document the deviation in DESIGN.md.

- [ ] **Step 6: Update PEER.md's live-path runbook**

Replace the "Two-phone bring-up" step 3 with the role step: both phones to Play → Live match, each picks its job (one calls, one assists), then PAIR on both, compare the code, CONFIRM on both. Add that START RALLY only lights on the calling phone, and only once the assisting phone's calibration has arrived.

- [ ] **Step 7: Commit**

```bash
git add ios/PEER.md CLAUDE.md
git commit -m "docs(peer): live-path runbook for the wired-up entry point"
```

---

## Self-Review Notes

**Spec coverage.** Architecture/ownership → Tasks 6, 9. Navigation → Task 9. DESIGN.md changes (all seven) → Task 5. Data flow steps 1–2 → Task 6; 3–5 → Task 6; 6–7 → Tasks 6–7; 8–11 → Task 7; 12–13 → Tasks 7, 10; 14–16 → Tasks 4, 7. `PeerSession` seams → Task 1. Orientation guard → Task 2. Stereo track → Task 3. API → Task 4. Error handling table → Tasks 6 (calibration rows), 2 (orientation), 7 (degraded, upload). Testing section → Tasks 1–7, 11.

**Known soft spots the implementer must resolve by reading, not guessing:**
- Task 3 Step 1's `goldenSamples(for:)` and Task 6 Step 1's `StubAPI` are described rather than written, because both must mirror fixtures that already exist in those files (`StereoEngineTests`'s `left`/`right` + `decode(_:)`, and `StereoDemo.localModelJSON`). Writing a second fixture instead of lifting the existing one would be a defect.
- Task 7 Step 1's `makePairedRig()` is likewise described: it must reuse `StereoWiringTests`'s tick loop (`for _ in 0..<40 { t += 0.1; ... }`) verbatim, since that is the established way to drive a loopback pair to `.ready`.

**Knowing divergence from the spec.** The spec says the sync manifest carries `clap_anchor_s` when anchored, otherwise `offset_series`. The plan always sends `offset_series`, because `ClockSync.anchor` is private and the client cannot distinguish the two. This is lossless — an anchored `estimate.offset` *is* the anchor value, and the server medians a one-element series back to it — and costs only the `seed.source` label in the fusion report. Exposing `ClockSync.isAnchored` to recover the label is a follow-up, not part of this plan.

**Type consistency.** `PeerRole` (`.primary` / `.secondary`) is the pairing role throughout; the server's `camera_role` string (`"a"` / `"b"`) appears only via `LiveSessionModel.cameraRole`. `StereoEvent.impact(_:track:)` is produced in Task 3 and consumed in Tasks 3 and 10. `LiveSessionModel.primaryTapped()` is defined in Task 6 and called in Task 8. `bind(record:)` is defined in Task 6 and called in Task 9.

**Biggest risk.** Task 9 changes `RecordView`'s ownership model, which is the one file the shipping single-camera path depends on. If it regresses, plain recording breaks — which is exactly the outcome the whole ownership design exists to prevent. Verify recording end-to-end (record → ResultsView → upload) before moving on, not just that the suite compiles.
