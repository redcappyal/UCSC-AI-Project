// ios/Tests/LiveSessionModelTests.swift
import XCTest
@testable import SquashLineCalling

/// Fetches whatever `cameraModel` says on success, throws it on failure. The
/// remaining `APIClientProtocol` requirements are never exercised by
/// `LiveSessionModel` (it only calls `latestCalibration` and
/// `fetchSolvedCameraModel`), so they throw loudly rather than fake success.
///
/// Deliberately NOT `StereoDemo.localModelJSON`: that fixture is pinned to its
/// own 1920x1080 landscape aspect (its doc comment: used *unadopted*, because
/// `adoptedForCapture()` would throw for it). `LiveSessionModel.prepare()`
/// calls `adoptedForCapture()`, which always targets
/// `CaptureSettings.frameWidth x frameHeight` (2160x3840, portrait). A
/// landscape fixture would make every "calibration ready" test fail with
/// `ScaleError.aspectMismatch` instead of reaching `.ready`.
/// `adoptableCameraModelJSON` below uses the same 1080x1920 portrait aspect
/// `CameraFrameSpaceTests.testAdoptedForCaptureLandsInTheLivePixelSpace`
/// already proves adopts cleanly (a clean 2x scale to 2160x3840).
private struct StubAPI: APIClientProtocol {
    static let adoptableCameraModelJSON = """
    {"focal_px":1600.0,"center_px":[1080.0,1920.0],\
    "rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],\
    "camera_center_ft":[9.0,31.95,7.0],"distortion":null,\
    "frame_width":1080.0,"frame_height":1920.0}
    """

    private static let fixtureCalibration = try! LatestCalibration(responseData: Data(
        #"{"ok": true, "run_id": "7", "calibration": {"lines": []}}"#.utf8))

    var cameraModel: Result<String, Error>

    func latestCalibration() async throws -> LatestCalibration { Self.fixtureCalibration }

    func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String {
        switch cameraModel {
        case .success(let json): return json
        case .failure(let error): throw error
        }
    }

    func upload(videoURL: URL) async throws -> UploadResponse { throw APIError.badResponse }

    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String?, cameraRole: String?,
                    peerVideoID: String?, syncManifestJSON: String?) async throws -> JobStatus {
        throw APIError.badResponse
    }

    func trackStatus(runID: String) async throws -> JobStatus { throw APIError.badResponse }
}

@MainActor
final class LiveSessionModelTests: XCTestCase {
    /// A model with no camera bound and a stub API — enough to assert every
    /// gate in §16's p-pair table without a radio or a camera. Defaults to a
    /// fresh, unretained loopback half: fine for the tests that never call
    /// `primaryTapped()`/`beginPairing()`; tests that do drive a real session
    /// pass their own `makeTransport` so they can keep the other half alive.
    private func makeModel(cameraModel: Result<String, Error> = .success(StubAPI.adoptableCameraModelJSON),
                          makeTransport: @escaping (String) -> PeerTransport = { _ in LoopbackTransport.pair().0 })
        -> (LiveSessionModel, StubAPI) {
        let api = StubAPI(cameraModel: cameraModel)
        let model = LiveSessionModel(api: api, makeTransport: makeTransport)
        return (model, api)
    }

    func testCalibrationInFlightKeepsPairDisabled() {
        let (model, _) = makeModel()
        XCTAssertEqual(model.calibration, .loading)
        XCTAssertEqual(model.linkStatus, "Checking this phone's court calibration…")
        XCTAssertFalse(model.primaryEnabled)
    }

    func testCalibrationFailureShowsTheReasonVerbatimAndBlocksPair() async {
        let (model, _) = makeModel(cameraModel: .failure(APIError.http(404, "No calibration on this phone.")))
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

    // MARK: - Fix 1: PAIR must be re-enabled after a pairing failure

    /// DESIGN.md §16's `p-pair` table: `Failed → PAIR (retry)`. Reached via
    /// the real failure that motivated the fix — `PeerSession`'s
    /// capture-orientation guard — by pairing this model's session (built
    /// portrait, via `beginPairing()`) against a hand-built secondary that
    /// advertises `.landscapeRight` instead. Both ends independently reject
    /// the other's hello, so no cooperation from a "remote app" is needed.
    ///
    /// `primaryTitle`/`primaryEnabled` are cached by `republish()`, a private
    /// method only invoked from a few call sites — notably `startPump()`'s
    /// 0.05 s timer (`pairing.refresh(); republish()`), since there is no
    /// Combine subscription from `LiveSessionModel` to `PeerSession.$phase`.
    /// So this spins the real run loop rather than poking any internals: the
    /// same timer `beginPairing()` already started picks up the failure and
    /// republishes it, exactly as it would on a device.
    func testPairIsReenabledAfterAPairingFailure() async {
        let pair = LoopbackTransport.pair()
        let (model, _) = makeModel(makeTransport: { _ in pair.0 })
        let record = RecordModel(detector: nil)
        model.bind(record: record)
        await model.prepare()
        XCTAssertEqual(model.calibration, .ready)
        model.role = .primary

        // Wired before the primary session exists, so its hello (sent by
        // beginPairing() below) is not lost to an unwired `onControl`.
        let secondary = PeerSession(transport: pair.1, isInitiator: false,
                                    orientation: .landscapeRight)

        model.primaryTapped()   // beginPairing(): builds the primary session (portrait)
                                 // and starts its 0.05 s pump
        secondary.start()       // sends the secondary's (landscape) hello — the primary
                                 // rejects it as an orientation mismatch and fails

        let deadline = Date().addingTimeInterval(2.0)
        while true {
            if case .failed = model.pairing.step { break }
            guard Date() < deadline else { break }
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }

        guard case .failed = model.pairing.step else {
            return XCTFail("expected the orientation mismatch to fail pairing, got \(model.pairing.step)")
        }
        XCTAssertEqual(model.primaryTitle, "PAIR")
        XCTAssertTrue(model.primaryEnabled)
        // Both ends independently reject the other's hello; checking the
        // secondary too (rather than just letting it go unused after
        // `.start()`) documents that and keeps it alive for the whole wait.
        guard case .failed = secondary.phase else {
            return XCTFail("expected the secondary to fail as well, got \(secondary.phase)")
        }
    }

    // MARK: - Fix 2: ending a session must detach the camera model

    #if DEBUG
    /// `RecordModel.peer`/`.stereoEngine` are private and unreachable from
    /// here, so this proves `endSession()` → `RecordModel.detachPeer()` on
    /// the observable published surface instead: populate the live-call
    /// state via the DEBUG stereo demo (no peer needed, same fields
    /// `detachPeer()` clears), then assert `endSession()` clears them.
    func testEndSessionClearsRecordModelsLiveCallState() {
        let (model, _) = makeModel()
        let record = RecordModel(detector: nil)
        model.bind(record: record)

        record.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                               remoteModelJSON: StereoDemo.remoteModelJSON)
        let deadline = Date().addingTimeInterval(3.0)
        while record.livePresentation == nil && Date() < deadline {
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }
        XCTAssertNotNil(record.livePresentation, "setup: the demo never produced a call")
        XCTAssertNotNil(record.flashPresentation, "setup: the flash never appeared")

        model.endSession()

        XCTAssertNil(record.livePresentation)
        XCTAssertNil(record.liveImpact)
        XCTAssertTrue(record.liveTrack.isEmpty)
        XCTAssertNil(record.flashPresentation)
        XCTAssertTrue(record.stereoEvents.isEmpty)
    }
    #endif

    // MARK: - Task 7: rally lifecycle and paired upload

    func testPrimaryBroadcastsRecordStartAndTheSecondaryFollows() async {
        let rig = await makePairedRig()          // two bound models over loopback
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertEqual(rig.secondary.rally, .recording)
    }

    /// `RecordModel.toggleRecording()`'s start branch can throw and leave
    /// `isRecording == false` with the reason swallowed into `errorText` —
    /// before this fix, nothing read that back, so `startRally()` proceeded
    /// straight into `rally == .recording` regardless. Reproducing the real
    /// throw needs no seam: `CameraController` is a concrete, non-injectable
    /// type, and `startRecording()`'s `AVAssetWriter` setup doesn't depend
    /// on a running capture session, so it doesn't reliably fail in a test
    /// host. So this simulates the state `toggleRecording()`'s catch branch
    /// leaves behind (`isRecording = false`, `errorText` set) and drives
    /// `reconcileRallyAfterStartAttempt()` directly — the exact
    /// reconciliation `startRally()` schedules for itself once its own
    /// toggle completes — rather than the real throw.
    func testAStartThatFailsToActuallyRecordDoesNotLeaveRallyBelievingItIsRecording() {
        let (model, _) = makeModel()
        let record = RecordModel(detector: nil)
        model.bind(record: record)

        model.startRally()
        XCTAssertEqual(model.rally, .recording, "setup: startRally() optimistically goes .recording")

        // The real toggle this startRally() call scheduled hasn't run yet —
        // it's chained behind a suspension point this synchronous test never
        // reaches — so overwriting this state first is race-free, not a
        // guess about scheduling order.
        record.isRecording = false
        record.errorText = "camera would not start"
        model.reconcileRallyAfterStartAttempt()

        XCTAssertEqual(model.rally, .failed("camera would not start"))
        // Before this fix, `rally` would still read `.recording` here, and
        // `stopRally()`'s own guard (`rally == .recording`) would then let a
        // "stop" through to a toggle that — with `isRecording` false — would
        // START the camera instead of stopping it, with no remaining exit.
        // Confirm that door is shut: `rally` no longer says `.recording`, so
        // `stopRally()` is a no-op.
        model.stopRally()
        XCTAssertEqual(model.rally, .failed("camera would not start"))
    }

    /// The guard this test targets is `startRally()`'s own `guard rally !=
    /// .recording else { return }`. Deleting it and asserting only the
    /// SECONDARY's `rally` (as this test used to) can't catch that: a
    /// second call would re-toggle the PRIMARY's own camera and re-send
    /// "start", and the secondary's own separate guard in
    /// `handleRemoteRecord` (`if rally != .recording { startLocalRecording()
    /// }`) ignores a repeated "start" regardless of whether the primary's
    /// guard exists — so the secondary's state can't distinguish a working
    /// guard from a deleted one. Assert the PRIMARY's own side instead:
    /// its `rally` is unchanged, and — critically — its `RecordModel
    /// .isRecording` is still true, i.e. the second call did not silently
    /// stop the camera it was already recording on.
    func testASecondStartWhileRecordingIsIgnored() async {
        let rig = await makePairedRig()
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        // Let the primary's own first-start toggle actually finish (not
        // just the secondary's delivery of it) before firing the second
        // call, so `isRecording` reflects a completed start rather than an
        // in-flight one.
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.startRally()
        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertTrue(rig.primaryRecord.isRecording,
                      "a second startRally() must not silently stop the camera it's already recording on")
    }

    #if DEBUG
    /// `handleDegraded(_:)` is `#if DEBUG`-only (production never calls it —
    /// the 20 Hz pump would revert its write within 50 ms anyway), matching
    /// this guard to it.
    func testSecondaryGainsALocalStopOnlyWhileDegraded() async {
        let rig = await makePairedRig()
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertFalse(rig.secondary.showsLocalStop)
        rig.secondary.handleDegraded(true)
        // Otherwise a dropped link leaves it recording 4K60 with no way out.
        XCTAssertTrue(rig.secondary.showsLocalStop)
        rig.secondary.handleDegraded(false)
        XCTAssertFalse(rig.secondary.showsLocalStop)
    }
    #endif

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

    // MARK: - The paired rig

    /// Spins until `done()` is true or a short deadline elapses.
    ///
    /// The hop `onRecord`/`onSessionManifest` use in
    /// `LiveSessionModel.beginPairing()` to reach a *different* model's
    /// main-actor-isolated state is a fresh, separate
    /// `Task { @MainActor in ... }`: loopback delivers the underlying
    /// control frame synchronously, inline in the caller's stack, but that
    /// final hop is not part of that stack, so a synchronous assertion
    /// immediately after e.g. `startRally()` would still see the
    /// pre-delivery state. Spinning the run loop (rather than guessing how
    /// many `Task.yield()`s the pending job needs) lets it actually run.
    private func settleCrossModelDelivery(until done: () -> Bool) async {
        let deadline = Date().addingTimeInterval(2.0)
        while !done() {
            guard Date() < deadline else { return }
            await Task.yield()
            RunLoop.main.run(until: Date().addingTimeInterval(0.01))
        }
    }

    /// Two `LiveSessionModel`s, each bound to its own `RecordModel`, paired
    /// over one `LoopbackTransport.pair()` and driven to `.ready` on both
    /// sides. `primaryRecord`/`secondaryRecord` come along for the ride and
    /// must be kept alive by the caller holding the whole tuple:
    /// `bind(record:)` stores its argument `weak`, so with nothing else
    /// retaining them they would be deallocated the instant this method
    /// returns, silently turning every later `beginPairing()` guard
    /// (`let record`) into a no-op.
    ///
    /// Does NOT hand-tick a synthetic clock the way `StereoWiringTests`/
    /// `PairingModelTests` do (`for _ in 0..<40 { t += 0.1; ...tick... }`).
    /// `beginPairing()` — reachable only through the public `primaryTapped()`
    /// — unconditionally starts `RecordModel.attachPeer`'s real 20 Hz
    /// `peer.tick(now: ClockSync.hostNow())` timer, and there is no seam to
    /// suppress it. Feeding the same `PeerSession` a synthetic near-zero `t`
    /// and then letting that real timer fire with a huge real host-uptime
    /// `t` would make `tick()`'s `t - lastPeerActivityAt > heartbeatTimeout`
    /// check see an enormous jump and degrade the link before anything could
    /// ever observe `.ready`. Spinning the real run loop instead — as
    /// `testPairIsReenabledAfterAPairingFailure` above already does — lets
    /// the real timer do all of the ticking, in one consistent time domain,
    /// exactly as production does.
    private func makePairedRig() async -> (primary: LiveSessionModel, secondary: LiveSessionModel,
                                            primaryRecord: RecordModel, secondaryRecord: RecordModel) {
        let pair = LoopbackTransport.pair()

        // beginPairing() constructs the PeerSession and calls its start()
        // (which sends this side's hello) in one call. Whichever model's
        // primaryTapped() below runs first would fire its hello before the
        // other side's PeerSession — and its transport.onControl — exists to
        // receive it; LoopbackTransport does not buffer, so an unwired
        // onControl just drops the frame. Buffer both directions until both
        // sessions exist, then release together, so neither hello is lost.
        // (PairingModelTests.makeRig() sidesteps the same hazard by
        // constructing both PeerSessions before calling either start(); that
        // seam isn't reachable through LiveSessionModel's public surface, so
        // this reproduces the same effect with a delivery hook instead.)
        var buffered: [() -> Void] = []
        pair.0.controlDeliveryHook = { frame, deliver in buffered.append { deliver(frame) } }
        pair.1.controlDeliveryHook = { frame, deliver in buffered.append { deliver(frame) } }

        let (primary, _) = makeModel(makeTransport: { _ in pair.0 })
        let (secondary, _) = makeModel(makeTransport: { _ in pair.1 })
        // No detector needed anywhere in this rig, and `SyntheticBallDetector`
        // is `#if DEBUG`-only — matching `testPairIsReenabledAfterAPairingFailure`'s
        // `RecordModel(detector: nil)` above keeps this file's one unguarded
        // construction of `RecordModel` from depending on a DEBUG-only type.
        let primaryRecord = RecordModel(detector: nil)
        let secondaryRecord = RecordModel(detector: nil)
        primary.bind(record: primaryRecord)
        secondary.bind(record: secondaryRecord)

        await primary.prepare()
        await secondary.prepare()
        primary.role = .primary
        secondary.role = .secondary

        primary.primaryTapped()    // beginPairing(): builds + starts the primary session
        secondary.primaryTapped()  // beginPairing(): builds + starts the secondary session

        pair.0.controlDeliveryHook = nil
        pair.1.controlDeliveryHook = nil
        let queuedHellos = buffered
        buffered = []
        for send in queuedHellos { send() }

        // Pull `pairing.step` up to date with the session's real phase
        // (already `.confirming` on both sides now that the hellos landed)
        // before deciding whether to confirm — `primaryTapped()`'s `.confirm`
        // branch reads `pairing.step`, and nothing has resynced it yet.
        primary.pairing.refresh()
        secondary.pairing.refresh()
        if case .confirm = primary.pairing.step { primary.primaryTapped() }
        if case .confirm = secondary.pairing.step { secondary.primaryTapped() }

        let deadline = Date().addingTimeInterval(5.0)
        var reachedReady = false
        while true {
            let primaryReady: Bool = { if case .ready = primary.pairing.step { return true }; return false }()
            let secondaryReady: Bool = { if case .ready = secondary.pairing.step { return true }; return false }()
            if primaryReady, secondaryReady,
               primary.sessionID != nil, secondary.sessionID == primary.sessionID {
                reachedReady = true
                break
            }
            guard Date() < deadline else { break }
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }

        // Assert the setup's own postcondition here, rather than letting a
        // broken pairing fall through to whatever assertion the calling
        // test happens to make first: that produces a confusing failure
        // about e.g. `rally` when the actual problem is that the rig never
        // reached `.ready` at all.
        if !reachedReady {
            XCTFail("""
                makePairedRig setup did not reach ready in time: \
                primary.step=\(primary.pairing.step), secondary.step=\(secondary.pairing.step), \
                primary.sessionID=\(String(describing: primary.sessionID)), \
                secondary.sessionID=\(String(describing: secondary.sessionID))
                """)
        }

        return (primary, secondary, primaryRecord, secondaryRecord)
    }
}
