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

/// Records the `.calibration` frames the secondary actually put on the wire.
///
/// A reference type rather than a captured local (the shape
/// `testASecondStartWhileRecordingIsIgnored` uses for its own counter) for two
/// reasons: the hook that fills it is installed inside `handshake(...)` while
/// the assertions live in the test that called it, and it has to survive a
/// *second* `handshake(...)` on the same two models — the re-pairing case —
/// and keep accumulating across both.
///
/// No locking: `LoopbackTransport` delivers synchronously on the caller's own
/// stack, and the only thing that sends a `.calibration` is
/// `LiveSessionModel.republish()`, which is `@MainActor`.
private final class CalibrationCounter {
    private(set) var profileIDs: [String] = []
    private(set) var payloads: [String] = []
    var count: Int { payloads.count }

    func record(profileID: String, payloadJSON: String) {
        profileIDs.append(profileID)
        payloads.append(payloadJSON)
    }
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

    /// `RecordModel`'s start branch can throw and leave `isRecording == false`
    /// with the reason swallowed into `errorText` — before this fix, nothing
    /// read that back, so `startRally()` proceeded straight into
    /// `rally == .recording` regardless. Reproducing the real throw needs no
    /// seam: `CameraController` is a concrete, non-injectable type, and
    /// `startRecording()`'s `AVAssetWriter` setup doesn't depend on a running
    /// capture session, so it doesn't reliably fail in a test host. So this
    /// simulates the state that catch branch leaves behind
    /// (`isRecording = false`, `errorText` set) and drives
    /// `reconcileRallyAfterStartAttempt()` directly — the exact
    /// reconciliation `startRally()` schedules for itself once its own
    /// transition completes — rather than the real throw.
    ///
    /// Teardown (added in review): `model.startRally()`'s own chained
    /// transition is real — there is no seam to stub the camera itself — so
    /// once this test's synchronous assertions above finish, that
    /// already-scheduled `Task` is still pending. Its guard
    /// (`canSetRecording(true, owner: .live)`, which the manual
    /// `isRecording = false` overwrite above still satisfies at the point
    /// that `Task` actually runs) passes, so it goes on to really open an
    /// `AVAssetWriter`, with nothing left to ever stop it once `record`
    /// deallocates at the end of this method. Rather than depend on the
    /// weak `self`/`record` captures inside the funnel /
    /// `reconcileRallyAfterStartAttempt` happening to resolve to nil before
    /// that `Task` runs — plausible, but nothing in this file pins down
    /// that ordering as guaranteed — `async` here buys an explicit place to
    /// wait for the pending `Task` to actually finish, then explicitly stop
    /// whatever it started before `record` goes out of scope.
    func testAStartThatFailsToActuallyRecordDoesNotLeaveRallyBelievingItIsRecording() async {
        let (model, _) = makeModel()
        let record = RecordModel(detector: nil)
        model.bind(record: record)

        model.startRally()
        XCTAssertEqual(model.rally, .recording, "setup: startRally() optimistically goes .recording")

        // The real transition this startRally() call scheduled hasn't run
        // yet — it's chained behind a suspension point this synchronous
        // section never reaches — so overwriting this state first is
        // race-free, not a guess about scheduling order.
        record.isRecording = false
        record.errorText = "camera would not start"
        model.reconcileRallyAfterStartAttempt()

        XCTAssertEqual(model.rally, .failed("camera would not start"))
        // Before this fix, `rally` would still read `.recording` here, and
        // `stopRally()`'s own guard (`rally == .recording`) would then let a
        // "stop" through against a camera that never started — back when the
        // call was a raw toggle, that STARTED it instead, with no remaining
        // exit. Confirm that door is shut: `rally` no longer says
        // `.recording`, so `stopRally()` is a no-op.
        model.stopRally()
        XCTAssertEqual(model.rally, .failed("camera would not start"))

        // Teardown: let startRally()'s own pending chained transition
        // actually run — its guard still matches the `isRecording = false`
        // forced above, so it really starts the camera (an `AVAssetWriter`
        // needs no running capture session — the same reason this whole test
        // can't make a *real* start fail deterministically) and sets
        // `isRecording = true` with `.live` as the owner. Then explicitly
        // stop it as that same owner, exactly as a real session would
        // eventually do, so no writer is left dangling once `record`
        // deallocates.
        await settleCrossModelDelivery(until: { record.isRecording })
        if record.isRecording {
            await record.setRecording(false, owner: .live)
        }
        XCTAssertFalse(record.isRecording, "teardown: no AVAssetWriter should be left running past this test")
    }

    /// The guard this test targets is `startRally()`'s own `guard rally !=
    /// .recording else { return }`. Two things (both found in review) make a
    /// weaker version of this test vacuous:
    ///
    /// 1. There is no suspension point between the second `startRally()`
    ///    call and an assertion right after it, and this test class is
    ///    `@MainActor`, so the unstructured `Task` `startRally()`
    ///    kicks off structurally cannot have run yet by the time such an
    ///    assertion executes — an `isRecording`-based assertion taken
    ///    immediately is incapable of having changed either way.
    /// 2. Even after settling for that `Task`, deleting the guard leaves
    ///    every *state* assertion passing anyway: the second call's own
    ///    transition would be skipped by `RecordModel.canSetRecording
    ///    (_:owner:)`'s "camera already agrees with what I intend" check,
    ///    `rally` would just be re-assigned the value it already had, and
    ///    the reconcile that follows would no-op. The *only* observable
    ///    difference a deleted
    ///    guard produces is an extra `goLive()` / `sendRecord(action:
    ///    "start", ...)` frame put on the wire — so that is what this test
    ///    counts, by wrapping the primary's own outgoing transport half
    ///    (`rig.pair.0`).
    ///
    /// Verified (by reading, not compiling) that this counter would
    /// genuinely increment if the guard were removed:
    /// `LoopbackTransport.sendControl` calls `controlDeliveryHook` (when
    /// set) synchronously and, whether hooked or not, `deliver` synchronously
    /// afterward — both inline in the caller's own call stack. `startRally()`
    /// calls `session?.sendRecord(action: "start", ...)` synchronously in its
    /// own body, and `sendRecord`'s only gate (`internalPhase == .live ||
    /// .ready`) is already satisfied by the first call's `goLive()`. So a
    /// second, guard-free `startRally()` call would synchronously push a
    /// second "start" frame through this exact hook before returning — no
    /// settling needed for the count itself, only (below) for the
    /// state-based assertions that DO depend on the cross-model `Task` hop.
    func testASecondStartWhileRecordingIsIgnored() async {
        let rig = await makePairedRig()

        var startFramesDelivered = 0
        rig.pair.0.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame), case .record("start", _) = message {
                startFramesDelivered += 1
            }
            deliver(frame)
        }

        rig.primary.startRally()
        XCTAssertEqual(startFramesDelivered, 1, "setup: the first startRally() must reach the secondary")
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        // Let the primary's own first-start transition actually finish (not
        // just the secondary's delivery of it) before firing the second
        // call, so `isRecording` reflects a completed start rather than an
        // in-flight one.
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })

        rig.primary.startRally()
        rig.pair.0.controlDeliveryHook = nil

        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertTrue(rig.primaryRecord.isRecording,
                      "a second startRally() must not silently stop the camera it's already recording on")
        XCTAssertEqual(startFramesDelivered, 1,
                      "a second startRally() while already recording must not put a second \"start\" frame on the wire")
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

    // MARK: - Task 10: `showsStop`, the `p-live` dead-end fix

    /// Before Task 10, a started rally had no way to end at all — this is
    /// the fix: the primary must be able to stop a rally it started
    /// regardless of link health, not only while degraded like the
    /// secondary's safety valve above.
    func testPrimaryAlwaysShowsStopWhileRecordingRegardlessOfLinkHealth() async {
        let rig = await makePairedRig()
        XCTAssertFalse(rig.primary.showsStop, "setup: no rally yet")

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.primary.rally == .recording })
        XCTAssertTrue(rig.primary.showsStop,
                      "the primary must always be able to end a rally it started")

        rig.primary.handleDegraded(true)
        XCTAssertTrue(rig.primary.showsStop, "degrading the link must not take the primary's STOP away")
        rig.primary.handleDegraded(false)
        XCTAssertTrue(rig.primary.showsStop)
    }

    /// The mirror of `testSecondaryGainsALocalStopOnlyWhileDegraded`, through
    /// the `showsStop` property `LiveStageView` actually reads: normally the
    /// secondary has no STOP of its own (it relies on the primary's
    /// `.record("stop", ...)` message), and only gains one once the link is
    /// degraded.
    func testSecondaryShowsStopOnlyWhileDegraded() async {
        let rig = await makePairedRig()
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertFalse(rig.secondary.showsStop,
                       "normally only the primary's STOP message ends the secondary's recording")

        rig.secondary.handleDegraded(true)
        XCTAssertTrue(rig.secondary.showsStop,
                      "a dropped link must not leave the secondary recording with no exit")
        rig.secondary.handleDegraded(false)
        XCTAssertFalse(rig.secondary.showsStop)
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

    // MARK: - Task 10b: one pairing, many rallies

    /// The one-rally-per-session dead end, end to end on the model.
    ///
    /// `startRally()` moved the peer session to `.live` and nothing ever moved
    /// it back — `goLive()` was `.ready → .live` only, `tick`'s `.ready, .live`
    /// branch never left `.live` except to degrade, and degrade-recovery
    /// restored it. So from the first rally onward `computedPrimaryTitle`
    /// returned "RALLY LIVE" and `computedPrimaryEnabled` returned `false`
    /// permanently: no second rally, and no re-pair either (the `.live` case
    /// short-circuits before `canPair` is consulted). `LiveSessionModel` is a
    /// `@StateObject` on `PlayRootView`, so backing out to Play did not clear
    /// it — only killing the app did.
    ///
    /// Rally 1 lands on `.failed("The rally produced no clip.")` rather than
    /// `.submitted`, and that is not a weakness here: a test host has no
    /// capture session, so `CameraController.stopRecording()` always ends in
    /// `CameraError.recordingEmpty` and no clip is ever published (the same
    /// reason `RecordModel.publishFinishedClipForTesting` exists). What this
    /// test needs is only that the rally *ended* — every terminal outcome goes
    /// through the same `finishRally`, which is where the session is released.
    ///
    /// `primaryEnabled` is deliberately not asserted: in `.ready` it also
    /// requires `engineReady`, which is set from `RecordModel.onStereoReady` —
    /// fired only when a peer `.calibration` message builds the StereoEngine.
    /// That gate is orthogonal to this fix (and is what Task 10c's
    /// `testThePrimarysEngineIsBuiltFromTheSecondarysCalibration` covers), so
    /// asserting on it here would say nothing about *this* one either way.
    /// `primaryTitle` is the value that was actually stuck, so that is what is
    /// asserted.
    func testASecondRallyStartsOnTheSamePairing() async {
        let rig = await makePairedRig()
        let mintedSessionID = rig.primary.sessionID
        XCTAssertNotNil(mintedSessionID, "setup: the primary mints one on reaching ready")

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        XCTAssertEqual(rig.primary.primaryTitle, "RALLY LIVE", "setup: rally 1 is live")
        XCTAssertTrue(rig.primary.showsStop)

        rig.primary.stopRally()
        // Two settles rather than one: the camera stop and the 0.05 s pump that
        // republishes `primaryTitle` from the refreshed `pairing.step` are
        // separate waits, and one shared deadline would let a slow stop eat the
        // budget for the title flip — the very thing under test.
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
        })
        await settleCrossModelDelivery(until: { rig.primary.primaryTitle == "START RALLY" })

        XCTAssertEqual(rig.primary.primaryTitle, "START RALLY",
                       "the pairing must offer the next rally, not strand on RALLY LIVE")
        guard case .ready = rig.primary.pairing.step else {
            return XCTFail("the session must be back to ready, got \(rig.primary.pairing.step)")
        }
        XCTAssertFalse(rig.primary.showsStop, "no rally is running, so there is nothing to stop")
        XCTAssertFalse(rig.primary.showsLocalStop)
        await settleCrossModelDelivery(until: { rig.secondary.rally != .recording })

        // The pairing's identity survives the rally: both phones' clips keep
        // sharing one `session_id`, which is what the server fuses on.
        XCTAssertEqual(rig.primary.sessionID, mintedSessionID, "sessionID must not be re-minted")
        XCTAssertEqual(rig.secondary.sessionID, mintedSessionID)

        rig.primary.startRally()
        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertTrue(rig.primary.showsStop)
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertEqual(rig.secondary.rally, .recording,
                       "the secondary must follow the primary into the second rally too")
        XCTAssertEqual(rig.primary.sessionID, mintedSessionID)

        // Teardown: rally 2 really is recording on both phones. Stop it here
        // rather than leaving two AVAssetWriters running past this test.
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            !rig.primaryRecord.isRecording && !rig.secondaryRecord.isRecording
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    #if DEBUG
    /// DESIGN.md §16's `p-live` table promises the mini-court "clears again
    /// when `START RALLY` begins the next one" — a promise that only became
    /// reachable once a session could run a second rally at all. Without the
    /// clear, rally 2 opens on rally 1's verdict: the banner still reads the
    /// old call and the mini-court still draws the old track.
    ///
    /// Uses the DEBUG stereo demo to populate exactly the fields a real call
    /// would (no peer, no ball model needed — the same seam
    /// `testEndSessionClearsRecordModelsLiveCallState` uses), then starts a
    /// rally on a `LiveSessionModel` bound to that same `RecordModel`. The
    /// assertions are taken synchronously right after `startRally()`, so the
    /// demo's own 0.05 s pump cannot slip another call in between.
    func testStartingTheNextRallyClearsThePreviousCall() async {
        let (model, _) = makeModel()
        let record = RecordModel(detector: nil)
        model.bind(record: record)

        record.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                               remoteModelJSON: StereoDemo.remoteModelJSON)
        await settleCrossModelDelivery(until: { record.livePresentation != nil })
        XCTAssertNotNil(record.livePresentation, "setup: the demo never produced a call")
        XCTAssertFalse(record.liveTrack.isEmpty, "setup: the demo never produced a track")

        model.startRally()

        XCTAssertNil(record.livePresentation, "a new rally must not open on the last one's verdict")
        XCTAssertNil(record.liveImpact)
        XCTAssertTrue(record.liveTrack.isEmpty)
        XCTAssertNil(record.flashPresentation)
        XCTAssertTrue(record.stereoEvents.isEmpty)

        // Teardown: `startRally()`'s chained start is real (there is no seam to
        // stub the camera), so stop what it began before `record` goes out of
        // scope — same reason as
        // `testAStartThatFailsToActuallyRecordDoesNotLeaveRallyBelievingItIsRecording`.
        await settleCrossModelDelivery(until: { record.isRecording })
        if record.isRecording { await record.setRecording(false, owner: .live) }
        XCTAssertFalse(record.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
    }
    #endif

    // MARK: - Task 10c: the secondary's calibration exchange

    /// The showstopper this closes: `sendCalibration` had no production caller
    /// at all. The receiving half was fully wired — `RecordModel.attachStereo`
    /// installs `peer.onCalibration`, which builds the primary's
    /// `StereoEngine` and fires `onStereoReady` → `engineReady` — but nothing
    /// ever sent, so on real hardware `engineReady` stayed `false` forever,
    /// `computedPrimaryEnabled`'s `.ready` case (`role == .primary &&
    /// engineReady`) never enabled START RALLY, and no rally could begin.
    ///
    /// Asserts the payload is the *solved, unadopted* camera-model JSON, byte
    /// for byte the string `prepare()` fetched and `attachStereo` was handed —
    /// then re-runs the receiver's own two decode steps
    /// (`CameraModel.fromJSON` + `adoptedForCapture()`) on it. A mismatch here
    /// is silent in production: `onCalibration`'s guard is one long `guard
    /// let ... else { return }`, so a payload it cannot parse simply never
    /// builds an engine and looks exactly like a message that never arrived.
    func testSecondarySendsItsSolvedCameraModelOnReachingReady() async {
        let rig = await makePairedRig()

        XCTAssertEqual(rig.calibrations.count, 1,
                       "the secondary must send its camera model on reaching .ready")
        XCTAssertEqual(rig.calibrations.payloads, [StubAPI.adoptableCameraModelJSON],
                       "the wire must carry the solved model verbatim, not a re-encoded or pre-adopted one")
        // The run ID `StubAPI.fixtureCalibration` reports is "7".
        XCTAssertEqual(rig.calibrations.profileIDs, ["calibration-run-7"],
                       "the profileID must name the calibration the model was solved from")

        guard let payload = rig.calibrations.payloads.first,
              let data = payload.data(using: .utf8),
              let model = try? CameraModel.fromJSON(data),
              (try? model.adoptedForCapture()) != nil else {
            return XCTFail("""
                the payload must survive exactly what the receiver does to it — \
                CameraModel.fromJSON then adoptedForCapture() — or the engine is \
                silently never built
                """)
        }
    }

    /// The assertion that actually proves a rally can start: the primary's
    /// engine gate flips, and START RALLY enables.
    ///
    /// This is the full production chain, not a stand-in for it — the
    /// secondary's `republish()` → `PeerSession.sendCalibration` → loopback →
    /// the primary's `handleControl` → `RecordModel.attachStereo`'s
    /// `onCalibration` → a real `StereoEngine` built from two real
    /// `CameraModel`s → `onStereoReady` → `LiveSessionModel.engineReady`.
    ///
    /// One honest limitation: both rig halves fetch the same
    /// `StubAPI.adoptableCameraModelJSON`, so the two camera models are
    /// identical and the pair has a zero baseline — a `StereoEngine` that
    /// could never triangulate anything. That is fine for what is under test
    /// (`StereoEngine.init` is non-throwing and validates nothing, so the gate
    /// flips on construction either way, exactly as on device), and the
    /// triangulation math itself is covered by `StereoEngineTests` /
    /// `StereoGoldenTests` against real two-camera fixtures. What this proves
    /// is the *wiring*: that a message is sent, arrives, parses, and enables
    /// the button.
    func testThePrimarysEngineIsBuiltFromTheSecondarysCalibration() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        XCTAssertTrue(rig.primary.engineReady,
                      "the secondary's calibration must build the primary's StereoEngine")
        XCTAssertFalse(rig.secondary.engineReady,
                       "only the primary owns an engine — attachStereo's handler guards on role")
        XCTAssertEqual(rig.primary.primaryTitle, "START RALLY")
        XCTAssertTrue(rig.primary.primaryEnabled,
                      "START RALLY must actually enable — this is the dead end being closed")
        XCTAssertNotEqual(rig.primary.linkStatus,
                          "Paired · waiting for the other phone's calibration",
                          "and the status line must stop saying it is still waiting")
    }

    /// `republish()` runs at 20 Hz for the whole life of a pairing and the
    /// secondary sits in `.ready` the entire time, so an unguarded send would
    /// put a fresh camera model on a constrained BLE link twenty times a
    /// second. Counts frames rather than state: a re-send is idempotent from
    /// the primary's point of view (it just rebuilds the engine), so the wire
    /// is the only place the difference shows.
    func testTheCalibrationIsSentOnceNotOnEveryRepublish() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertEqual(rig.calibrations.count, 1, "setup: exactly one send so far")

        spinPump(seconds: 0.5)   // ~10 more turns of the 0.05 s republish pump

        XCTAssertEqual(rig.calibrations.count, 1,
                       "the 20 Hz republish must not re-send the camera model")
    }

    /// The second-rally path (`finishRally` → `PeerSession.endRally()`) returns
    /// the session `.live → .ready`, and reaching `.ready` is what triggers the
    /// send — so the guard has to distinguish "first time ready" from "ready
    /// again after a rally". The primary's engine already exists by then; a
    /// re-send would rebuild it for nothing.
    ///
    /// Note the guard's job is even broader on the secondary than the name of
    /// this test suggests: only the primary ever calls `goLive()`, so the
    /// secondary's own session never leaves `.ready` at all — the entire rally
    /// is one long stretch of `.ready` republishes. The primary's `endRally()`
    /// round trip is covered here too by driving a real rally through both.
    ///
    /// Rally 1 ends on `.failed("The rally produced no clip.")` rather than
    /// `.submitted`, for the reason `testASecondRallyStartsOnTheSamePairing`
    /// documents: a test host has no capture session, so `stopRecording()`
    /// always ends in `CameraError.recordingEmpty`. Irrelevant here — every
    /// terminal outcome goes through the same `finishRally`, which is where
    /// the session is handed back to `.ready`.
    func testReturningToReadyAfterARallyDoesNotResendTheCalibration() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertEqual(rig.calibrations.count, 1, "setup: exactly one send so far")

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
        })
        await settleCrossModelDelivery(until: {
            rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        await settleCrossModelDelivery(until: { rig.primary.primaryTitle == "START RALLY" })
        guard case .ready = rig.primary.pairing.step else {
            return XCTFail("setup: the primary's session must be back to ready, got \(rig.primary.pairing.step)")
        }
        guard case .ready = rig.secondary.pairing.step else {
            return XCTFail("setup: the secondary's session must be ready, got \(rig.secondary.pairing.step)")
        }

        spinPump(seconds: 0.4)

        XCTAssertEqual(rig.calibrations.count, 1,
                       "returning to .ready after a rally must not re-send — the engine already exists")
        XCTAssertTrue(rig.primary.engineReady,
                      "and the engine the one exchange built must still be there for rally 2")

        // Teardown: both cameras really rolled above. Confirm the rally's own
        // stop took them down, rather than leaving AVAssetWriters running
        // past this test.
        await settleCrossModelDelivery(until: {
            !rig.primaryRecord.isRecording && !rig.secondaryRecord.isRecording
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    /// The other half of "once per session": once, but once *per session*. A
    /// guard that never reset would not fix the dead end, only defer it by one
    /// pairing — `endSession()` calls `record.detachPeer()`, which throws the
    /// primary's `StereoEngine` and `localModel` away, so the second pairing
    /// needs the exchange just as much as the first did.
    ///
    /// Re-pairs the same two models over the same (stateless) loopback pair,
    /// which is why `handshake(...)` is factored out of `makePairedRig`. The
    /// flag itself is private and unreadable from here; the wire count and
    /// `engineReady` are the observable surface, and they are the two things
    /// that actually matter.
    func testAFreshSessionAfterTeardownSendsTheCalibrationAgain() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertEqual(rig.calibrations.count, 1, "setup: exactly one send so far")
        XCTAssertTrue(rig.primary.engineReady, "setup: the first pairing built the engine")

        rig.primary.endSession()
        rig.secondary.endSession()
        XCTAssertFalse(rig.primary.engineReady,
                       "setup: teardown drops the engine gate along with the engine itself")

        let reachedReady = await handshake(primary: rig.primary, secondary: rig.secondary,
                                           pair: rig.pair, calibrations: rig.calibrations)
        XCTAssertTrue(reachedReady, """
            setup: the second pairing did not reach ready in time: \
            primary.step=\(rig.primary.pairing.step), secondary.step=\(rig.secondary.pairing.step)
            """)

        XCTAssertEqual(rig.calibrations.count, 2,
                       "a re-pairing must perform the exchange again, or the guard has just moved the dead end")
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertTrue(rig.primary.engineReady,
                      "the second pairing must rebuild the engine detachPeer() threw away")
        XCTAssertTrue(rig.primary.primaryEnabled,
                      "so START RALLY enables on the second pairing too")
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
    /// The handshake itself — and the reasoning about why it spins the real
    /// run loop instead of hand-ticking a synthetic clock — lives in
    /// `handshake(...)` below, which a test can also run a second time on
    /// these same models to re-pair them.
    ///
    /// `pair` (the raw `LoopbackTransport` half-pair) comes along too, past
    /// the point where `handshake(...)`'s own use of `pair.0`'s
    /// `controlDeliveryHook` has been cleared (`nil`ed out there once the
    /// buffered hellos are released) — so a caller can install its own hook
    /// on that half afterward, e.g. to count wire frames, without stepping on
    /// setup's. `pair.1`'s hook is deliberately *not* free: `handshake(...)`
    /// leaves the `.calibration` counter installed on it (see `calibrations`).
    private func makePairedRig() async -> (primary: LiveSessionModel, secondary: LiveSessionModel,
                                            primaryRecord: RecordModel, secondaryRecord: RecordModel,
                                            pair: (LoopbackTransport, LoopbackTransport),
                                            calibrations: CalibrationCounter) {
        let pair = LoopbackTransport.pair()
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

        let calibrations = CalibrationCounter()
        let reachedReady = await handshake(primary: primary, secondary: secondary,
                                           pair: pair, calibrations: calibrations)
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

        return (primary, secondary, primaryRecord, secondaryRecord, pair, calibrations)
    }

    /// The pairing handshake itself: PAIR on both models, exchange hellos,
    /// confirm, and spin the real run loop until both sides are `.ready` and
    /// share a `sessionID`. Returns whether that happened before the deadline.
    ///
    /// Extracted from `makePairedRig` so a test can run it a *second* time
    /// against the same two models after `endSession()`. That re-pairing is
    /// the only way to observe from outside that the once-per-session
    /// calibration guard really resets — the flag itself is private, and a
    /// second fresh rig would prove nothing (its flag starts clear anyway).
    /// Reusing the same `LoopbackTransport` pair is sound: it is stateless
    /// (`stop()` only emits `.disconnected`, nothing latches), each new
    /// `PeerSession`'s `init` re-points `onControl`/`onStateChange` at itself,
    /// and `start()` re-emits `.connected` — so a second pairing genuinely
    /// runs the whole handshake again.
    ///
    /// beginPairing() constructs the PeerSession and calls its start()
    /// (which sends this side's hello) in one call. Whichever model's
    /// primaryTapped() below runs first would fire its hello before the
    /// other side's PeerSession — and its transport.onControl — exists to
    /// receive it; LoopbackTransport does not buffer, so an unwired
    /// onControl just drops the frame. Buffer both directions until both
    /// sessions exist, then release together, so neither hello is lost.
    /// (PairingModelTests.makeRig() sidesteps the same hazard by
    /// constructing both PeerSessions before calling either start(); that
    /// seam isn't reachable through LiveSessionModel's public surface, so
    /// this reproduces the same effect with a delivery hook instead.)
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
    private func handshake(primary: LiveSessionModel, secondary: LiveSessionModel,
                           pair: (LoopbackTransport, LoopbackTransport),
                           calibrations: CalibrationCounter) async -> Bool {
        var buffered: [() -> Void] = []
        pair.0.controlDeliveryHook = { frame, deliver in buffered.append { deliver(frame) } }
        pair.1.controlDeliveryHook = { frame, deliver in buffered.append { deliver(frame) } }

        primary.primaryTapped()    // beginPairing(): builds + starts the primary session
        secondary.primaryTapped()  // beginPairing(): builds + starts the secondary session

        pair.0.controlDeliveryHook = nil
        // Counts what the secondary actually puts on the wire, installed here
        // rather than left to the calling test: the calibration goes out
        // during the wait loop below (`republish()` sends it the moment the
        // secondary's own pump observes `.ready`), so a hook installed after
        // this method returned would count zero and prove nothing.
        pair.1.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame),
               case .calibration(let profileID, let payloadJSON) = message {
                calibrations.record(profileID: profileID, payloadJSON: payloadJSON)
            }
            deliver(frame)
        }
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
        while true {
            let primaryReady: Bool = { if case .ready = primary.pairing.step { return true }; return false }()
            let secondaryReady: Bool = { if case .ready = secondary.pairing.step { return true }; return false }()
            if primaryReady, secondaryReady,
               primary.sessionID != nil, secondary.sessionID == primary.sessionID {
                return true
            }
            guard Date() < deadline else { return false }
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }
    }

    /// Spins the real run loop with nothing to wait for, so the 0.05 s
    /// republish pump gets many chances to do something it must not do.
    /// `settleCrossModelDelivery` cannot express that: it returns on its first
    /// pass when the condition is already true, which for a "must not happen"
    /// assertion is zero pump ticks of evidence.
    private func spinPump(seconds: TimeInterval) {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            RunLoop.main.run(until: Date().addingTimeInterval(0.01))
        }
    }
}
