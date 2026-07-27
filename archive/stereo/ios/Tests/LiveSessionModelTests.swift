// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Tests/LiveSessionModelTests.swift
import XCTest
@testable import SquashLineCalling

/// Fetches whatever `cameraModel` says on success, throws it on failure. The
/// remaining `APIClientProtocol` requirements are never exercised by
/// `LiveSessionModel` (it only calls `latestCalibration` and
/// `fetchSolvedCameraModel`), so they throw loudly rather than fake success.
///
/// `LiveSessionModel.prepare()` calls `adoptedForCapture()`, which always
/// targets `CaptureSettings.frameWidth x frameHeight` — now 3840x2160,
/// landscape, since capture is landscape-only. A fixture of any other aspect
/// makes every "calibration ready" test fail with `ScaleError.aspectMismatch`
/// instead of reaching `.ready`, so `adoptableCameraModelJSON` below is the
/// same 1920x1080 16:9 source
/// `CameraFrameSpaceTests.testAdoptedForCaptureLandsInTheLivePixelSpace`
/// already proves adopts cleanly (a clean 2x scale to 3840x2160). It was a
/// 1080x1920 portrait model until capture flipped to landscape.
///
/// Deliberately still a fixture of its own rather than `StereoDemo`'s pair:
/// those two are the *geometry* goldens, used unadopted on purpose, and
/// borrowing one here would couple this suite's calibration gate to whatever
/// the stereo goldens do next.
private struct StubAPI: APIClientProtocol {
    static let adoptableCameraModelJSON = """
    {"focal_px":1600.0,"center_px":[960.0,540.0],\
    "rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],\
    "camera_center_ft":[9.0,31.95,7.0],"distortion":null,\
    "frame_width":1920.0,"frame_height":1080.0}
    """

    private static let fixtureCalibration = try! LatestCalibration(responseData: Data(
        #"{"ok": true, "run_id": "7", "calibration": {"lines": []}}"#.utf8))

    var cameraModel: Result<String, Error>

    /// When true, only the *first* `fetchSolvedCameraModel` fails; every later
    /// one returns the adoptable model. Lets a test drive
    /// `prepare()` → `.failed` → retry → `.ready` through the real code path
    /// rather than by poking `calibration` directly.
    var recoversAfterFirstFailure = false

    /// How many times `fetchSolvedCameraModel` has actually been called. A
    /// reference type so it survives the struct being copied into
    /// `LiveSessionModel` — a retry that silently no-ops and a retry that
    /// really re-fetches are otherwise indistinguishable from outside.
    /// `@unchecked Sendable` for the same reason `MockAPIClient.Cursor` is:
    /// `APIClientProtocol` is `Sendable`, and every call here is made from
    /// `@MainActor` code.
    final class Attempts: @unchecked Sendable { var count = 0 }
    let attempts = Attempts()

    func latestCalibration() async throws -> LatestCalibration { Self.fixtureCalibration }

    func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String {
        attempts.count += 1
        if recoversAfterFirstFailure, attempts.count > 1 { return Self.adoptableCameraModelJSON }
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

/// Records the `.sessionManifest` frames a phone actually put on the wire,
/// split by the protocol's own discriminator: an *arming* announce carries an
/// empty `videoID`, a *completion* manifest carries the uploader's real run ID
/// (`PeerSession.onSessionManifest`). Same reference-type reasoning as
/// `CalibrationCounter` above — installed inside a helper, asserted on from the
/// test that called it, and it must survive a second pairing.
///
/// No locking, for the same reason: `LoopbackTransport` delivers synchronously
/// on the caller's own stack, and every manifest send originates from
/// `@MainActor` code (`LiveSessionModel.armRallyIdentity()`, or
/// `finishRecordingAndSubmit()`'s completion branch).
private final class ManifestCounter {
    private(set) var arming: [String] = []
    private(set) var completions: [(sessionID: String, videoID: String)] = []

    func record(sessionID: String, videoID: String) {
        if videoID.isEmpty { arming.append(sessionID) }
        else { completions.append((sessionID: sessionID, videoID: videoID)) }
    }

    /// Installs on a `LoopbackTransport` half, which intercepts that half's
    /// *outgoing* frames (`sendControl` runs the hook before `deliver`).
    /// Passes everything through — this only observes.
    func attach(to transport: LoopbackTransport) {
        transport.controlDeliveryHook = { [self] frame, deliver in
            if let message = ControlMessage.decode(frame),
               case .sessionManifest(let sessionID, let videoID) = message {
                record(sessionID: sessionID, videoID: videoID)
            }
            deliver(frame)
        }
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

    /// Amended for the calibration-retry fix: this used to assert
    /// `XCTAssertFalse(model.primaryEnabled)`, which pinned down the dead end
    /// rather than the design. DESIGN.md §16 and the spec's error table both
    /// promise `Failed → PAIR (retry)`, and `prepare()` runs from exactly one
    /// place (`PairingView`'s `.task`, once per appearance) — so a disabled
    /// PAIR here meant a failed fetch was permanent until the user backed out
    /// to Play and re-entered. The reason-verbatim half is unchanged and still
    /// asserted; only the enabled-ness expectation flipped.
    ///
    /// Enabled with `role` still nil, deliberately: the retry re-fetches this
    /// phone's own camera model, which is independent of which job it will do.
    func testCalibrationFailureShowsTheReasonVerbatimAndOffersARetry() async {
        let (model, _) = makeModel(cameraModel: .failure(APIError.http(404, "No calibration on this phone.")))
        await model.prepare()
        XCTAssertEqual(model.calibration, .failed("No calibration on this phone."))
        XCTAssertEqual(model.linkStatus, "No calibration on this phone.")
        XCTAssertNil(model.role, "setup: no role picked, and the retry must not need one")
        XCTAssertEqual(model.primaryTitle, "PAIR")
        XCTAssertTrue(model.primaryEnabled,
                      "§16's Failed row offers PAIR as the retry — a disabled one is a dead end")
    }

    /// The retry actually retrying. `primaryEnabled` reading `true` above is
    /// only half the promise: §7 forbids a primary that advertises a tap which
    /// cannot fire, and before this fix `primaryTapped()` on a failed
    /// calibration fell through to `beginPairing()`, whose own
    /// `guard case .ready = calibration` made it a silent no-op.
    ///
    /// Counts real fetches rather than watching state alone — a tap that
    /// no-ops and a tap that re-fetches both leave `calibration` where it was
    /// if the second fetch happens to fail the same way, so the count is what
    /// distinguishes them.
    func testTappingPairAfterACalibrationFailureRefetchesAndRecovers() async {
        let api = StubAPI(cameraModel: .failure(APIError.http(503, "Server unreachable.")),
                          recoversAfterFirstFailure: true)
        let model = LiveSessionModel(api: api, makeTransport: { _ in LoopbackTransport.pair().0 })

        await model.prepare()
        XCTAssertEqual(model.calibration, .failed("Server unreachable."),
                       "setup: the first fetch must fail")
        XCTAssertEqual(api.attempts.count, 1)
        XCTAssertTrue(model.primaryEnabled, "setup: §16's retry must be offered")

        model.primaryTapped()   // the retry — `prepare()` again, not `beginPairing()`
        // `primaryTapped()` schedules `prepare()` on a `Task`, so let it run.
        await settleCrossModelDelivery(until: { model.calibration == .ready })

        XCTAssertEqual(api.attempts.count, 2, """
            the tap must re-run prepare(): before this fix it fell through to beginPairing(), \
            whose own `guard case .ready = calibration` silently refused, so the only recovery \
            was backing out to Play and re-entering
            """)
        XCTAssertEqual(model.calibration, .ready)
        XCTAssertEqual(model.linkStatus, "Pick this phone's job to start.",
                       "and the row moves on to the next thing actually blocking a pairing")
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
    /// capture-orientation guard — by pairing this model's session against a
    /// hand-built secondary mounted the other way round. `beginPairing()`
    /// builds its session from `record.camera.orientation`, which for a
    /// `RecordModel` that has never run `startCamera()` is
    /// `CameraController`'s `.landscapeRight` default, so the secondary is
    /// built `.landscapeLeft` to mismatch it. Both mounts share one frame
    /// size, so it is `Hello.captureOrientation` — not the dimensions — that
    /// this trips. Both ends independently reject the other's hello, so no
    /// cooperation from a "remote app" is needed.
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
                                    captureOrientation: .landscapeLeft)

        model.primaryTapped()   // beginPairing(): builds the primary session
                                // (.landscapeRight) and starts its 0.05 s pump
        secondary.start()       // sends the secondary's .landscapeLeft hello — the
                                // primary rejects it as a mount mismatch and fails

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
        // The teardown below depends on the pending start really succeeding,
        // so this needs the deterministic mount — see `makeRecordModel()`.
        let record = makeRecordModel()
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

    // MARK: - Task 10e: the uploaded identity is per rally, not per pairing

    /// Nothing is minted at pairing time any more. The old code broadcast one
    /// id when the pairing first reached `.ready`; the server pairs runs on
    /// `session_id` + `camera_role` and assumes one rally per id, so that made
    /// every rally after the first either unfusable (`maybe_start_stereo_fuse`
    /// claims `stereo-<session_id>` with `exist_ok=False`, so rallies 2..N hit
    /// a swallowed `FileExistsError`) or fused against the *wrong* rally
    /// (`session_runs` picks the newest complete run per role independently).
    func testNoIdentityIsMintedOrBroadcastAtPairingTime() async {
        let rig = await makePairedRig()
        let manifests = ManifestCounter()
        manifests.attach(to: rig.pair.0)

        XCTAssertNil(rig.primary.rallySessionID,
                     "a pairing with no rally running has no identity to upload under")
        XCTAssertNil(rig.secondary.rallySessionID)

        spinPump(seconds: 0.3)   // ~6 turns of the 20 Hz republish pump

        XCTAssertTrue(manifests.arming.isEmpty,
                      "reaching .ready must not put a session manifest on the wire — that broadcast " +
                      "now belongs to startRally(), once per rally")
        XCTAssertTrue(manifests.completions.isEmpty,
                      "and no completion manifest either: nothing has uploaded")
        XCTAssertNil(rig.primary.rallySessionID)
    }

    /// The core of the fix: rally 2 must not upload under rally 1's id.
    ///
    /// Asserts on the wire as well as on the model. The model side alone would
    /// still pass if the primary minted per rally but never told the secondary;
    /// the wire side alone would still pass if it broadcast an id it did not
    /// itself upload under.
    func testEachRallyMintsAndBroadcastsADistinctIdentity() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        let manifests = ManifestCounter()
        manifests.attach(to: rig.pair.0)

        rig.primary.startRally()
        let firstID = rig.primary.rallySessionID
        XCTAssertNotNil(firstID, "rally 1 must mint an identity")
        XCTAssertEqual(manifests.arming.count, 1, "and put exactly one arming announce on the wire")
        XCTAssertEqual(manifests.arming.first, firstID, "carrying exactly the id it will upload under")
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })

        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
        })
        await settleCrossModelDelivery(until: {
            rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        // The rally ended, so its identity is spent: nothing may upload under
        // it again, which is exactly what `stereo-<session_id>`'s one-shot
        // `mkdir` claim requires.
        XCTAssertNil(rig.primary.rallySessionID,
                     "a finished rally's identity must be consumed, not left lying around for the next one")
        XCTAssertNil(rig.secondary.rallySessionID)
        await settleCrossModelDelivery(until: { rig.primary.primaryTitle == "START RALLY" })

        rig.primary.startRally()
        let secondID = rig.primary.rallySessionID
        XCTAssertNotNil(secondID, "rally 2 must mint one of its own")
        XCTAssertNotEqual(secondID, firstID, """
            rally 2 must not upload under rally 1's session_id: the server claims one fuse run per \
            id (mkdir exist_ok=False) and pairs runs per role independently, so a shared id means \
            rally 2 is either never fused or fused against rally 1's clip
            """)
        XCTAssertEqual(manifests.arming.count, 2, "one arming announce per rally, no more")
        XCTAssertEqual(manifests.arming.last, secondID)

        // Teardown: rally 2 really is rolling on both phones. Wait for each
        // rally to reach a terminal state rather than for `isRecording` alone —
        // the latter reads `false` both before a queued start has run and after
        // the stop behind it, so waiting on it directly can return on the wrong
        // side of the chain and leave a writer open.
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    /// Distinct per rally is only half of it — the two phones must still agree,
    /// or `session_runs` finds one role and never fuses at all.
    ///
    /// Also pins the *ordering* the scheme rests on: the arming manifest is
    /// sent before the `record("start")` frame, so the secondary is holding
    /// this rally's identity by the time it even begins recording — a margin of
    /// the whole rally over the upload that eventually reads it.
    func testBothPhonesAgreeOnOneRallysIdentity() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        var order: [String] = []
        rig.pair.0.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame) {
                if case .sessionManifest(_, let videoID) = message, videoID.isEmpty {
                    order.append("manifest")
                } else if case .record("start", _) = message {
                    order.append("start")
                }
            }
            deliver(frame)
        }

        rig.primary.startRally()
        rig.pair.0.controlDeliveryHook = nil
        XCTAssertEqual(order, ["manifest", "start"], """
            the arming manifest must precede the record("start") frame — transports deliver control \
            frames in order, so this is what guarantees the secondary is never recording a rally it \
            has no identity for
            """)

        await settleCrossModelDelivery(until: { rig.secondary.rallySessionID != nil })
        XCTAssertNotNil(rig.primary.rallySessionID)
        XCTAssertEqual(rig.secondary.rallySessionID, rig.primary.rallySessionID,
                       "both clips must upload under one session_id or the server has nothing to pair")

        // Through the accessor `finishRecordingAndSubmit()` itself uses, so
        // this asserts the exact triple that reaches `RunSubmission.submit` —
        // including the `camera_role` the server pairs on, which the published
        // property above does not carry. Consuming, which is why it comes last:
        // the rally is being torn down from here on.
        let primaryUpload = rig.primary.takeRallyUploadIdentity()
        let secondaryUpload = rig.secondary.takeRallyUploadIdentity()
        XCTAssertNotNil(primaryUpload.sessionID)
        XCTAssertEqual(primaryUpload.sessionID, secondaryUpload.sessionID)
        XCTAssertEqual(primaryUpload.cameraRole, "a")
        XCTAssertEqual(secondaryUpload.cameraRole, "b")

        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    /// The failure mode that makes "consume it when the rally ends" load
    /// bearing: rally 2's arming manifest is dropped on the wire, so the
    /// secondary never learns rally 2's identity.
    ///
    /// It must upload *unpaired* — both `session_id` and `camera_role` omitted,
    /// which `/api/track` explicitly allows ("All four are absent for a
    /// single-camera run") — and must not fall back on rally 1's id. Falling
    /// back is the actively harmful outcome: it hands the server a role-`b` run
    /// for a rally the primary already fused, so `session_runs` would pair
    /// rally 2's secondary clip with rally 1's primary clip and triangulate two
    /// different rallies as one.
    func testARallyWhoseManifestNeverArrivedUploadsUnpairedRatherThanReusingTheLastOne() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        // Rally 1, delivered normally, so the secondary really does hold an id
        // that a broken implementation could fall back on.
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rallySessionID != nil })
        let firstID = rig.secondary.rallySessionID
        XCTAssertNotNil(firstID, "setup: the secondary must have rally 1's identity to be able to leak it")
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })

        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        await settleCrossModelDelivery(until: { rig.primary.primaryTitle == "START RALLY" })

        // Rally 2: swallow the arming announce only. The record frames still
        // get through, so the secondary records a rally it has no identity for
        // — exactly the case the fallback would have papered over.
        var manifestsDropped = 0
        rig.pair.0.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame),
               case .sessionManifest(_, let videoID) = message, videoID.isEmpty {
                manifestsDropped += 1
                return
            }
            deliver(frame)
        }

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertEqual(manifestsDropped, 1, "setup: rally 2's arming manifest must actually have been dropped")
        XCTAssertEqual(rig.secondary.rally, .recording,
                       "setup: the secondary must still be recording rally 2 — only the manifest was lost")

        XCTAssertNil(rig.secondary.rallySessionID, """
            the secondary must not reuse rally 1's identity for rally 2 — that would pair rally 2's \
            role-b clip with rally 1's role-a run and triangulate two different rallies as one
            """)
        let upload = rig.secondary.takeRallyUploadIdentity()
        XCTAssertNil(upload.sessionID)
        XCTAssertNil(upload.cameraRole,
                     "both or neither: the server rejects a run carrying camera_role without session_id")
        XCTAssertNil(upload.peerVideoID)
        // The primary is unaffected — it minted and sent; only the delivery was
        // lost, which it cannot know. Its own upload stays paired, and lands on
        // the server as a role-a run that simply never finds a partner.
        XCTAssertNotNil(rig.primary.rallySessionID)
        XCTAssertNotEqual(rig.primary.rallySessionID, firstID)

        rig.pair.0.controlDeliveryHook = nil
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    /// Fix 3, on the half of it that has a deterministic seam: the identity is
    /// latched from the send's *outcome*, never optimistically.
    ///
    /// Degrading the primary's transport synchronously (no run-loop spin in
    /// between, so `tick`'s recovery branch cannot fire) puts the session's
    /// authoritative `internalPhase` outside `sendSessionManifest`'s
    /// `.live || .ready` gate. The old shape — assign the id, then send —
    /// would have kept the id anyway and uploaded a role-`a` run under an
    /// identity the secondary was never told, which then waits forever for a
    /// role `b`. Reading the return value instead makes it upload unpaired.
    func testAnIdentityWhoseBroadcastWasDroppedIsNotLatched() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        var manifestFramesSent = 0
        rig.pair.0.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame),
               case .sessionManifest = message {
                manifestFramesSent += 1
            }
            deliver(frame)
        }
        // Degrade the primary's own session. `stop()` fires `.disconnected` on
        // this half only, so the secondary stays healthy — and everything from
        // here to the assertions is synchronous on the main actor, so the 20 Hz
        // pump cannot tick the session back out of `.degraded` in between.
        rig.pair.0.stop()

        rig.primary.startRally()

        XCTAssertEqual(manifestFramesSent, 0,
                       "setup: the session's own gate must have refused the frame")
        XCTAssertNil(rig.primary.rallySessionID, """
            an id must not be latched when the broadcast that makes it agreed never went out — the \
            secondary is not recording this rally either (the same gate refused the "start" frame), \
            so a paired upload would leave a role-a run waiting for a partner that is never coming
            """)
        let upload = rig.primary.takeRallyUploadIdentity()
        XCTAssertNil(upload.sessionID)
        XCTAssertNil(upload.cameraRole)

        rig.pair.0.controlDeliveryHook = nil
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
    }

    // MARK: - Task 10e: a beginPairing() retry must reset ALL per-pairing state

    /// DESIGN.md §16's `Failed → PAIR (retry)` row calls `beginPairing()`
    /// straight through — the exact path `primaryTapped()`'s `.failed` branch
    /// takes, without `endSession()` running first. `beginPairing()` used to
    /// clear a hand-picked subset of the previous pairing's state, and this
    /// pins down the three things that subset missed:
    ///
    /// * `engineReady` — the worst of them. It appeared *only* in
    ///   `endSession()`, and neither `attachPeer` nor `attachStereo` nils
    ///   `RecordModel.stereoEngine`, so a retry after a previously successful
    ///   pairing reached `.ready` with the gate already open. START RALLY
    ///   enabled before the new secondary's calibration had arrived, the
    ///   "waiting for the other phone's calibration" status stopped showing,
    ///   and a rally started in that window ran against the *previous*
    ///   pairing's `StereoEngine` — whose `remoteToLocal` closure holds a weak
    ///   reference to the deallocated old `PeerSession`, so remote samples
    ///   could not be time-mapped and the rally produced no calls at all.
    /// * `rally` — a retry taken mid-rally kept `.recording` from the dead
    ///   session, which `startRally()`'s own guard then refuses forever while
    ///   `primaryEnabled` reads true: an enabled primary whose tap cannot fire
    ///   (DESIGN.md §7 forbids it). Covered by
    ///   `testARetryTakenMidRallyResetsTheRallyAndStopsTheCamera` below.
    /// * the upload identity — now per rally, so this asserts the weaker but
    ///   correct thing: the new pairing's first rally must not reuse the dead
    ///   pairing's id.
    ///
    /// Reaches the failure the same way `testPairIsReenabledAfterAPairingFailure`
    /// does — a real `PeerSession.Phase.failed` — but *after* both sides have
    /// already reached `.ready` and built the engine, which that test cannot
    /// reach (its failure fires during the very first hello, before `.ready`
    /// is possible). There is no production seam for "make an already-paired
    /// link renegotiate hello and fail," so this drives the same
    /// `handleControl` `.hello` guard `testMismatchedPeerFrameSizeIsRejected`
    /// exercises, directly over the raw transport halves the rig exposes —
    /// bypassing each side's own (still-healthy) `PeerSession` object
    /// entirely, exactly as a stray/duplicated hello delivered by a
    /// misbehaving transport would. A wrong `protoVersion` is the simplest
    /// guard to trip: `PeerSession.handleControl` checks it before frame size,
    /// so there is no need to compute a correctly-mismatched orientation.
    func testARetryAfterAPairingFailureDoesNotCarryEngineReadyOrTheOldIdentity() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertTrue(rig.primary.engineReady, "setup: the first pairing really did build an engine")

        // A rally on the first pairing, so there is a real previous identity
        // available to leak into the retry.
        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rallySessionID != nil })
        let deadPairingRallyID = rig.primary.rallySessionID
        XCTAssertNotNil(deadPairingRallyID, "setup: the first pairing must have run a rally")
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })

        // Fail both real sessions independently, without ever calling
        // `endSession()` — the exact gap this fix closes. Delivered straight
        // over the raw transport halves (`rig.pair`), not through either
        // side's own `PeerSession`, so this reaches `handleControl` even
        // though both sessions are already `.ready` and would otherwise
        // never receive another `.hello` at all.
        let bogusHello = ControlMessage.hello(
            Hello(protoVersion: peerProtoVersion + 1, appVersion: "dev",
                  deviceModel: "test", nonce: 0, frameW: 0, frameH: 0))
        let bogusFrame = try! ControlMessage.encode(bogusHello)
        rig.pair.1.sendControl(bogusFrame)   // delivered to the primary's session
        rig.pair.0.sendControl(bogusFrame)   // delivered to the secondary's session

        await settleCrossModelDelivery(until: {
            if case .failed = rig.primary.pairing.step, case .failed = rig.secondary.pairing.step {
                return true
            }
            return false
        })
        guard case .failed = rig.primary.pairing.step else {
            return XCTFail("setup: expected the primary to fail, got \(rig.primary.pairing.step)")
        }
        guard case .failed = rig.secondary.pairing.step else {
            return XCTFail("setup: expected the secondary to fail, got \(rig.secondary.pairing.step)")
        }
        XCTAssertTrue(rig.primary.primaryEnabled, "setup: §16's Failed row must still offer PAIR (retry)")
        XCTAssertEqual(rig.primary.primaryTitle, "PAIR")
        // The bug's precondition: reaching `.failed` alone — with no
        // `beginPairing()` retry yet — must not by itself reset anything. The
        // reset has to be the retry's own doing, or this test would pass for
        // the wrong reason.
        XCTAssertTrue(rig.primary.engineReady,
                      "setup: failing must not itself clear the engine gate")

        // The retry: `primaryTapped()` on a `.failed` step calls
        // `beginPairing()` directly, exactly as `handshake(...)` reproduces
        // here by re-running the same helper `makePairedRig` used the first
        // time — this is the only thing that changed between the two calls.
        // (Driving `primaryTapped()` by hand *before* this would consume the
        // `.failed` step `handshake` needs, leaving its own tap on `.searching`
        // where `primaryTapped()` is a no-op. The window inside the retry is
        // asserted separately, in
        // `testARetryDropsThePreviousPairingsEngineGateImmediately`.)
        let reachedReady = await handshake(primary: rig.primary, secondary: rig.secondary,
                                           pair: rig.pair, calibrations: rig.calibrations)
        XCTAssertTrue(reachedReady, """
            the retry did not reach ready in time: \
            primary.step=\(rig.primary.pairing.step), secondary.step=\(rig.secondary.pairing.step)
            """)
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertTrue(rig.primary.engineReady,
                      "the retry's own calibration exchange must rebuild the engine detachPeer() threw away")

        // And the new pairing's first rally gets a genuinely new identity —
        // reusing the dead pairing's would hand the server a second run under
        // an id whose `stereo-<session_id>` fuse directory already exists.
        rig.primary.startRally()
        XCTAssertNotNil(rig.primary.rallySessionID)
        XCTAssertNotEqual(rig.primary.rallySessionID, deadPairingRallyID,
                          "a rally on the new pairing must not reuse an id from the old one")

        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.secondaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
    }

    /// The `engineReady` window itself, which the test above cannot see: by the
    /// time a full re-handshake reports `.ready`, the *new* secondary's
    /// calibration has already rebuilt the engine and set the gate back to true
    /// legitimately. The bug was that it was never false in between — START
    /// RALLY stayed enabled across the whole retry, on an engine belonging to a
    /// pairing that no longer exists.
    ///
    /// `primaryTapped()` runs `beginPairing()` synchronously, so reading
    /// immediately after it cannot miss the window. Only the primary is failed
    /// and retried here; nothing needs to reach `.ready` again.
    func testARetryDropsThePreviousPairingsEngineGateImmediately() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })
        XCTAssertTrue(rig.primary.engineReady, "setup: the first pairing built an engine")
        XCTAssertTrue(rig.primary.primaryEnabled, "setup: so START RALLY is enabled")

        let bogusHello = ControlMessage.hello(
            Hello(protoVersion: peerProtoVersion + 1, appVersion: "dev",
                  deviceModel: "test", nonce: 0, frameW: 0, frameH: 0))
        rig.pair.1.sendControl(try! ControlMessage.encode(bogusHello))
        await settleCrossModelDelivery(until: {
            if case .failed = rig.primary.pairing.step { return true }
            return false
        })
        guard case .failed = rig.primary.pairing.step else {
            return XCTFail("setup: expected the primary's session to fail, got \(rig.primary.pairing.step)")
        }
        XCTAssertTrue(rig.primary.engineReady,
                      "setup: failing must not itself clear the gate — the retry has to be what does it")

        rig.primary.primaryTapped()   // §16's Failed → PAIR (retry)

        XCTAssertFalse(rig.primary.engineReady, """
            a retry must not inherit the previous pairing's engine gate: START RALLY would enable \
            before the new secondary's calibration arrived, and a rally started in that window would \
            run against the dead pairing's StereoEngine — whose remoteToLocal closure weakly \
            references a deallocated PeerSession, so it can time-map nothing and calls nothing
            """)
        XCTAssertFalse(rig.primary.primaryEnabled,
                       "so the button must be disabled again while the new pairing searches")
        XCTAssertEqual(rig.primary.linkStatus, "Looking for the other phone…",
                       "and the status line must go back to searching, not keep the dead pairing's")
        XCTAssertNil(rig.primary.rallySessionID,
                     "and no upload identity carries over from the pairing that just died")
    }

    /// The other half of Fix 2: a retry taken *mid-rally*.
    ///
    /// `beginPairing()` left `rally` and `rallyGeneration` alone, so the new
    /// pairing inherited `.recording` from a session that no longer exists.
    /// `startRally()`'s own `guard rally != .recording` then refuses every
    /// start on the new pairing — permanently — while `computedPrimaryEnabled`
    /// happily reads `.ready` and enables the button. DESIGN.md §7 forbids
    /// exactly that: a primary that advertises a tap which cannot fire.
    ///
    /// The camera is the other half of the same omission. `endSession()` always
    /// scheduled a stop through `RecordModel`'s funnel; `beginPairing()` did
    /// not, so a retry mid-rally left an `AVAssetWriter` rolling with nothing
    /// left that was allowed to stop it — which also locks the plain record
    /// stage out of the camera it shares.
    func testARetryTakenMidRallyResetsTheRallyAndStopsTheCamera() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        XCTAssertEqual(rig.primary.rally, .recording, "setup: a rally really is running")
        XCTAssertTrue(rig.primaryRecord.isRecording, "setup: and the camera really is rolling")

        // Retry straight out of the running rally. `primaryTapped()` would read
        // `.live` here and do nothing (§16 has no PAIR in `p-live`), so this
        // reproduces the reachable route: the link fails under a rally, §16's
        // `Failed` row offers PAIR, and the user takes it.
        let bogusHello = ControlMessage.hello(
            Hello(protoVersion: peerProtoVersion + 1, appVersion: "dev",
                  deviceModel: "test", nonce: 0, frameW: 0, frameH: 0))
        rig.pair.1.sendControl(try! ControlMessage.encode(bogusHello))
        await settleCrossModelDelivery(until: {
            if case .failed = rig.primary.pairing.step { return true }
            return false
        })
        guard case .failed = rig.primary.pairing.step else {
            return XCTFail("setup: expected the primary's session to fail, got \(rig.primary.pairing.step)")
        }
        XCTAssertEqual(rig.primary.rally, .recording,
                       "setup: failing the link does not itself end the rally — that is the point")

        rig.primary.primaryTapped()   // §16's Failed → PAIR (retry)

        XCTAssertEqual(rig.primary.rally, .idle, """
            a retry must not inherit `.recording` from the dead session: startRally() would refuse \
            every start on the new pairing forever while primaryEnabled still reads true
            """)
        XCTAssertFalse(rig.primary.showsStop, "and nothing is left to stop")
        XCTAssertFalse(rig.primary.showsLocalStop)

        // The camera stop is chained through `RecordModel`'s funnel, so it
        // lands on the next turn rather than synchronously.
        await settleCrossModelDelivery(until: { !rig.primaryRecord.isRecording })
        XCTAssertFalse(rig.primaryRecord.isRecording, """
            a retry mid-rally must not leave an AVAssetWriter running: nothing on the new pairing \
            owns it, and it locks the record stage out of the camera it shares
            """)
        // Deliberately not asserted: that the abandoned clip is *discarded*. A
        // test host has no capture session, so the stop always ends in
        // `CameraError.recordingEmpty` and no clip is ever published — the
        // assertion would hold for a reason unrelated to the code under test.
        // The discard is the same `Task { [record] ... liveClip = nil }`
        // `endSession()` has always run, now shared via `teardownPairing()`.

        // §16's Searching row, which is where a fresh `beginPairing()` lands:
        // the label is back to PAIR and the control is disabled while the new
        // session looks for a peer — not "START RALLY, enabled" on a pairing
        // that has not even said hello yet.
        XCTAssertEqual(rig.primary.primaryTitle, "PAIR")
        XCTAssertFalse(rig.primary.primaryEnabled)
        XCTAssertEqual(rig.primary.linkStatus, "Looking for the other phone…")

        // Teardown: the *secondary* followed the primary into this rally and
        // nothing on the retried pairing can end it — the primary's "stop"
        // would have travelled over a session that no longer exists. Stop it
        // through its own §16 safety valve so no writer outlives this test.
        rig.secondary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.secondaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
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

        rig.primary.startRally()
        let firstRallyID = rig.primary.rallySessionID
        XCTAssertNotNil(firstRallyID, "setup: rally 1 mints its own identity")
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

        // The *pairing* survives the rally; its upload identity does not — that
        // is per rally, and Task 10e's own tests cover it. Asserted here only as
        // the precondition for the second start below.
        XCTAssertNil(rig.primary.rallySessionID, "rally 1's identity is spent")

        rig.primary.startRally()
        XCTAssertEqual(rig.primary.rally, .recording)
        XCTAssertTrue(rig.primary.showsStop)
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        XCTAssertEqual(rig.secondary.rally, .recording,
                       "the secondary must follow the primary into the second rally too")
        XCTAssertNotEqual(rig.primary.rallySessionID, firstRallyID,
                          "rally 2 is a different rally, so a different session_id")

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

    // MARK: - A rally's terminal outcome has to be visible somewhere

    /// `RallyState.failed(String)` was set and rendered *nowhere*.
    /// `LiveStageView` reads only `rally == .recording`, `PairingView` does not
    /// read `rally` at all, and `computedLinkStatus` never consulted it — so a
    /// failed upload, which on a court means the ordinary "server unreachable",
    /// looked exactly like a successful one: `p-live` popped and `p-pair` read
    /// a cheerful "Paired · sync ±1.4 ms" over a rally that was silently lost.
    ///
    /// Asserts against the reason `rally` actually carries rather than a
    /// hardcoded string: in a test host the stop always ends in
    /// `CameraError.recordingEmpty` (no capture session), so the reason is "The
    /// rally produced no clip." rather than an upload error. Which failure
    /// produced it is not what is under test — that the row shows *the* reason,
    /// verbatim, is.
    func testAFailedRallyShowsItsReasonInTheLinkStatusRow() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
        })
        await settleCrossModelDelivery(until: { rig.primary.primaryTitle == "START RALLY" })

        guard case .failed(let reason) = rig.primary.rally else {
            return XCTFail("setup: expected the rally to end .failed, got \(rig.primary.rally)")
        }
        // Stated rather than assumed: the row only speaks for the rally while
        // the session is `.ready` (every other step has something more urgent
        // to say), and `.ready` is where `finishRally` leaves it. A `.degraded`
        // link here would make the assertion below fail for an unrelated
        // reason, so name the precondition.
        guard case .ready = rig.primary.pairing.step else {
            return XCTFail("setup: the session must be back to ready, got \(rig.primary.pairing.step)")
        }
        XCTAssertEqual(rig.primary.linkStatus, reason, """
            a rally that ended badly must say so, verbatim, the same way a calibration failure \
            does — otherwise the only record of it is a RallyState nothing renders
            """)
        // §7: the row reports, it does not take the tap away. START RALLY is
        // still the phase's one primary and it still fires.
        XCTAssertEqual(rig.primary.primaryTitle, "START RALLY")
        XCTAssertTrue(rig.primary.primaryEnabled,
                      "a failed rally must not disable the next one — that would be a second dead end")

        // And the row clears when the next rally begins, rather than carrying a
        // dead rally's verdict into a live one (§16's rule for the mini-court,
        // applied to the same seam).
        rig.primary.startRally()
        XCTAssertNotEqual(rig.primary.linkStatus, reason,
                          "starting the next rally must clear the previous one's failure")

        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        rig.primary.stopRally()
        await settleCrossModelDelivery(until: {
            rig.primary.rally != .recording && rig.primary.rally != .submitting
                && rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertFalse(rig.primaryRecord.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
        XCTAssertFalse(rig.secondaryRecord.isRecording)
    }

    // MARK: - Teardown must not leave the peer recording

    /// `teardownPairing()` stopped the local camera but never told the peer, so
    /// a pairing torn down mid-rally left the secondary recording 4K60 until
    /// its own 3 s heartbeat timeout noticed the link was gone — a rally's
    /// worth of 4K60 on a phone that thinks it is still working.
    ///
    /// Driven through `endSession()` rather than §16's `Failed → PAIR (retry)`
    /// path, because that is the only route on which the fix is *observable*:
    /// both callers share `teardownPairing()` verbatim, but on the retry path
    /// the session is already `.failed`, so `sendRecord`'s own
    /// `.live || .ready` gate correctly refuses the frame and the secondary's
    /// degraded-link STOP (§16) is what saves it instead. Here the session is
    /// `.live`, so the frame goes out and the secondary follows it down.
    func testTearingDownAPairingMidRallyTellsThePeerToStopRecording() async {
        let rig = await makePairedRig()
        await settleCrossModelDelivery(until: { rig.primary.engineReady })

        rig.primary.startRally()
        await settleCrossModelDelivery(until: { rig.secondary.rally == .recording })
        await settleCrossModelDelivery(until: { rig.primaryRecord.isRecording })
        XCTAssertEqual(rig.secondary.rally, .recording, "setup: the secondary really is recording")

        var stopFramesDelivered = 0
        rig.pair.0.controlDeliveryHook = { frame, deliver in
            if let message = ControlMessage.decode(frame), case .record("stop", _) = message {
                stopFramesDelivered += 1
            }
            deliver(frame)
        }

        rig.primary.endSession()
        rig.pair.0.controlDeliveryHook = nil

        XCTAssertEqual(stopFramesDelivered, 1, """
            teardown must send .record("stop") while it still has a session to send it through — \
            without it the secondary keeps recording until its 3 s heartbeat timeout
            """)
        await settleCrossModelDelivery(until: {
            rig.secondary.rally != .recording && rig.secondary.rally != .submitting
        })
        XCTAssertNotEqual(rig.secondary.rally, .recording,
                          "and the secondary must actually act on it, not just receive it")

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
        // Same as above: the teardown stops a recording that must actually
        // have started, so the mount cannot be left to the ambient scene.
        let record = makeRecordModel()
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

    /// A `RecordModel` every rally in this file can actually start a recording
    /// on.
    ///
    /// `detector: nil` — no detections are needed anywhere here, and
    /// `SyntheticBallDetector` is `#if DEBUG`-only, so this file's `RecordModel`
    /// construction must not depend on a DEBUG-only type.
    ///
    /// `mountResolver` is the part that matters. `RecordModel.applyRecording`'s
    /// start path refuses to start unless the interface is resolved to a
    /// landscape mount, and in production that comes from the live window
    /// scene — ambient state a unit test does not control. Every test in this
    /// file that starts a rally would otherwise be racing the simulator's
    /// rotation, and the ones that wait on `isRecording` through
    /// `settleCrossModelDelivery` would burn their full 2 s deadline before
    /// failing on something that has nothing to do with what they assert.
    /// The refusal itself is production behaviour and stays exactly as it is;
    /// this supplies a resolved mount, it does not remove the requirement for
    /// one. `.landscapeRight` is what
    /// `CameraController.orientation` already defaults to, so the mount the
    /// rig's `beginPairing()` puts on the wire is unchanged — which is what
    /// keeps `makePairedRig()`'s two halves matching at the handshake.
    private func makeRecordModel() -> RecordModel {
        RecordModel(detector: nil,
                    mountResolver: { CaptureSettings.CaptureOrientation.landscapeRight })
    }

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
        // Both halves record: the primary from its own START RALLY, the
        // secondary from the relayed `.record` control message. So both need
        // the deterministic mount — see `makeRecordModel()`.
        let primaryRecord = makeRecordModel()
        let secondaryRecord = makeRecordModel()
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
                calibrations=\(calibrations.count)
                """)
        }

        return (primary, secondary, primaryRecord, secondaryRecord, pair, calibrations)
    }

    /// The pairing handshake itself: PAIR on both models, exchange hellos,
    /// confirm, and spin the real run loop until both sides are `.ready` and
    /// the secondary's one calibration frame has actually gone out. Returns
    /// whether that happened before the deadline.
    ///
    /// The calibration is the postcondition because, since Task 10e, nothing
    /// else is exchanged at pairing time: the session manifest used to be
    /// minted and broadcast on first reaching `.ready`, and waiting for both
    /// sides to agree on it was what made this loop's exit meaningful. It is
    /// now minted per *rally* (`LiveSessionModel.armRallyIdentity()`), so
    /// waiting on it here would wait forever. `calibrations.count` rising is
    /// the equivalent proof that a full round of both models' 20 Hz
    /// `republish()` has run against a real `.ready` — a `baseline` rather
    /// than `== 1` because `testAFreshSessionAfterTeardownSendsTheCalibration\
    /// Again` runs this a second time on an accumulating counter.
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
        let calibrationBaseline = calibrations.count
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
            if primaryReady, secondaryReady, calibrations.count > calibrationBaseline {
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
