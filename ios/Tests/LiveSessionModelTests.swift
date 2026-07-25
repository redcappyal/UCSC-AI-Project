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
}
