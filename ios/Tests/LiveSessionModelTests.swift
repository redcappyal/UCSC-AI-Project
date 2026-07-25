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
        case .success: return Self.adoptableCameraModelJSON
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
