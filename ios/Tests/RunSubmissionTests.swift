import XCTest
@testable import SquashLineCalling

private struct MockAPIClient: APIClientProtocol {
    private static let defaultCalibration = try! LatestCalibration(responseData: Data(
        #"{"ok": true, "run_id": "7", "calibration": {"lines": []}}"#.utf8))
    private static let defaultStatuses = [
        JobStatus(ok: true, status: "complete", runID: "run-9", stage: nil,
                  progress: 100, processedFrames: 1, totalFrames: 1,
                  message: "done", error: nil, hits: nil)
    ]

    var calibration: Result<LatestCalibration, Error> = .success(MockAPIClient.defaultCalibration)
    var statuses: [JobStatus] = MockAPIClient.defaultStatuses   // consumed by successive trackStatus calls

    final class Cursor: @unchecked Sendable { var index = 0 }
    let cursor = Cursor()

    // Records what the most recent startTrack call carried, so tests can
    // assert on paired-session fields without inspecting network traffic.
    final class RecordedStartTrack: @unchecked Sendable {
        var sessionID: String?
        var cameraRole: String?
        var peerVideoID: String?
        var syncManifestJSON: String?
    }
    let recorded = RecordedStartTrack()

    var lastSessionID: String? { recorded.sessionID }
    var lastCameraRole: String? { recorded.cameraRole }
    var lastPeerVideoID: String? { recorded.peerVideoID }
    var lastSyncManifestJSON: String? { recorded.syncManifestJSON }

    func latestCalibration() async throws -> LatestCalibration {
        try calibration.get()
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        UploadResponse(ok: true, videoID: "vid-1", fps: 30, frameCount: 900, duration: 30)
    }

    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String?, cameraRole: String?,
                    peerVideoID: String?, syncManifestJSON: String?) async throws -> JobStatus {
        recorded.sessionID = sessionID
        recorded.cameraRole = cameraRole
        recorded.peerVideoID = peerVideoID
        recorded.syncManifestJSON = syncManifestJSON
        return statuses[0]
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        cursor.index = min(cursor.index + 1, statuses.count - 1)
        return statuses[cursor.index]
    }

    func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String {
        "{}"
    }
}

@MainActor
final class RunSubmissionTests: XCTestCase {
    private func calibration() throws -> LatestCalibration {
        try LatestCalibration(responseData: Data(
            #"{"ok": true, "run_id": "7", "calibration": {"lines": []}}"#.utf8))
    }

    private func status(_ status: String, hits: [Hit]? = nil) -> JobStatus {
        JobStatus(ok: true, status: status, runID: "run-9", stage: nil,
                  progress: 50, processedFrames: 1, totalFrames: 2,
                  message: "msg", error: nil, hits: hits)
    }

    func testHappyPathReachesComplete() async throws {
        let api = MockAPIClient(
            calibration: .success(try calibration()),
            statuses: [status("queued"), status("running"), status("complete")])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .complete(let job) = submission.phase else {
            return XCTFail("expected complete, got \(submission.phase)")
        }
        XCTAssertEqual(job.runID, "run-9")
        XCTAssertEqual(submission.completedRunID, "run-9")
    }

    func testMissingCalibrationFailsWithActionableMessage() async {
        let api = MockAPIClient(calibration: .failure(APIError.noCalibration), statuses: [])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected failed")
        }
        XCTAssertTrue(message.contains("Calibrate"))
    }

    func testServerFailureSurfacesError() async throws {
        let failed = JobStatus(ok: true, status: "failed", runID: "run-9", stage: nil,
                               progress: nil, processedFrames: nil, totalFrames: nil,
                               message: nil, error: "Tracking failed hard.", hits: nil)
        let api = MockAPIClient(calibration: .success(try calibration()),
                          statuses: [status("queued"), failed])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected failed")
        }
        XCTAssertEqual(message, "Tracking failed hard.")
    }

    func testUnpairedSubmissionSendsNoPairedFields() async {
        let api = MockAPIClient()
        let submission = RunSubmission(api: api, pollInterval: .milliseconds(1))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/a.mp4"), duration: 3)
        XCTAssertNil(api.lastSessionID)
        XCTAssertNil(api.lastCameraRole)
        XCTAssertNil(api.lastPeerVideoID)
        XCTAssertNil(api.lastSyncManifestJSON)
    }

    func testPairedSubmissionCarriesSessionAndRole() async {
        let api = MockAPIClient()
        let submission = RunSubmission(api: api, pollInterval: .milliseconds(1))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/a.mp4"), duration: 3,
                                sessionID: "S-1", cameraRole: "b",
                                peerVideoID: "V-7", syncManifestJSON: "{\"clap_anchor_s\":0.01}")
        XCTAssertEqual(api.lastSessionID, "S-1")
        XCTAssertEqual(api.lastCameraRole, "b")
        XCTAssertEqual(api.lastPeerVideoID, "V-7")
        XCTAssertEqual(api.lastSyncManifestJSON, "{\"clap_anchor_s\":0.01}")
    }
}
