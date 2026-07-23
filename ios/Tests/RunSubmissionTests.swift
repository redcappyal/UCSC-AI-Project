import XCTest
@testable import SquashLineCalling

private struct FakeAPI: APIClientProtocol {
    var calibration: Result<LatestCalibration, Error>
    var statuses: [JobStatus]   // consumed by successive trackStatus calls

    final class Cursor: @unchecked Sendable { var index = 0 }
    let cursor = Cursor()

    func latestCalibration() async throws -> LatestCalibration {
        try calibration.get()
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        UploadResponse(ok: true, videoID: "vid-1", fps: 30, frameCount: 900, duration: 30)
    }

    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        statuses[0]
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        cursor.index = min(cursor.index + 1, statuses.count - 1)
        return statuses[cursor.index]
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
        let api = FakeAPI(
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
        let api = FakeAPI(calibration: .failure(APIError.noCalibration), statuses: [])
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
        let api = FakeAPI(calibration: .success(try calibration()),
                          statuses: [status("queued"), failed])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected failed")
        }
        XCTAssertEqual(message, "Tracking failed hard.")
    }
}
