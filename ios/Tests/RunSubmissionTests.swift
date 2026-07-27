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

    /// The poll loop used to be `while job.status == "queued" || "running"`
    /// with no cap. A job whose worker died still answers `"running"` forever —
    /// the HTTP layer is perfectly healthy, so nothing throws and `submit`
    /// never returns.
    ///
    /// That is not a hang the user can back out of: the sheet presenting this
    /// submission never resolves, so only killing the app recovers.
    ///
    /// A short `pollTimeout` here rather than the 20-minute production default:
    /// the bound under test is that one exists and is honoured, not its value.
    /// `MockAPIClient` with a single `"running"` status answers that way
    /// forever — `trackStatus` clamps its cursor to the last element — so
    /// without the cap this test would never finish.
    func testAJobThatNeverFinishesIsCappedInsteadOfPollingForever() async throws {
        let api = MockAPIClient(calibration: .success(try calibration()),
                                statuses: [status("running")])
        let submission = RunSubmission(api: api, pollInterval: .milliseconds(5),
                                       pollTimeout: .milliseconds(200))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)

        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected the poll to give up and fail, got \(submission.phase)")
        }
        // Routed through the same `catch` every other failure uses, so the
        // caller sees one shape of outcome.
        XCTAssertTrue(message.contains("stopped reporting"),
                      "the message must say the server went quiet, not invent a server error: \(message)")
        XCTAssertTrue(message.contains("Matches"), """
            and it must stay honest that the run may still finish server-side — giving up on the \
            poll does not cancel the job: \(message)
            """)

        // The *production* bound's rendering, asserted directly: the 200 ms
        // timeout above rounds down to "0 min", so the minutes arithmetic that
        // a real user reads is not otherwise covered by anything.
        XCTAssertEqual(
            APIError.trackingTimedOut(.seconds(20 * 60)).errorDescription,
            "The server stopped reporting on this clip after 20 min. "
                + "It may still finish on its own — check Matches.")
    }

    /// The cap must not fire on a job that is merely slow. Same mock shape as
    /// the happy path, but with the poll bound left at a value many multiples
    /// of the poll interval, so a cap that started counting in the wrong place
    /// (or measured the upload against the tracking bound) would show up here.
    func testASlowButProgressingJobIsNotCapped() async throws {
        let api = MockAPIClient(
            calibration: .success(try calibration()),
            statuses: [status("queued"), status("running"), status("running"), status("complete")])
        let submission = RunSubmission(api: api, pollInterval: .milliseconds(1),
                                       pollTimeout: .seconds(30))
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .complete = submission.phase else {
            return XCTFail("expected complete, got \(submission.phase)")
        }
    }
}
