// ios/Tests/RecordingFunnelTests.swift
import XCTest
@testable import SquashLineCalling

/// `RecordModel.setRecording(_:)` is the one funnel every start and stop goes
/// through. These assert its rule directly — no view — which is the point: the
/// holes this replaced were missed because the invariant lived in a `View`'s
/// private computed properties, where nothing could assert it.
///
/// Was `RecordingOwnershipTests`, which additionally arbitrated between two
/// consumers of one camera (the record stage and the archived live layer's
/// rally lifecycle). That arbitration went with the live layer on 2026-07-27 —
/// see archive/stereo/README.md — and the original file is preserved at
/// archive/stereo/ios/Tests/RecordingOwnershipTests.swift. What remains here is
/// everything the funnel still does with one consumer: absolute intent, real
/// state re-read at execution time, and issue-order serialization.
///
/// What the funnel can and cannot do in a test host: a start really does open
/// an `AVAssetWriter` (that setup does not depend on a running capture
/// session), but no frame ever reaches it, so a stop always ends in
/// `CameraError.recordingEmpty` — which cancels the writer, deletes the file,
/// and throws. So "did it act" is fully exercisable here; "what clip did it
/// produce" is not.
@MainActor
final class RecordingFunnelTests: XCTestCase {
    /// `detector: nil` — nothing here needs detections, and
    /// `SyntheticBallDetector` is `#if DEBUG`-only.
    ///
    /// `mountResolver` is what makes the starts below deterministic.
    /// `applyRecording`'s start path refuses unless the interface is resolved
    /// to a landscape mount ("Hold the phone in landscape to start
    /// recording."), and in production that reads the live window scene — so
    /// without this seam every start here would depend on whether the
    /// simulator's rotation had settled by the time XCTest ran, which is a
    /// race, not a test. `.landscapeRight` matches both `RecordModel`'s own
    /// `captureOrientation` default and `CameraController.orientation`'s, so
    /// nothing else about these models shifts. The refusal itself is
    /// production behaviour and is deliberately still there — this supplies a
    /// resolved mount, it does not remove the requirement for one.
    private func makeRecord() -> RecordModel {
        RecordModel(detector: nil,
                    mountResolver: { CaptureSettings.CaptureOrientation.landscapeRight })
    }

    /// A start that the test depends on succeeding. `CameraController` names
    /// its output file from a whole-second timestamp, so a start can fail if
    /// an earlier test left a file of the same name behind — fail with the
    /// real reason rather than a confusing downstream assertion.
    @discardableResult
    private func startOrFail(_ record: RecordModel,
                             file: StaticString = #filePath, line: UInt = #line) async -> Bool {
        let started = await record.setRecording(true)
        if !started {
            XCTFail("setup: the camera would not start — \(record.errorText ?? "no reason given")",
                    file: file, line: line)
        }
        return started
    }

    // MARK: - A transition that is not needed must not happen

    func testAStopWithNothingRecordingIsANoOp() async {
        let record = makeRecord()
        XCTAssertFalse(record.isRecording, "setup")

        let acted = await record.setRecording(false)

        XCTAssertFalse(acted, "there was nothing to stop")
        XCTAssertFalse(record.isRecording)
        XCTAssertNil(record.recordingStartedAt)
        // The guard returns before the camera is touched at all, so a no-op
        // must not leave an error on screen either.
        XCTAssertNil(record.errorText)
        XCTAssertNil(record.singleCameraClip)
    }

    func testAStartWhileAlreadyRecordingIsANoOp() async {
        let record = makeRecord()
        guard await startOrFail(record) else { return }
        let startedAt = record.recordingStartedAt

        let again = await record.setRecording(true)

        XCTAssertFalse(again, "the camera is already recording")
        XCTAssertTrue(record.isRecording)
        XCTAssertEqual(record.recordingStartedAt, startedAt,
                       "a redundant start must not restart the clock")

        await record.setRecording(false)   // teardown
        XCTAssertFalse(record.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
    }

    // MARK: - Interleaving

    /// All four are enqueued synchronously, before any suspension point, so
    /// this pins down the order the funnel resolves them in rather than
    /// whatever order the underlying `Task`s happen to be scheduled.
    ///
    /// The failure this rules out: if a later request could reach the camera
    /// before an earlier one, a stop could find nothing recording and no-op,
    /// and the start would then leave the camera rolling with nothing left to
    /// stop it.
    func testInterleavedRequestsNeverInvert() async {
        let record = makeRecord()

        let first = record.enqueueSetRecording(true)
        let redundant = record.enqueueSetRecording(true)
        let stop = record.enqueueSetRecording(false)
        let stopAgain = record.enqueueSetRecording(false)

        let started = await first.value
        let startedTwice = await redundant.value
        let stopped = await stop.value
        let stoppedTwice = await stopAgain.value

        guard started else {
            return XCTFail("setup: the camera would not start — \(record.errorText ?? "no reason given")")
        }
        XCTAssertFalse(startedTwice, "a start behind a start must not act")
        XCTAssertTrue(stopped, "the stop must act, and must land behind the start")
        XCTAssertFalse(stoppedTwice, "a stop behind a stop must not act")

        XCTAssertFalse(record.isRecording,
                       "no AVAssetWriter should be left running past this test")
    }

    // MARK: - The predicate the screen asks

    /// `RecordView` offers its button only when this says the model would act,
    /// so the two cannot drift. Asserted here rather than in the view because
    /// the invariant has to be assertable without standing up a view.
    func testCanSetRecordingMatchesWhatTheFunnelWillDo() async {
        let record = makeRecord()

        XCTAssertTrue(record.canSetRecording(true))
        XCTAssertFalse(record.canSetRecording(false),
                       "nothing is recording, so there is no stop to offer")

        guard await startOrFail(record) else { return }

        XCTAssertFalse(record.canSetRecording(true),
                       "already recording, so there is no start to offer")
        XCTAssertTrue(record.canSetRecording(false))

        await record.setRecording(false)   // teardown
        XCTAssertFalse(record.isRecording)
        XCTAssertTrue(record.canSetRecording(true),
                      "the record stage is usable again the moment the camera stops")
    }
}
