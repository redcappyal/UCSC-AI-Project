// ios/Tests/RecordingOwnershipTests.swift
import XCTest
@testable import SquashLineCalling

/// `RecordModel.setRecording(_:owner:)` is the one funnel every start and stop
/// goes through, now that the record stage and the live layer share a single
/// camera. These assert the funnel's rule directly — no view, no radio — which
/// is the point: the holes this replaced were missed because the invariant
/// lived in a `View`'s private computed properties, where nothing could
/// assert it.
///
/// What the funnel can and cannot do in a test host: a start really does open
/// an `AVAssetWriter` (that setup does not depend on a running capture
/// session), but no frame ever reaches it, so a stop always ends in
/// `CameraError.recordingEmpty` — which cancels the writer, deletes the file,
/// and throws. So "did it act" is fully exercisable here; "what clip did it
/// produce" is not, and the routing test below uses the DEBUG seam
/// `publishFinishedClipForTesting` for exactly that one step.
@MainActor
final class RecordingOwnershipTests: XCTestCase {
    /// `detector: nil` — nothing here needs detections, and
    /// `SyntheticBallDetector` is `#if DEBUG`-only.
    private func makeRecord() -> RecordModel { RecordModel(detector: nil) }

    /// A start that the test depends on succeeding. `CameraController` names
    /// its output file from a whole-second timestamp, so a start can fail if
    /// an earlier test left a file of the same name behind — fail with the
    /// real reason rather than a confusing downstream assertion.
    private func startOrFail(_ record: RecordModel, owner: RecordingOwner,
                             file: StaticString = #filePath, line: UInt = #line) async -> Bool {
        let started = await record.setRecording(true, owner: owner)
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

        let acted = await record.setRecording(false, owner: .single)

        XCTAssertFalse(acted, "there was nothing to stop")
        XCTAssertFalse(record.isRecording)
        XCTAssertNil(record.recordingOwner)
        XCTAssertNil(record.recordingStartedAt)
        // The guard returns before the camera is touched at all, so a no-op
        // must not leave an error on screen either.
        XCTAssertNil(record.errorText)
        XCTAssertNil(record.singleCameraClip)
        XCTAssertNil(record.liveClip)
    }

    func testAStartWhileAlreadyRecordingIsANoOp() async {
        let record = makeRecord()
        guard await startOrFail(record, owner: .single) else { return }
        let startedAt = record.recordingStartedAt
        XCTAssertEqual(record.recordingOwner, .single)

        let again = await record.setRecording(true, owner: .single)

        XCTAssertFalse(again, "the camera is already recording")
        XCTAssertTrue(record.isRecording)
        XCTAssertEqual(record.recordingOwner, .single)
        XCTAssertEqual(record.recordingStartedAt, startedAt,
                       "a redundant start must not restart the clock")

        // The cross-owner case is the one the old navigation gate existed to
        // prevent, and it is the exact sequence a user can walk into now that
        // both hero cards are plainly navigable: record → back → live match →
        // START RALLY. The live start must not act, and — critically — must
        // not take ownership of a recording it did not start, or `stopRally()`
        // would later stop and submit the record stage's clip as rally footage.
        let liveStart = await record.setRecording(true, owner: .live)
        XCTAssertFalse(liveStart)
        XCTAssertEqual(record.recordingOwner, .single)

        await record.setRecording(false, owner: .single)   // teardown
        XCTAssertFalse(record.isRecording,
                       "teardown: no AVAssetWriter should be left running past this test")
    }

    // MARK: - Interleaving

    /// All four are enqueued synchronously, before any suspension point, so
    /// this pins down the order the funnel resolves them in rather than
    /// whatever order the underlying `Task`s happen to be scheduled — the
    /// property the live layer depends on, since `startRally()` fires its
    /// start off without awaiting it while `stopRally()`'s path awaits.
    ///
    /// The failure this rules out: if a later request could reach the camera
    /// before an earlier one, the stop below would find nothing recording and
    /// no-op, and the start would then leave the camera rolling with nothing
    /// left to stop it.
    func testInterleavedLiveAndSingleRequestsNeverInvert() async {
        let record = makeRecord()

        let liveStart = record.enqueueSetRecording(true, owner: .live)
        let singleStart = record.enqueueSetRecording(true, owner: .single)
        let singleStop = record.enqueueSetRecording(false, owner: .single)
        let liveStop = record.enqueueSetRecording(false, owner: .live)

        let liveStarted = await liveStart.value
        let singleStarted = await singleStart.value
        let singleStopped = await singleStop.value
        let liveStopped = await liveStop.value

        guard liveStarted else {
            return XCTFail("setup: the camera would not start — \(record.errorText ?? "no reason given")")
        }
        XCTAssertFalse(singleStarted, "a start behind a start must not act")
        XCTAssertFalse(singleStopped,
                       "the record stage must not stop a recording the live layer owns")
        XCTAssertTrue(liveStopped, "the owner's own stop must act, and must be last")

        XCTAssertFalse(record.isRecording)
        XCTAssertNil(record.recordingOwner,
                     "no AVAssetWriter should be left running past this test")
        // Nothing the record stage ever owned, so nothing it may see.
        XCTAssertNil(record.singleCameraClip)
    }

    // MARK: - Clip routing

    #if DEBUG
    func testALiveOwnedClipIsNotSurfacedToTheSingleCameraConsumer() {
        let record = makeRecord()
        let rally = FinishedClip(url: URL(fileURLWithPath: "/tmp/rally.mp4"), duration: 4)

        record.publishFinishedClipForTesting(rally, owner: .live)

        XCTAssertNil(record.singleCameraClip,
                     "a live-owned rally must never open ResultsView in the plain judge flow")
        XCTAssertEqual(record.liveClip?.id, rally.id)

        // And the plain flow's own dismissal — `.sheet(item:)` writing nil
        // through the binding — must not consume the live layer's clip.
        record.singleCameraClip = nil
        XCTAssertEqual(record.liveClip?.id, rally.id)

        // The mirror case: a plain recording must never be submitted as
        // paired rally footage.
        let plain = FinishedClip(url: URL(fileURLWithPath: "/tmp/plain.mp4"), duration: 2)
        record.publishFinishedClipForTesting(plain, owner: .single)

        XCTAssertNil(record.liveClip)
        XCTAssertEqual(record.singleCameraClip?.id, plain.id)

        record.liveClip = nil
        XCTAssertEqual(record.singleCameraClip?.id, plain.id)
    }
    #endif

    // MARK: - The predicate the screen asks

    /// `RecordView` offers its button only when this says the model would act,
    /// so the two cannot drift. Asserted here rather than in the view for the
    /// reason `LiveSessionModel`'s own type doc gives: the invariant has to be
    /// assertable without standing up a view.
    func testCanSetRecordingMatchesWhatTheFunnelWillDo() async {
        let record = makeRecord()

        XCTAssertTrue(record.canSetRecording(true, owner: .single))
        XCTAssertFalse(record.canSetRecording(false, owner: .single),
                       "nothing is recording, so there is no stop to offer")

        guard await startOrFail(record, owner: .live) else { return }

        XCTAssertFalse(record.canSetRecording(true, owner: .single))
        XCTAssertFalse(record.canSetRecording(false, owner: .single),
                       "the record stage must not be offered a stop for a live-owned rally")
        XCTAssertTrue(record.canSetRecording(false, owner: .live))
        XCTAssertTrue(record.isRecordingOwned(by: .live))
        XCTAssertFalse(record.isRecordingOwned(by: .single))

        await record.setRecording(false, owner: .live)   // teardown
        XCTAssertFalse(record.isRecording)
        XCTAssertTrue(record.canSetRecording(true, owner: .single),
                      "the record stage is usable again the moment the rally's camera stops")
    }
}
