// ios/Tests/LiveWiringTests.swift
import XCTest
import simd
@testable import SquashLineCalling

@MainActor
final class LiveWiringTests: XCTestCase {
    func testLivePresentationIsNilBeforeAnyEvent() {
        let model = RecordModel(detector: nil)
        XCTAssertNil(model.livePresentation)
    }

    #if DEBUG

    // MARK: - The demo fixture's own geometry

    /// The demo is only worth anything if its two eyes see a physically
    /// consistent ball. Project a court-feet track through both models, feed
    /// the engine, and the impact must come back on the front wall — proving
    /// triangulation recovered the geometry we started from.
    ///
    /// Fully synchronous: `StereoEngine.flushForTesting` drains the engine's
    /// own queue, and nothing here hops to main.
    func testDemoTrajectoryTriangulatesBackToAFrontWallCall() throws {
        let local = try CameraModel.fromJSON(Data(StereoDemo.localModelJSON.utf8))
        let remote = try CameraModel.fromJSON(Data(StereoDemo.remoteModelJSON.utf8))
        let tracks = StereoDemo.pixelTracks(local: local, remote: remote)

        // Every sample must be in front of both cameras, or the demo is
        // silently running on a partial trajectory.
        XCTAssertEqual(tracks.local.count, StereoDemo.courtTrack().count)
        XCTAssertEqual(tracks.remote.count, StereoDemo.courtTrack().count)

        let engine = StereoEngine(localModel: local, remoteModel: remote,
                                  remoteToLocal: { $0 })
        var events: [StereoEvent] = []
        engine.onEvent = { events.append($0) }
        for sample in tracks.local { engine.addLocalPixel(sample.px, tS: sample.tS) }
        engine.addRemote(StereoDemo.remoteTuples(tracks.remote))
        engine.processIfDue(now: 100.0)
        engine.flushForTesting()

        guard case .impact(let impact)? = events.first else {
            return XCTFail("demo trajectory emitted no impact")
        }
        XCTAssertEqual(impact.surface, "front_wall")
        XCTAssertEqual(impact.call, "in")
        XCTAssertEqual(impact.tS, StereoDemo.impactAtS, accuracy: 0.05)
        // Recovered within a couple of inches of where the arc was authored.
        XCTAssertEqual(impact.pointFt.z, 5.5 - 1.2 * StereoDemo.impactAtS, accuracy: 0.25)
    }

    /// The two eyes must actually disagree — identical pixel tracks would mean
    /// the baseline collapsed and triangulation is meaningless.
    func testTheTwoEyesSeeDifferentPixels() throws {
        let local = try CameraModel.fromJSON(Data(StereoDemo.localModelJSON.utf8))
        let remote = try CameraModel.fromJSON(Data(StereoDemo.remoteModelJSON.utf8))
        let tracks = StereoDemo.pixelTracks(local: local, remote: remote)
        let separations = zip(tracks.local, tracks.remote).map {
            simd_length($0.px - $1.px)
        }
        XCTAssertGreaterThan(separations.max() ?? 0, 10.0,
                             "3 ft of baseline should move the ball well over 10 px")
    }

    // MARK: - The wiring

    /// `startStereoDemo` sets `livePresentation` from the engine's `onEvent`
    /// after a `DispatchQueue.main.async` hop, and drives the engine from a
    /// `Timer` on the main run loop. A synchronous test method *occupies* main,
    /// so neither would ever run — `flushForTesting` drains the engine's queue,
    /// not main's. Spin the run loop instead.
    private func spinMainUntil(timeout: TimeInterval,
                               _ done: () -> Bool) {
        let deadline = Date().addingTimeInterval(timeout)
        while !done() && Date() < deadline {
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }
    }

    func testStereoDemoDrivesAVerdictAllTheWayToLivePresentation() {
        let model = RecordModel(detector: nil)
        model.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                              remoteModelJSON: StereoDemo.remoteModelJSON)

        spinMainUntil(timeout: 3.0) { model.livePresentation != nil }

        guard let presentation = model.livePresentation else {
            return XCTFail("the demo never produced a presentation")
        }
        XCTAssertEqual(presentation.word, "IN")
        XCTAssertEqual(presentation.detail, "high confidence")
        XCTAssertTrue(presentation.isVerdict)
        XCTAssertEqual(presentation.color, Theme.inCall)
    }

    func testStereoDemoSurvivesBeingStartedTwice() {
        let model = RecordModel(detector: nil)
        model.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                              remoteModelJSON: StereoDemo.remoteModelJSON)
        // Restarting must replace the timer, not stack a second one.
        model.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                              remoteModelJSON: StereoDemo.remoteModelJSON)
        XCTAssertNil(model.livePresentation, "restart clears the previous call")

        spinMainUntil(timeout: 3.0) { model.livePresentation != nil }
        XCTAssertNotNil(model.livePresentation)
    }

    func testMalformedCameraModelsFailLoudlyAndDoNotGoLive() {
        let model = RecordModel(detector: nil)
        model.startStereoDemo(localModelJSON: "{}", remoteModelJSON: "{}")
        XCTAssertNil(model.livePresentation)
        XCTAssertNotNil(model.errorText)
    }

    #endif
}
