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

    /// The invariant the original bug broke: RecordModel labelled every
    /// detection tuple with hardcoded constants while PeerSession advertised
    /// its own, so the space tuples were expressed in could disagree with the
    /// space the peer was told to read them in.
    ///
    /// Be honest about the reach of this test. Both mounts currently yield the
    /// same (3840, 2160) — `CaptureSettings.frameSize(for:)` ignores its
    /// argument by design — so the frame-size assertions compare equal
    /// constants today. What keeps them from being vacuous by construction is
    /// that `model` and `session` derive the same arranged orientation (the
    /// loop's `orientation`) through independent paths, rather than one side
    /// reading the other's output: the model's path runs through its own
    /// `init` and `detectionFrameSize`; the session's runs through its own
    /// `init` and `advertisedHello`. Once `frameSize(for:)` distinguishes the
    /// mounts, a break in how `captureOrientation` reaches
    /// `detectionFrameSize` shows up as a mismatch here instead of being
    /// masked by both sides reading the same subject twice.
    ///
    /// The mount assertion is the one that bites today: it fails if `init`
    /// ignores the parameter, which is the wiring this task adds.
    func testDetectionFrameSpaceMatchesTheAdvertisedHello() {
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let model = RecordModel(detector: nil, captureOrientation: orientation)
            XCTAssertEqual(model.captureOrientation, orientation,
                           "init must honour the requested mount")
            let (transport, _) = LoopbackTransport.pair()
            let session = PeerSession(transport: transport, isInitiator: true, now: { 0 },
                                      captureOrientation: orientation)
            XCTAssertEqual(model.detectionFrameSize.width,
                           session.advertisedHello.frameW, "\(orientation)")
            XCTAssertEqual(model.detectionFrameSize.height,
                           session.advertisedHello.frameH, "\(orientation)")
        }
    }

    func testCaptureOrientationDefaultsToLandscapeRight() {
        XCTAssertEqual(RecordModel(detector: nil).captureOrientation, .landscapeRight)
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
        // zip yields one tuple per element, so this takes a single parameter —
        // `$0.px - $1.px` would not compile.
        let separations = zip(tracks.local, tracks.remote).map { pair in
            simd_length(pair.0.px - pair.1.px)
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

    /// §10 caps the wash at ≤ 500 ms and §16 gates the mini-court on "once the
    /// flash clears". Binding the flash to the persistent banner value leaves
    /// a verdict wash over the camera feed forever, so the two must be
    /// separate state with only the flash auto-clearing.
    func testFlashClearsButTheBannerPersists() {
        let model = RecordModel(detector: nil)
        model.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                              remoteModelJSON: StereoDemo.remoteModelJSON)

        spinMainUntil(timeout: 3.0) { model.flashPresentation != nil }
        XCTAssertNotNil(model.flashPresentation, "the wash never appeared")

        spinMainUntil(timeout: 2.0) { model.flashPresentation == nil }
        XCTAssertNil(model.flashPresentation, "the wash never cleared")
        XCTAssertNotNil(model.livePresentation,
                        "the banner must survive the flash clearing")
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
