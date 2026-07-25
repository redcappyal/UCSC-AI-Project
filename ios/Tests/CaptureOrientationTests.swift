// ios/Tests/CaptureOrientationTests.swift
import XCTest
@testable import SquashLineCalling

final class CaptureOrientationTests: XCTestCase {
    func testBothMountsShareOneUprightLandscapeFrameSpace() {
        // Identical dimensions for both mounts is the whole reason the wire
        // needs an explicit orientation field: width/height cannot tell a
        // landscape-left mount from a landscape-right one.
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let s = CaptureSettings.frameSize(for: orientation)
            XCTAssertEqual(s.width, CaptureSettings.sensorWidth, "\(orientation)")
            XCTAssertEqual(s.height, CaptureSettings.sensorHeight, "\(orientation)")
        }
    }

    func testRotationNormalizesEachMountUpright() {
        // Landscape-right matches the sensor's native readout; landscape-left
        // is that readout upside down, so it needs 180 to record upright.
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeRight), 0)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeLeft), 180)
    }

    func testFrameConstantsAreLandscape() {
        XCTAssertEqual(CaptureSettings.frameWidth, 3840)
        XCTAssertEqual(CaptureSettings.frameHeight, 2160)
    }

    func testMismatchedPeerFrameSizeIsRejected() {
        let pair = LoopbackTransport.pair()
        // Corrupt the incoming hello's frame size to simulate a peer in the
        // other orientation: same pixels, transposed — silently fatal for 3D.
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var h)? = ControlMessage.decode(frame) else { return deliver(frame) }
            (h.frameW, h.frameH) = (h.frameH, h.frameW)
            deliver(try! ControlMessage.encode(.hello(h)))
        }
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failure, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation")
                      || why.lowercased().contains("frame"), "unhelpful reason: \(why)")
    }
}
