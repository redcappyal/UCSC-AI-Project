// ios/Tests/CaptureOrientationTests.swift
import XCTest
@testable import SquashLineCalling

final class CaptureOrientationTests: XCTestCase {
    func testPortraitMatchesTodaysGeometry() {
        let s = CaptureSettings.frameSize(for: .portrait)
        XCTAssertEqual(s.width, CaptureSettings.sensorHeight)
        XCTAssertEqual(s.height, CaptureSettings.sensorWidth)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .portrait), 90)
    }

    func testLandscapeIsSensorNativeAndUnrotated() {
        let s = CaptureSettings.frameSize(for: .landscapeRight)
        XCTAssertEqual(s.width, CaptureSettings.sensorWidth)
        XCTAssertEqual(s.height, CaptureSettings.sensorHeight)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeRight), 0)
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
