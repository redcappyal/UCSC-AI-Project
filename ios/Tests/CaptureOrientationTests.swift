// ios/Tests/CaptureOrientationTests.swift
import XCTest
@testable import SquashLineCalling

/// The capture frame space itself. The peer-handshake half of this file -- the
/// `Hello` frame-space/mount guard, which needed `PeerSession` -- moved to
/// archive/stereo/ios/Tests/PeerMountGuardTests.swift on 2026-07-27.
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
}
