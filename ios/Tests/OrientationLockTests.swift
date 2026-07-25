// ios/Tests/OrientationLockTests.swift
import XCTest
import UIKit
@testable import SquashLineCalling

final class OrientationLockTests: XCTestCase {
    func testPlayTabIsLandscapeOnly() {
        // The preview is live while the operator aims the phone in its mount,
        // so a sideways preview is worst exactly where it matters most.
        XCTAssertEqual(OrientationLock.mask(for: .play), .landscape)
    }

    func testWebTabsRotateFreely() {
        // Matches and Coach are portrait mobile web UI.
        XCTAssertEqual(OrientationLock.mask(for: .matches), .all)
        XCTAssertEqual(OrientationLock.mask(for: .coach), .all)
    }

    func testPinnedMaskNarrowsToTheResolvedMount() {
        XCTAssertEqual(OrientationLock.pinnedMask(for: .landscapeRight), .landscapeRight)
        XCTAssertEqual(OrientationLock.pinnedMask(for: .landscapeLeft), .landscapeLeft)
    }

    /// Pinning must only ever narrow what the Play tab already permits — a
    /// pinned mask outside it would ask UIKit to rotate somewhere forbidden.
    func testPinnedMaskIsAlwaysASubsetOfThePlayMask() {
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            XCTAssertTrue(OrientationLock.mask(for: .play)
                .contains(OrientationLock.pinnedMask(for: orientation)), "\(orientation)")
        }
    }

    func testLandscapeInterfaceMapsToTheMatchingMount() {
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeRight), .landscapeRight)
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeLeft), .landscapeLeft)
    }

    func testPortraitInterfaceHasNoMount() {
        // Portrait is not a capture mode; callers fall back rather than guess.
        XCTAssertNil(OrientationLock.captureOrientation(for: .portrait))
        XCTAssertNil(OrientationLock.captureOrientation(for: .portraitUpsideDown))
        XCTAssertNil(OrientationLock.captureOrientation(for: .unknown))
    }
}
