// ios/Tests/OrientationLockTests.swift
import XCTest
import UIKit
@testable import SquashLineCalling

// OrientationPolicy is @MainActor-isolated; XCTestCase is not, so its test
// methods are nonisolated by default. Two tests below reference
// OrientationPolicy.shared and AppDelegate().application(...) directly —
// isolating the whole case keeps those references on the actor they
// require. Inert for the other, pure-value-type tests.
@MainActor
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
            // `.landscape.contains([])` is true, so the subset check alone
            // would pass vacuously for an arm that returned an empty mask —
            // e.g. a future CaptureOrientation case nobody wired up. Close
            // that hole explicitly.
            XCTAssertFalse(OrientationLock.pinnedMask(for: orientation).isEmpty, "\(orientation)")
        }
    }

    func testLandscapeInterfaceMapsToTheMatchingMount() {
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeRight), .landscapeRight)
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeLeft), .landscapeLeft)
    }

    func testPortraitInterfaceHasNoMount() {
        // Portrait is not a capture mode; `RecordModel.startCamera` is the
        // caller that falls back to a default rather than guessing a mount.
        XCTAssertNil(OrientationLock.captureOrientation(for: .portrait))
        XCTAssertNil(OrientationLock.captureOrientation(for: .portraitUpsideDown))
        XCTAssertNil(OrientationLock.captureOrientation(for: .unknown))
    }

    /// The delegate forwards `OrientationPolicy.shared.mask` rather than
    /// tracking its own state. `.portrait` is deliberately not a mask any tab
    /// or default produces (Play is `.landscape`, the web tabs and the launch
    /// seed are `.all`/`.landscape`), so this can't pass vacuously off a
    /// value the mask already happened to hold.
    func testDelegateServesThePolicysMask() {
        let previous = OrientationPolicy.shared.mask
        defer { OrientationPolicy.shared.apply(previous) }
        OrientationPolicy.shared.apply(.portrait)
        XCTAssertEqual(
            AppDelegate().application(UIApplication.shared, supportedInterfaceOrientationsFor: nil),
            .portrait)
    }

    /// `apply` assigns `mask` before it ever looks at `UIApplication.shared`'s
    /// scenes, so the delegate answers correctly on the very next query
    /// regardless of whether a foreground-active scene exists — which is the
    /// launch-time case FIX 1 exists for. `.portraitUpsideDown` is likewise a
    /// mask nothing else in the app ever applies, so a no-op `apply` would
    /// leave `mask` at whatever it was and this would fail rather than pass
    /// by coincidence.
    func testApplyUpdatesTheMaskRegardlessOfSceneState() {
        let previous = OrientationPolicy.shared.mask
        defer { OrientationPolicy.shared.apply(previous) }
        OrientationPolicy.shared.apply(.portraitUpsideDown)
        XCTAssertEqual(OrientationPolicy.shared.mask, .portraitUpsideDown)
    }

    /// `applyForTab` must prefer a live capture pin over the Play tab's
    /// resting `.landscape` mask — that preference is what keeps a running
    /// capture session pinned through a Play → Matches → Play round trip,
    /// since nothing else re-asserts the pin on tab exit and re-entry.
    /// `OrientationPolicy.shared` is a singleton other tests read, so save
    /// and restore both `mask` and `capturePin` with `defer`.
    func testApplyForTabPreservesTheCapturePinOnReturnToPlay() {
        let previousMask = OrientationPolicy.shared.mask
        let previousPin = OrientationPolicy.shared.capturePin
        defer {
            if let previousPin {
                OrientationPolicy.shared.pinForCapture(previousPin)
            } else {
                OrientationPolicy.shared.releaseCapturePin()
            }
            OrientationPolicy.shared.apply(previousMask)
        }

        OrientationPolicy.shared.pinForCapture(.landscapeRight)
        OrientationPolicy.shared.applyForTab(.play)
        XCTAssertEqual(OrientationPolicy.shared.mask, .landscapeRight,
                       "a live capture pin must win over the tab's resting mask")

        OrientationPolicy.shared.applyForTab(.matches)
        XCTAssertEqual(OrientationPolicy.shared.mask, .all)

        OrientationPolicy.shared.releaseCapturePin()
        OrientationPolicy.shared.applyForTab(.play)
        XCTAssertEqual(OrientationPolicy.shared.mask, .landscape,
                       "with the pin released, Play falls back to its resting mask")
    }
}
