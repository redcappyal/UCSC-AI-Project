import XCTest
import CoreMedia
@testable import SquashLineCalling

/// The court-capture contract: pinned 4K60, and an exposure solve that trades
/// ISO for shutter so the ball freezes instead of smearing.
final class CaptureSettingsTests: XCTestCase {

    // MARK: format selection

    func testPicksExactFourKSixtyWhenAvailable() {
        let specs = [
            CaptureSettings.FormatSpec(width: 1920, height: 1080, maxFrameRate: 60),
            CaptureSettings.FormatSpec(width: 3840, height: 2160, maxFrameRate: 60),
            CaptureSettings.FormatSpec(width: 1280, height: 720, maxFrameRate: 120),
        ]
        XCTAssertEqual(CaptureSettings.bestFormatIndex(specs), 1)
    }

    func testPrefersSixtyFPSOverHigherResolution() {
        // 4K30 is useless to us: the whole point of 60 is trajectory density.
        let specs = [
            CaptureSettings.FormatSpec(width: 3840, height: 2160, maxFrameRate: 30),
            CaptureSettings.FormatSpec(width: 1920, height: 1080, maxFrameRate: 60),
        ]
        XCTAssertEqual(CaptureSettings.bestFormatIndex(specs), 1)
    }

    func testDoesNotPickOversizedFormatOverFourK() {
        // Stills-sized formats blow the encoder and the ANE budget for no
        // gain — the detector downsamples to 960 anyway.
        let specs = [
            CaptureSettings.FormatSpec(width: 8064, height: 6048, maxFrameRate: 60),
            CaptureSettings.FormatSpec(width: 3840, height: 2160, maxFrameRate: 60),
        ]
        XCTAssertEqual(CaptureSettings.bestFormatIndex(specs), 1)
    }

    func testFallsBackToBestResolutionWhenNothingReachesSixty() {
        let specs = [
            CaptureSettings.FormatSpec(width: 1920, height: 1080, maxFrameRate: 30),
            CaptureSettings.FormatSpec(width: 3840, height: 2160, maxFrameRate: 30),
        ]
        XCTAssertEqual(CaptureSettings.bestFormatIndex(specs), 1)
    }

    func testReturnsNilWhenNoFormatsOffered() {
        XCTAssertNil(CaptureSettings.bestFormatIndex([]))
    }

    // MARK: exposure solve

    func testRaisesISOToHoldExposureAtTargetShutter() {
        // Metered 1/60 @ ISO 100. Going to 1/1000 is 16.67x less light, so
        // ISO has to climb by the same factor to land on the same exposure.
        let solution = CaptureSettings.solveExposure(
            meteredDuration: CMTime(value: 1, timescale: 60), meteredISO: 100,
            deviceMinISO: 34, deviceMaxISO: 6144)

        XCTAssertEqual(solution.duration.seconds, 1.0 / 1000, accuracy: 1e-9)
        XCTAssertEqual(solution.iso, 1666.67, accuracy: 0.1)
        XCTAssertFalse(solution.underexposed)
    }

    func testDropsToFallbackShutterWhenTargetWouldBlowTheISOCap() {
        // Metered 1/120 @ ISO 400: 1/1000 needs ISO 3333 (over the 2500 cap),
        // 1/500 needs 1667 (under it), so the solve gives up the shutter
        // rather than the noise floor.
        let solution = CaptureSettings.solveExposure(
            meteredDuration: CMTime(value: 1, timescale: 120), meteredISO: 400,
            deviceMinISO: 34, deviceMaxISO: 6144)

        XCTAssertEqual(solution.duration.seconds, 1.0 / 500, accuracy: 1e-9)
        XCTAssertEqual(solution.iso, 1666.67, accuracy: 0.1)
        XCTAssertFalse(solution.underexposed)
    }

    func testFlagsUnderexposedWhenEvenTheFallbackShutterCannotHoldExposure() {
        // Metered 1/60 @ ISO 1600 is a dim court. Both shutters overrun the
        // cap; we keep the fallback shutter, clip ISO, and say so.
        let solution = CaptureSettings.solveExposure(
            meteredDuration: CMTime(value: 1, timescale: 60), meteredISO: 1600,
            deviceMinISO: 34, deviceMaxISO: 6144)

        XCTAssertEqual(solution.duration.seconds, 1.0 / 500, accuracy: 1e-9)
        XCTAssertEqual(solution.iso, CaptureSettings.preferredMaxISO, accuracy: 0.1)
        XCTAssertTrue(solution.underexposed)
    }

    func testClampsUpToDeviceMinimumISOInBrightLight() {
        // Metered 1/2000 @ ISO 30 wants ISO 15 at 1/1000 — below what the
        // sensor accepts, so it pins to the device floor.
        let solution = CaptureSettings.solveExposure(
            meteredDuration: CMTime(value: 1, timescale: 2000), meteredISO: 30,
            deviceMinISO: 34, deviceMaxISO: 6144)

        XCTAssertEqual(solution.duration.seconds, 1.0 / 1000, accuracy: 1e-9)
        XCTAssertEqual(solution.iso, 34, accuracy: 0.001)
        XCTAssertFalse(solution.underexposed)
    }

    func testHonoursADeviceMaxISOTighterThanThePreferredCap() {
        // deviceMaxISO 800 binds before the 2500 preference does.
        let solution = CaptureSettings.solveExposure(
            meteredDuration: CMTime(value: 1, timescale: 60), meteredISO: 100,
            deviceMinISO: 34, deviceMaxISO: 800)

        XCTAssertEqual(solution.duration.seconds, 1.0 / 500, accuracy: 1e-9)
        XCTAssertEqual(solution.iso, 800, accuracy: 0.001)
        XCTAssertTrue(solution.underexposed)
    }

    // MARK: operator readout

    func testSummaryReportsTheLockedShutterAndISO() {
        let solution = CaptureSettings.ExposureSolution(
            duration: CMTime(value: 1, timescale: 1000), iso: 1450, underexposed: false)

        XCTAssertEqual(CaptureSettings.summary(for: solution), "Locked 1/1000 · ISO 1450")
    }

    func testSummaryCallsOutADimCourt() {
        let solution = CaptureSettings.ExposureSolution(
            duration: CMTime(value: 1, timescale: 500), iso: 2500, underexposed: true)

        let summary = CaptureSettings.summary(for: solution)
        XCTAssertTrue(summary.contains("1/500"), summary)
        XCTAssertTrue(summary.contains("ISO 2500"), summary)
        XCTAssertTrue(summary.lowercased().contains("dim"), summary)
    }
}
