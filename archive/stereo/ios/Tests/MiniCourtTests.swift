// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Tests/MiniCourtTests.swift
import XCTest
import simd
@testable import SquashLineCalling

final class MiniCourtTests: XCTestCase {
    private let eps = 1e-9

    /// `CGPoint` stores `CGFloat`, and `XCTAssertEqual(_:_:accuracy:)` is
    /// generic over a single `FloatingPoint` — mixing `CGFloat` and `Double`
    /// in one call does not compile. Everything goes through here.
    private func assertClose(_ got: CGFloat, _ want: Double,
                             _ message: String = "",
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Double(got), want, accuracy: eps, message, file: file, line: line)
    }

    // MARK: - Plan view (from above)

    /// Court feet: x = width 0…21 left to right, y = depth 0 **at the front
    /// wall** to 32 at the back. Getting y's origin backwards mirrors every
    /// plot front-to-back, so pin all four corners.
    func testPlanViewMapsTheFourCourtCornersToTheUnitSquare() {
        let corners: [(SIMD3<Double>, CGPoint)] = [
            (SIMD3(0, 0, 0),   CGPoint(x: 0, y: 0)),   // front-left
            (SIMD3(21, 0, 0),  CGPoint(x: 1, y: 0)),   // front-right
            (SIMD3(0, 32, 0),  CGPoint(x: 0, y: 1)),   // back-left
            (SIMD3(21, 32, 0), CGPoint(x: 1, y: 1)),   // back-right
        ]
        for (point, expected) in corners {
            let got = CourtProjection.planView(point)
            assertClose(got.x, Double(expected.x), "x for \(point)")
            assertClose(got.y, Double(expected.y), "y for \(point)")
        }
    }

    func testPlanViewPutsTheFrontWallAtTheTop() {
        // The front wall is where the calls happen; it reads at the top.
        XCTAssertLessThan(CourtProjection.planView(SIMD3(10, 1, 0)).y,
                          CourtProjection.planView(SIMD3(10, 30, 0)).y)
    }

    // MARK: - Side elevation

    func testSideElevationPutsTheFloorOnTheBottomEdge() {
        // Screen space: y grows downward, so the floor is y = 1.
        assertClose(CourtProjection.sideElevation(SIMD3(10, 16, 0)).y, 1.0)
    }

    func testSideElevationPutsTheOutLineOnTheTopEdge() {
        assertClose(
            CourtProjection.sideElevation(SIMD3(10, 16, StereoMath.outLineHeightFt)).y,
            0.0)
    }

    func testSideElevationPlacesTheTinJustAboveTheFloor() {
        let tin = CourtProjection.sideElevation(SIMD3(10, 0, StereoMath.tinTopHeightFt)).y
        assertClose(tin, 1.0 - (19.0 / 12.0) / StereoMath.outLineHeightFt)
        XCTAssertGreaterThan(tin, 0.85)   // sanity: the tin is low on the wall
    }

    func testSideElevationRunsFrontWallToBack() {
        assertClose(CourtProjection.sideElevation(SIMD3(10, 0, 0)).x, 0.0)
        assertClose(CourtProjection.sideElevation(SIMD3(10, 32, 0)).x, 1.0)
    }

    // MARK: - Clamping

    /// The real `no_call` golden point: raw unsnapped triangulation puts it
    /// 1.67 ft *behind* the front wall. Unclamped it would draw outside the
    /// panel; the projection must fold it onto the edge instead.
    func testUnresolvedGoldenPointClampsRatherThanEscaping() {
        let rogue = SIMD3(13.63, -1.67, 5.07)
        let plan = CourtProjection.planView(rogue)
        assertClose(plan.y, 0.0)
        XCTAssertTrue((0...1).contains(plan.x))
        let side = CourtProjection.sideElevation(rogue)
        assertClose(side.x, 0.0)
        XCTAssertTrue((0...1).contains(side.y))
    }

    func testEveryProjectionStaysInsideTheUnitSquare() {
        let wild: [SIMD3<Double>] = [
            SIMD3(-50, -50, -50), SIMD3(500, 500, 500),
            SIMD3(21.5, 32.5, 22), SIMD3(-0.001, 32.0001, 15.0001),
        ]
        for point in wild {
            for projected in [CourtProjection.planView(point),
                              CourtProjection.sideElevation(point)] {
                XCTAssertTrue((0...1).contains(projected.x), "x escaped for \(point)")
                XCTAssertTrue((0...1).contains(projected.y), "y escaped for \(point)")
            }
        }
    }

    func testProjectionsUseTheRealCourtConstants() {
        // Guards against a hardcoded 21/32 drifting from StereoMath.
        XCTAssertEqual(StereoMath.courtWidthFt, 21.0, accuracy: eps)
        XCTAssertEqual(StereoMath.courtLengthFt, 32.0, accuracy: eps)
        XCTAssertEqual(CourtProjection.maxHeightFt, StereoMath.outLineHeightFt, accuracy: eps)
    }
}
