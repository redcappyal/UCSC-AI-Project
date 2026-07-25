import XCTest
import CoreVideo
@testable import SquashLineCalling

#if DEBUG
@MainActor
final class SyntheticDetectorTests: XCTestCase {
    private func buffer() -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &pb)
        return pb!
    }

    func testProducesObservationAtEveryTimestamp() {
        let d = SyntheticBallDetector()
        for t in stride(from: 0.0, through: 2.0, by: 0.25) {
            XCTAssertNotNil(d.detect(buffer(), timestamp: t), "no observation at \(t)")
        }
    }

    func testDeterministicForTheSameTimestamp() {
        let a = SyntheticBallDetector().detect(buffer(), timestamp: 0.7)
        let b = SyntheticBallDetector().detect(buffer(), timestamp: 0.7)
        XCTAssertEqual(a, b)
    }

    func testStaysInsideTheNormalizedFrame() {
        let d = SyntheticBallDetector()
        for t in stride(from: -6.0, through: 6.0, by: 0.05) {
            let r = d.detect(buffer(), timestamp: t)!.rect
            XCTAssertGreaterThanOrEqual(r.minX, 0); XCTAssertLessThanOrEqual(r.maxX, 1)
            XCTAssertGreaterThanOrEqual(r.minY, 0); XCTAssertLessThanOrEqual(r.maxY, 1)
        }
    }

    func testSweepsAcrossTheFrameAndArcs() {
        let d = SyntheticBallDetector(period: 2.0)
        let start = d.detect(buffer(), timestamp: 0.0)!.rect.midX
        let mid   = d.detect(buffer(), timestamp: 1.0)!.rect
        let end   = d.detect(buffer(), timestamp: 1.99)!.rect.midX
        XCTAssertLessThan(start, end, "ball should travel across frame")
        // Vision origin is bottom-left, so the arc's apex is a LARGER v.
        XCTAssertGreaterThan(mid.midY, d.detect(buffer(), timestamp: 0.0)!.rect.midY)
    }

    func testDetectorKindReportsSynthetic() {
        let model = RecordModel(detector: SyntheticBallDetector())
        XCTAssertEqual(model.detectorKind, .synthetic)
        XCTAssertFalse(model.detectorMissing)
    }

    func testDetectorKindReportsNoneWhenNil() {
        XCTAssertEqual(RecordModel(detector: nil).detectorKind, .none)
    }
}
#endif
