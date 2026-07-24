import XCTest
@testable import SquashLineCalling

final class DetectionTupleTests: XCTestCase {
    private func sample(_ seq: UInt32) -> DetectionTuple {
        DetectionTuple(seq: seq, ptsNs: 123_456_789_000, x: 540.5, y: 960.25,
                       conf: Float16(0.87), bboxH: Float16(22.0))
    }

    func testEncodedSizeIsHeaderPlus24PerTuple() {
        XCTAssertEqual(DetectionBatch.encode([sample(1), sample(2)]).count, 3 + 48)
    }

    func testRoundTrip() {
        let tuples = [sample(7), sample(8), sample(9)]
        XCTAssertEqual(DetectionBatch.decode(DetectionBatch.encode(tuples)), tuples)
    }

    func testEmptyBatchRoundTrips() {
        XCTAssertEqual(DetectionBatch.decode(DetectionBatch.encode([])), [])
    }

    func testMalformedReturnsNil() {
        XCTAssertNil(DetectionBatch.decode(Data([0x01, 5, 0, 1, 2])))   // truncated
        XCTAssertNil(DetectionBatch.decode(Data([0x02, 0, 0])))          // unknown type
        XCTAssertNil(DetectionBatch.decode(Data()))                       // empty
    }
}
