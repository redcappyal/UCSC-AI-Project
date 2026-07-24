import XCTest
@testable import SquashLineCalling

final class DetectionStreamTests: XCTestCase {
    func testMapperConvertsNormalizedRectToPixels() {
        let observation = BallObservation(
            timestamp: 1.5,
            rect: CGRect(x: 0.4, y: 0.7, width: 0.1, height: 0.05),  // Vision bottom-left
            confidence: 0.9)
        let tuple = DetectionMapper.tuple(seq: 3, observation: observation,
                                          frameW: 1080, frameH: 1920)
        XCTAssertEqual(tuple.seq, 3)
        XCTAssertEqual(tuple.ptsNs, 1_500_000_000)
        XCTAssertEqual(tuple.x, 0.45 * 1080, accuracy: 0.01)          // midX
        // Vision midY 0.725 from bottom → row from top = (1 − 0.725) × 1920
        XCTAssertEqual(tuple.y, (1 - 0.725) * 1920, accuracy: 0.01)
        XCTAssertEqual(Float(tuple.bboxH), Float(0.05 * 1920), accuracy: 0.1)
    }

    func testRemoteStoreKeepsNewest900() {
        let store = RemoteDetectionStore()
        let tuples = (0..<1000).map {
            DetectionTuple(seq: UInt32($0), ptsNs: UInt64($0), x: 0, y: 0,
                           conf: Float16(1), bboxH: Float16(1))
        }
        store.append(tuples)
        XCTAssertEqual(store.recent.count, 900)
        XCTAssertEqual(store.recent.first?.seq, 100)
        XCTAssertEqual(store.recent.last?.seq, 999)
    }
}
