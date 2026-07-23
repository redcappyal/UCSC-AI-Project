import XCTest
@testable import SquashLineCalling

final class RingBufferTests: XCTestCase {
    func testAppendsInOrderBelowCapacity() {
        var buffer = RingBuffer<Int>(capacity: 3)
        buffer.append(1); buffer.append(2)
        XCTAssertEqual(buffer.elements, [1, 2])
        XCTAssertEqual(buffer.count, 2)
    }

    func testWrapsKeepingNewestOldestFirst() {
        var buffer = RingBuffer<Int>(capacity: 3)
        for value in 1...5 { buffer.append(value) }
        XCTAssertEqual(buffer.elements, [3, 4, 5])
        XCTAssertEqual(buffer.count, 3)
    }
}
