// ios/Tests/BLEChunkerTests.swift
import XCTest
@testable import SquashLineCalling

final class BLEChunkerTests: XCTestCase {
    func testChunksReassembleToOriginalFrame() throws {
        let payload = Data((0..<1000).map { UInt8($0 % 256) })
        let frame = FrameCodec.encode(payload)
        let chunks = BLEChunker.chunks(frame, maxWriteLength: 180)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 180 })
        XCTAssertEqual(chunks.count, Int(ceil(Double(frame.count) / 180.0)))
        let reassembler = FrameReassembler()
        var frames: [Data] = []
        for chunk in chunks { frames.append(contentsOf: try reassembler.ingest(chunk)) }
        XCTAssertEqual(frames, [payload])
    }

    func testSmallFrameIsSingleChunk() {
        XCTAssertEqual(BLEChunker.chunks(Data([1, 2, 3]), maxWriteLength: 180).count, 1)
    }

    func testNonPositiveMaxWriteLengthDoesNotTrap() {
        XCTAssertEqual(BLEChunker.chunks(Data([1, 2]), maxWriteLength: 0).count, 2)
    }
}
