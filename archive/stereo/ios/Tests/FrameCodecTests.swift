// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import XCTest
@testable import SquashLineCalling

final class FrameCodecTests: XCTestCase {
    func testRoundTripSingleFrame() throws {
        let payload = Data("hello".utf8)
        let wire = FrameCodec.encode(payload)
        let reassembler = FrameReassembler()
        XCTAssertEqual(try reassembler.ingest(wire), [payload])
    }

    func testSplitAcrossChunksAndCoalesced() throws {
        let a = Data("alpha".utf8), b = Data("bravo-long-payload".utf8)
        var wire = FrameCodec.encode(a); wire.append(FrameCodec.encode(b))
        let reassembler = FrameReassembler()
        // Feed one byte at a time: frames must pop out exactly twice, in order.
        var frames: [Data] = []
        for byte in wire { frames.append(contentsOf: try reassembler.ingest(Data([byte]))) }
        XCTAssertEqual(frames, [a, b])
    }

    func testEmptyPayloadAllowed() throws {
        let reassembler = FrameReassembler()
        XCTAssertEqual(try reassembler.ingest(FrameCodec.encode(Data())), [Data()])
    }

    func testOversizeFrameThrows() {
        var header = withUnsafeBytes(of: UInt32(2_000_000).littleEndian) { Data($0) }
        header.append(Data(repeating: 0, count: 8))
        let reassembler = FrameReassembler()
        XCTAssertThrowsError(try reassembler.ingest(header)) { error in
            XCTAssertEqual(error as? FrameCodecError, .oversizeFrame)
        }
    }
}
