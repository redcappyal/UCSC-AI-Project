import Foundation

enum FrameCodecError: Error, Equatable { case oversizeFrame }

/// u32 little-endian payload length + payload. Shared by the TCP control
/// stream and the BLE control characteristic.
enum FrameCodec {
    static let maxFrameLength = 1 << 20   // 1 MiB: calibration JSON fits with margin

    static func encode(_ payload: Data) -> Data {
        var out = withUnsafeBytes(of: UInt32(payload.count).littleEndian) { Data($0) }
        out.append(payload)
        return out
    }
}

/// Stateful: feed chunks in arrival order, get complete payloads out.
final class FrameReassembler {
    private var buffer = Data()

    func ingest(_ chunk: Data) throws -> [Data] {
        buffer.append(chunk)
        var frames: [Data] = []
        while buffer.count >= 4 {
            let length = buffer.prefix(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
            guard length <= FrameCodec.maxFrameLength else { throw FrameCodecError.oversizeFrame }
            let total = 4 + Int(length)
            guard buffer.count >= total else { break }
            frames.append(Data(buffer[buffer.startIndex + 4 ..< buffer.startIndex + total]))
            buffer.removeFirst(total)
        }
        return frames
    }
}
