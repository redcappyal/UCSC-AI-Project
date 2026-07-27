// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import Foundation

struct DetectionTuple: Equatable {
    var seq: UInt32
    var ptsNs: UInt64
    var x: Float
    var y: Float
    var conf: Float16
    var bboxH: Float16
}

enum DetectionBatch {
    static let typeByte: UInt8 = 0x01
    static let tupleSize = 24

    static func encode(_ tuples: [DetectionTuple]) -> Data {
        // Wire format's count field is a u16; clamp rather than trap on oversize input.
        let tuples = tuples.prefix(Int(UInt16.max))
        var out = Data([typeByte])
        out.append(withUnsafeBytes(of: UInt16(tuples.count).littleEndian) { Data($0) })
        for t in tuples {
            out.append(withUnsafeBytes(of: t.seq.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.ptsNs.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.x.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.y.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.conf.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.bboxH.bitPattern.littleEndian) { Data($0) })
        }
        return out
    }

    static func decode(_ data: Data) -> [DetectionTuple]? {
        guard data.count >= 3, data[data.startIndex] == typeByte else { return nil }
        let body = data.dropFirst(1)
        let count = Int(body.prefix(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian)
        let tuplesData = body.dropFirst(2)
        guard tuplesData.count == count * tupleSize else { return nil }
        var tuples: [DetectionTuple] = []
        tuples.reserveCapacity(count)
        var offset = tuplesData.startIndex
        func read<T: FixedWidthInteger>(_: T.Type) -> T {
            let value = tuplesData[offset ..< offset + MemoryLayout<T>.size]
                .withUnsafeBytes { $0.loadUnaligned(as: T.self) }
            offset += MemoryLayout<T>.size
            return T(littleEndian: value)
        }
        for _ in 0..<count {
            tuples.append(DetectionTuple(
                seq: read(UInt32.self),
                ptsNs: read(UInt64.self),
                x: Float(bitPattern: read(UInt32.self)),
                y: Float(bitPattern: read(UInt32.self)),
                conf: Float16(bitPattern: read(UInt16.self)),
                bboxH: Float16(bitPattern: read(UInt16.self))))
        }
        return tuples
    }
}
