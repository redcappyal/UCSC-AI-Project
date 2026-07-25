import Foundation

let peerProtoVersion = 1

enum PeerRole: String, Codable { case primary, secondary }

struct Hello: Codable, Equatable {
    var protoVersion: Int
    var appVersion: String
    var deviceModel: String
    var nonce: UInt32
    var frameW: Int
    var frameH: Int
    /// Which way the phone is mounted. Both mounts share one frame size, so
    /// frameW/frameH cannot distinguish them — this can.
    ///
    /// Optional is load-bearing, not laziness. A required field makes a hello
    /// from a build predating it THROW in JSONDecoder; ControlMessage.decode
    /// returns nil and handleControl guards that away with no branch, so the
    /// frame is dropped silently and pairing hangs in .searching with no
    /// diagnostic. Optional decodes cleanly: a peer predating this field
    /// advertises no mount at all, which cannot equal any real orientation
    /// and is refused as a mismatch — not read as any particular capture
    /// mode.
    var captureOrientation: CaptureSettings.CaptureOrientation? = nil
}

struct SyncPing: Codable, Equatable { var pingID: UInt32; var t1: Double }
struct SyncPong: Codable, Equatable { var pingID: UInt32; var t1: Double; var t2: Double; var t3: Double }

enum ControlMessage: Codable, Equatable {
    case hello(Hello)
    case role(PeerRole)
    case calibration(profileID: String, calibrationJSON: String)
    case syncPing(SyncPing)
    case syncPong(SyncPong)
    case clapAnchor(hostTime: Double)
    case record(action: String, ptsNs: UInt64)
    case event(rallyID: UInt32, json: String)
    case heartbeat(seq: UInt32)
    case sessionManifest(sessionID: String, videoID: String)

    static func encode(_ message: ControlMessage) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(message)
    }

    static func decode(_ data: Data) -> ControlMessage? {
        try? JSONDecoder().decode(ControlMessage.self, from: data)
    }
}

/// Same 4 digits on both screens; users visually confirm. XOR makes it
/// order-independent.
func pairingCode(_ a: Hello, _ b: Hello) -> String {
    String(format: "%04d", (a.nonce ^ b.nonce) % 10000)
}
