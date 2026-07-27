// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import Foundation

/// In-memory transport pair for tests and DEBUG benches. Synchronous
/// delivery by default; hooks let tests delay, reorder, or drop.
final class LoopbackTransport: PeerTransport {
    let name = "loopback"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?
    var controlDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?
    var datagramDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?

    private weak var peer: LoopbackTransport?

    /// Ownership contract: the caller owns both returned ends and must retain
    /// them for as long as the link is needed — `pair()` returns both
    /// precisely so the caller can hold them. `peer` is weak on both sides so
    /// pairing itself creates no reference cycle; if one end deallocates,
    /// deliveries to it silently no-op via the weak reference.
    static func pair() -> (LoopbackTransport, LoopbackTransport) {
        let a = LoopbackTransport(), b = LoopbackTransport()
        a.peer = b; b.peer = a
        return (a, b)
    }

    func startInitiator() { onStateChange?(.connected) }
    func startResponder() { onStateChange?(.connected) }

    func sendControl(_ frame: Data) {
        let deliver: (Data) -> Void = { [weak peer] in peer?.onControl?($0) }
        if let hook = controlDeliveryHook { hook(frame, deliver) } else { deliver(frame) }
    }

    func sendDatagram(_ datagram: Data) {
        let deliver: (Data) -> Void = { [weak peer] in peer?.onDatagram?($0) }
        if let hook = datagramDeliveryHook { hook(datagram, deliver) } else { deliver(datagram) }
    }

    func stop() { onStateChange?(.disconnected("stopped")) }
}
