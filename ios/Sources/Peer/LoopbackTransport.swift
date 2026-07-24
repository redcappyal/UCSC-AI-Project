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

    static func pair() -> (LoopbackTransport, LoopbackTransport) {
        let a = LoopbackTransport(), b = LoopbackTransport()
        a.peer = b; b.peer = a
        // Hold strong references through captured closures so `weak var peer`
        // stays alive as long as either end is.
        a.retainedPeer = b; b.retainedPeer = a
        return (a, b)
    }
    private var retainedPeer: LoopbackTransport?

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
