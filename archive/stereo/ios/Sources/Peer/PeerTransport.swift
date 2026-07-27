// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import Foundation

enum TransportState: Equatable { case idle, searching, connected, disconnected(String) }

protocol PeerTransport: AnyObject {
    var name: String { get }
    var onControl: ((Data) -> Void)? { get set }
    var onDatagram: ((Data) -> Void)? { get set }
    var onStateChange: ((TransportState) -> Void)? { get set }
    func startInitiator()
    func startResponder()
    func sendControl(_ frame: Data)
    func sendDatagram(_ datagram: Data)
    func stop()
}
