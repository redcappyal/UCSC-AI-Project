// ios/Sources/Peer/WiFiP2PTransport.swift
import Foundation
import Network

/// Peer-to-peer Wi-Fi (AWDL) transport. Responder: Bonjour TCP listener +
/// ephemeral UDP listener. Initiator: Bonjour browser + TCP and UDP
/// connections. The first control frame each way is the private
/// "UDP:<port>" announcement; everything after is opaque to this class.
/// All mutable state is confined to `queue`: public entry points dispatch
/// onto it, and Network.framework callbacks are configured to fire on it.
final class WiFiP2PTransport: PeerTransport {
    static let bonjourType = "_crosscourt._tcp"

    let name = "wifi-p2p"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?

    private let queue: DispatchQueue
    private var reassembler = FrameReassembler()
    private var tcpListener: NWListener?
    private var udpListener: NWListener?
    private var browser: NWBrowser?
    private var control: NWConnection?
    private var datagramOut: NWConnection?
    private var datagramIn: NWConnection?
    private var announcedLocalUDPPort = false
    /// Set once `tearDown()` runs; stays true for the life of the instance
    /// (a torn-down transport is dead — new sessions create new transports).
    /// Lets callbacks distinguish a stale, already-handled completion (e.g.
    /// Network.framework's async cancel finishing after we've already
    /// emitted our own `.disconnected`) from a live one.
    private var isTearingDown = false

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.wifi")) {
        self.queue = queue
    }

    private static func p2pParameters(_ base: NWParameters) -> NWParameters {
        base.includePeerToPeer = true
        return base
    }

    // MARK: start

    func startResponder() {
        queue.async { [weak self] in
            guard let self else { return }
            self.onStateChange?(.searching)
            self.startUDPListener()
            do {
                let listener = try NWListener(using: Self.p2pParameters(.tcp))
                listener.service = NWListener.Service(name: UUID().uuidString, type: Self.bonjourType)
                listener.newConnectionHandler = { [weak self] connection in
                    self?.adoptControl(connection)
                }
                listener.start(queue: self.queue)
                self.tcpListener = listener
            } catch {
                self.onStateChange?(.disconnected("listener failed: \(error.localizedDescription)"))
                self.tearDown()
            }
        }
    }

    func startInitiator() {
        queue.async { [weak self] in
            guard let self else { return }
            self.onStateChange?(.searching)
            self.startUDPListener()
            let browser = NWBrowser(for: .bonjour(type: Self.bonjourType, domain: nil),
                                    using: Self.p2pParameters(.tcp))
            browser.browseResultsChangedHandler = { [weak self] results, _ in
                guard let self, self.control == nil, let first = results.first else { return }
                self.browser?.cancel()
                self.adoptControl(NWConnection(to: first.endpoint, using: Self.p2pParameters(.tcp)))
            }
            browser.start(queue: self.queue)
            self.browser = browser
        }
    }

    private func adoptControl(_ connection: NWConnection) {
        control = connection
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.announceUDPPortIfNeeded()
            case .failed(let error):
                guard !self.isTearingDown else { return }
                self.onStateChange?(.disconnected(error.localizedDescription))
                self.tearDown()
            case .cancelled:
                guard !self.isTearingDown else { return }
                self.onStateChange?(.disconnected("cancelled"))
                self.tearDown()
            default:
                break
            }
        }
        receiveControlLoop(connection)
        connection.start(queue: queue)
    }

    private func startUDPListener() {
        do {
            let listener = try NWListener(using: Self.p2pParameters(.udp))
            listener.newConnectionHandler = { [weak self] connection in
                self?.datagramIn = connection
                self?.receiveDatagramLoop(connection)
                connection.start(queue: self?.queue ?? .main)
            }
            listener.start(queue: queue)
            udpListener = listener
        } catch {
            onStateChange?(.disconnected("udp listener failed: \(error.localizedDescription)"))
            tearDown()
        }
    }

    private func announceUDPPortIfNeeded() {
        guard !announcedLocalUDPPort, let port = udpListener?.port else { return }
        announcedLocalUDPPort = true
        rawSendControl(FrameCodec.encode(Data("UDP:\(port.rawValue)".utf8)))
    }

    // MARK: receive

    private func receiveControlLoop(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self, !self.isTearingDown else { return }
            if let data { self.ingestControlChunk(data) }
            if done || error != nil {
                self.onStateChange?(.disconnected(error?.localizedDescription ?? "control closed"))
                self.tearDown()
            } else {
                self.receiveControlLoop(connection)
            }
        }
    }

    private func ingestControlChunk(_ chunk: Data) {
        guard let frames = try? reassembler.ingest(chunk) else {
            onStateChange?(.disconnected("control stream corrupted"))
            tearDown()
            return
        }
        for frame in frames {
            if frame.starts(with: Data("UDP:".utf8)) {
                openDatagramPath(portFrame: frame)
            } else {
                onControl?(frame)
            }
        }
    }

    private func openDatagramPath(portFrame: Data) {
        guard let text = String(data: portFrame, encoding: .utf8),
              let rawPort = UInt16(text.dropFirst(4)),
              let port = NWEndpoint.Port(rawValue: rawPort),
              let control = control,
              case .hostPort(let host, _)? = control.currentPath?.remoteEndpoint else { return }
        let connection = NWConnection(host: host, port: port, using: Self.p2pParameters(.udp))
        connection.stateUpdateHandler = { [weak self] state in
            if case .ready = state { self?.onStateChange?(.connected) }
        }
        connection.start(queue: queue)
        datagramOut = connection
    }

    private func receiveDatagramLoop(_ connection: NWConnection) {
        connection.receiveMessage { [weak self] data, _, _, error in
            guard let self, !self.isTearingDown else { return }
            if let data { self.onDatagram?(data) }
            if error == nil { self.receiveDatagramLoop(connection) }
        }
    }

    // MARK: send

    func sendControl(_ frame: Data) {
        queue.async { [weak self] in
            self?.rawSendControl(FrameCodec.encode(frame))
        }
    }

    private func rawSendControl(_ encoded: Data) {
        control?.send(content: encoded, completion: .contentProcessed { _ in })
    }

    func sendDatagram(_ datagram: Data) {
        queue.async { [weak self] in
            self?.datagramOut?.send(content: datagram, completion: .contentProcessed { _ in })
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.tearDown()
            self.onStateChange?(.disconnected("stopped"))
        }
    }

    /// Callers must be on `queue`. Instances are single-use: a torn-down
    /// transport is discarded, never restarted.
    private func tearDown() {
        isTearingDown = true
        browser?.cancel(); tcpListener?.cancel(); udpListener?.cancel()
        control?.cancel(); datagramOut?.cancel(); datagramIn?.cancel()
        browser = nil; tcpListener = nil; udpListener = nil
        control = nil; datagramOut = nil; datagramIn = nil
        reassembler = FrameReassembler()
        announcedLocalUDPPort = false
    }
}
