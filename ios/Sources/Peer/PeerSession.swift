// ios/Sources/Peer/PeerSession.swift
import Foundation

/// Owns the pairing lifecycle over an abstract transport. Threading: all
/// callbacks arrive on the transport's delivery context and are processed
/// inline (transports guarantee serial delivery); @Published writes hop to
/// main. `tick` must be called from one consistent queue.
final class PeerSession: ObservableObject {
    enum Phase: Equatable {
        case idle, searching, confirming(code: String), syncing, ready, live,
             degraded(String), ended, failed(String)
    }

    /// UI mirror; written on the main thread only. Internal logic reads
    /// internalPhase (authoritative, lock-guarded) — never this.
    @Published private(set) var phase: Phase = .idle
    private var internalPhase: Phase = .idle

    var role: PeerRole? { stateLock.lock(); defer { stateLock.unlock() }; return _role }
    let clockSync = ClockSync()
    var peerHello: Hello? { stateLock.lock(); defer { stateLock.unlock() }; return _peerHello }
    var onRemoteDetections: (([DetectionTuple]) -> Void)?
    /// Fired on the transport delivery context (like onRemoteDetections) when
    /// a .calibration control message arrives. Phase 3: the payload carries
    /// the SOLVED camera-model JSON (CameraModel.fromJSON-parseable), not
    /// raw wizard taps.
    var onCalibration: ((_ profileID: String, _ payloadJSON: String) -> Void)?

    private let transport: PeerTransport
    private let isInitiator: Bool
    private let now: () -> TimeInterval
    private let heartbeatTimeout: TimeInterval

    // Loopback delivers synchronously: a send from inside a locked section
    // can re-enter handleControl on the same thread (our ping → peer's pong
    // → our handler, all in one call stack). Recursive lock keeps that
    // safe; production transports deliver on their own serial queues.
    private let stateLock = NSRecursiveLock()

    private var _role: PeerRole?
    private var _peerHello: Hello?

    private var myHello: Hello
    private var pendingPings: [UInt32: Double] = [:]   // pingID → t1
    private var nextPingID: UInt32 = 0
    private var nextHeartbeatSeq: UInt32 = 0
    private var burstPingsRemaining = 0
    private var lastPingAt: TimeInterval = -.infinity
    private var lastPeerActivityAt: TimeInterval = 0
    // Bug fix (beyond the coordinator's concurrency amendments): tick's `t`
    // and the injected `now()` closure are the same real-world clock in
    // production (the owner sources both from one wall clock), but tests
    // deliberately decouple them (`now: { 0 }` vs. a manually-advanced `t`)
    // to make ClockSync's NTP math deterministic. `lastPeerActivityAt` is
    // compared against `t` in tick's ready/live/degraded branches, so it
    // must be refreshed in `t`'s domain — using `now()` here (as literally
    // written in the task brief) desyncs the two clocks under the test
    // harness and spuriously degrades a healthy link. `lastTickAt` mirrors
    // the most recent `t` so `handleControl` (which has no `t` of its own)
    // can stamp activity consistently with tick's scheduling clock.
    private var lastTickAt: TimeInterval = 0
    // Bug fix (beyond the coordinator's concurrency amendments): the brief's
    // ready/live branch gates outgoing heartbeats on "haven't heard from the
    // peer in a while" (t - lastPeerActivityAt > heartbeatTimeout / 2). That
    // is self-suppressing whenever the two sides' tick cadence is even
    // loosely correlated (guaranteed under a shared driving loop, likely in
    // practice too): whichever side's gap crosses the threshold first sends
    // a heartbeat that refreshes the *other* side's gap, so the other side's
    // own trigger never fires — the first side's own inbound gap then grows
    // unbounded and it spuriously degrades (confirmed by trace: the
    // initiator alone re-sent 20 consecutive heartbeats while the responder
    // never sent one back, then the initiator hit heartbeatTimeout). The
    // brief's own prose says "missing heartbeat*s*" (plural) past the
    // timeout — i.e. each side is meant to emit on its own periodic cadence
    // regardless of what it has heard, the conventional heartbeat pattern.
    // `lastHeartbeatSentAt` makes the send trigger self-referential (mirrors
    // `lastPingAt`) instead of peer-reactive; the message type and the
    // degrade/recovery conditions are unchanged.
    private var lastHeartbeatSentAt: TimeInterval = -.infinity
    private var phaseBeforeDegraded: Phase?
    private var myClapOnset: Double?
    private var peerClapOnset: Double?

    static let syncBurstCount = 30
    static let burstInterval = 0.05        // 20 Hz during the burst
    static let steadyInterval = 10.0       // re-sync cadence in ready/live
    static let readyUncertainty = 0.005    // 5 ms gate to leave syncing

    init(transport: PeerTransport, isInitiator: Bool,
         now: @escaping () -> TimeInterval = ClockSync.hostNow,
         heartbeatTimeout: TimeInterval = 3.0) {
        self.transport = transport
        self.isInitiator = isInitiator
        self.now = now
        self.heartbeatTimeout = heartbeatTimeout
        self.myHello = Hello(protoVersion: peerProtoVersion,
                             appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                             deviceModel: ProcessInfo.processInfo.hostName,
                             nonce: UInt32.random(in: .min ... .max),
                             frameW: 1080, frameH: 1920)
        transport.onControl = { [weak self] in self?.handleControl($0) }
        transport.onDatagram = { [weak self] in self?.handleDatagram($0) }
        transport.onStateChange = { [weak self] in self?.handleTransportState($0) }
    }

    // MARK: lifecycle

    func start() {
        stateLock.lock(); defer { stateLock.unlock() }
        // A synchronously-delivered peer hello (loopback tests; fast radios)
        // can move us to .confirming before our own start() runs — never
        // regress an established phase back to .searching.
        if internalPhase == .idle { setPhase(.searching) }
        isInitiator ? transport.startInitiator() : transport.startResponder()
    }

    func confirmPairing() {
        stateLock.lock(); defer { stateLock.unlock() }
        guard case .confirming = internalPhase else { return }
        setPhase(.syncing)
        burstPingsRemaining = PeerSession.syncBurstCount
    }

    func goLive() {
        stateLock.lock(); defer { stateLock.unlock() }
        if internalPhase == .ready { setPhase(.live) }
    }

    func end() {
        stateLock.lock(); defer { stateLock.unlock() }
        setPhase(.ended)
        transport.stop()
    }

    // MARK: outbound

    func sendDetections(_ tuples: [DetectionTuple]) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        transport.sendDatagram(DetectionBatch.encode(tuples))
    }

    /// Phase 3: the payload carries the SOLVED camera-model JSON
    /// (CameraModel.fromJSON-parseable), not raw wizard taps.
    func sendCalibration(profileID: String, payloadJSON: String) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        sendControl(.calibration(profileID: profileID, calibrationJSON: payloadJSON))
    }

    func sendClapAnchor(localOnset: TimeInterval) {
        stateLock.lock(); defer { stateLock.unlock() }
        myClapOnset = localOnset
        sendControl(.clapAnchor(hostTime: localOnset))
        tryApplyAnchor()
    }

    /// Pump from the owner's queue. Drives pings + timeout detection.
    func tick(now t: TimeInterval) {
        stateLock.lock(); defer { stateLock.unlock() }
        lastTickAt = t
        switch internalPhase {
        case .syncing:
            if burstPingsRemaining > 0, t - lastPingAt >= PeerSession.burstInterval {
                sendPing(at: t); burstPingsRemaining -= 1
            }
            if let estimate = clockSync.estimate, estimate.uncertainty <= PeerSession.readyUncertainty {
                sendHeartbeat()
                lastPeerActivityAt = t
                lastHeartbeatSentAt = t
                setPhase(.ready)
            }
        case .ready, .live:
            if t - lastPingAt >= PeerSession.steadyInterval { sendPing(at: t) }
            if t - lastHeartbeatSentAt > heartbeatTimeout / 2 {
                sendHeartbeat()
                lastHeartbeatSentAt = t
            }
            if t - lastPeerActivityAt > heartbeatTimeout {
                phaseBeforeDegraded = internalPhase
                setPhase(.degraded("link lost"))
            }
        case .degraded:
            sendHeartbeat()
            if t - lastPeerActivityAt <= heartbeatTimeout {
                setPhase(phaseBeforeDegraded ?? .ready)
            }
        default: break
        }
    }

    // MARK: inbound

    private func handleTransportState(_ state: TransportState) {
        stateLock.lock(); defer { stateLock.unlock() }
        switch state {
        case .connected:
            sendControl(.hello(myHello))
        case .disconnected(let reason):
            // Bug fix (beyond the coordinator's concurrency amendments):
            // transport.stop() delivers .disconnected synchronously and can
            // re-enter this handler on the same call stack that just set
            // .failed (e.g. a version-mismatch hello handler calls
            // transport.stop() right after setPhase(.failed(...))). .failed
            // is terminal like .ended/.idle and must not be clobbered.
            switch internalPhase {
            case .ended, .idle, .failed: break
            default:
                // Reviewer fix: a second .disconnected while already
                // .degraded(reasonA) must not capture .degraded(reasonA) as
                // the "before" phase — recovery would then restore
                // .degraded(reasonA) forever. Only capture a non-degraded
                // phase; the tick degrade branch is already safe by
                // construction (it only runs from .ready/.live).
                if case .degraded = internalPhase {} else {
                    phaseBeforeDegraded = internalPhase
                }
                setPhase(.degraded(reason))
            }
        case .idle, .searching: break
        }
    }

    private func handleControl(_ frame: Data) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let message = ControlMessage.decode(frame) else { return }
        lastPeerActivityAt = lastTickAt
        switch message {
        case .hello(let theirs):
            guard theirs.protoVersion == peerProtoVersion else {
                setPhase(.failed("protocol version mismatch — update both apps"))
                transport.stop()
                return
            }
            _peerHello = theirs
            _role = isInitiator ? .primary : .secondary
            if isInitiator { sendControl(.role(.secondary)) }
            setPhase(.confirming(code: pairingCode(myHello, theirs)))
        case .role(let assigned):
            _role = assigned
        case .syncPing(let ping):
            let t2 = now()
            sendControl(.syncPong(SyncPong(pingID: ping.pingID, t1: ping.t1, t2: t2, t3: now())))
            // Symmetric sync: the responder also learns the offset from its own pings.
            if case .syncing = internalPhase, burstPingsRemaining == 0 {
                burstPingsRemaining = PeerSession.syncBurstCount
            }
        case .syncPong(let pong):
            guard pendingPings.removeValue(forKey: pong.pingID) != nil else { return }
            clockSync.addSample(t1: pong.t1, t2: pong.t2, t3: pong.t3, t4: now())
        case .clapAnchor(let hostTime):
            peerClapOnset = hostTime
            tryApplyAnchor()
        case .heartbeat:
            if internalPhase == .syncing, clockSync.estimate != nil { setPhase(.ready) }
        case .calibration(let profileID, let calibrationJSON):
            // Phase 3: the payload carries the SOLVED camera-model JSON
            // (CameraModel.fromJSON-parseable), not raw wizard taps.
            onCalibration?(profileID, calibrationJSON)
        case .record, .event, .sessionManifest:
            break   // consumed by Phase 4/5 code; parsing is already validated
        }
    }

    private func handleDatagram(_ datagram: Data) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let tuples = DetectionBatch.decode(datagram), !tuples.isEmpty else { return }
        onRemoteDetections?(tuples)
    }

    // MARK: helpers

    private func sendPing(at t: TimeInterval) {
        lastPingAt = t
        let ping = SyncPing(pingID: nextPingID, t1: now())
        pendingPings[ping.pingID] = ping.t1
        nextPingID += 1
        sendControl(.syncPing(ping))
    }

    // Reviewer fold-in: was duplicated at all three tick call sites
    // (.syncing's ready transition, .ready/.live's periodic send, .degraded's
    // unconditional send). Unlocked like the other helpers — only ever
    // called from tick, an already-locked entry point.
    private func sendHeartbeat() {
        sendControl(.heartbeat(seq: nextHeartbeatSeq))
        nextHeartbeatSeq += 1
    }

    private func tryApplyAnchor() {
        guard let mine = myClapOnset, let theirs = peerClapOnset else { return }
        clockSync.applyAnchor(localEventTime: mine, remoteEventTime: theirs)
    }

    private func sendControl(_ message: ControlMessage) {
        guard let data = try? ControlMessage.encode(message) else { return }
        transport.sendControl(data)
    }

    private func setPhase(_ newPhase: Phase) {
        internalPhase = newPhase
        if Thread.isMainThread { phase = newPhase }
        else { DispatchQueue.main.async { self.phase = newPhase } }
    }
}
