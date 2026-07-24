#if DEBUG
import SwiftUI

/// Dev-only bench harness: pick transport + role, pair, watch live stats,
/// run a timed bench, share the JSON report. Not part of shipped UI
/// (DESIGN.md phases arrive in spec Phase 4).
struct PeerBenchView: View {
    @StateObject private var model = PeerBenchModel()

    var body: some View {
        List {
            Section("Setup") {
                Picker("Transport", selection: $model.transportName) {
                    Text("Bluetooth").tag("ble")
                    Text("Wi-Fi P2P").tag("wifi-p2p")
                }
                Picker("Role", selection: $model.isInitiator) {
                    Text("Primary (initiator)").tag(true)
                    Text("Secondary").tag(false)
                }
                Button(model.running ? "Stop" : "Start pairing") { model.toggle() }
            }
            Section("Status") {
                Text("Phase: \(model.phaseText)")
                if let code = model.pairingCode {
                    Text("Pairing code: \(code)")
                        .font(.title2.monospacedDigit())
                    Button("Codes match — confirm") { model.confirm() }
                }
                Text("Offset: \(model.offsetText)")
                Text("RTT median/p95/max: \(model.rttText)")
                Text("Datagram loss: \(model.lossText)")
                Text("Thermal: \(model.thermalText)")
            }
            Section("Bench") {
                Button("Run 60 s datagram bench (120 Hz)") { model.runBench() }
                    .disabled(!model.canBench)
                Button("Arm clap anchor") { model.armClap() }
                    .disabled(!model.canBench)
                if let url = model.reportURL {
                    ShareLink(item: url) { Text("Share report JSON") }
                }
            }
        }
        .navigationTitle("Peer Bench")
    }
}

/// Owns a PeerSession over the chosen transport, pumps `tick` on a timer,
/// counts echoed datagrams for RTT/loss, and writes BenchReport JSON to
/// the Documents directory. ~150 lines of glue; no protocol logic —
/// everything measurable is delegated to PeerSession/ClockSync/BenchReport.
final class PeerBenchModel: ObservableObject {
    @Published var transportName = "ble"
    @Published var isInitiator = true
    @Published var running = false
    @Published var phaseText = "idle"
    @Published var pairingCode: String?
    @Published var offsetText = "—"
    @Published var rttText = "—"
    @Published var lossText = "—"
    @Published var thermalText = "nominal"
    @Published var reportURL: URL?
    var canBench: Bool { running && (phaseText.hasPrefix("ready") || phaseText.hasPrefix("live")) }

    private var session: PeerSession?
    private var transport: PeerTransport?
    private var timer: Timer?
    private var rtts: [Double] = []
    private var sent = 0, received = 0
    private var benchStart: Date?
    private var thermal: [String] = []
    private var clapArmed = false
    private let clapDetector = ClapDetector()
    private let camera = CameraController()   // audio-only use for the clap

    func toggle() { running ? stop() : start() }

    /// Called from the "Codes match — confirm" button once the human has
    /// visually compared the pairing code on both phones. Without this the
    /// bench (and real usage) wedges forever in `.confirming` — PeerSession
    /// only advances past it via an explicit confirmPairing() call.
    func confirm() { session?.confirmPairing() }

    func start() {
        let transport: PeerTransport = transportName == "ble" ? BLETransport() : WiFiP2PTransport()
        let session = PeerSession(transport: transport, isInitiator: isInitiator)
        self.transport = transport; self.session = session
        session.onRemoteDetections = { [weak self] tuples in
            guard let self else { return }
            // Fires on the transport's delivery queue, not main — pump()
            // (Timer-driven, main thread) also reads/mutates `received` and
            // `rtts`, so all counter state must be confined to main. Capture
            // the receive timestamp here, first, so RTT isn't inflated by
            // queue-hop latency before the async hop.
            let now = ClockSync.hostNow()
            // Echo bench: initiator sends, responder reflects, initiator times.
            if !self.isInitiator {
                // Reflecting stays on the delivery queue — PeerSession is
                // internally lock-guarded, so calling sendDetections here
                // (off main) is safe. Only the counter needs main.
                session.sendDetections(tuples)
                DispatchQueue.main.async { self.received += tuples.count }
            } else {
                let localRtts = tuples.map { (now - Double($0.ptsNs) / 1e9) * 1000 }
                DispatchQueue.main.async {
                    self.rtts.append(contentsOf: localRtts)
                    self.received += tuples.count
                }
            }
        }
        session.start()
        running = true
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.pump()
        }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        session?.end(); session = nil; transport = nil
        running = false
    }

    private func pump() {
        guard let session else { return }
        session.tick(now: ClockSync.hostNow())
        phaseText = "\(session.phase)"
        if case .confirming(let code) = session.phase { pairingCode = code } else { pairingCode = nil }
        if let estimate = session.clockSync.estimate {
            offsetText = String(format: "%.2f ms ± %.2f ms",
                                estimate.offset * 1000, estimate.uncertainty * 1000)
        }
        if session.phase == .ready || session.phase == .live {
            session.goLive()
        }
        if let benchStart, session.phase == .live {
            // Only the primary originates synthetic traffic and times the
            // echo — the responder's report is per-device (offset, thermal,
            // reflected-echo count) with empty rtts / zero sent, per the
            // task-11 reviewer fix.
            if isInitiator {
                // 3 datagrams x 2 tuples/datagram per 50 ms tick = 120 Hz
                // tuple rate, 60 Hz datagram rate — mirrors RecordModel's
                // production batching pattern (>= 2 tuples flushed per
                // datagram, ~30 ms cadence) instead of 20 Hz singletons.
                // `sent` counts tuples (not datagrams) so loss% math is
                // unchanged.
                for _ in 0..<3 {
                    var batch: [DetectionTuple] = []
                    for _ in 0..<2 {
                        batch.append(DetectionTuple(seq: UInt32(sent), ptsNs: UInt64(ClockSync.hostNow() * 1e9),
                                                     x: 0, y: 0, conf: Float16(1), bboxH: Float16(1)))
                        sent += 1
                    }
                    session.sendDetections(batch)
                }
                let sorted = rtts.sorted()
                if !sorted.isEmpty {
                    rttText = String(format: "%.0f / %.0f / %.0f ms",
                                     sorted[sorted.count / 2],
                                     sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))],
                                     sorted.last!)
                }
                lossText = String(format: "%.1f %%",
                                  sent == 0 ? 0 : Double(sent - received) / Double(sent) * 100)
            }
            if Date().timeIntervalSince(benchStart) >= 60 { finishBench() }
        }
        let state = ProcessInfo.processInfo.thermalState
        thermalText = ["nominal", "fair", "serious", "critical"][state.rawValue]
    }

    func runBench() {
        rtts.removeAll(); sent = 0; received = 0
        // Start/end thermal timeline: one reading now, one more in
        // finishBench() — not a 30 s cadence sample.
        thermal = [thermalText]
        benchStart = Date()
    }

    func armClap() {
        guard !clapArmed else { return }
        clapArmed = true
        camera.onAudioSample = { [weak self] buffer in
            guard let self, let onset = self.clapDetector.process(sampleBuffer: buffer) else { return }
            self.session?.sendClapAnchor(localOnset: onset)
            DispatchQueue.main.async {
                self.camera.onAudioSample = nil
                self.clapArmed = false
            }
        }
        Task { try? await camera.configure(); camera.start() }
    }

    private func finishBench() {
        defer { benchStart = nil }
        guard let session, let startedAt = benchStart else { return }
        thermal.append(thermalText)
        let report = BenchReport.build(
            transport: transport?.name ?? "?", startedAt: startedAt, durationS: 60,
            rtts: rtts, sent: sent, received: received,
            estimate: session.clockSync.estimate,
            anchorDelta: nil,   // filled manually: compare pre/post-anchor offsets in the report pair
            thermal: thermal)
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("peer-bench-\(Int(startedAt.timeIntervalSince1970)).json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? (try? encoder.encode(report))?.write(to: url)
        reportURL = url
    }
}
#endif
