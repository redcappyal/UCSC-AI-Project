// ios/Sources/Live/LiveSessionModel.swift
import Foundation

/// Owns the live two-camera lifecycle: the calibration gate, the role choice,
/// the PeerSession, and (Task 7) the rally and its paired upload.
///
/// Deliberately NOT part of `RecordModel`. DESIGN.md §16 makes "pairing adds
/// capability, never gates it" a hard requirement, so the camera model is owned
/// above this one and merely lent here — a defect in this file cannot reach
/// plain single-camera recording.
///
/// Publishes §16's p-pair table as flat view state so the whole table is
/// assertable without standing up a view, the same reason `PairingModel` exists.
@MainActor
final class LiveSessionModel: ObservableObject {
    enum Calibration: Equatable { case loading, ready, failed(String) }

    @Published private(set) var calibration: Calibration = .loading
    @Published private(set) var linkStatus = "Checking this phone's court calibration…"
    @Published private(set) var primaryTitle = "PAIR"
    @Published private(set) var primaryEnabled = false
    @Published private(set) var pairing = PairingModel(session: nil)
    /// Set once the primary's engine exists. The gate that stops START RALLY
    /// from recording a rally nothing can ever call.
    @Published private(set) var engineReady = false

    /// No default: two phones both defaulting to primary would both open a
    /// CBCentralManager and hang with nothing honest to show for it.
    ///
    /// DESIGN.md scopes the role segment to the idle state only: once
    /// `beginPairing()` has built a session, `PeerSession.isInitiator` is
    /// already fixed from whatever `role` was at that moment, so a later
    /// change here would let this published value disagree with the
    /// session's actual wiring. Revert instead of accepting it — settable
    /// freely before a session exists, a no-op after.
    @Published var role: PeerRole? {
        didSet {
            guard session == nil else {
                if !isRevertingRole && role != oldValue {
                    isRevertingRole = true
                    role = oldValue
                    isRevertingRole = false
                }
                return
            }
            republish()
        }
    }
    private var isRevertingRole = false

    #if DEBUG
    @Published var transportName = "ble"
    #endif

    private let api: APIClientProtocol
    private let makeTransport: (String) -> PeerTransport
    private weak var record: RecordModel?
    private var session: PeerSession?
    private var localModelJSON: String?
    private var pumpTimer: Timer?
    private var isPreparing = false

    init(api: APIClientProtocol = APIClient(),
         makeTransport: @escaping (String) -> PeerTransport = LiveSessionModel.defaultTransport) {
        self.api = api
        self.makeTransport = makeTransport
    }

    /// `nonisolated`: this is the default value for a non-isolated
    /// `(String) -> PeerTransport` parameter. Left `@MainActor`-isolated (the
    /// class default), converting it to that non-isolated function type loses
    /// the actor annotation — diagnosed as a warning in Swift 5.9 language
    /// mode, and free to silence since the body touches no instance state.
    nonisolated static func defaultTransport(_ name: String) -> PeerTransport {
        name == "wifi-p2p" ? WiFiP2PTransport() : BLETransport()
    }

    /// Idempotent. The camera model is owned by `PlayRootView`, not by this
    /// object — see the type doc.
    func bind(record: RecordModel) {
        guard self.record == nil else { return }
        self.record = record
        record.onStereoReady = { [weak self] in
            Task { @MainActor in
                self?.engineReady = true
                self?.republish()
            }
        }
    }

    // MARK: - Calibration gate

    /// Fetch and validate this phone's solved camera model. Validating
    /// adoption here rather than inside `attachStereo` is the point: an
    /// unusable calibration must fail before two people walk to opposite
    /// corners of the court.
    ///
    /// A view can plausibly call this from both `.task` and `.onAppear`. The
    /// in-flight guard makes the second concurrent call a no-op instead of
    /// running two overlapping fetches — without it, a late completion of
    /// the first could reset an already-`.failed` result from the second
    /// back to `.loading`.
    func prepare() async {
        guard calibration != .ready else { return }
        guard !isPreparing else { return }
        isPreparing = true
        defer { isPreparing = false }
        calibration = .loading
        republish()
        do {
            let latest = try await api.latestCalibration()
            let json = try await api.fetchSolvedCameraModel(calibrationJSON: latest.calibrationJSON)
            guard let data = json.data(using: .utf8) else { throw APIError.badResponse }
            _ = try CameraModel.fromJSON(data).adoptedForCapture()
            localModelJSON = json
            calibration = .ready
        } catch {
            // §16 shows a failure reason verbatim, so prefer the server's own
            // words. `APIError.errorDescription` already returns
            // `message ?? "Server error (code)."` for `.http` — the same rule
            // this used to duplicate — so `localizedDescription` is enough.
            calibration = .failed(error.localizedDescription)
        }
        republish()
    }

    // MARK: - The one primary (§7)

    func primaryTapped() {
        guard session != nil else { return beginPairing() }
        switch pairing.step {
        case .idle, .failed: beginPairing()
        case .confirm:       pairing.confirm()
        case .ready:         startRally()          // Task 7
        default:             break
        }
    }

    private func beginPairing() {
        guard case .ready = calibration, let role, let localModelJSON,
              let record else { return }
        #if DEBUG
        let transport = makeTransport(transportName)
        #else
        let transport = makeTransport("ble")
        #endif
        let session = PeerSession(transport: transport,
                                  isInitiator: role == .primary,
                                  orientation: record.camera.orientation)
        self.session = session
        // Order is load-bearing: attachStereo installs peer.onCalibration, so
        // attachPeer's `self.peer = peer` has to happen first, and both must
        // precede start() or a fast radio can deliver before we are listening.
        record.attachPeer(session)
        record.attachStereo(localModelJSON: localModelJSON)
        let pairing = PairingModel(session: session)
        pairing.onSessionEnded = { [weak self] in self?.endSession() }
        self.pairing = pairing
        pairing.start()
        startPump()
        republish()
    }

    func endSession() {
        pumpTimer?.invalidate(); pumpTimer = nil
        session = nil
        engineReady = false
        pairing = PairingModel(session: nil)
        // Tear down the camera side too: without this, RecordModel's own
        // 20 Hz pump keeps ticking a session nobody owns anymore, and a
        // stale call can keep showing for a rally that was just cancelled.
        record?.detachPeer()
        republish()
    }

    /// `RecordModel.attachPeer` already ticks the peer at 20 Hz, so this pump
    /// only republishes — ticking here too would double-drive the timers.
    private func startPump() {
        pumpTimer?.invalidate()
        pumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pairing.refresh()
                self?.republish()
            }
        }
    }

    /// Task 7 wires the actual rally start (recording + `PeerSession.sendRecord`).
    /// Stubbed here so the calibration/role/pairing gates in this file compile
    /// and are testable standalone.
    private func startRally() {}

    // MARK: - §16's p-pair table, as flat state

    private func republish() {
        let status = computedLinkStatus
        let title = computedPrimaryTitle
        let enabled = computedPrimaryEnabled
        if status != linkStatus { linkStatus = status }
        if title != primaryTitle { primaryTitle = title }
        if enabled != primaryEnabled { primaryEnabled = enabled }
    }

    private var computedLinkStatus: String {
        switch calibration {
        case .loading:              return "Checking this phone's court calibration…"
        case .failed(let reason):   return reason
        case .ready:                break
        }
        if session == nil && role == nil { return "Pick this phone's job to start." }
        if case .ready = pairing.step {
            if role == .primary {
                return engineReady ? pairing.statusLine
                                   : "Paired · waiting for the other phone's calibration"
            }
            return "Paired · the other phone starts the rally"
        }
        return pairing.statusLine
    }

    private var computedPrimaryTitle: String {
        guard session != nil else { return "PAIR" }
        switch pairing.step {
        case .idle, .searching, .failed: return "PAIR"
        case .confirm, .syncing:         return "CONFIRM"
        case .ready, .degraded:          return "START RALLY"
        case .live:                      return "RALLY LIVE"
        }
    }

    private var computedPrimaryEnabled: Bool {
        guard case .ready = calibration, role != nil else { return false }
        guard session != nil else { return true }
        switch pairing.step {
        case .searching, .syncing, .live, .degraded: return false
        case .idle:                                  return pairing.canPair
        // `.failed` differs from `.idle` on purpose: `pairing.canPair` answers
        // "can THIS spent session pair again", which is always false once a
        // session has failed (`PeerSession` never resets). But PAIR here
        // means "build a fresh session", and `beginPairing()` is the owner
        // that constructs one — DESIGN.md §16's `Failed → PAIR (retry)` needs
        // this to read `true`, not the spent session's own answer.
        case .failed:                                return true
        case .confirm:                               return pairing.canConfirm
        // Only the primary can honestly know a rally is callable, and only
        // once its engine exists.
        case .ready:                                 return role == .primary && engineReady
        }
    }

    deinit { pumpTimer?.invalidate() }
}
