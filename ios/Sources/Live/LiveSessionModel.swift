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

    // MARK: - Rally lifecycle and paired upload (Task 7)

    enum RallyState: Equatable { case idle, recording, submitting, submitted, failed(String) }

    @Published private(set) var rally: RallyState = .idle
    /// Only shown while the link is degraded: without it a dropped link
    /// leaves the secondary recording 4K60 with no way to end the rally.
    @Published private(set) var showsLocalStop = false
    @Published private(set) var sessionID: String?

    private var peerVideoID: String?
    private var submission = RunSubmission()
    /// Chains `RecordModel.toggleRecording()` calls so a fire-and-forget
    /// start and an awaited stop can never land out of the order they were
    /// issued in. See `toggleRecordingChained()`.
    private var recordingTransition: Task<Void, Never>?

    /// The server's paired-run role. Fixed by the pairing role: initiator = a.
    var cameraRole: String? { role.map { $0 == .primary ? "a" : "b" } }

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
        // Task 7: rally start/stop and the paired-upload session identity.
        // Both fire on the transport delivery context (BLE/WiFi queues in
        // production), so both hop to main before touching any published
        // state.
        session.onRecord = { [weak self] action, _ in
            Task { @MainActor in self?.handleRemoteRecord(action) }
        }
        session.onSessionManifest = { [weak self] sessionID, videoID in
            Task { @MainActor in
                self?.sessionID = sessionID
                if !videoID.isEmpty { self?.peerVideoID = videoID }
            }
        }
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
        // And the rally identity: a later re-pairing must be free to mint a
        // fresh sessionID (republish()'s mint guard is `sessionID == nil`),
        // and a stale RallyState/local-stop flag from the session that just
        // ended must not bleed into the next one.
        rally = .idle
        sessionID = nil
        peerVideoID = nil
        showsLocalStop = false
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

    /// Primary-only, and only once the engine exists — both enforced by
    /// `computedPrimaryEnabled`. Recording is started locally first so a
    /// dropped message can never leave this phone not recording.
    func startRally() {
        guard rally != .recording else { return }
        rally = .recording
        toggleRecordingChained()
        session?.goLive()
        session?.sendRecord(action: "start", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        republish()
    }

    func stopRally() {
        guard rally == .recording else { return }
        session?.sendRecord(action: "stop", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        finishRecordingAndSubmit()
    }

    private func handleRemoteRecord(_ action: String) {
        switch action {
        case "start": if rally != .recording { startLocalRecording() }
        case "stop":  if rally == .recording { finishRecordingAndSubmit() }
        default:      break
        }
    }

    /// The secondary's half of a remote start — no goLive broadcast, no
    /// re-send, or the two phones would ping-pong record messages forever.
    private func startLocalRecording() {
        rally = .recording
        toggleRecordingChained()
        republish()
    }

    /// `RecordModel.toggleRecording()` is a toggle, not start/stop.
    /// `startRally()`'s kickoff is fire-and-forget while `stopRally()`'s path
    /// awaits, so a fast stop right behind a start could otherwise race two
    /// independent unstructured `Task`s against `toggleRecording()` with no
    /// guaranteed order — whichever lands second would flip the *other*
    /// direction (a "stop" landing first turns into a second start; the
    /// queued "start" landing after that then turns into a stop). Chaining
    /// through one stored `Task` — each call awaits whatever the previous
    /// call started before issuing its own `toggleRecording()` — makes the
    /// order exactly the call order, regardless of how the two Tasks happen
    /// to be scheduled.
    @discardableResult
    private func toggleRecordingChained() -> Task<Void, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            await self?.record?.toggleRecording()
        }
        recordingTransition = next
        return next
    }

    private func finishRecordingAndSubmit() {
        rally = .submitting
        republish()
        let toggle = toggleRecordingChained()
        Task { [weak self] in
            guard let self else { return }
            await toggle.value
            guard let record = self.record else { return }
            guard let clip = record.finishedClip else {
                self.rally = .failed("The rally produced no clip.")
                return self.republish()
            }
            // finishedClip is shared state with the single-camera flow:
            // RecordView presents ResultsView whenever it is non-nil. The
            // live path reports its own outcome through `rally`, so once
            // it's been read here, clear it — otherwise a later
            // single-camera visit to this same RecordModel would reopen a
            // results sheet for a rally ResultsView never showed.
            record.finishedClip = nil
            await self.submission.submit(videoURL: clip.url, duration: clip.duration,
                                         sessionID: self.sessionID,
                                         cameraRole: self.cameraRole,
                                         peerVideoID: self.peerVideoID,
                                         syncManifestJSON: self.syncManifestJSON())
            switch self.submission.phase {
            case .complete:
                self.rally = .submitted
                // Best-effort enrichment only: fusion pairs on session_id +
                // camera_role, so a lost manifest never blocks it.
                if let sessionID = self.sessionID, let videoID = self.submission.completedRunID {
                    self.session?.sendSessionManifest(sessionID: sessionID, videoID: videoID)
                }
            case .failed(let message): self.rally = .failed(message)
            default:                   self.rally = .failed("Upload did not finish.")
            }
            self.republish()
        }
    }

    /// Seeds the server's offset refinement with what the phones measured.
    /// Primary-only — `job_runner` reads the manifest from whichever run
    /// carries it, so sending it twice would be redundant.
    ///
    /// Always `offset_series`, never `clap_anchor_s`: `ClockSync.anchor` is
    /// private, so the client cannot tell an anchored estimate from a network
    /// one. It costs nothing — when the anchor is applied, `estimate.offset`
    /// *is* the anchor value, and the server takes the median of a one-element
    /// series, which returns it exactly. Only the report's `seed.source` label
    /// differs.
    private func syncManifestJSON() -> String? {
        guard role == .primary, let estimate = session?.clockSync.estimate else { return nil }
        let payload: [String: Any] = ["offset_series": [estimate.offset]]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    /// Single rule for the §16 "still recording, no way out" safety valve —
    /// shown only while the rally is genuinely recording *and* the link is
    /// down. `republish()` supplies `degraded` from the real `pairing.step`
    /// at 20 Hz: the production path, automatic, wired to no button anywhere.
    /// `handleDegraded(_:)` is a test seam that supplies it directly, so a
    /// test can assert the safety valve without starving a real link for
    /// `heartbeatTimeout` seconds. Both funnel through here so the rule
    /// itself — not just the inputs to it — is written exactly once; keeping
    /// two independent assignments to `showsLocalStop` would leave whichever
    /// ran most recently as the value, an easy way to reintroduce a second
    /// source of truth by accident.
    private func updateShowsLocalStop(degraded: Bool) {
        showsLocalStop = degraded && rally == .recording
    }

    /// Test seam for the degraded-link path, which needs no radio to assert.
    func handleDegraded(_ degraded: Bool) {
        updateShowsLocalStop(degraded: degraded)
    }

    // MARK: - §16's p-pair table, as flat state

    private func republish() {
        let status = computedLinkStatus
        let title = computedPrimaryTitle
        let enabled = computedPrimaryEnabled
        if status != linkStatus { linkStatus = status }
        if title != primaryTitle { primaryTitle = title }
        if enabled != primaryEnabled { primaryEnabled = enabled }

        // Mint exactly once: the guard is `sessionID == nil`, and the
        // assignment right below happens synchronously with no `await` in
        // between — so the very next 20 Hz republish() (this same private
        // method, called from the pump timer) already sees a non-nil
        // sessionID and takes neither branch. One mint, one broadcast, per
        // paired session; `endSession()` is what clears it for the next one.
        if role == .primary, case .ready = pairing.step, sessionID == nil {
            let minted = UUID().uuidString
            sessionID = minted
            session?.sendSessionManifest(sessionID: minted, videoID: "")
        }
        updateShowsLocalStop(degraded: isPairingDegraded)
    }

    /// The production input to `updateShowsLocalStop` — derived from the
    /// real session's phase, unlike `handleDegraded(_:)`'s test-supplied one.
    private var isPairingDegraded: Bool {
        if case .degraded = pairing.step { return true }
        return false
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
