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
    /// issued in, and reports whether its own call to `toggleRecording()`
    /// actually ran. See `toggleRecordingChained(ifRecordingIs:)` —
    /// `endSession()` reads that to know whether it produced a clip worth
    /// discarding.
    private var recordingTransition: Task<Bool, Never>?

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
        // attachPeer's `self.peer = peer` has to happen first, and
        // attachPeer/attachStereo/onRecord/onSessionManifest must *all*
        // precede start() or a fast radio can deliver before we are
        // listening for it.
        record.attachPeer(session)
        record.attachStereo(localModelJSON: localModelJSON)
        // Task 7: rally start/stop and the paired-upload session identity.
        // Both fire on the transport delivery context (BLE/WiFi queues in
        // production), so both hop to main before touching any published
        // state. Wired here — before pairing.start() — for the same reason
        // attachPeer/attachStereo are: a fast radio must never be able to
        // deliver a "record" or session-manifest control frame before these
        // closures are installed to receive it.
        session.onRecord = { [weak self] action, _ in
            Task { @MainActor in self?.handleRemoteRecord(action) }
        }
        session.onSessionManifest = { [weak self] sessionID, videoID in
            Task { @MainActor in
                self?.sessionID = sessionID
                if !videoID.isEmpty { self?.peerVideoID = videoID }
            }
        }
        let pairing = PairingModel(session: session)
        pairing.onSessionEnded = { [weak self] in self?.endSession() }
        self.pairing = pairing
        pairing.start()
        startPump()
        republish()
    }

    func endSession() {
        // A running recording — or one that is about to start via a
        // chained toggle still queued behind an earlier one — must never
        // be abandoned: left alone, the camera keeps rolling (or starts
        // rolling) with no path back to it once `session`/`pairing` are
        // torn down below, and the *next* rally start would toggle the
        // abandoned recording *off* and publish its clip into the shared
        // `finishedClip` — the exact inversion this fix exists to prevent.
        //
        // Unconditional — deliberately NOT gated on a preceding
        // `record?.isRecording == true` (an earlier version's mistake):
        // that read is taken *now*, at call time, but what must be
        // cancelled is a *chained* toggle that can still run *later*.
        // Failing sequence that gate missed: startRally() enqueues a start
        // toggle and returns (fire-and-forget) → endSession() runs while
        // the camera hasn't actually started yet, so an `isRecording`-gated
        // `if` here reads false, skips scheduling a stop, and lets teardown
        // proceed → the queued start toggle then executes after teardown,
        // into a session nobody owns anymore. `toggleRecordingChained
        // (ifRecordingIs:)` re-reads `record.isRecording` itself, at the
        // moment it actually runs (after awaiting whatever was already
        // chained ahead of it) — so calling it unconditionally here is a
        // correctly-timed no-op when nothing ends up recording, and a
        // correctly-timed stop otherwise.
        let toggle = toggleRecordingChained(ifRecordingIs: true)
        // Captures `record` itself (not `self`/`self.record`) so the clip
        // this stop produces is discarded even if this `LiveSessionModel`
        // — or its binding to this `RecordModel` — is gone by the time the
        // toggle finishes. Clears `finishedClip` only when the toggle
        // actually fired (per its own reported return value): `finishedClip`
        // is shared with the plain single-camera flow (see
        // `finishRecordingAndSubmit()`'s own comment on the same field), so
        // when this call was skipped — nothing here was ever recording —
        // any clip already sitting in `finishedClip` belongs to that other
        // flow and must not be nilled out from under it.
        Task { [record] in
            let toggled = await toggle.value
            if toggled {
                // Ending a session is a cancellation, not a completed
                // rally: never submit this clip. Clearing it also keeps it
                // from leaking into a later single-camera visit to this
                // same RecordModel, the same reason
                // `finishRecordingAndSubmit()` clears it after reading it.
                record?.finishedClip = nil
            }
        }
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
        // and a stale RallyState from the session that just ended must not
        // bleed into the next one.
        rally = .idle
        sessionID = nil
        peerVideoID = nil
        // No direct `showsLocalStop = false` here: republish() below
        // recomputes it from `rally` (now `.idle`) on the very next line, so
        // a second writer here would just be the "last one wins" pattern
        // `updateShowsLocalStop`'s own doc argues against.
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
        let toggle = toggleRecordingChained(ifRecordingIs: false)
        session?.goLive()
        session?.sendRecord(action: "start", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        republish()
        // The toggle above is fire-and-forget; reconcile once it actually
        // finishes rather than assuming it did what was asked — see
        // `reconcileRallyAfterStartAttempt()`.
        Task { [weak self] in
            await toggle.value
            self?.reconcileRallyAfterStartAttempt()
        }
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
        let toggle = toggleRecordingChained(ifRecordingIs: false)
        republish()
        Task { [weak self] in
            await toggle.value
            self?.reconcileRallyAfterStartAttempt()
        }
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
    ///
    /// `expected` guards the toggle itself, not just its ordering:
    /// `record.isRecording` must read exactly that, right before the call,
    /// for `toggleRecording()` to do what this call intends (`false` for a
    /// start, `true` for a stop). `toggleRecording()` is a raw flip with no
    /// notion of "start" or "stop" of its own, so calling it against a
    /// camera state that already contradicts the intent flips it the wrong
    /// way — most importantly, a stop issued while the camera never
    /// actually confirmed it had started would *start* it instead. When
    /// the guard fails, the call is skipped rather than forced;
    /// `reconcileRallyAfterStartAttempt()` is what turns a skipped or
    /// failed start into a visible `.failed` rally instead of a silent one.
    ///
    /// Returns whether this call's own `toggleRecording()` actually ran —
    /// `false` when the guard above skipped it. No longer
    /// `@discardableResult`: every call site now reads or awaits this to
    /// decide something (`endSession()` needs it to know whether it
    /// produced a clip worth discarding; the others await it purely for
    /// sequencing, which reading the property still allows).
    private func toggleRecordingChained(ifRecordingIs expected: Bool) -> Task<Bool, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            guard let self, let record = self.record, record.isRecording == expected else { return false }
            await record.toggleRecording()
            return true
        }
        recordingTransition = next
        return next
    }

    /// Reconciles `rally` against the camera's real state after a start
    /// attempt's chained toggle finishes, rather than assuming the toggle
    /// did what it was asked. `RecordModel.toggleRecording()`'s start
    /// branch can throw and leave `isRecording == false` with the reason
    /// swallowed into `errorText` — nobody else reads that back, so without
    /// this a failed start would silently proceed as though the rally were
    /// live: the camera never started, `rally == .recording` says
    /// otherwise, and the next stop would then toggle the camera the wrong
    /// way (see `toggleRecordingChained(ifRecordingIs:)`'s doc for why).
    ///
    /// Internal rather than private: this is exactly the reconciliation
    /// `startRally()`/`startLocalRecording()` schedule for themselves once
    /// their own toggle completes, exposed only so a test can drive it
    /// synchronously. There is no seam to make the real
    /// `CameraController.startRecording()` throw in a test host — its
    /// `AVAssetWriter` setup does not depend on a running capture session,
    /// so it does not reliably fail outside a device — so
    /// `LiveSessionModelTests` simulates the state `toggleRecording()`'s
    /// catch branch leaves behind and calls this directly instead of
    /// exercising the real throw.
    func reconcileRallyAfterStartAttempt() {
        guard rally == .recording else { return }
        guard record?.isRecording != true else { return }
        // The peer may already be acting as though the rally is live —
        // startRally() sends "start" before this phone's own camera is
        // confirmed, and handleRemoteRecord's "start" case (reached only on
        // the far side of that same send) starts the peer's own local
        // recording independently, before either phone knows whether this
        // one's camera actually started. If it didn't, the rally cannot
        // proceed, and the peer must be told to retract *before* this phone
        // marks itself `.failed` below: once `.failed`, `stopRally()`'s own
        // `guard rally == .recording` blocks this phone from ever sending
        // "stop" again, so this is the last chance to do it. The send has
        // to live here rather than in `stopRally()` because `stopRally()`
        // is never reached on this path — `rally` never actually made it to
        // a stoppable state, only the optimistic assumption (set at the top
        // of `startRally()`/`startLocalRecording()`) that it had.
        session?.sendRecord(action: "stop", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        rally = .failed(record?.errorText ?? "Recording did not start.")
        republish()
    }

    private func finishRecordingAndSubmit() {
        rally = .submitting
        republish()
        // `ifRecordingIs: true`: a stop must depend on the camera's real
        // state, never just on `rally` saying `.recording` — if the camera
        // never actually confirmed it started, this call is skipped rather
        // than issuing a toggle that would START it instead of stopping it.
        let toggle = toggleRecordingChained(ifRecordingIs: true)
        Task { [weak self] in
            guard let self else { return }
            await toggle.value
            guard let record = self.record else {
                self.rally = .failed("The camera model was no longer available.")
                return self.republish()
            }
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

    #if DEBUG
    /// Test seam for the degraded-link path, which needs no radio to
    /// assert. `#if DEBUG`-gated so production code can never call a
    /// method whose write is guaranteed to be reverted: `republish()`'s
    /// 20 Hz pump always calls `updateShowsLocalStop(degraded:
    /// isPairingDegraded)` right behind it (within 50 ms), deriving the
    /// same field from the real `pairing.step` — so outside a synchronous
    /// test, whatever this writes doesn't stick. It exists purely to let a
    /// test assert the safety valve without starving a real link for
    /// `heartbeatTimeout` seconds; it is not a state the pump can't
    /// otherwise derive on its own.
    func handleDegraded(_ degraded: Bool) {
        updateShowsLocalStop(degraded: degraded)
    }
    #endif

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
