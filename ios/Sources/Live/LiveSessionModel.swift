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
    /// §16's `p-live` STOP — the whole rule for who gets one, published flat
    /// (same reason as `primaryTitle`/`primaryEnabled`) so `LiveStageView`
    /// has no state logic of its own: the primary can always end a rally it
    /// started; the secondary only gains a local STOP once `showsLocalStop`
    /// already says the link is degraded. See `updateShowsStop()`.
    @Published private(set) var showsStop = false

    /// The `session_id` **this rally's** clip uploads under — per rally, never
    /// per pairing.
    ///
    /// That distinction is the whole of Task 10e's first fix. `session_id` +
    /// `camera_role` are the *only* keys the server pairs runs on
    /// (`peer_video_id` is stored by `/api/track` and never read by fusion),
    /// and both of the server's consumers assume one rally per id:
    ///
    /// * `job_runner.maybe_start_stereo_fuse` claims its work by
    ///   `mkdir(RUNS_DIR / "stereo-<session_id>", exist_ok=False)` — one fuse
    ///   run per id, *ever*. With one id per pairing, rally 1 created it and
    ///   rallies 2..N hit `FileExistsError`, which is swallowed: their clips
    ///   uploaded, tracked, and were silently never fused.
    /// * `job_runner.session_runs` picks the newest complete run **per role
    ///   independently**. So with one id per pairing, a secondary still queued
    ///   on rally 1 while the primary's rally 2 completes yields the claim
    ///   `{a: rally2, b: rally1}` — two different rallies triangulated as one
    ///   stereo pair. A wrong answer, not just a missing one.
    ///
    /// Minted and broadcast by the primary at `startRally()` (see
    /// `armRallyIdentity()`), received by the secondary (see
    /// `receiveSessionManifest`), and **consumed** — read and nil'd — when the
    /// rally ends. Nil therefore means "this rally has no agreed identity", and
    /// the upload goes out unpaired (both `session_id` and `camera_role`
    /// omitted, which `/api/track` explicitly allows) rather than corrupting a
    /// previous rally's fuse with a stale one.
    @Published private(set) var rallySessionID: String?

    /// The pairing's own identity, minted locally by whichever phone built the
    /// session and **never broadcast**. Pure bookkeeping: it prefixes every
    /// rally id this pairing mints, so a run's `session_id` on the server is
    /// traceable back to the pairing that produced it, and so two pairings can
    /// never collide on a rally id even if their generation counters agree.
    ///
    /// Not sent, and not derivable by the secondary — which is the point. The
    /// secondary learns each rally's id only by receiving it, because a
    /// derivation that drifts by one on one phone is exactly the mismatched
    /// `{a: rallyN, b: rallyN-1}` pair described above.
    private var pairingID: String?

    private var peerVideoID: String?

    /// Which rally on this pairing the model's state currently belongs to.
    /// Bumped by `beginRally()`, and by `teardownPairing()` so a pairing torn
    /// down mid-rally — by `endSession()` *or* by a `beginPairing()` retry —
    /// cannot be written to afterwards either.
    ///
    /// A rally's *finishing* work — the upload plus the server-side tracking
    /// poll — can outlive its recording by minutes, and (see `finishRally`) it
    /// is genuinely still running on the secondary when the next rally starts.
    /// Everything scheduled by one rally therefore captures this value and
    /// refuses to write anything once it no longer matches.
    private var rallyGeneration = 0

    /// The server's paired-run role. Fixed by the pairing role: initiator = a.
    var cameraRole: String? { role.map { $0 == .primary ? "a" : "b" } }

    private let api: APIClientProtocol
    private let makeTransport: (String) -> PeerTransport
    private weak var record: RecordModel?
    private var session: PeerSession?
    private var localModelJSON: String?
    /// Names *which* calibration `localModelJSON` was solved from: the server
    /// run ID `prepare()` already has in hand, prefixed so the value is
    /// self-describing on the wire and in a log.
    ///
    /// The receiving side ignores it today (`RecordModel.attachStereo`'s
    /// `onCalibration` handler binds it to `_`), so this is chosen to be
    /// *honest* rather than load-bearing. The spec's eventual identity for
    /// this field is a per-fin `camera_id` like `"<court>-left-fin"` — an
    /// optional calibration-v2 field that nothing in this client stores yet,
    /// so inventing one here would be a label with no provenance behind it.
    /// The calibration run ID is provenance this phone genuinely has.
    private var localProfileID: String?
    /// Whether this pairing's one `.calibration` exchange has already gone out
    /// — set from the send's own outcome, never optimistically. Cleared in
    /// `teardownPairing()`, i.e. wherever `session` is replaced. See
    /// `republish()`.
    private var didSendCalibration = false
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
            // `json`, not the adopted model: this is the string both
            // `record.attachStereo(localModelJSON:)` and (on the secondary)
            // the `.calibration` message carry, and the receiving side runs
            // its own `adoptedForCapture()` against its own capture space.
            // Sending an already-adopted model would be adopting it twice.
            localModelJSON = json
            localProfileID = "calibration-run-\(latest.runID)"
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
        case .confirm:       pairing.confirm(); republish()
        case .ready:         startRally()          // Task 7
        default:             break
        }
    }

    /// Builds a fresh pairing. Reached both from a cold PAIR tap and — via
    /// `primaryTapped()`'s `.failed` branch — from DESIGN.md §16's
    /// `Failed → PAIR (retry)` row, which does *not* run `endSession()` first.
    ///
    /// That retry path is why the whole of the previous pairing's state is torn
    /// down here through the same `teardownPairing()` `endSession()` uses,
    /// rather than through a hand-picked subset. The subset was the bug: it
    /// cleared `sessionID`/`peerVideoID`/`didSendCalibration` but left
    /// `engineReady`, `rally`, `rallyGeneration` and the camera alone, so a
    /// retry after a *previously successful* pairing reached `.ready` with
    /// `engineReady` still true from the pairing that just died. START RALLY
    /// then enabled before the new secondary's calibration had arrived, the
    /// "waiting for the other phone's calibration" status stopped showing, and
    /// a rally started in that window ran against the **previous** pairing's
    /// `StereoEngine` — whose `remoteToLocal` closure holds a `weak` reference
    /// to the deallocated old `PeerSession`, so remote samples could not be
    /// time-mapped and the rally produced no calls at all. A retry taken
    /// mid-rally was worse still: `rally` stayed `.recording` from the dead
    /// session, permanently blocking `startRally()` on the new one while
    /// `primaryEnabled` read true — an enabled primary whose tap cannot fire,
    /// which DESIGN.md §7 forbids outright.
    ///
    /// The guard runs *before* the teardown on purpose: a tap that cannot build
    /// a session (no calibration, no role, no camera) must change nothing at
    /// all, least of all stop a recording.
    private func beginPairing() {
        guard case .ready = calibration, let role, let localModelJSON,
              let record else { return }
        teardownPairing()
        #if DEBUG
        let transport = makeTransport(transportName)
        #else
        let transport = makeTransport("ble")
        #endif
        let session = PeerSession(transport: transport,
                                  isInitiator: role == .primary,
                                  orientation: record.camera.orientation)
        self.session = session
        // Fresh per pairing, and never reused: `armRallyIdentity()` prefixes
        // every rally id with it, so ids minted by two different pairings
        // cannot collide even though `rallyGeneration` restarts low.
        pairingID = UUID().uuidString
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
            Task { @MainActor in self?.receiveSessionManifest(sessionID: sessionID, videoID: videoID) }
        }
        let pairing = PairingModel(session: session)
        pairing.onSessionEnded = { [weak self] in self?.endSession() }
        self.pairing = pairing
        pairing.start()
        startPump()
        republish()
    }

    func endSession() {
        teardownPairing()
        republish()
    }

    /// Everything one pairing owns, released in one place.
    ///
    /// Two callers, deliberately identical: `endSession()` (the user leaving,
    /// or `PairingModel.onSessionEnded`) and `beginPairing()` (§16's
    /// `Failed → PAIR (retry)`, which never routes through `endSession()`).
    /// Splitting them — which is what the code did before — is how
    /// `engineReady`, `rally` and `rallyGeneration` came to survive a retry;
    /// see `beginPairing()`'s doc for what that cost. Anything per-pairing
    /// belongs here and nowhere else, so the two paths cannot drift again.
    ///
    /// Deliberately *not* reset here, each for a reason:
    /// * `calibration` / `localModelJSON` / `localProfileID` — this phone's own
    ///   solved camera model. Per phone, not per pairing; `beginPairing()`
    ///   requires it to already be `.ready` and would refuse to rebuild.
    /// * `role` — the user's choice, and `PeerSession.isInitiator` is fixed
    ///   from it. Clearing it would silently drop the tap that got here.
    /// * `record` — the camera model is owned above this object and merely
    ///   lent; `bind(record:)` is idempotent and outlives any one pairing.
    /// * `linkStatus` / `primaryTitle` / `primaryEnabled` / `showsStop` /
    ///   `showsLocalStop` — derived, and recomputed by the `republish()` both
    ///   callers run immediately afterwards. A direct write here would be a
    ///   second source of truth for a value `republish()` owns.
    /// * `transportName` (DEBUG) — a bench setting, not pairing state.
    /// * `api` / `makeTransport` — injected `let`s, not state.
    /// * `isPreparing` — scoped to one `prepare()` call by its own `defer`, and
    ///   never true while a pairing exists (`beginPairing()` requires
    ///   `calibration == .ready`, which only a finished `prepare()` produces).
    /// * `isRevertingRole` — a one-statement re-entrancy latch inside `role`'s
    ///   `didSet`, always false outside it.
    /// * `cameraRole` — computed from `role`, so it follows it.
    private func teardownPairing() {
        // A running recording — or one that is about to start via a
        // transition still queued behind an earlier one — must never be
        // abandoned: left alone, the camera keeps rolling (or starts
        // rolling) with no path back to it once `session`/`pairing` are
        // torn down below, and it would go on owning the camera with
        // nothing left that is allowed to stop it, which also locks the
        // record stage out of the camera it shares.
        //
        // Unconditional — deliberately NOT gated on a preceding
        // `record?.isRecording == true` (an earlier version's mistake):
        // that read is taken *now*, at call time, but what must be
        // cancelled is a *chained* transition that can still run *later*.
        // Failing sequence that gate missed: startRally() enqueues a start
        // and returns (fire-and-forget) → endSession() runs while the
        // camera hasn't actually started yet, so an `isRecording`-gated
        // `if` here reads false, skips scheduling a stop, and lets teardown
        // proceed → the queued start then executes after teardown, into a
        // session nobody owns anymore. `RecordModel`'s funnel re-reads the
        // camera's real state itself, at the moment it actually runs (after
        // awaiting whatever was already chained ahead of it) — so calling
        // it unconditionally here is a correctly-timed no-op when nothing
        // ends up recording, and a correctly-timed stop otherwise.
        let stop = setRecordingChained(false)
        // Captures `record` itself (not `self`/`self.record`) so the clip
        // this stop produces is discarded even if this `LiveSessionModel`
        // — or its binding to this `RecordModel` — is gone by the time the
        // stop finishes. `liveClip`, not the raw clip: a clip the record
        // stage owns is invisible through this accessor and cannot be
        // nilled out from under that flow. The `stopped` check is the same
        // rule stated a second time — kept because it also says, plainly,
        // that a skipped stop produced nothing to discard.
        Task { [record] in
            let stopped = await stop.value
            if stopped {
                // Ending a session is a cancellation, not a completed
                // rally: never submit this clip. Clearing it also keeps it
                // from leaking into a later live visit to this same
                // RecordModel, the same reason
                // `finishRecordingAndSubmit()` clears it after reading it.
                record?.liveClip = nil
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
        // And the rally state. A stale `RallyState` from the pairing that just
        // ended must not bleed into the next one: on the retry path it would
        // arrive as `.recording` on a session with no camera rolling, which
        // `startRally()`'s own `guard rally != .recording` then refuses
        // forever — while `computedPrimaryEnabled` happily reads `.ready` and
        // enables the tap. §7's one primary must never advertise a tap that
        // cannot fire.
        //
        // The generation bump is what makes that `.idle` stick: a rally whose
        // upload/tracking was still in flight when the pairing ended would
        // otherwise land minutes later and overwrite it with `.submitted` /
        // `.failed` for a pairing that no longer exists. Same mechanism the
        // next rally uses to disown the previous one's finishing work — see
        // `finishRally`. It is also what makes the ids `armRallyIdentity()`
        // mints monotonic across a pairing.
        rallyGeneration += 1
        rally = .idle
        // The identities, both halves. `pairingID` is this pairing's; a new
        // one is minted in `beginPairing()`. `rallySessionID`/`peerVideoID`
        // belonged to whatever rally was in flight — carrying either into the
        // next pairing would upload a rally under a dead pairing's identity,
        // which on the server means it either never fuses or fuses against
        // some *other* rally's run. Unpaired is the correct fallback; stale is
        // never one.
        pairingID = nil
        rallySessionID = nil
        peerVideoID = nil
        // And the calibration exchange, for the same reason and by the same
        // rule as the identities above: it belonged to the pairing that just
        // ended. `record?.detachPeer()` a few lines up threw away the
        // primary's `StereoEngine` and its `localModel`, so a re-pairing that
        // did not send again would leave `engineReady` false forever and
        // START RALLY permanently disabled — the exact dead end this send
        // exists to close, merely deferred by one pairing.
        didSendCalibration = false
        // No direct `showsLocalStop = false` here: both callers `republish()`
        // immediately, which recomputes it from `rally` (now `.idle`), so a
        // second writer here would just be the "last one wins" pattern
        // `updateShowsLocalStop`'s own doc argues against.
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

    /// Everything a new rally needs to be true before it begins, in one place
    /// so the primary's own start and the secondary's remote-driven one cannot
    /// drift apart.
    ///
    /// The `clearLiveCall()` is DESIGN.md §16's `p-live` promise that the
    /// mini-court and call banner "clear again when `START RALLY` begins the
    /// next one" — unreachable while a session could only ever run one rally,
    /// and now the thing that keeps rally 2 from opening on rally 1's verdict.
    private func beginRally() {
        rallyGeneration += 1
        rally = .recording
        record?.clearLiveCall()
    }

    // MARK: - The per-rally upload identity
    //
    // Ordering, stated once here because the whole scheme rests on it:
    //
    // 1. `startRally()` runs synchronously on main: `beginRally()` (bumps the
    //    generation this id is built from) → enqueue the camera start →
    //    `goLive()` → `armRallyIdentity()` → `sendRecord("start")`. So the
    //    manifest frame is handed to the transport strictly before the record
    //    frame.
    // 2. Transports guarantee serial, in-order delivery and `PeerSession`
    //    processes control frames inline on that context, so the secondary's
    //    `onSessionManifest` is *invoked* before its `onRecord`.
    // 3. Both handlers then hop to main with their own
    //    `Task { @MainActor in ... }`, and cross-`Task` ordering is NOT relied
    //    on: `startLocalRecording()` never reads or writes `rallySessionID`, so
    //    the two hops landing in either order changes nothing.
    // 4. What must be true is only that the manifest lands before the
    //    secondary *uploads*, and the upload is in `finishRecordingAndSubmit()`
    //    — reached from the primary's later "stop" frame, i.e. one whole rally
    //    plus another control round trip afterwards. The margin is the rally,
    //    not a scheduling coincidence.
    // 5. The identity is *consumed* — read and nil'd — synchronously at the top
    //    of `finishRecordingAndSubmit()`, before any `await`. So rally N+1's
    //    manifest cannot retarget rally N's in-flight upload, and rally N+1
    //    cannot inherit rally N's id: if N+1's manifest never arrives, the
    //    field is already nil and the clip uploads unpaired.

    /// Mints this rally's uploaded identity and tells the secondary. Primary
    /// only — §10e's rule is that the agreement is *driven* by the primary and
    /// the secondary never derives one of its own, because a derivation that
    /// drifts by a single rally is precisely the `{a: rallyN, b: rallyN-1}`
    /// mispairing `rallySessionID`'s doc describes.
    ///
    /// The id is set from the send's *outcome*, not optimistically. If the
    /// frame was dropped (`PeerSession`'s senders gate on the authoritative
    /// `internalPhase`, which the `pairing.step` mirror can lag), the secondary
    /// never learns this rally's identity — and the very same gate has already
    /// dropped, or is about to drop, the `sendRecord("start")` that follows, so
    /// the secondary is not recording this rally at all. Uploading paired
    /// anyway would leave a role-`a` run on the server waiting forever for a
    /// role `b` that is never coming. Unpaired is the honest outcome.
    private func armRallyIdentity() {
        // A complete no-op off the primary — including the clears below. The
        // secondary reaches its own rallies through `startLocalRecording()`,
        // which never comes here, but `startRally()` itself carries no role
        // guard of its own; clearing unconditionally would let a stray call on
        // a secondary wipe the identity its arming manifest had just delivered.
        guard role == .primary else { return }
        // Explicit rather than relying on the consume at the previous rally's
        // end: this is the field the whole "never upload under a stale
        // identity" rule turns on, so the mint site states it rather than
        // inheriting it.
        rallySessionID = nil
        peerVideoID = nil
        guard let session, let pairingID else { return }
        let minted = "\(pairingID)-r\(rallyGeneration)"
        guard session.sendSessionManifest(sessionID: minted, videoID: "") else { return }
        rallySessionID = minted
    }

    /// The receiving half. Runs on main (hopped in `beginPairing()`).
    ///
    /// `videoID` is the protocol's own discriminator, per
    /// `PeerSession.onSessionManifest`: empty on the announce that *arms* a
    /// rally, non-empty once the sender has uploaded. Reading it that way is
    /// what keeps a post-rally completion manifest from being mistaken for the
    /// next rally's arming one.
    ///
    /// The arming branch is secondary-only: the primary minted this rally's id
    /// itself and must never let a peer overwrite it. The completion branch
    /// requires the id to match the rally still in hand — `peerVideoID` is
    /// best-effort enrichment (`/api/track` stores `peer_video_id` and fusion
    /// never reads it), so attaching one from a *different* rally would be
    /// strictly worse than attaching none.
    private func receiveSessionManifest(sessionID: String, videoID: String) {
        guard !sessionID.isEmpty else { return }
        if videoID.isEmpty {
            guard role == .secondary else { return }
            rallySessionID = sessionID
            peerVideoID = nil
        } else if sessionID == rallySessionID {
            peerVideoID = videoID
        }
    }

    /// Reads this rally's upload identity and clears it in one step, so the
    /// same identity can never be handed to two uploads.
    ///
    /// Returns all-nil unless *both* halves are present. The server refuses a
    /// half-paired run outright ("A paired run needs both session_id and
    /// camera_role"), and `APIClient.startTrack` already drops both unless both
    /// are non-nil — stating it here too keeps the rule where the decision is
    /// actually made. `peerVideoID` rides along: meaningless without a session.
    ///
    /// Internal rather than private, for the same reason
    /// `reconcileRallyAfterStartAttempt()` is: this tuple *is* what
    /// `finishRecordingAndSubmit()` hands to `RunSubmission.submit`, and there
    /// is no seam to observe that call — `finishRecordingAndSubmit()`
    /// constructs its own `RunSubmission()` on the real `APIClient`, and in a
    /// test host the rally never produces a clip to submit in the first place.
    /// So a test that wants to assert what a rally *would* upload under calls
    /// this directly. Consuming, so a test must call it where a real stop
    /// would have.
    func takeRallyUploadIdentity()
        -> (sessionID: String?, cameraRole: String?, peerVideoID: String?) {
        let id = rallySessionID
        let peer = peerVideoID
        rallySessionID = nil
        peerVideoID = nil
        // Not named `role`: that is already a stored property of a *different*
        // type (`PeerRole?`) on this object, and shadowing it here would make
        // the return line read as though it were sending the pairing role.
        guard let id, let serverRole = cameraRole else {
            return (sessionID: nil, cameraRole: nil, peerVideoID: nil)
        }
        return (sessionID: id, cameraRole: serverRole, peerVideoID: peer)
    }

    /// Primary-only, and only once the engine exists — both enforced by
    /// `computedPrimaryEnabled`. Recording is started locally first so a
    /// dropped message can never leave this phone not recording.
    ///
    /// `.submitting` is refused alongside `.recording`, so this phone never
    /// starts a rally on top of one of its own that is still finishing. §7
    /// already keeps the tap from existing — the peer session stays `.live`
    /// until `finishRally` releases it, and `computedPrimaryEnabled` reads
    /// `.live` as disabled — so this guard is about anything reaching the
    /// model around the screen. Starting anyway would work mechanically
    /// (`rallyGeneration` disowns the old rally's finish), but the previous
    /// rally's outcome and its session manifest would then be dropped without
    /// ever being shown. The secondary's `startLocalRecording()` deliberately
    /// does *not* copy this guard — see there.
    func startRally() {
        guard rally != .recording, rally != .submitting else { return }
        beginRally()
        let start = setRecordingChained(true)
        session?.goLive()
        // Before the "start" frame, deliberately — see `armRallyIdentity()`.
        armRallyIdentity()
        session?.sendRecord(action: "start", ptsNs: UInt64(ClockSync.hostNow() * 1_000_000_000))
        republish()
        // The start above is fire-and-forget; reconcile once it actually
        // finishes rather than assuming it did what was asked — see
        // `reconcileRallyAfterStartAttempt()`.
        Task { [weak self] in
            await start.value
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
    ///
    /// Unlike `startRally()` this must follow the primary into the next rally
    /// even from `.submitting`: the two phones upload and track independently,
    /// the primary only ever learns when *its own* submission finished, and the
    /// wire protocol carries no "I'm still busy" message. Refusing here would
    /// silently leave the secondary not recording a rally the primary is
    /// calling. `rallyGeneration` is what makes that safe — the still-running
    /// finish work for the previous rally can no longer write this model.
    private func startLocalRecording() {
        beginRally()
        let start = setRecordingChained(true)
        republish()
        Task { [weak self] in
            await start.value
            self?.reconcileRallyAfterStartAttempt()
        }
    }

    /// Every recording this layer asks for goes through `RecordModel`'s one
    /// funnel, tagged `.live`.
    ///
    /// The serialization and the state check now live in `RecordModel`, where
    /// the camera is — this layer is no longer the only consumer of that
    /// camera, so a chain private to this object could only order *its own*
    /// calls and was blind to the record stage's. The funnel re-reads the
    /// camera's real state at execution time and refuses a transition that is
    /// not the one actually needed, so a start issued while something else is
    /// already recording, or a stop issued against a recording this layer does
    /// not own, is skipped rather than forced.
    ///
    /// `enqueueSetRecording`, not `Task { await record.setRecording(...) }`:
    /// the position in the chain has to be taken synchronously, here, at issue
    /// time. `startRally()`'s start is fire-and-forget while `stopRally()`'s
    /// path awaits, so a fast stop right behind a start would otherwise reach
    /// the funnel first, no-op against a camera that has not started yet, and
    /// leave the queued start rolling with nothing left to stop it.
    ///
    /// Returns whether this call's own transition actually ran — `false` when
    /// the funnel skipped it. `endSession()` reads that to know whether it
    /// produced a clip worth discarding; the others await it for sequencing.
    private func setRecordingChained(_ shouldRecord: Bool) -> Task<Bool, Never> {
        guard let record else { return Task { false } }
        return record.enqueueSetRecording(shouldRecord, owner: .live)
    }

    /// Reconciles `rally` against the camera's real state after a start
    /// attempt's chained transition finishes, rather than assuming it did
    /// what it was asked. `RecordModel`'s start branch can throw and leave
    /// `isRecording == false` with the reason swallowed into `errorText` —
    /// nobody else reads that back, so without this a failed start would
    /// silently proceed as though the rally were live: the camera never
    /// started, `rally == .recording` says otherwise, and the rally would
    /// then be driven to a stop against a camera that was never rolling.
    ///
    /// `isRecordingOwned(by: .live)`, not the bare `isRecording`: the record
    /// stage shares this camera, so a plain recording already in progress
    /// makes `isRecording` read `true` for a start that the funnel actually
    /// refused. Reading the bare flag here is what would let `stopRally()`
    /// later stop — and submit — someone else's clip as rally footage.
    ///
    /// Internal rather than private: this is exactly the reconciliation
    /// `startRally()`/`startLocalRecording()` schedule for themselves once
    /// their own transition completes, exposed only so a test can drive it
    /// synchronously. There is no seam to make the real
    /// `CameraController.startRecording()` throw in a test host — its
    /// `AVAssetWriter` setup does not depend on a running capture session,
    /// so it does not reliably fail outside a device — so
    /// `LiveSessionModelTests` simulates the state `applyRecording`'s start
    /// catch branch leaves behind and calls this directly instead of
    /// exercising the real throw.
    func reconcileRallyAfterStartAttempt() {
        guard rally == .recording else { return }
        guard record?.isRecordingOwned(by: .live) != true else { return }
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
        // Trivially the current generation: this runs only while `rally ==
        // .recording` (guarded above), and neither `startRally()` nor
        // `handleRemoteRecord`'s "start" branch can begin another rally from
        // there — so nothing can have bumped it since this attempt began.
        // Routed through `finishRally` anyway because this is a rally ending,
        // and every rally ending has to release the peer session.
        finishRally(.failed(record?.errorText ?? "Recording did not start."),
                    generation: rallyGeneration)
    }

    /// Applies a rally's terminal outcome and hands the peer session back to
    /// `.ready`, so the same pairing can run another rally. The one exit from
    /// a rally — both the normal stop path and the failed-start reconcile
    /// above go through it — so no path can leave `PeerSession` stuck `.live`.
    ///
    /// `generation` is what makes it safe for a rally's finishing work to land
    /// late. On the secondary that work really is still running when the next
    /// rally starts: each phone uploads and tracks its own clip, the primary
    /// only knows when its own finished, and the wire protocol is fixed. An
    /// ungated late write would set `rally = .submitted` for a rally that is
    /// currently recording — `PlayRootView` would pop `p-live` out from under
    /// a rolling camera, `stopRally()`'s own `rally == .recording` guard would
    /// then refuse to stop it, and the dead end this change closes would be
    /// back in a worse form. A superseded finish therefore writes nothing at
    /// all — not `rally`, and not the session phase, which the running rally
    /// owns. The `Bool` says whether it applied.
    @discardableResult
    private func finishRally(_ outcome: RallyState, generation: Int) -> Bool {
        guard generation == rallyGeneration else { return false }
        rally = outcome
        // This rally's identity dies with it. `finishRecordingAndSubmit()` has
        // normally consumed it already (and holds its own copy), but this is
        // also the exit `reconcileRallyAfterStartAttempt()` takes when the
        // camera never started — a path with no upload and therefore no
        // consume. Left set, the *next* rally would find it and, if that
        // rally's own manifest were lost, upload under it.
        //
        // Safe under the generation guard above: a superseded finish returns
        // before this line, so a late-landing rally N can never clear the id
        // rally N+1 is currently holding.
        rallySessionID = nil
        peerVideoID = nil
        session?.endRally()
        republish()
        return true
    }

    private func finishRecordingAndSubmit() {
        let generation = rallyGeneration
        // Taken here, synchronously, before the first `await` below — see the
        // ordering note above `armRallyIdentity()`, point 5. Capturing it as a
        // local is what makes a manifest for the *next* rally, arriving while
        // this upload is still in flight, unable to retarget it.
        let upload = takeRallyUploadIdentity()
        rally = .submitting
        republish()
        // One `RunSubmission` per rally rather than one per model. `submit`
        // runs an upload plus a poll of the server-side tracking job, and on
        // the secondary the previous rally's is genuinely still running when
        // the next one's begins (see `finishRally`). Two concurrent `submit`
        // calls on one object interleave their writes to the single `phase`
        // this method reads straight back as the rally's outcome — a rally
        // could report the other one's result. Nothing observes this object
        // (it is not `@Published` and no view reaches it), so scoping it to
        // the rally costs nothing and removes the shared state entirely.
        let submission = RunSubmission()
        // The stop depends on the camera's real state and on this layer
        // owning it, never on `rally` saying `.recording` — if the camera
        // never actually confirmed it started, or is rolling for the record
        // stage, the funnel skips this call and the `no clip` branch below
        // is what the rally reports.
        let stop = setRecordingChained(false)
        Task { [weak self] in
            guard let self else { return }
            await stop.value
            guard let record = self.record else {
                // Two statements, not `return self.finishRally(...)`: this
                // closure returns Void and `finishRally` returns Bool, so the
                // old `return self.republish()` shape does not carry over.
                self.finishRally(.failed("The camera model was no longer available."),
                                 generation: generation)
                return
            }
            // `liveClip`, not the raw clip: a clip the record stage owns is
            // invisible here, so a plain recording can never be submitted as
            // this rally's paired footage.
            guard let clip = record.liveClip else {
                self.finishRally(.failed("The rally produced no clip."), generation: generation)
                return
            }
            // The live path reports its own outcome through `rally`, so once
            // the clip has been read here, clear it — otherwise a later live
            // visit to this same RecordModel would find a stale one waiting.
            record.liveClip = nil
            await submission.submit(videoURL: clip.url, duration: clip.duration,
                                    sessionID: upload.sessionID,
                                    cameraRole: upload.cameraRole,
                                    peerVideoID: upload.peerVideoID,
                                    syncManifestJSON: self.syncManifestJSON())
            switch submission.phase {
            case .complete:
                // The *completion* manifest — a non-empty videoID, which is
                // the protocol's discriminator (see `receiveSessionManifest`),
                // so the peer can never read it as an arming announce for a
                // new rally.
                //
                // `upload.sessionID`, not the model's current one: the model's
                // was consumed at the top of this method and may by now belong
                // to a *later* rally. This names the rally that just uploaded,
                // which is the only thing this videoID means anything about.
                //
                // Order is load-bearing and verified, not assumed: `finishRally`
                // has just put the session back in `.ready`, and
                // `sendSessionManifest`'s gate is `.live || .ready` — the same
                // gate every other `PeerSession` sender uses — so this still
                // goes out. Best-effort enrichment either way: fusion pairs on
                // session_id + camera_role and never reads `peer_video_id`, so
                // a lost one blocks nothing. Skipped entirely when
                // `finishRally` reports the rally was superseded.
                if self.finishRally(.submitted, generation: generation),
                   let sessionID = upload.sessionID, let videoID = submission.completedRunID {
                    self.session?.sendSessionManifest(sessionID: sessionID, videoID: videoID)
                }
            case .failed(let message):
                self.finishRally(.failed(message), generation: generation)
            default:
                self.finishRally(.failed("Upload did not finish."), generation: generation)
            }
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

    /// §16's `p-live` STOP rule, kept alongside `updateShowsLocalStop` (its
    /// only two call sites are the same two as that method's) rather than
    /// folded into it: this one also depends on `role`, and keeping the two
    /// separate keeps each one's doc honest about what it actually derives.
    /// A view reading `rally`/`role`/`showsLocalStop` directly and combining
    /// them itself would duplicate this exact rule outside the model — the
    /// same reason `computedPrimaryEnabled` etc. live here instead of in
    /// `PairingView`.
    private func updateShowsStop() {
        showsStop = rally == .recording && (role == .primary || showsLocalStop)
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
        updateShowsStop()
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

        // NOTE: there is deliberately no session-manifest broadcast here any
        // more. It used to mint one id per pairing on first reaching `.ready`;
        // the server pairs runs on `session_id` + `camera_role` and assumes one
        // rally per id, so that made every rally after the first silently
        // unfusable — or, worse, fusable against the wrong rally. The broadcast
        // now lives in `armRallyIdentity()`, once per rally. See
        // `rallySessionID`.

        // The secondary's calibration exchange, and the one thing that makes a
        // paired rally possible at all: this message is what builds the
        // primary's `StereoEngine` (`RecordModel.attachStereo`'s
        // `onCalibration` handler), which fires `onStereoReady`, which sets
        // `engineReady`, which is half of `computedPrimaryEnabled`'s `.ready`
        // case. Without it START RALLY never enables on real hardware and no
        // rally can ever begin.
        //
        // `.ready` is the earliest correct moment as well as the specified
        // one: `PeerSession.sendCalibration` gates on `.live || .ready` and
        // silently drops anything sent before that.
        //
        // The flag is set from the send's **return value**, never
        // optimistically. `pairing.step` is a mirror of `PeerSession`'s
        // `@Published phase`, which lags the authoritative `internalPhase`
        // whenever `setPhase` ran off-main (it hops to main asynchronously).
        // Latching the flag before knowing the outcome therefore had a real
        // window: the mirror still says `.ready`, the sender's own gate reads
        // an `internalPhase` that has already left it and drops the frame, and
        // because the flag never re-arms, `engineReady` stays false and START
        // RALLY is permanently disabled with no recovery short of ending the
        // session. Reporting the outcome makes the 20 Hz pump retry instead —
        // a dropped send costs one more attempt 50 ms later and nothing else.
        //
        // Still exactly once per pairing on the success path: the flag is set
        // synchronously with the successful send, with no `await` between, so
        // the very next pump turn takes neither branch. That is also what makes
        // *returning* to `.ready` a no-op rather than a re-send — `endRally()`
        // puts the session back to `.ready` after every rally, and the
        // primary's engine already exists by then. The flag is cleared only in
        // `teardownPairing()` (i.e. wherever `self.session` is replaced), never
        // by a phase change, so "ready again" is never mistaken for "first
        // ready".
        //
        // On the secondary the session in fact never leaves `.ready` at all —
        // only the primary calls `goLive()` — so this guard is load-bearing
        // for the whole length of every rally, not just at the seam.
        if role == .secondary, case .ready = pairing.step, !didSendCalibration,
           let localModelJSON, let localProfileID {
            didSendCalibration = session?.sendCalibration(profileID: localProfileID,
                                                          payloadJSON: localModelJSON) ?? false
        }
        updateShowsLocalStop(degraded: isPairingDegraded)
        updateShowsStop()
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
