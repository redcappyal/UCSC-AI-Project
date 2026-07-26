import Foundation
import UIKit

struct FinishedClip: Identifiable {
    let id = UUID()
    let url: URL
    let duration: Double
}

/// Which consumer asked for a recording — the record stage's own button, or
/// the live layer's rally lifecycle.
///
/// One `RecordModel` (one camera) is shared by both, so "who started this
/// recording" is not derivable from anything else: `LiveSessionModel.rally`
/// is that layer's *belief*, which is exactly what must never be trusted here
/// (DESIGN.md §16 — pairing adds capability, never gates it). The owner is
/// recorded where the camera actually lives, for the duration of the
/// recording, and it decides two things: who is allowed to stop it, and which
/// consumer its clip is surfaced to.
enum RecordingOwner: Equatable { case single, live }

@MainActor
final class RecordModel: ObservableObject {
    let camera = CameraController()
    let tracker: BallTracker

    @Published var trail: [BallObservation] = []
    @Published var isRecording = false
    @Published var recordingStartedAt: Date?
    @Published var errorText: String?
    /// Resolved capture exposure, shown once the court lock lands.
    @Published var exposureNote: String?

    /// Who started the recording currently running, or nil when the camera is
    /// not recording. Written only by `applyRecording(_:owner:)`.
    @Published private(set) var recordingOwner: RecordingOwner?

    /// The clip the last completed recording produced, and who it belongs to.
    ///
    /// Deliberately not readable as one shared field any more: both consumers
    /// used to read the same `finishedClip`, so a live-owned rally could open
    /// `ResultsView` in the plain judge flow, and a plain recording could be
    /// submitted as paired rally footage. Each consumer now reads its own
    /// owner-filtered view of it (`singleCameraClip` / `liveClip`) and cannot
    /// see — or clear — the other's.
    @Published private(set) var finishedClip: FinishedClip?
    @Published private(set) var finishedClipOwner: RecordingOwner?

    /// The record stage's clip: non-nil presents `ResultsView`. Settable so
    /// `RecordView`'s `.sheet(item:)` can bind to it; writing anything but
    /// `nil`, or clearing a clip this consumer does not own, is refused.
    var singleCameraClip: FinishedClip? {
        get { clip(for: .single) }
        set { setClip(newValue, for: .single) }
    }

    /// The live layer's clip, read by `LiveSessionModel` when a rally's stop
    /// lands. Same rules as `singleCameraClip`, mirrored.
    var liveClip: FinishedClip? {
        get { clip(for: .live) }
        set { setClip(newValue, for: .live) }
    }

    private func clip(for owner: RecordingOwner) -> FinishedClip? {
        finishedClipOwner == owner ? finishedClip : nil
    }

    private func setClip(_ newValue: FinishedClip?, for owner: RecordingOwner) {
        guard newValue == nil, finishedClipOwner == owner else { return }
        finishedClip = nil
        finishedClipOwner = nil
    }

    private static let trailLength = 15

    var detectorMissing: Bool { !tracker.isEnabled }

    enum DetectorKind: Equatable { case none, synthetic, model }

    /// `detectorMissing` only knows nil-vs-non-nil, so it cannot tell a real
    /// model from the DEBUG stand-in. The UI must never imply a real detection
    /// when the source is synthetic.
    @Published private(set) var detectorKind: DetectorKind

    /// How this model learns which way the phone is mounted: the interface
    /// orientation mapped to a capture mount, or `nil` when the interface is
    /// not resolved to either landscape — no window scene at all, or a
    /// portrait transient mid-rotation. Both readers are on the recording
    /// path: `startCamera`'s mount seed, and `applyRecording`'s start path,
    /// which refuses to start at all on `nil`.
    ///
    /// A seam rather than a direct call, because the value comes from ambient
    /// UIKit scene state that `RecordModel` would otherwise be reaching out of
    /// itself to read. Every test that starts a recording would then depend on
    /// whether the simulator's rotation had settled by the time XCTest ran it,
    /// and the ones that *wait* on `isRecording` would burn their full
    /// deadline before failing — a confusing failure for a race that has
    /// nothing to do with what they assert. `interfaceMount` below is the
    /// production implementation and the default, so the shipping path is
    /// exactly what it was.
    typealias MountResolver = @MainActor () -> CaptureSettings.CaptureOrientation?

    /// The production mount resolution, and `init`'s default: `OrientationPolicy`'s
    /// two-tier scene lookup mapped through `OrientationLock.captureOrientation(for:)`.
    ///
    /// Two-tier (foreground-active, else any scene) rather than a strict
    /// foreground-active-only check: a transient system interruption (an
    /// alert, Control Center) can leave every scene `.foregroundInactive` for
    /// a moment, and a strict check landing on `nil` there would send
    /// `startCamera`'s first seed straight to its `.landscapeRight` fallback —
    /// indistinguishable from a real landscape-right resolution — and would
    /// make `applyRecording` refuse a start the operator was entitled to.
    ///
    /// Passes `requiresKeyWindow: false`: only `interfaceOrientation` is read
    /// here, and that is available on a scene with no key window yet, unlike
    /// `OrientationPolicy.apply`'s use of the same lookup. One lookup for both
    /// readers rather than two that could quietly diverge — a divergent second
    /// lookup was already a review finding on this branch.
    static func interfaceMount() -> CaptureSettings.CaptureOrientation? {
        OrientationPolicy.activeWindowScene(requiresKeyWindow: false)
            .flatMap { OrientationLock.captureOrientation(for: $0.interfaceOrientation) }
    }

    private let mountResolver: MountResolver

    init(detector: BallDetecting? = CoreMLBallDetector(),
         captureOrientation: CaptureSettings.CaptureOrientation = .landscapeRight,
         mountResolver: @escaping MountResolver = { RecordModel.interfaceMount() }) {
        self.mountResolver = mountResolver
        // Production overwrites this in `startCamera` once the interface
        // orientation is known; the parameter exists so tests can pick a mount
        // without a live window scene.
        self.captureOrientation = captureOrientation
        if detector == nil {
            detectorKind = .none
        } else {
            #if DEBUG
            detectorKind = detector is SyntheticBallDetector ? .synthetic : .model
            #else
            detectorKind = .model
            #endif
        }
        tracker = BallTracker(detector: detector)
        tracker.subscribe { [weak self] observation in
            guard let self else { return }
            trail.append(observation)
            if trail.count > Self.trailLength { trail.removeFirst() }
        }
        // Inference must never block the capture callback: that queue also
        // feeds the AVAssetWriter, and a synchronous Core ML pass (~20-60 ms)
        // would drop recorded frames. Hop to a dedicated queue and skip
        // frames while a detection is in flight — the overlay can afford
        // missed frames, the recording cannot.
        let inferenceQueue = DispatchQueue(label: "slc.record.inference")
        let inFlight = DispatchSemaphore(value: 1)
        camera.onVideoSample = { [tracker] pixelBuffer, timestamp in
            guard inFlight.wait(timeout: .now()) == .success else { return }
            inferenceQueue.async {
                tracker.process(pixelBuffer, timestamp: timestamp)
                inFlight.signal()
            }
        }
    }

    // MARK: peer streaming

    let remoteDetections = RemoteDetectionStore()
    private var peer: PeerSession?
    private var peerPumpTimer: Timer?
    /// The DEBUG stereo demo has no peer, so it cannot share `peerPumpTimer`.
    /// Declared unconditionally so `deinit` doesn't have to fork on `#if`.
    private var demoPumpTimer: Timer?
    private var peerSubscribed = false
    private var nextDetectionSeq: UInt32 = 0
    private var pendingTuples: [DetectionTuple] = []
    private var lastFlushAt: TimeInterval = 0
    /// The mount this session is currently using. While no recording is
    /// running, `startCamera` freely re-seeds this with an unpinned guess
    /// every time Play appears — there is nothing to protect yet, and the
    /// operator is entitled to re-mount the phone between rallies.
    /// `applyRecording`'s start path is what commits to a mount: it only
    /// assigns this AFTER `camera.startRecording()` actually succeeds, and
    /// only then does the re-seed stop touching it — a tab round trip
    /// mid-recording still re-runs `startCamera`, and would otherwise stomp
    /// the recording's committed mount with a fresh guess. Two things stop
    /// that: `startCamera` bails out whole on `isRecording`, and the re-seed
    /// itself runs on the recording funnel's own transition chain and
    /// re-checks `isRecording` when its turn comes — see `enqueueMountReseed`
    /// for why the chain, not the flag, is what makes it airtight.
    /// The peer detection path reads this (`detectionFrameSize` below, and
    /// the tuples it sizes); the preview overlay and camera-model adoption
    /// still key off the `CaptureSettings` statics directly, since both
    /// mounts share one frame size and neither needed the extra indirection.
    @Published private(set) var captureOrientation: CaptureSettings.CaptureOrientation

    /// The pixel space local detections are expressed in — the same space the
    /// peer is told to read them in via `Hello`. Derived from the resolved
    /// mount rather than from a constant, which is exactly what went wrong
    /// before: a constant frame size labelled every tuple regardless of the
    /// mount the session actually captured in.
    var detectionFrameSize: (width: Int, height: Int) {
        CaptureSettings.frameSize(for: captureOrientation)
    }

    // MARK: stereo (Phase 3, Plan B2)

    private var localModel: CameraModel?
    private var stereoEngine: StereoEngine?
    /// Fired once the primary's StereoEngine is built from both camera models.
    /// Until then the primary cannot make a call, and START RALLY must say so.
    var onStereoReady: (() -> Void)?
    /// Newest-first, capped at 20. Populated on the primary as its engine
    /// emits impacts, and mirrored on the secondary from relayed .event
    /// messages — same list either way, so the UI doesn't need to know
    /// which role it's running as.
    @Published var stereoEvents: [String] = []
    private static let stereoEventsCap = 20

    // MARK: live call rendering (Phase 4)

    /// What `p-live` should be showing, or nil for "nothing called yet".
    ///
    /// Set from the engine's `onEvent` on the primary and from the relayed
    /// `.event` mirror on the secondary, both after a main hop, and both
    /// through `CallPresentation` so the `no_call` gate is written once.
    /// Deliberately never derived from `stereoEvents`: mapping the wire JSON
    /// where it arrives (`attachStereo`) keeps that gate at the one entry
    /// point, where re-parsing a display list later could not.
    @Published private(set) var livePresentation: CallPresentation?

    /// The 3D track behind the most recent call, for the §8.10 post-rally
    /// replay. Never derived from `stereoEvents` — that list is relayed JSON
    /// with no geometry in it.
    @Published private(set) var liveTrack: [TrackPoint3D] = []
    /// The impact that produced `livePresentation`. `MiniCourtView` colours its
    /// marker from this, so the replay always agrees with the call that was
    /// made — the same reason `CallPresentation.color` is the one mapping.
    @Published private(set) var liveImpact: StereoImpact?

    /// The §8.17 wash, which is *transient* where `livePresentation` is
    /// persistent. Separate state on purpose: binding the full-stage flash to
    /// the banner's value leaves an 82%-opacity verdict wash covering the
    /// camera feed from the first call onward, and §16 gates the mini-court on
    /// "once the flash clears".
    @Published private(set) var flashPresentation: CallPresentation?
    private var flashClearWork: DispatchWorkItem?
    /// §10 budgets the whole flash at ≤ 500 ms, and `CallFlashView` spends
    /// 150 ms of that easing in — so the hold is the remainder, not 500 ms.
    static let flashHoldS = 0.35

    private func showFlash(_ presentation: CallPresentation) {
        flashPresentation = presentation
        flashClearWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.flashPresentation = nil }
        flashClearWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.flashHoldS, execute: work)
    }

    /// Wire a paired session. Safe to call again with a new session; the
    /// tracker subscription is registered once. Subscriber runs on the main
    /// queue (BallTracker's fan-out queue). The timer pump keeps
    /// heartbeats/sync alive even when no ball is detected — a primary with
    /// zero local detections must still tick.
    func attachPeer(_ peer: PeerSession) {
        self.peer = peer
        peer.onRemoteDetections = { [weak self] tuples in
            // remoteDetections is lock-guarded and latency-sensitive — stays
            // on the delivery queue. stereoEngine is a plain @MainActor
            // reference also written (in attachStereo's onCalibration
            // handler) and read (pump timer, tracker.subscribe) on main, so
            // only the reference read needs to hop; the engine's own
            // addRemote is queue-confined internally.
            self?.remoteDetections.append(tuples)
            DispatchQueue.main.async { [weak self] in self?.stereoEngine?.addRemote(tuples) }
        }
        peerPumpTimer?.invalidate()
        peerPumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self, weak peer] _ in
            peer?.tick(now: ClockSync.hostNow())
            self?.stereoEngine?.processIfDue(now: ClockSync.hostNow())
        }
        // A new peer session expects a fresh seq space.
        nextDetectionSeq = 0
        pendingTuples.removeAll()
        lastFlushAt = 0
        guard !peerSubscribed else { return }
        peerSubscribed = true
        tracker.subscribe { [weak self] observation in
            guard let self, let peer = self.peer else { return }
            if peer.role == .secondary {
                let tuple = DetectionMapper.tuple(seq: self.nextDetectionSeq,
                                                  observation: observation,
                                                  frameW: self.detectionFrameSize.width,
                                                  frameH: self.detectionFrameSize.height)
                self.nextDetectionSeq += 1
                self.pendingTuples.append(tuple)
                let now = observation.timestamp
                if self.pendingTuples.count >= 2 || now - self.lastFlushAt >= 0.030 {
                    peer.sendDetections(self.pendingTuples)
                    self.pendingTuples.removeAll(keepingCapacity: true)
                    self.lastFlushAt = now
                }
            } else if peer.role == .primary, let engine = self.stereoEngine {
                engine.addLocalObservation(observation,
                                           frameW: self.detectionFrameSize.width,
                                           frameH: self.detectionFrameSize.height)
            }
        }
    }

    /// Decodes this device's own solved camera model and arms the peer's
    /// calibration handler to build the live StereoEngine once the remote
    /// model arrives. Primary-only: the secondary never owns an engine — it
    /// streams local detections to the primary (see `attachPeer` above) and
    /// mirrors emitted events for display via `peer.onEvent` below.
    func attachStereo(localModelJSON: String) {
        guard let data = localModelJSON.data(using: .utf8),
              let model = try? CameraModel.fromJSON(data) else {
            errorText = "stereo: malformed local camera model"
            return
        }
        // A solved model is only meaningful in the pixel space it was solved
        // in. Everything downstream (detections, StereoEngine) works in
        // CaptureSettings' space, so re-express it here or refuse: a model
        // of unknown or differently-cropped origin cannot be scaled, and a
        // wrong-but-silent line call is worse than no call.
        do {
            localModel = try model.adoptedForCapture()
        } catch {
            errorText = "stereo: local camera model — \(error.localizedDescription)"
            return
        }
        // onCalibration fires on the transport delivery context, not main
        // (like onRemoteDetections) — hop before touching any RecordModel
        // state (self.peer, self.localModel, self.stereoEngine), all of
        // which the pump timer and tracker.subscribe also read/write, on
        // main, elsewhere.
        peer?.onCalibration = { [weak self] _, payloadJSON in
            DispatchQueue.main.async {
                guard let self, let peer = self.peer, peer.role == .primary,
                      let localModel = self.localModel,
                      let remoteData = payloadJSON.data(using: .utf8),
                      let remoteModel = try? CameraModel.fromJSON(remoteData) else { return }
                // Same adoption as the local model above — the peer sends
                // whatever space its own calibration was solved in. Refusing
                // to go live is the correct outcome when it can't be scaled.
                let adoptedRemote: CameraModel
                do {
                    adoptedRemote = try remoteModel.adoptedForCapture()
                } catch {
                    self.errorText =
                        "stereo: remote camera model — \(error.localizedDescription)"
                    return
                }
                let engine = StereoEngine(localModel: localModel, remoteModel: adoptedRemote,
                                          remoteToLocal: { [weak peer] in peer?.clockSync.remoteToLocal($0) })
                engine.onEvent = { [weak self, weak peer] event in
                    guard case .impact(let impact, let track) = event else { return }
                    let payload: [String: Any] = [
                        "surface": impact.surface,
                        "call": impact.call,
                        "margin_ft": impact.marginFt,
                        "confidence": impact.confidence,
                        "t_s": impact.tS,
                    ]
                    guard let data = try? JSONSerialization.data(withJSONObject: payload),
                          let json = String(data: data, encoding: .utf8) else { return }
                    // Rally segmentation is Phase 4; Plan B2 has one
                    // implicit rally per session, so rallyID stays a
                    // constant 0.
                    let rallyID: UInt32 = 0
                    // engine.onEvent fires on StereoEngine's own private
                    // queue — sendEvent is lock-guarded (safe off-main, same
                    // convention as PeerSession's other send* methods), but
                    // the @Published append still needs the main hop.
                    peer?.sendEvent(rallyID: rallyID, json: json)
                    // Built here, off-main: the mapping is pure, and doing it
                    // from the impact struct rather than from `json` keeps the
                    // no_call gate in exactly one place.
                    let presentation = CallPresentation.from(impact)
                    DispatchQueue.main.async {
                        self?.appendStereoEvent(json)
                        self?.livePresentation = presentation
                        self?.liveTrack = track
                        self?.liveImpact = impact
                        self?.showFlash(presentation)
                    }
                }
                self.stereoEngine = engine
                self.onStereoReady?()
            }
        }
        // Secondary-side mirror: relayed events land here (never fires on
        // the primary — it never receives .event, only sends it). Also
        // fires off-main, so every published write below is hopped.
        //
        // The banner and the flash are *not* primary-only. DESIGN.md §16's
        // `p-live` states table draws no role distinction for either, and the
        // spec's v1 narrowing is only about the mini-court — so `liveTrack`
        // and `liveImpact` stay untouched here (the payload carries no
        // geometry, and `MiniCourtView` would have nothing to draw), while
        // `livePresentation`/`showFlash` get exactly what the primary gets.
        // Without this the secondary's banner read "No call yet" for an
        // entire rally, on both phones' account of the same call.
        //
        // Mapped off-main, before the hop, for the same reason the primary's
        // own presentation is: the mapping is pure, and routing it through
        // `CallPresentation` — rather than reading `call` here and deciding
        // anything about it — keeps the no_call / unknown-word / floor-bounce
        // gate in exactly one place. A payload that isn't one of those, or
        // isn't parseable at all, yields nil and changes nothing on screen:
        // it is never rendered as a confident verdict.
        peer?.onEvent = { [weak self] _, json in
            let presentation = CallPresentation.fromRelayedEvent(json: json)
            DispatchQueue.main.async {
                self?.appendStereoEvent(json)
                guard let presentation else { return }
                self?.livePresentation = presentation
                self?.showFlash(presentation)
            }
        }
    }

    /// Undoes `attachPeer`/`attachStereo` when a session ends. Without this,
    /// `peerPumpTimer` keeps calling `peer?.tick()` and
    /// `stereoEngine?.processIfDue()` forever against a session nobody owns
    /// anymore, and a stale `livePresentation`/`flashPresentation` can keep
    /// showing a verdict for a rally the user just cancelled.
    ///
    /// Does NOT clear `onStereoReady`: `bind(record:)` installs that once and
    /// the binding outlives any one session.
    ///
    /// `tracker.subscribe`'s closure (registered once in `attachPeer`, guarded
    /// by `peerSubscribed` so it is never registered twice) cannot be
    /// unsubscribed — `BallTracker` has no such API. That's fine: its body
    /// opens with `guard let self, let peer = self.peer else { return }`, so
    /// once `peer` is nil below it is a no-op on every subsequent detection.
    /// Confirmed by reading that closure, not assumed.
    func detachPeer() {
        peerPumpTimer?.invalidate()
        peerPumpTimer = nil
        peer = nil
        stereoEngine = nil
        localModel = nil

        clearLiveCall()

        nextDetectionSeq = 0
        pendingTuples.removeAll()
        lastFlushAt = 0
    }

    /// Clears the call *readout* — banner, flash, mini-court geometry, engine
    /// event list — without touching the peer, the engine, or the local model
    /// behind it. That split is the whole point: `detachPeer()` above tears the
    /// session down and reuses this for the display half, while a new rally on
    /// an existing pairing needs exactly this half and none of the rest.
    ///
    /// DESIGN.md §16's `p-live` table promises the mini-court "clears again
    /// when `START RALLY` begins the next one". Until a session could run more
    /// than one rally that promise had no call site; `LiveSessionModel`'s rally
    /// start is now it. Written once, here, so the two callers cannot drift
    /// into clearing different subsets of the same readout.
    func clearLiveCall() {
        livePresentation = nil
        liveTrack = []
        liveImpact = nil
        flashClearWork?.cancel()
        flashClearWork = nil
        flashPresentation = nil
        stereoEvents = []
    }

    #if DEBUG
    /// Drive the whole live path on one device, with no peer and no ball model.
    ///
    /// Builds a `StereoEngine` directly from two solved camera models — the
    /// same shape `StereoEngineTests` uses, deliberately *not* a spoofed
    /// `PeerSession` handshake — and feeds it a court-feet trajectory
    /// projected through both models, so the two eyes see a physically
    /// consistent ball. Both eyes matter: `StereoEngine.process` bails unless
    /// local *and* remote samples overlap, so a local-only synthetic detector
    /// would emit nothing, forever.
    ///
    /// The models are used unadopted, in their own solve pixel space. Calling
    /// `adoptedForCapture()` here would not throw — the goldens' 1920×1080
    /// shares the capture target's 16:9 aspect, so it is a clean 2x scale —
    /// but the synthetic detections below are produced in that same
    /// 1920×1080 space, so adopting the models while leaving the detections
    /// unadopted would mismatch the two.
    func startStereoDemo(localModelJSON: String, remoteModelJSON: String) {
        // The demo's models are the goldens in their own unadopted 1920×1080
        // space. Installing its engine over a paired session's would keep
        // attachPeer's pump and onRemoteDetections feeding 4K capture-space
        // detections through the wrong geometry — silently wrong calls, which
        // is the one failure mode this whole layer exists to avoid.
        guard peer == nil else {
            errorText = "Stereo demo: not available while paired."
            return
        }
        guard let localData = localModelJSON.data(using: .utf8),
              let remoteData = remoteModelJSON.data(using: .utf8),
              let local = try? CameraModel.fromJSON(localData),
              let remote = try? CameraModel.fromJSON(remoteData) else {
            errorText = "Stereo demo: could not parse the camera models."
            return
        }

        // Same clock for both eyes — there is no peer to sync against.
        let engine = StereoEngine(localModel: local, remoteModel: remote,
                                  remoteToLocal: { $0 })
        engine.onEvent = { [weak self] event in
            guard case .impact(let impact, let track) = event else { return }
            let presentation = CallPresentation.from(impact)
            DispatchQueue.main.async {
                self?.livePresentation = presentation
                self?.liveTrack = track
                self?.liveImpact = impact
                self?.showFlash(presentation)
            }
        }

        let tracks = StereoDemo.pixelTracks(local: local, remote: remote)
        for sample in tracks.local { engine.addLocalPixel(sample.px, tS: sample.tS) }
        engine.addRemote(StereoDemo.remoteTuples(tracks.remote))

        stereoEngine = engine
        livePresentation = nil
        liveTrack = []
        liveImpact = nil
        flashClearWork?.cancel()
        flashPresentation = nil

        // `attachPeer` owns the shared pump, but the demo has no peer, so it
        // needs its own. Invalidate first so repeated taps don't stack timers.
        demoPumpTimer?.invalidate()
        demoPumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak engine] _ in
            engine?.processIfDue(now: ClockSync.hostNow())
        }
    }
    #endif

    /// Newest-first insert with the 20-entry cap. Main-thread only; callers
    /// hop before calling.
    private func appendStereoEvent(_ json: String) {
        stereoEvents.insert(json, at: 0)
        if stereoEvents.count > Self.stereoEventsCap {
            stereoEvents.removeLast(stereoEvents.count - Self.stereoEventsCap)
        }
    }

    deinit {
        peerPumpTimer?.invalidate()
        demoPumpTimer?.invalidate()
        flashClearWork?.cancel()
    }

    /// Guards the *configuration* half of `startCamera()`, and only that half.
    /// Three callers now: `RecordView`'s own `.task`, `PlayRootView`'s
    /// `p-pair` destination (the live path configures the camera when pairing
    /// is entered — a rally starts from a tap on `p-pair`, before `p-live`
    /// exists), and `p-live`'s own `.task` as a backstop. Configuring is
    /// expensive, and re-metering the court mid-session would change exposure
    /// under a rally, so the second and third callers must not re-run it.
    ///
    /// Deliberately does NOT guard the mount seed. Those are two different
    /// concerns and conflating them is a real bug: the seed has to re-run on
    /// every appearance (see `startCamera`).
    private var cameraStarted = false

    func startCamera() async {
        // `isRecording` first, and it bails out whole: every entry point
        // re-fires this (`RecordView`'s `.task`, `p-pair`'s and `p-live`'s),
        // including a return from Matches/Coach while a recording that
        // `applyRecording`'s start path already resolved and pinned is still
        // running. Re-seeding an unpinned guess below would overwrite the
        // mount a running recording committed to, which is exactly what
        // `captureOrientation`'s doc says stops happening once that path
        // commits. Deliberately does NOT clear `cameraStarted`: the camera is
        // running, and this is a skipped re-entry, not a failed one.
        guard !isRecording else { return }

        // Every appearance after the first re-seeds the mount, and does
        // nothing else. Session configuration is one-shot; the mount is not,
        // and freezing it at the app's first-ever camera start is a bug in
        // two visible places. The preview and the court-exposure meter would
        // keep working from a mount the operator has since changed; worse,
        // `LiveSessionModel.beginPairing()` reads `record.camera.orientation`
        // to build the peer `Hello`, so an operator who opened the record
        // stage before physically mounting the phone would advertise a mount
        // they no longer have — defeating the frame-space handshake guard in
        // precisely the case it exists for.
        //
        // No fallback on this path: a resolution that fails *now* must leave
        // whatever a previous, successful resolution established rather than
        // downgrade it to the `.landscapeRight` literal below. The first seed
        // has no such previous value to keep; this one does.
        guard !cameraStarted else {
            await enqueueMountReseed(fallback: nil).value
            return
        }
        cameraStarted = true      // cleared again in the `catch`, so a failure retries
        do {
            // The first seed, before `configure()` — `configureSession()`
            // reads `camera.orientation` to set the video connection's initial
            // rotation, so this has to have landed by then, which is what
            // awaiting the chained transition guarantees.
            //
            // Purely so the preview and the court-exposure meter below have a
            // sensible mount to work with. Deliberately NOT pinned, here or on
            // any later re-seed: the Play tab stays at both-landscape
            // (`.landscape`) through the whole framing window, which is what
            // lets the operator flip the mount before recording starts.
            // `applyRecording`'s start path is where the mount actually gets
            // committed and pinned — see there for why record start, not
            // camera start, has to be the point of no return.
            //
            // `.landscapeRight` is a last resort only, for the genuinely-no-
            // scene case (or a portrait interface, which is not a capture
            // mode): a fallback default, not a resolved orientation. It costs
            // only a wrong preview/metering default, since nothing here pins;
            // `applyRecording`'s own resolution at record start is what has to
            // get the real mount right, and it refuses rather than falling
            // back at all.
            await enqueueMountReseed(fallback: .landscapeRight).value

            // `CameraController.configure()` is repeatable: it tears its own
            // inputs/outputs down before re-adding them, so re-entering here
            // after a mid-configuration failure below really does retry
            // rather than throwing `configurationFailed` forever.
            try await camera.configure()
            camera.start()
            // Meter the court once from the mounted position, then freeze
            // exposure/WB/focus for the session. Everything after this point
            // is shot under identical conditions, which is what keeps the
            // footage usable as training data.
            exposureNote = CaptureSettings.summary(for: try await camera.lockForCourt())
        } catch {
            cameraStarted = false      // let a retry re-enter
            errorText = error.localizedDescription
        }
    }

    // MARK: - The one recording funnel

    /// Serializes every start/stop against the camera — and every mount
    /// re-seed — in the order they were issued. Held here, not in any
    /// consumer, because the camera is here.
    private var recordingTransition: Task<Bool, Never>?

    /// Re-resolves the mount from the interface and applies it to the camera,
    /// serialized against every recording start and stop.
    ///
    /// Being *on this chain* is the whole safety argument, and a flag would
    /// not do. `applyRecording`'s start path resolves a mount, awaits
    /// `camera.updateOrientation(mount)`, opens an `AVAssetWriter` against
    /// that rotation, and only then assigns `captureOrientation` and pins the
    /// mask. `isRecording` is still `false` for all of that, and the
    /// `updateOrientation` await is a real suspension point — so a re-seed
    /// checking `isRecording` alone could interleave right there and leave the
    /// video connection rotated for one mount while the recording it is
    /// feeding was committed and pinned to the other. Chained, a re-seed runs
    /// strictly before or strictly after a transition and never inside one.
    ///
    /// `fallback` is what to use when the interface cannot be resolved to a
    /// mount: `.landscapeRight` for `startCamera`'s first seed, which has
    /// nothing better, and `nil` for every later re-seed, which does — the
    /// mount an earlier successful resolution already established.
    ///
    /// Reports `false` unconditionally. The chain is typed by the recording
    /// funnel, whose `Bool` means "did this start or stop a recording"; a
    /// re-seed never does either. Callers await it to sequence, not for the
    /// value.
    @discardableResult
    private func enqueueMountReseed(fallback: CaptureSettings.CaptureOrientation?) -> Task<Bool, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            guard let self else { return false }
            await self.applyMountReseed(fallback: fallback)
            return false
        }
        recordingTransition = next
        return next
    }

    /// Runs only from the chain above, so `isRecording` read here is the
    /// camera's state *after* everything issued earlier landed.
    private func applyMountReseed(fallback: CaptureSettings.CaptureOrientation?) async {
        // Re-checked here, at execution time, never in advance — the same rule
        // `canSetRecording` states for the recording transitions this shares a
        // chain with. A recording running by the time this gets its turn owns
        // the mount outright: it resolved one, rotated the connection to it,
        // pinned the orientation mask to it, and is writing frames under it.
        // `isRecording` also stays true across the whole stop path — it is
        // cleared only after `camera.stopRecording()` has returned — so this
        // cannot land mid-stop either, when the writer is still finishing.
        guard !isRecording else { return }
        guard let mount = mountResolver() ?? fallback else { return }
        captureOrientation = mount
        // Routed through `updateOrientation` rather than a direct
        // `camera.orientation = ...` assignment, so `orientation` has one
        // writer, always on `sessionQueue`, instead of a direct main-actor
        // write racing `applyRecording`'s queued one. Before `configure()` has
        // run there is no connection yet, so it simply records the value and
        // cannot fail; afterwards it rotates the live connection — see
        // `updateOrientation(_:)`'s doc for both branches.
        await camera.updateOrientation(mount)
    }

    /// Whether a recording is running that `owner` started.
    ///
    /// On the model, not in a `View`'s private computed properties, so the
    /// invariant is assertable without standing up a view — the same reason
    /// `LiveSessionModel` publishes §16's table as flat state.
    func isRecordingOwned(by owner: RecordingOwner) -> Bool {
        isRecording && recordingOwner == owner
    }

    /// The whole rule, written once: `owner` may start only when nothing is
    /// recording, and may stop only a recording it started itself.
    ///
    /// `applyRecording` enforces exactly this predicate, and `RecordView` asks
    /// exactly this predicate before offering its button — so what the screen
    /// offers and what the model will do cannot drift apart.
    ///
    /// Evaluate it at the moment the transition would run, never in advance:
    /// a queued transition ahead of this one can change the answer.
    func canSetRecording(_ shouldRecord: Bool, owner: RecordingOwner) -> Bool {
        shouldRecord ? !isRecording : isRecordingOwned(by: owner)
    }

    /// The single funnel. Chains onto any in-flight transition, re-reads the
    /// camera's real state at execution time, performs the start or stop only
    /// if that is actually the needed transition, and returns whether it acted.
    ///
    /// Absolute (`shouldRecord`), never a toggle: a toggle issued against a
    /// camera state that already contradicts the intent flips it the wrong
    /// way, which is how a stop could once turn into a start.
    @discardableResult
    func setRecording(_ shouldRecord: Bool, owner: RecordingOwner) async -> Bool {
        await enqueueSetRecording(shouldRecord, owner: owner).value
    }

    /// `setRecording`'s synchronous face, for callers that must fire and
    /// forget but still need the order fixed at *issue* time — `startRally()`
    /// kicks a start off without awaiting it, and a stop issued right behind
    /// it must land behind it. Wrapping `setRecording` in an unstructured
    /// `Task` at the call site would not do: the enqueue would then happen
    /// whenever that task got scheduled, so two calls could reach the chain in
    /// the opposite order from the one they were made in.
    func enqueueSetRecording(_ shouldRecord: Bool, owner: RecordingOwner) -> Task<Bool, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            guard let self else { return false }
            return await self.applyRecording(shouldRecord, owner: owner)
        }
        recordingTransition = next
        return next
    }

    /// Runs only from the chain above, so `isRecording`/`recordingOwner` read
    /// here are the camera's state *after* everything issued earlier landed.
    ///
    /// This is also where the capture mount is committed and the orientation
    /// mask pinned — see the start path below. Both halves of that (resolve,
    /// then pin only once `startRecording()` succeeded) came from the
    /// landscape-only capture change and are unchanged by the ownership
    /// funnel wrapped around them; the only difference is that a refusal
    /// returns `false` to the caller instead of returning silently.
    private func applyRecording(_ shouldRecord: Bool, owner: RecordingOwner) async -> Bool {
        guard canSetRecording(shouldRecord, owner: owner) else { return false }
        if shouldRecord {
            // Re-resolve the mount from the CURRENT interface orientation
            // here, at record start — not whatever `startCamera` last seeded —
            // because the window between camera start and record start is
            // exactly when the operator is meant to be able to flip the mount.
            // Reads `mountResolver`, the same seam `startCamera`'s seed reads,
            // rather than a second lookup of its own: a divergent second
            // lookup was already a review finding on this branch, and in a
            // test host this is also the only way to make a start deterministic
            // (see `MountResolver`).
            guard let mount = mountResolver() else {
                // The interface isn't resolved to either landscape mount
                // right now — no scene, or a portrait transient mid-rotation.
                // Pinning a guessed mount here is exactly the failure this
                // change exists to prevent, so refuse the start rather than
                // falling back to a default. `false`, so a live caller's
                // `reconcileRallyAfterStartAttempt()` turns it into a visible
                // `.failed` rally rather than a silent no-rally.
                errorText = "Hold the phone in landscape to start recording."
                return false
            }
            // Remembered so a failed attempt below can put the connection
            // back rather than leave it describing a mount no recording ever
            // committed to. Read straight from `camera`, not from
            // `captureOrientation`: this model property is intentionally not
            // updated until a recording actually starts (see below), so
            // before the first successful recording it can already differ
            // from what the connection is currently rotated to.
            let priorOrientation = camera.orientation
            // Must land, and actually apply to the live connection, before
            // startRecording() below sizes the asset writer: getting the
            // order wrong would leave the connection rotated for the old
            // mount while the writer already assumes the new one. Checked,
            // not just awaited — an unsupported angle leaves the connection
            // (and `camera.orientation`) at the previous mount, and pinning
            // or recording against that silently-still-wrong rotation would
            // be exactly the divergence this change exists to prevent.
            guard await camera.updateOrientation(mount) else {
                errorText = "Could not switch the camera to this mount."
                return false
            }
            do {
                try camera.startRecording()
                // Only commit `captureOrientation` and the pin once the
                // recording has actually started. Doing either before this
                // point and having `startRecording()` then throw would
                // strand the operator locked to this one mount with no
                // running recording to ever release it from (the stop path
                // below, and so `releaseCapturePin()`, is only reachable
                // while `isRecording` is true) — the same one-shot-per-launch
                // failure this whole change exists to remove, just relocated
                // to the failure path instead of removed.
                captureOrientation = mount
                OrientationPolicy.shared.pinForCapture(OrientationLock.pinnedMask(for: mount))
                isRecording = true
                recordingOwner = owner
                recordingStartedAt = Date()
                errorText = nil
                return true
            } catch {
                // The recording never actually started: nothing above was
                // committed (captureOrientation/pin are untouched), but the
                // connection's rotation already changed. Put it back so
                // `camera.orientation` doesn't keep describing a mount no
                // recording committed to.
                await camera.updateOrientation(priorOrientation)
                // Nothing started, so nothing is owned. The caller learns this
                // from the `false`; the live layer turns it into a visible
                // `.failed` rally via `reconcileRallyAfterStartAttempt()`.
                errorText = error.localizedDescription
                return false
            }
        }
        // Whether the write below finishes cleanly or fails, this recording is
        // over — release the pin so the operator can re-mount before the next
        // rally. Ahead of `stopRecording()`, not after it: the pin must lift on
        // the failure path too, and `stopRecording()` throws.
        OrientationPolicy.shared.releaseCapturePin()
        do {
            let url = try await camera.stopRecording()
            let duration = recordingStartedAt.map { Date().timeIntervalSince($0) } ?? 0
            isRecording = false
            recordingOwner = nil
            recordingStartedAt = nil
            finishedClip = FinishedClip(url: url, duration: duration)
            finishedClipOwner = owner
            return true
        } catch {
            // The recording is over either way (`stopRecording()` clears the
            // writer before it can throw), so this counts as having acted —
            // there is just no clip to hand anyone.
            isRecording = false
            recordingOwner = nil
            recordingStartedAt = nil
            errorText = error.localizedDescription
            return true
        }
    }

    #if DEBUG
    /// Test seam for clip routing. A test host has no capture session, so no
    /// frame ever reaches the writer and a real stop always ends in
    /// `CameraError.recordingEmpty` — the success branch of
    /// `applyRecording`'s stop, the one that publishes a clip, is unreachable
    /// outside a device. This publishes exactly what that branch would, so the
    /// owner-routing rule itself can still be asserted.
    func publishFinishedClipForTesting(_ clip: FinishedClip, owner: RecordingOwner) {
        finishedClip = clip
        finishedClipOwner = owner
    }
    #endif
}

final class RemoteDetectionStore {
    private let lock = NSLock()
    private var buffer = RingBuffer<DetectionTuple>(capacity: BallTracker.bufferCapacity)

    func append(_ tuples: [DetectionTuple]) {
        lock.lock(); defer { lock.unlock() }
        for tuple in tuples { buffer.append(tuple) }
    }

    var recent: [DetectionTuple] {
        lock.lock(); defer { lock.unlock() }
        return buffer.elements
    }
}

enum DetectionMapper {
    /// Vision-normalized (bottom-left) rect → pixel tuple in the LOCAL
    /// frame (frameW/frameH). y flips to top-left row for rolling shutter.
    static func tuple(seq: UInt32, observation: BallObservation,
                      frameW: Int, frameH: Int) -> DetectionTuple {
        DetectionTuple(
            seq: seq,
            ptsNs: UInt64(observation.timestamp * 1_000_000_000),
            x: Float(observation.rect.midX) * Float(frameW),
            y: Float(1 - observation.rect.midY) * Float(frameH),
            conf: Float16(observation.confidence),
            bboxH: Float16(Float(observation.rect.height) * Float(frameH)))
    }
}
