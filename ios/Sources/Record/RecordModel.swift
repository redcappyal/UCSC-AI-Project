import Foundation

struct FinishedClip: Identifiable {
    let id = UUID()
    let url: URL
    let duration: Double
}

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
    @Published var finishedClip: FinishedClip?   // non-nil presents ResultsView

    private static let trailLength = 15

    var detectorMissing: Bool { !tracker.isEnabled }

    enum DetectorKind: Equatable { case none, synthetic, model }

    /// `detectorMissing` only knows nil-vs-non-nil, so it cannot tell a real
    /// model from the DEBUG stand-in. The UI must never imply a real detection
    /// when the source is synthetic.
    @Published private(set) var detectorKind: DetectorKind

    init(detector: BallDetecting? = CoreMLBallDetector()) {
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
    // Must match the Hello this device advertises (PeerSession) — the peer
    // reads detections in these pixel units, so a mismatch skews stereo.
    private let peerFrameW = CaptureSettings.frameWidth
    private let peerFrameH = CaptureSettings.frameHeight

    // MARK: stereo (Phase 3, Plan B2)

    private var localModel: CameraModel?
    private var stereoEngine: StereoEngine?
    /// Newest-first, capped at 20. Populated on the primary as its engine
    /// emits impacts, and mirrored on the secondary from relayed .event
    /// messages — same list either way, so the UI doesn't need to know
    /// which role it's running as.
    @Published var stereoEvents: [String] = []
    private static let stereoEventsCap = 20

    // MARK: live call rendering (Phase 4)

    /// What `p-live` should be showing, or nil for "nothing called yet".
    ///
    /// Set from the engine's `onEvent` after a main hop. Deliberately never
    /// derived from `stereoEvents`: that list is relayed JSON, and re-parsing
    /// it would be a second place for the `no_call` gate to be forgotten.
    @Published private(set) var livePresentation: CallPresentation?

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
                                                  frameW: self.peerFrameW, frameH: self.peerFrameH)
                self.nextDetectionSeq += 1
                self.pendingTuples.append(tuple)
                let now = observation.timestamp
                if self.pendingTuples.count >= 2 || now - self.lastFlushAt >= 0.030 {
                    peer.sendDetections(self.pendingTuples)
                    self.pendingTuples.removeAll(keepingCapacity: true)
                    self.lastFlushAt = now
                }
            } else if peer.role == .primary, let engine = self.stereoEngine {
                engine.addLocalObservation(observation, frameW: self.peerFrameW, frameH: self.peerFrameH)
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
                    guard case .impact(let impact) = event else { return }
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
                        self?.showFlash(presentation)
                    }
                }
                self.stereoEngine = engine
            }
        }
        // Secondary-side mirror: relayed events land here (never fires on
        // the primary — it never receives .event, only sends it). Also
        // fires off-main; only touches stereoEvents, already hopped.
        peer?.onEvent = { [weak self] _, json in
            DispatchQueue.main.async { self?.appendStereoEvent(json) }
        }
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
            guard case .impact(let impact) = event else { return }
            let presentation = CallPresentation.from(impact)
            DispatchQueue.main.async {
                self?.livePresentation = presentation
                self?.showFlash(presentation)
            }
        }

        let tracks = StereoDemo.pixelTracks(local: local, remote: remote)
        for sample in tracks.local { engine.addLocalPixel(sample.px, tS: sample.tS) }
        engine.addRemote(StereoDemo.remoteTuples(tracks.remote))

        stereoEngine = engine
        livePresentation = nil
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

    func startCamera() async {
        do {
            try await camera.configure()
            camera.start()
            // Meter the court once from the mounted position, then freeze
            // exposure/WB/focus for the session. Everything after this point
            // is shot under identical conditions, which is what keeps the
            // footage usable as training data.
            exposureNote = CaptureSettings.summary(for: try await camera.lockForCourt())
        } catch {
            errorText = error.localizedDescription
        }
    }

    func toggleRecording() async {
        if isRecording {
            do {
                let url = try await camera.stopRecording()
                let duration = recordingStartedAt.map {
                    Date().timeIntervalSince($0)
                } ?? 0
                isRecording = false
                recordingStartedAt = nil
                finishedClip = FinishedClip(url: url, duration: duration)
            } catch {
                isRecording = false
                recordingStartedAt = nil
                errorText = error.localizedDescription
            }
        } else {
            do {
                try camera.startRecording()
                isRecording = true
                recordingStartedAt = Date()
                errorText = nil
            } catch {
                errorText = error.localizedDescription
            }
        }
    }
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
