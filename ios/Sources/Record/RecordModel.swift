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

    init(detector: BallDetecting? = CoreMLBallDetector()) {
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
                    DispatchQueue.main.async { self?.appendStereoEvent(json) }
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
