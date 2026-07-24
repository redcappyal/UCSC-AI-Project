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
    private let peerFrameW = 1080, peerFrameH = 1920   // matches Hello until Phase 4

    /// Wire a paired session. Safe to call again with a new session; the
    /// tracker subscription is registered once. Subscriber runs on the main
    /// queue (BallTracker's fan-out queue). The timer pump keeps
    /// heartbeats/sync alive even when no ball is detected — a primary with
    /// zero local detections must still tick.
    func attachPeer(_ peer: PeerSession) {
        self.peer = peer
        peer.onRemoteDetections = { [weak self] tuples in
            self?.remoteDetections.append(tuples)
        }
        peerPumpTimer?.invalidate()
        peerPumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak peer] _ in
            peer?.tick(now: ClockSync.hostNow())
        }
        // A new peer session expects a fresh seq space.
        nextDetectionSeq = 0
        pendingTuples.removeAll()
        lastFlushAt = 0
        guard !peerSubscribed else { return }
        peerSubscribed = true
        tracker.subscribe { [weak self] observation in
            guard let self, let peer = self.peer, peer.role == .secondary else { return }
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
        }
    }

    deinit {
        peerPumpTimer?.invalidate()
    }

    func startCamera() async {
        do {
            try await camera.configure()
            camera.start()
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
