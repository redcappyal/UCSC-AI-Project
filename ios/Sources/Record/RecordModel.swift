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
