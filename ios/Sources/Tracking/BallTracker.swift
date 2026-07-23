import CoreVideo
import Foundation

struct BallObservation: Equatable {
    let timestamp: TimeInterval
    /// Vision-normalized bounding box: [0,1] with origin at BOTTOM-left.
    let rect: CGRect
    let confidence: Float
}

protocol BallDetecting {
    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation?
}

/// One producer (the capture queue), many consumers. v1 consumers: the live
/// overlay and a ring buffer. The real-time in/out phase adds a bounce
/// detector as a third subscriber — that is the whole migration path, so
/// keep this class free of UI or network concerns.
final class BallTracker {
    static let bufferCapacity = 900   // ~30 s at 30 fps

    private let detector: BallDetecting?
    private let lock = NSLock()
    private var buffer = RingBuffer<BallObservation>(capacity: BallTracker.bufferCapacity)
    private var subscribers: [(BallObservation) -> Void] = []

    var isEnabled: Bool { detector != nil }

    init(detector: BallDetecting?) {
        self.detector = detector
    }

    func subscribe(_ subscriber: @escaping (BallObservation) -> Void) {
        lock.lock(); defer { lock.unlock() }
        subscribers.append(subscriber)
    }

    var recent: [BallObservation] {
        lock.lock(); defer { lock.unlock() }
        return buffer.elements
    }

    /// Called on the capture queue for every frame.
    func process(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        guard let observation = detector?.detect(pixelBuffer, timestamp: timestamp) else { return }
        lock.lock()
        buffer.append(observation)
        let currentSubscribers = subscribers
        lock.unlock()
        DispatchQueue.main.async {
            for subscriber in currentSubscribers { subscriber(observation) }
        }
    }
}
