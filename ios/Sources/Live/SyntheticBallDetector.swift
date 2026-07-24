#if DEBUG
import CoreVideo
import Foundation

/// A deterministic ballistic arc in Vision-normalized space (origin
/// bottom-left). Ignores the pixel buffer: it exists so the pairing, live-call
/// and replay surfaces can be exercised before the Core ML model exists.
/// Never compiled into a release build.
final class SyntheticBallDetector: BallDetecting {
    private let period: TimeInterval
    private let ballSize: Double

    init(period: TimeInterval = 2.0, ballSize: Double = 0.012) {
        self.period = max(0.1, period)
        self.ballSize = ballSize
    }

    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        let phase = (timestamp.truncatingRemainder(dividingBy: period)) / period
        let u = 0.1 + 0.8 * phase
        // Apex mid-flight; Vision's y grows upward, so the arc peaks high.
        let v = min(0.98, max(0.02, 0.2 + 4.0 * 0.6 * phase * (1.0 - phase)))
        let rect = CGRect(x: u - ballSize / 2, y: v - ballSize / 2,
                          width: ballSize, height: ballSize)
        return BallObservation(timestamp: timestamp, rect: rect, confidence: 0.9)
    }
}
#endif
