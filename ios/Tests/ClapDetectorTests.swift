import XCTest
@testable import SquashLineCalling

final class ClapDetectorTests: XCTestCase {
    private let rate = 44_100.0

    private func quiet(_ seconds: Double) -> [Float] {
        (0..<Int(seconds * rate)).map { _ in Float.random(in: -0.004...0.004) }
    }

    /// Sharp transient at a known sample index.
    private func withClap(at second: Double, total: Double) -> [Float] {
        var samples = quiet(total)
        let start = Int(second * rate)
        for i in 0..<Int(0.02 * rate) {   // 20 ms burst, decaying
            samples[start + i] = 0.8 * Float(pow(0.9995, Double(i))) * (i.isMultiple(of: 2) ? 1 : -1)
        }
        return samples
    }

    func testQuietAudioNeverTriggers() {
        let detector = ClapDetector()
        XCTAssertNil(detector.process(samples: quiet(2.0), startTime: 100.0, sampleRate: rate))
    }

    func testClapOnsetWithinTwoMilliseconds() {
        let detector = ClapDetector()
        _ = detector.process(samples: quiet(1.0), startTime: 100.0, sampleRate: rate)   // learn floor
        let onset = detector.process(samples: withClap(at: 0.5, total: 1.0),
                                     startTime: 101.0, sampleRate: rate)
        XCTAssertNotNil(onset)
        XCTAssertEqual(onset ?? -1, 101.5, accuracy: 0.002)
    }

    func testRefractorySuppressesDoubleFire() {
        let detector = ClapDetector()
        _ = detector.process(samples: quiet(1.0), startTime: 100.0, sampleRate: rate)
        var samples = withClap(at: 0.1, total: 1.0)
        let echo = withClap(at: 0.3, total: 1.0)   // second transient inside 500 ms
        for i in 0..<samples.count { samples[i] = max(samples[i], echo[i]) }
        var onsets: [Double] = []
        // Feed in 100 ms chunks like the capture path would.
        let chunk = Int(0.1 * rate)
        for (index, start) in stride(from: 0, to: samples.count, by: chunk).enumerated() {
            let slice = Array(samples[start ..< min(start + chunk, samples.count)])
            if let t = detector.process(samples: slice,
                                        startTime: 101.0 + Double(index) * 0.1,
                                        sampleRate: rate) { onsets.append(t) }
        }
        XCTAssertEqual(onsets.count, 1)
    }
}
