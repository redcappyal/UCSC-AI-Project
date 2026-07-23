import XCTest
import CoreVideo
@testable import SquashLineCalling

private final class ScriptedDetector: BallDetecting {
    var results: [BallObservation?]
    init(results: [BallObservation?]) { self.results = results }
    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        results.isEmpty ? nil : results.removeFirst()
    }
}

final class BallTrackerTests: XCTestCase {
    private func pixelBuffer() -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &buffer)
        return buffer!
    }

    private func observation(_ t: TimeInterval) -> BallObservation {
        BallObservation(timestamp: t,
                        rect: CGRect(x: 0.4, y: 0.5, width: 0.02, height: 0.02),
                        confidence: 0.9)
    }

    func testHitNotifiesSubscribersAndBuffers() {
        let tracker = BallTracker(detector: ScriptedDetector(
            results: [observation(1.0), nil, observation(2.0)]))
        var received: [BallObservation] = []
        let expectation = expectation(description: "two notifications")
        expectation.expectedFulfillmentCount = 2
        tracker.subscribe { received.append($0); expectation.fulfill() }

        let buffer = pixelBuffer()
        tracker.process(buffer, timestamp: 1.0)   // hit
        tracker.process(buffer, timestamp: 1.5)   // miss: no notify, no buffer
        tracker.process(buffer, timestamp: 2.0)   // hit

        wait(for: [expectation], timeout: 1.0)
        XCTAssertEqual(received.map(\.timestamp), [1.0, 2.0])
        XCTAssertEqual(tracker.recent.map(\.timestamp), [1.0, 2.0])
    }

    func testNilDetectorDisablesTracking() {
        let tracker = BallTracker(detector: nil)
        XCTAssertFalse(tracker.isEnabled)
        tracker.process(pixelBuffer(), timestamp: 1.0)
        XCTAssertTrue(tracker.recent.isEmpty)
    }
}
