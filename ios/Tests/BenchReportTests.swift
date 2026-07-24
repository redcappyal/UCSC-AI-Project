import XCTest
@testable import SquashLineCalling

final class BenchReportTests: XCTestCase {
    func testPercentilesAndLoss() {
        let rtts = (1...100).map { Double($0) }   // 1..100 ms
        let report = BenchReport.build(
            transport: "loopback", startedAt: Date(timeIntervalSince1970: 0),
            durationS: 60, rtts: rtts, sent: 200, received: 150,
            estimate: ClockSyncEstimate(offset: 0.0021, uncertainty: 0.0009, sampleCount: 40),
            anchorDelta: 0.0004, thermal: ["nominal"])
        XCTAssertEqual(report.rttMedianMs, 50.5, accuracy: 1.0)
        XCTAssertEqual(report.rttP95Ms, 95.0, accuracy: 1.0)
        XCTAssertEqual(report.rttMaxMs, 100.0)
        XCTAssertEqual(report.lossPercent, 25.0)
        XCTAssertEqual(report.offsetMs, 2.1, accuracy: 1e-9)
        XCTAssertEqual(report.clapDeltaMs ?? -1, 0.4, accuracy: 1e-9)
    }

    func testEmptyRTTsProduceZeros() {
        let report = BenchReport.build(transport: "ble", startedAt: Date(), durationS: 0,
                                       rtts: [], sent: 0, received: 0,
                                       estimate: nil, anchorDelta: nil, thermal: [])
        XCTAssertEqual(report.rttMedianMs, 0)
        XCTAssertEqual(report.lossPercent, 0)
    }
}
