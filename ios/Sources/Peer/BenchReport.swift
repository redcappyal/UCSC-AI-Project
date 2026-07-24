import Foundation

struct BenchReport: Codable {
    var transport: String
    var startedAt: Date
    var durationS: Double
    var rttMedianMs: Double
    var rttP95Ms: Double
    var rttMaxMs: Double
    var datagramsSent: Int
    var datagramsReceived: Int
    var lossPercent: Double
    var offsetMs: Double
    var offsetUncertaintyMs: Double
    var clapDeltaMs: Double?
    var thermalStates: [String]

    static func build(transport: String, startedAt: Date, durationS: Double,
                      rtts: [Double], sent: Int, received: Int,
                      estimate: ClockSyncEstimate?, anchorDelta: Double?,
                      thermal: [String]) -> BenchReport {
        let sorted = rtts.sorted()
        func percentile(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let index = min(sorted.count - 1, Int(Double(sorted.count) * p))
            return sorted[index]
        }
        let median = sorted.isEmpty ? 0 :
            (sorted.count.isMultiple(of: 2)
                ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
                : sorted[sorted.count / 2])
        return BenchReport(
            transport: transport, startedAt: startedAt, durationS: durationS,
            rttMedianMs: median, rttP95Ms: percentile(0.95), rttMaxMs: sorted.last ?? 0,
            datagramsSent: sent, datagramsReceived: received,
            lossPercent: sent == 0 ? 0 : Double(sent - received) / Double(sent) * 100,
            offsetMs: (estimate?.offset ?? 0) * 1000,
            offsetUncertaintyMs: (estimate?.uncertainty ?? 0) * 1000,
            clapDeltaMs: anchorDelta.map { $0 * 1000 },
            thermalStates: thermal)
    }
}
