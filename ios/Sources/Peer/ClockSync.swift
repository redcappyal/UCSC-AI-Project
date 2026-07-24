// ios/Sources/Peer/ClockSync.swift
import CoreMedia
import Foundation

struct ClockSyncEstimate: Equatable {
    var offset: Double
    var uncertainty: Double
    var sampleCount: Int
}

/// Pure estimator: feed NTP-style quadruples, read an offset. All times are
/// seconds on each device's capture host clock. "remote − local" sign.
final class ClockSync {
    static let minimumSamples = 5
    static let windowSize = 200
    static let anchorUncertainty = 0.0005   // 0.5 ms: acoustic resolution + T geometry

    private struct Sample { var rtt: Double; var theta: Double }
    private let lock = NSLock()
    private var samples: [Sample] = []
    private var anchor: Double?              // authoritative offset from clap

    static func hostNow() -> TimeInterval {
        CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
    }

    func addSample(t1: Double, t2: Double, t3: Double, t4: Double) {
        let rtt = (t4 - t1) - (t3 - t2)
        guard rtt >= 0 else { return }       // clock weirdness: discard
        lock.lock(); defer { lock.unlock() }
        samples.append(Sample(rtt: rtt, theta: ((t2 - t1) + (t3 - t4)) / 2))
        if samples.count > ClockSync.windowSize { samples.removeFirst() }
    }

    func applyAnchor(localEventTime: Double, remoteEventTime: Double) {
        lock.lock(); defer { lock.unlock() }
        anchor = remoteEventTime - localEventTime
    }

    /// Non-nil once ≥ 5 network samples have accumulated — or immediately
    /// once an anchor is applied, which is authoritative regardless of
    /// sample count.
    var estimate: ClockSyncEstimate? {
        lock.lock(); defer { lock.unlock() }
        return computeEstimate()
    }

    func remoteToLocal(_ tRemote: Double) -> Double? {
        lock.lock(); defer { lock.unlock() }
        guard let estimate = computeEstimate() else { return nil }
        return tRemote - estimate.offset
    }

    /// Unlocked; callers must hold `lock`.
    private func computeEstimate() -> ClockSyncEstimate? {
        if let anchor {
            return ClockSyncEstimate(offset: anchor,
                                     uncertainty: ClockSync.anchorUncertainty,
                                     sampleCount: samples.count)
        }
        guard samples.count >= ClockSync.minimumSamples else { return nil }
        let best = samples.sorted { $0.rtt < $1.rtt }.prefix(5)
        let thetas = best.map(\.theta).sorted()
        let median = thetas[thetas.count / 2]
        let spread = (thetas.last ?? 0) - (thetas.first ?? 0)
        let minRTT = best.first.map(\.rtt) ?? 0
        return ClockSyncEstimate(offset: median,
                                 uncertainty: max(minRTT / 2, spread),
                                 sampleCount: samples.count)
    }
}
