// ios/Sources/Stereo/StereoEngine.swift
import Foundation
import simd

enum StereoEvent: Equatable {
    case impact(StereoImpact)
}

/// Live stereo fusion on the primary phone. Owner-pumped: call
/// `processIfDue` on any cadence ≥ processIntervalS (RecordModel's peer pump
/// is the natural driver). All mutable state is confined to `queue`;
/// `onEvent` fires on `queue` — hop before touching UI.
final class StereoEngine {
    static let windowS = 4.0
    static let processHz = 120.0
    static let processIntervalS = 0.25
    static let emitDedupeS = 0.05

    var onEvent: ((StereoEvent) -> Void)?

    private let queue = DispatchQueue(label: "slc.stereo.engine")
    private let localModel: CameraModel
    private let remoteModel: CameraModel
    private let remoteToLocal: (Double) -> Double?
    private var localSamples: [TrackSample] = []
    private var remoteSamples: [TrackSample] = []
    private var emitted: [(tS: Double, surface: String)] = []
    private var lastProcessAt: TimeInterval = -.infinity

    init(localModel: CameraModel, remoteModel: CameraModel,
         remoteToLocal: @escaping (Double) -> Double?) {
        self.localModel = localModel
        self.remoteModel = remoteModel
        self.remoteToLocal = remoteToLocal
    }

    func addLocalObservation(_ observation: BallObservation, frameW: Int, frameH: Int) {
        let px = SIMD2(Double(observation.rect.midX) * Double(frameW),
                       (1.0 - Double(observation.rect.midY)) * Double(frameH))
        addLocalPixel(px, tS: observation.timestamp)
    }

    func addLocalPixel(_ px: SIMD2<Double>, tS: Double) {
        queue.async {
            self.localSamples.append(TrackSample(tS: tS, px: px))
            self.prune()
        }
    }

    func addRemote(_ tuples: [DetectionTuple]) {
        queue.async {
            for tuple in tuples {
                guard let tS = self.remoteToLocal(Double(tuple.ptsNs) / 1_000_000_000.0) else {
                    continue
                }
                self.remoteSamples.append(
                    TrackSample(tS: tS, px: SIMD2(Double(tuple.x), Double(tuple.y))))
            }
            self.prune()
        }
    }

    func processIfDue(now: TimeInterval) {
        queue.async {
            guard now - self.lastProcessAt >= Self.processIntervalS else { return }
            self.lastProcessAt = now
            self.process()
        }
    }

    /// Test seam: synchronous barrier so assertions observe queued work.
    func flushForTesting() {
        queue.sync {}
    }

    private func prune() {
        let newest = max(localSamples.last?.tS ?? -.infinity,
                         remoteSamples.last?.tS ?? -.infinity)
        guard newest.isFinite else { return }
        let horizon = newest - Self.windowS
        localSamples.removeAll { $0.tS < horizon }
        remoteSamples.removeAll { $0.tS < horizon }
        emitted.removeAll { $0.tS < horizon }
    }

    private func process() {
        guard let localFirst = localSamples.first?.tS,
              let localLast = localSamples.last?.tS,
              let remoteFirst = remoteSamples.first?.tS,
              let remoteLast = remoteSamples.last?.tS else { return }
        let tLo = max(localFirst, remoteFirst)
        let tHi = min(localLast, remoteLast)
        guard tHi > tLo else { return }
        // np.arange-equivalent grid: tLo + i/hz for i in 0..<count, strictly < tHi.
        let count = Int(((tHi - tLo) * Self.processHz).rounded(.down))
        guard count >= 3 else { return }
        let timeline = (0..<count).map { tLo + Double($0) / Self.processHz }
        let impacts = StereoTrack.detectImpacts(localModel, localSamples,
                                                remoteModel, remoteSamples,
                                                timelineS: timeline)
        for impact in impacts {
            let isDuplicate = emitted.contains {
                $0.surface == impact.surface && abs($0.tS - impact.tS) < Self.emitDedupeS
            }
            if isDuplicate { continue }
            emitted.append((impact.tS, impact.surface))
            onEvent?(.impact(impact))
        }
    }
}
