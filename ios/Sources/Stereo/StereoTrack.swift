// ios/Sources/Stereo/StereoTrack.swift
import Foundation
import simd

struct TrackSample { let tS: Double; let px: SIMD2<Double> }
struct TrackPoint3D { let tS: Double; let pointFt: SIMD3<Double>; let gapFt: Double }

struct StereoImpact: Equatable {
    let tS: Double
    let surface: String
    let pointFt: SIMD3<Double>
    let call: String
    let marginFt: Double
    let confidence: String
    let snapDisagreementFt: Double?
}

/// Track interpolation + impact detection — mirrors stereo_engine.py
/// (eval_pixel_track / build_track3d / detect_impacts) exactly, including
/// quirks. Do not "fix" behavior here without changing the Python authority.
enum StereoTrack {
    static let fitWindowSamples = 7
    static let minFitSamples = 4
    static let windowGapRatioMax = 3.0
    static let snapDisagreementMaxFt = 0.3
    static let impactProximityFt = 1.5
    static let impactMergeS = 0.060
    static let preImpactWindowS = 0.25
    static let preImpactGuardS = 1.0 / 240.0

    static func evalPixelTrack(_ samples: [TrackSample], tS: Double,
                               window: Int = fitWindowSamples) -> SIMD2<Double>? {
        guard !samples.isEmpty else { return nil }
        let nearest = samples.sorted { abs($0.tS - tS) < abs($1.tS - tS) }.prefix(window)
        guard nearest.count >= minFitSamples else { return nil }
        // Locality guard — mirrors stereo_engine.py eval_pixel_track: inert
        // when the median internal gap is 0 (duplicate timestamps), by design.
        let sortedTs = nearest.map(\.tS).sorted()
        let gaps = zip(sortedTs.dropFirst(), sortedTs).map(-)
        if !gaps.isEmpty {
            let sortedGaps = gaps.sorted()
            let median = sortedGaps.count % 2 == 1
                ? sortedGaps[sortedGaps.count / 2]
                : (sortedGaps[sortedGaps.count / 2 - 1] + sortedGaps[sortedGaps.count / 2]) / 2.0
            if median > 0.0, sortedGaps.last! > windowGapRatioMax * median { return nil }
        }
        let ts = nearest.map { $0.tS - tS }
        func fitAtZero(_ values: [Double]) -> Double {
            var s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0
            var b0 = 0.0, b1 = 0.0, b2 = 0.0
            for (t, v) in zip(ts, values) {
                let t2 = t * t
                s0 += 1; s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2
                b0 += v; b1 += v * t; b2 += v * t2
            }
            // Normal equations for v = c0 + c1·t + c2·t²; value at t=0 is c0.
            let m = simd_double3x3(rows: [SIMD3(s0, s1, s2),
                                          SIMD3(s1, s2, s3),
                                          SIMD3(s2, s3, s4)])
            let coeffs = m.inverse * SIMD3(b0, b1, b2)
            return coeffs.x
        }
        return SIMD2(fitAtZero(nearest.map(\.px.x)), fitAtZero(nearest.map(\.px.y)))
    }

    static func buildTrack3D(_ a: CameraModel, _ samplesA: [TrackSample],
                             _ b: CameraModel, _ samplesB: [TrackSample],
                             timelineS: [Double]) -> [TrackPoint3D] {
        guard !samplesA.isEmpty, !samplesB.isEmpty else { return [] }
        var track: [TrackPoint3D] = []
        for tS in timelineS {
            guard let pxA = evalPixelTrack(samplesA, tS: tS),
                  let pxB = evalPixelTrack(samplesB, tS: tS),
                  let result = StereoMath.triangulate(a, b, pxA: pxA, pxB: pxB) else {
                continue
            }
            track.append(TrackPoint3D(tS: tS, pointFt: result.point, gapFt: result.gapFt))
        }
        return track
    }

    private static func preImpactEval(_ samples: [TrackSample], tImpact: Double)
        -> SIMD2<Double>? {
        let window = samples.filter {
            $0.tS >= tImpact - preImpactWindowS && $0.tS <= tImpact - preImpactGuardS
        }
        guard window.count >= minFitSamples else { return nil }
        return evalPixelTrack(window, tS: tImpact, window: window.count)
    }

    static func detectImpacts(_ a: CameraModel, _ samplesA: [TrackSample],
                              _ b: CameraModel, _ samplesB: [TrackSample],
                              timelineS: [Double]) -> [StereoImpact] {
        let track = buildTrack3D(a, samplesA, b, samplesB, timelineS: timelineS)
        guard track.count >= 3 else { return [] }
        var impacts: [StereoImpact] = []
        for surface in StereoMath.surfaces {
            let dists = track.map { StereoMath.planeDistance(surface: surface, point: $0.pointFt) }
            for i in 1..<(track.count - 1) {
                guard dists[i] <= impactProximityFt,
                      dists[i] <= dists[i - 1], dists[i] <= dists[i + 1],
                      dists[i] - dists[i - 1] < 0.0, dists[i + 1] - dists[i] > 0.0 else {
                    continue
                }
                let tImpact = track[i].tS
                var snapA: SIMD3<Double>?
                var snapB: SIMD3<Double>?
                if let px = preImpactEval(samplesA, tImpact: tImpact) {
                    snapA = StereoMath.snapToPlane(a, px: px, surface: surface)
                }
                if let px = preImpactEval(samplesB, tImpact: tImpact) {
                    snapB = StereoMath.snapToPlane(b, px: px, surface: surface)
                }
                var disagreement: Double?
                let confidence: String
                let point: SIMD3<Double>
                if let sa = snapA, let sb = snapB {
                    let gap = simd_length(sa - sb)
                    disagreement = gap
                    confidence = gap <= snapDisagreementMaxFt ? "high" : "one_view"
                    point = StereoMath.fuseSnaps(sa, sb)!
                } else if let lone = StereoMath.fuseSnaps(snapA, snapB) {
                    confidence = "one_view"
                    point = lone
                } else {
                    confidence = "no_call"
                    point = track[i].pointFt
                }
                let (call, margin) = StereoMath.callForImpact(surface: surface, point: point)
                impacts.append(StereoImpact(tS: tImpact, surface: surface, pointFt: point,
                                            call: call, marginFt: margin,
                                            confidence: confidence,
                                            snapDisagreementFt: disagreement))
            }
        }
        impacts.sort { ($0.tS, $0.surface) < ($1.tS, $1.surface) }
        var merged: [StereoImpact] = []
        for impact in impacts {
            if let last = merged.last, last.surface == impact.surface,
               impact.tS - last.tS < impactMergeS {
                let prevDepth = abs(StereoMath.planeDistance(surface: impact.surface,
                                                             point: last.pointFt))
                let curDepth = abs(StereoMath.planeDistance(surface: impact.surface,
                                                            point: impact.pointFt))
                if curDepth < prevDepth { merged[merged.count - 1] = impact }
                continue
            }
            merged.append(impact)
        }
        return merged
    }
}
