// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Sources/Live/StereoDemo.swift
#if DEBUG
import Foundation
import simd

/// A model-free, phone-free stereo rally for exercising the live surfaces.
///
/// Two things make this honest rather than a mock:
///
///  1. The camera pair is the **validated golden pair** verbatim — two 1920×1080
///     cameras 3 ft apart on the back wall, the same geometry the engine's
///     parity tests run against. Hand-authored rotation matrices would be a
///     silent-wrong-answer machine.
///  2. The trajectory is generated in **court feet and projected through both
///     models**, so the two eyes see a physically consistent ball and
///     triangulation recovers the point it started from. Feeding one synthetic
///     pixel track to both eyes would triangulate to nonsense — and
///     `StereoEngine.process` needs *both* eyes populated to emit anything at
///     all, so a local-only synthetic detector produces zero impacts forever.
///
/// The models are used **unadopted**: the demo works entirely in the goldens'
/// own 1920×1080 space. `adoptedForCapture()` would not throw here — 1920×1080
/// and the 4K capture target share the same 16:9 aspect, so it is a clean 2x
/// scale — but the synthetic detections below are produced in the goldens'
/// own 1920×1080 space too, so adopting the models while leaving the
/// detections unadopted would mismatch the two.
enum StereoDemo {
    /// Golden left camera, back wall at x = 9 ft.
    static let localModelJSON = """
    {"camera_center_ft":[9.0,31.95,7.0],"center_px":[960.0,540.0],\
    "distortion":null,"focal_px":1600.0,"frame_height":1080.0,"frame_width":1920.0,\
    "rotation":[[-0.9988997444076146,-0.046896701615380974,0.0],\
    [-0.0029266849359595812,0.062338389135939073,-0.9980507801485964],\
    [0.04680528963362692,-0.9969526691962534,-0.06240705284483589]]}
    """

    /// Golden right camera, back wall at x = 12 ft — a 3 ft baseline.
    static let remoteModelJSON = """
    {"camera_center_ft":[12.0,31.95,7.0],"center_px":[960.0,540.0],\
    "distortion":null,"focal_px":1600.0,"frame_height":1080.0,"frame_width":1920.0,\
    "rotation":[[-0.9988997444076146,0.046896701615380974,0.0],\
    [0.0029266849359595812,0.062338389135939073,-0.9980507801485964],\
    [-0.04680528963362692,-0.9969526691962534,-0.06240705284483589]]}
    """

    static let sampleRateHz = 60.0
    static let durationS = 0.8
    /// The ball reaches the front wall here — a clean velocity reversal in y,
    /// which is the kink impact detection looks for.
    static let impactAtS = 0.4

    /// A ball driven into the front wall and rebounding, in court feet.
    ///
    /// At `impactAtS` it sits at y ≈ 0.25 ft (on the front wall) and
    /// z ≈ 5 ft — comfortably between the tin (1.58 ft) and the out line
    /// (15 ft), so the demo resolves to a confident IN rather than an edge
    /// case. x and z drift slightly so no axis fit is degenerate.
    static func courtTrack() -> [(tS: Double, pointFt: SIMD3<Double>)] {
        let count = Int(durationS * sampleRateHz)
        return (0...count).map { index in
            let t = Double(index) / sampleRateHz
            let point = SIMD3(9.0 + 1.5 * t,
                              abs(impactAtS - t) * 12.0 + 0.25,
                              5.5 - 1.2 * t)
            return (t, point)
        }
    }

    /// The court track as each eye sees it. Points behind a camera project to
    /// nil and are dropped — none should be, for this trajectory.
    static func pixelTracks(local: CameraModel, remote: CameraModel)
        -> (local: [(tS: Double, px: SIMD2<Double>)],
            remote: [(tS: Double, px: SIMD2<Double>)]) {
        var localTrack: [(tS: Double, px: SIMD2<Double>)] = []
        var remoteTrack: [(tS: Double, px: SIMD2<Double>)] = []
        for sample in courtTrack() {
            if let px = local.project(sample.pointFt) {
                localTrack.append((sample.tS, px))
            }
            if let px = remote.project(sample.pointFt) {
                remoteTrack.append((sample.tS, px))
            }
        }
        return (localTrack, remoteTrack)
    }

    /// The remote eye packaged as wire tuples, so the demo drives the same
    /// `addRemote` path a real secondary phone would.
    static func remoteTuples(_ track: [(tS: Double, px: SIMD2<Double>)]) -> [DetectionTuple] {
        track.enumerated().map { index, sample in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64(sample.tS * 1_000_000_000),
                           x: Float(sample.px.x), y: Float(sample.px.y),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
    }
}
#endif
