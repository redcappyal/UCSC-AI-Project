// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import Foundation
import simd

/// Division-model lens distortion — the JS<->Python<->Swift contract
/// (mirrors court_model.undistort_point).
///
/// Invariant: `normPx` is always > 0. Python resolves court_model's
/// `or 1000.0` fallback lazily, at each use; Swift resolves it once, in
/// `CameraModel.fromJSON`, so every later consumer (`undistort`, `scaled`)
/// can use the stored value directly and still agree with Python.
struct Distortion {
    let k1: Double
    let centerPx: SIMD2<Double>
    let normPx: Double
}

/// Calibrated pinhole camera in court coordinates (FEET). Mirrors
/// court_model.CameraModel: `project` returns UNDISTORTED pixels and `ray`
/// expects them; callers undistort raw observations first.
struct CameraModel {
    let focalPx: Double
    let centerPx: SIMD2<Double>
    let rotation: simd_double3x3        // world -> camera
    let cameraCenterFt: SIMD3<Double>
    let distortion: Distortion?
    /// The pixel space every px-valued field above is expressed in — the
    /// resolution the model was SOLVED at, not the one it is being used at.
    /// nil for a model stored before court_model recorded it; unknown must
    /// stay unknown, because assuming a default is exactly the silent
    /// failure this field exists to prevent (see `scaled(toWidth:height:)`).
    let frameWidth: Double?
    let frameHeight: Double?

    init(focalPx: Double, centerPx: SIMD2<Double>, rotation: simd_double3x3,
         cameraCenterFt: SIMD3<Double>, distortion: Distortion?,
         frameWidth: Double? = nil, frameHeight: Double? = nil) {
        self.focalPx = focalPx
        self.centerPx = centerPx
        self.rotation = rotation
        self.cameraCenterFt = cameraCenterFt
        self.distortion = distortion
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
    }

    enum DecodeError: Error { case malformed(String) }

    static func fromJSON(_ data: Data) throws -> CameraModel {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let focal = obj["focal_px"] as? Double,
              let center = obj["center_px"] as? [Double], center.count == 2,
              let rows = obj["rotation"] as? [[Double]], rows.count == 3,
              rows.allSatisfy({ $0.count == 3 }),
              let cc = obj["camera_center_ft"] as? [Double], cc.count == 3 else {
            throw DecodeError.malformed("missing or malformed camera model fields")
        }
        var distortion: Distortion?
        if let d = obj["distortion"] as? [String: Any] {
            guard let k1 = d["k1"] as? Double,
                  let dc = d["center_px"] as? [Double], dc.count == 2 else {
                throw DecodeError.malformed("malformed distortion")
            }
            // Mirrors court_model's `or 1000.0` falsy semantics: nil OR an
            // explicit non-positive value (e.g. 0) falls back to 1000.0.
            let rawNorm = d["norm_px"] as? Double
            distortion = Distortion(k1: k1, centerPx: SIMD2(dc[0], dc[1]),
                                    normPx: (rawNorm.flatMap { $0 > 0 ? $0 : nil }) ?? 1000.0)
        }
        let rotation = simd_double3x3(rows: [
            SIMD3(rows[0][0], rows[0][1], rows[0][2]),
            SIMD3(rows[1][0], rows[1][1], rows[1][2]),
            SIMD3(rows[2][0], rows[2][1], rows[2][2]),
        ])
        return CameraModel(focalPx: focal, centerPx: SIMD2(center[0], center[1]),
                           rotation: rotation, cameraCenterFt: SIMD3(cc[0], cc[1], cc[2]),
                           distortion: distortion,
                           // Absent on older stored models — left nil, never guessed.
                           frameWidth: obj["frame_width"] as? Double,
                           frameHeight: obj["frame_height"] as? Double)
    }

    func undistort(_ px: SIMD2<Double>) -> SIMD2<Double> {
        guard let d = distortion else { return px }
        let delta = px - d.centerPx
        let r2 = simd_length_squared(delta) / (d.normPx * d.normPx)
        let factor = 1.0 + d.k1 * r2
        precondition(abs(factor) > 1e-9, "Distortion factor collapsed to zero.")
        return d.centerPx + delta / factor
    }

    func project(_ courtXYZ: SIMD3<Double>) -> SIMD2<Double>? {
        let cameraPoint = rotation * (courtXYZ - cameraCenterFt)
        guard cameraPoint.z > 1e-9 else { return nil }
        return centerPx + focalPx * SIMD2(cameraPoint.x / cameraPoint.z,
                                          cameraPoint.y / cameraPoint.z)
    }

    func ray(_ undistortedPx: SIMD2<Double>) -> (origin: SIMD3<Double>, dir: SIMD3<Double>) {
        let cameraDir = SIMD3((undistortedPx.x - centerPx.x) / focalPx,
                              (undistortedPx.y - centerPx.y) / focalPx,
                              1.0)
        let worldDir = simd_normalize(rotation.transpose * cameraDir)
        return (cameraCenterFt, worldDir)
    }
}

// MARK: - pixel space

extension CameraModel {
    /// Aspect ratios equal to within this are the same framing at a
    /// different resolution. Mirrors court_model.CAMERA_ASPECT_TOLERANCE.
    static let aspectTolerance = 1e-3

    enum ScaleError: Error, Equatable, LocalizedError {
        case unknownFrameSize
        case nonPositiveTarget
        case aspectMismatch(from: SIMD2<Double>, to: SIMD2<Double>)

        var errorDescription: String? {
            switch self {
            case .unknownFrameSize:
                return "Camera model has no frame size: the pixel space it "
                     + "was solved in is unknown, so it cannot be scaled."
            case .nonPositiveTarget:
                return "Target frame size must be positive."
            case .aspectMismatch(let from, let to):
                return "Camera model aspect \(from.x / from.y) "
                     + "(\(from.x)x\(from.y)) does not match target aspect "
                     + "\(to.x / to.y) (\(to.x)x\(to.y)): different crop or "
                     + "FOV, not a resolution change."
            }
        }
    }

    /// Re-express a solved model in a different pixel space.
    ///
    /// Mirrors court_model.scale_camera_model exactly — same guards, same
    /// arithmetic, pinned by the `scaled_model` golden case.
    ///
    /// Pure resolution change only. Throws if the model has no frame size
    /// (unknown source space) or if the aspect ratio differs by more than
    /// 1e-3 — a different aspect means a different crop/FOV, which no scale
    /// factor can fix, so guessing would silently corrupt the geometry.
    ///
    /// Everything measured in pixels scales by s = toWidth / frameWidth:
    /// focal, principal point, and the distortion's own center and
    /// normalizer. Everything else is deliberately left alone — `rotation`
    /// and `cameraCenterFt` are world-space, and `k1` stays dimensionless
    /// precisely because r² = |p−c|²/norm² is invariant once `normPx`
    /// scales with the rest. (Python also scales `fit_rms_px`, a pixel
    /// residual; Swift's model does not carry it.)
    func scaled(toWidth: Double, height toHeight: Double) throws -> CameraModel {
        guard let frameWidth, let frameHeight else { throw ScaleError.unknownFrameSize }
        guard toWidth > 0, toHeight > 0 else { throw ScaleError.nonPositiveTarget }
        guard abs(toWidth / toHeight - frameWidth / frameHeight) <= Self.aspectTolerance else {
            throw ScaleError.aspectMismatch(from: SIMD2(frameWidth, frameHeight),
                                            to: SIMD2(toWidth, toHeight))
        }
        let scale = toWidth / frameWidth
        // fromJSON already resolved court_model's `or 1000.0` fallback for a
        // missing normPx, so scaling the stored value is the same
        // materialization Python does inside scale_camera_model.
        let lens = distortion.map {
            Distortion(k1: $0.k1, centerPx: $0.centerPx * scale, normPx: $0.normPx * scale)
        }
        return CameraModel(focalPx: focalPx * scale, centerPx: centerPx * scale,
                           rotation: rotation, cameraCenterFt: cameraCenterFt,
                           distortion: lens,
                           frameWidth: toWidth, frameHeight: toHeight)
    }

    /// Re-express a solved model in the live capture pixel space.
    ///
    /// Every consumer on the live path — overlay mapping, peer detection
    /// tuples, StereoEngine — works in CaptureSettings.frameWidth ×
    /// frameHeight, but a calibration may have been solved on a clip of any
    /// resolution. Adopting a model without this step is the silent failure:
    /// a 1080×1920 solve against 4K detections is wrong by 2× and raises
    /// nothing. Throws rather than guessing when the source space is unknown
    /// or the aspect differs.
    func adoptedForCapture() throws -> CameraModel {
        try scaled(toWidth: Double(CaptureSettings.frameWidth),
                   height: Double(CaptureSettings.frameHeight))
    }
}
