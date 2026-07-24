import Foundation
import simd

/// Division-model lens distortion — the JS<->Python<->Swift contract
/// (mirrors court_model.undistort_point).
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
                           distortion: distortion)
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
