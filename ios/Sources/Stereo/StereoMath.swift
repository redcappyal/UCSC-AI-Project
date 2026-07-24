// ios/Sources/Stereo/StereoMath.swift
import Foundation
import simd

/// Pure stereo geometry — mirrors stereo_engine.py exactly (constants,
/// algorithms, and quirks). Do not "improve" without changing the Python
/// authority and regenerating goldens.
///
/// One deliberate mapping: Python's `triangulate`/`snap_to_plane` return
/// `(None, inf)` / `None` for both near-parallel rays and closest-approach
/// (or plane intersection) landing behind a camera — distinguishing the two
/// rejection causes was never observed anywhere downstream. Swift folds both
/// into a single `nil`; Task 4's consumers only branch on nil-ness, same as
/// Python callers only branch on `point is None`.
enum StereoMath {
    static let parallelEps = 1e-9
    static let outLineHeightFt = 15.0
    static let tinTopHeightFt = 19.0 / 12.0
    static let backWallOutHeightFt = 7.0
    static let courtWidthFt = 21.0
    static let courtLengthFt = 32.0
    static let boundsSlackFt = 0.5
    /// Ordered exactly like Python's SURFACES tuple — parity contract.
    static let surfaces = ["floor", "front_wall", "back_wall", "left_wall", "right_wall"]

    private static let planes: [String: (SIMD3<Double>, SIMD3<Double>)] = [
        "floor": (SIMD3(0, 0, 0), SIMD3(0, 0, 1)),
        "front_wall": (SIMD3(0, 0, 0), SIMD3(0, 1, 0)),
        "back_wall": (SIMD3(0, courtLengthFt, 0), SIMD3(0, -1, 0)),
        "left_wall": (SIMD3(0, 0, 0), SIMD3(1, 0, 0)),
        "right_wall": (SIMD3(courtWidthFt, 0, 0), SIMD3(-1, 0, 0)),
    ]

    static func surfacePlane(_ name: String) -> (point: SIMD3<Double>, normal: SIMD3<Double>) {
        let (p, n) = planes[name]!
        return (p, n)
    }

    /// WSF side-wall out line: 15 ft at the front wall, 7 ft at the back.
    static func sideWallOutHeightFt(_ yFt: Double) -> Double {
        outLineHeightFt + (backWallOutHeightFt - outLineHeightFt) * (yFt / courtLengthFt)
    }

    /// (call, margin_ft) for an impact point known to lie on `surface`.
    /// margin_ft is the distance to the deciding line — how clear the call is.
    static func callForImpact(surface: String, point: SIMD3<Double>) -> (call: String, marginFt: Double) {
        let (_, y, z) = (point.x, point.y, point.z)
        switch surface {
        case "floor":
            return ("bounce", 0.0)
        case "front_wall":
            if z >= outLineHeightFt { return ("out", z - outLineHeightFt) }
            if z <= tinTopHeightFt { return ("down", tinTopHeightFt - z) }
            return ("in", min(outLineHeightFt - z, z - tinTopHeightFt))
        case "left_wall", "right_wall":
            let line = sideWallOutHeightFt(y)
            return z >= line ? ("out", z - line) : ("in", line - z)
        case "back_wall":
            return z >= backWallOutHeightFt
                ? ("out", z - backWallOutHeightFt) : ("in", backWallOutHeightFt - z)
        default:
            fatalError("Unknown surface: \(surface)")
        }
    }

    static func planeDistance(surface: String, point: SIMD3<Double>) -> Double {
        let (planePoint, normal) = planes[surface]!
        return simd_dot(point - planePoint, normal)
    }

    /// Closest-approach midpoint of the two viewing rays.
    ///
    /// Returns nil for near-parallel rays or when the closest approach lies
    /// behind either camera (s or t <= 0) — mirrors Python's (None, inf).
    static func triangulate(_ a: CameraModel, _ b: CameraModel,
                            pxA: SIMD2<Double>, pxB: SIMD2<Double>)
        -> (point: SIMD3<Double>, gapFt: Double)? {
        let (o1, d1) = a.ray(a.undistort(pxA))
        let (o2, d2) = b.ray(b.undistort(pxB))
        let w0 = o1 - o2
        let bDot = simd_dot(d1, d2)
        let d = simd_dot(d1, w0)
        let e = simd_dot(d2, w0)
        let denom = 1.0 - bDot * bDot          // a = c = 1 for unit directions
        guard denom >= parallelEps else { return nil }
        let s = (bDot * e - d) / denom
        let t = (e - bDot * d) / denom
        guard s > 0.0, t > 0.0 else { return nil }
        let p1 = o1 + s * d1
        let p2 = o2 + t * d2
        return ((p1 + p2) / 2.0, simd_length(p1 - p2))
    }

    private static func inSurfaceBounds(surface: String, point: SIMD3<Double>) -> Bool {
        let lo = -boundsSlackFt
        switch surface {
        case "floor":
            return point.x >= lo && point.x <= courtWidthFt + boundsSlackFt
                && point.y >= lo && point.y <= courtLengthFt + boundsSlackFt
        case "front_wall", "back_wall":
            return point.x >= lo && point.x <= courtWidthFt + boundsSlackFt && point.z >= lo
        default:
            return point.y >= lo && point.y <= courtLengthFt + boundsSlackFt && point.z >= lo
        }
    }

    /// Intersect the (undistorted) viewing ray with a court surface plane.
    static func snapToPlane(_ model: CameraModel, px: SIMD2<Double>, surface: String)
        -> SIMD3<Double>? {
        let (planePoint, normal) = planes[surface]!
        let (origin, dir) = model.ray(model.undistort(px))
        let denom = simd_dot(dir, normal)
        guard abs(denom) >= parallelEps else { return nil }
        let t = simd_dot(planePoint - origin, normal) / denom
        guard t > 0.0 else { return nil }
        let point = origin + t * dir
        return inSurfaceBounds(surface: surface, point: point) ? point : nil
    }

    static func fuseSnaps(_ a: SIMD3<Double>?, _ b: SIMD3<Double>?) -> SIMD3<Double>? {
        switch (a, b) {
        case (nil, let b): return b
        case (let a, nil): return a
        case (let a?, let b?): return (a + b) / 2.0
        }
    }
}
