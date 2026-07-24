import XCTest
import simd
@testable import SquashLineCalling

final class StereoGoldenTests: XCTestCase {
    static var goldens: [String: Any] = [:]
    static var left: CameraModel!
    static var right: CameraModel!

    override class func setUp() {
        super.setUp()
        let url = Bundle(for: StereoGoldenTests.self)
            .url(forResource: "stereo_goldens", withExtension: "json")!
        let data = try! Data(contentsOf: url)
        goldens = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        let cameras = goldens["cameras"] as! [String: Any]
        func decode(_ key: String) -> CameraModel {
            let json = try! JSONSerialization.data(withJSONObject: cameras[key]!)
            return try! CameraModel.fromJSON(json)
        }
        left = decode("left"); right = decode("right")
    }

    private func vec3(_ any: Any) -> SIMD3<Double> {
        let a = any as! [Double]; return SIMD3(a[0], a[1], a[2])
    }
    private func vec2(_ any: Any) -> SIMD2<Double> {
        let a = any as! [Double]; return SIMD2(a[0], a[1])
    }

    func testSchemaIsV2() {
        XCTAssertEqual(Self.goldens["schema"] as? String, "stereo-goldens-v2")
    }

    func testProjectMatchesGoldenPixels() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let point = vec3(case_["point_ft"]!)
            let pxA = Self.left.project(point)!
            let pxB = Self.right.project(point)!
            XCTAssertEqual(pxA.x, vec2(case_["px_a"]!).x, accuracy: 1e-7)
            XCTAssertEqual(pxA.y, vec2(case_["px_a"]!).y, accuracy: 1e-7)
            XCTAssertEqual(pxB.x, vec2(case_["px_b"]!).x, accuracy: 1e-7)
            XCTAssertEqual(pxB.y, vec2(case_["px_b"]!).y, accuracy: 1e-7)
        }
    }

    func testRayPassesThroughGoldenPoint() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let point = vec3(case_["point_ft"]!)
            let (origin, dir) = Self.left.ray(vec2(case_["px_a"]!))
            // Distance from the golden point to the ray must be ~0.
            let w = point - origin
            let dist = simd_length(w - simd_dot(w, dir) * dir)
            XCTAssertLessThan(dist, 1e-7)
        }
    }

    func testProjectBehindCameraReturnsNil() {
        // Behind the back wall relative to the camera's view direction.
        XCTAssertNil(Self.left.project(SIMD3(10.5, 40.0, 5.0)))
    }

    func testUndistortIdentityWhenNil() {
        let px = SIMD2(123.4, 567.8)
        XCTAssertEqual(Self.left.undistort(px), px)
    }

    func testUndistortDivisionModel() {
        let distorted = CameraModel(
            focalPx: 1000, centerPx: SIMD2(960, 540),
            rotation: matrix_identity_double3x3, cameraCenterFt: SIMD3(0, 0, 0),
            distortion: Distortion(k1: -0.1, centerPx: SIMD2(960, 540), normPx: 1000))
        let px = SIMD2<Double>(1160.0, 540.0)   // 200 px right of center; r2 = 0.04
        let undistorted = distorted.undistort(px)
        XCTAssertEqual(undistorted.x, 960.0 + 200.0 / (1.0 - 0.1 * 0.04), accuracy: 1e-9)
        XCTAssertEqual(undistorted.y, 540.0, accuracy: 1e-9)
    }

    func testTriangulationGoldenParity() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let result = StereoMath.triangulate(Self.left, Self.right,
                                                pxA: vec2(case_["px_a"]!),
                                                pxB: vec2(case_["px_b"]!))!
            let expected = vec3(case_["point_ft"]!)
            XCTAssertLessThan(simd_length(result.point - expected), 1e-7)
            XCTAssertEqual(result.gapFt, case_["gap_ft"] as! Double, accuracy: 1e-7)
        }
    }

    func testSnapGoldenParity() {
        for case_ in Self.goldens["snap_cases"] as! [[String: Any]] {
            let model = (case_["camera"] as! String) == "left" ? Self.left! : Self.right!
            let snap = StereoMath.snapToPlane(model, px: vec2(case_["px"]!),
                                              surface: case_["surface"] as! String)!
            XCTAssertLessThan(simd_length(snap - vec3(case_["point_ft"]!)), 1e-7)
        }
    }

    func testCallGoldenParity() {
        for case_ in Self.goldens["call_cases"] as! [[String: Any]] {
            let (call, margin) = StereoMath.callForImpact(
                surface: case_["surface"] as! String, point: vec3(case_["point_ft"]!))
            XCTAssertEqual(call, case_["call"] as! String)
            XCTAssertEqual(margin, case_["margin_ft"] as! Double, accuracy: 1e-7)
        }
    }

    func testSurfaceOrderMirrorsPython() {
        XCTAssertEqual(StereoMath.surfaces,
                       ["floor", "front_wall", "back_wall", "left_wall", "right_wall"])
    }
}
