// ios/Tests/CameraFrameSpaceTests.swift
//
// Mirrors tests/test_camera_model.py's frame-space section. A solved model
// that does not record its own pixel space is a silent failure: a 1080x1920
// calibration used verbatim against 4K detections bends every ray by 2x and
// raises nothing.
import XCTest
import simd
@testable import SquashLineCalling

final class CameraFrameSpaceTests: XCTestCase {
    /// A back-wall camera solved at 1080x1920, as the wizard would emit it.
    private func sourceJSON(frameSize: (Double, Double)? = (1080, 1920),
                            distortion: Bool = true) -> Data {
        var obj: [String: Any] = [
            "focal_px": 900.0,
            "center_px": [540.0, 960.0],
            "rotation": [[-1.0, 0.0, 0.0],
                         [0.0, 0.0816932394452715, -0.9966575212323125],
                         [0.0, -0.9966575212323125, -0.0816932394452715]],
            "camera_center_ft": [10.5, 30.5, 7.0],
        ]
        if distortion {
            obj["distortion"] = ["model": "division_k1", "k1": -0.08,
                                 "center_px": [548.0, 952.0], "norm_px": 1100.0]
        }
        if let frameSize {
            obj["frame_width"] = frameSize.0
            obj["frame_height"] = frameSize.1
        }
        return try! JSONSerialization.data(withJSONObject: obj)
    }

    private func model(frameSize: (Double, Double)? = (1080, 1920),
                       distortion: Bool = true) throws -> CameraModel {
        try CameraModel.fromJSON(sourceJSON(frameSize: frameSize, distortion: distortion))
    }

    func testFromJSONParsesFrameSize() throws {
        let camera = try model()
        XCTAssertEqual(camera.frameWidth, 1080)
        XCTAssertEqual(camera.frameHeight, 1920)
    }

    func testFromJSONToleratesMissingFrameSize() throws {
        // A model stored before frame size was recorded. Unknown stays
        // unknown — guessing a default is the failure this field prevents.
        let camera = try model(frameSize: nil)
        XCTAssertNil(camera.frameWidth)
        XCTAssertNil(camera.frameHeight)
    }

    func testScaledScalesOnlyPixelQuantities() throws {
        let camera = try model()
        let scaled = try camera.scaled(toWidth: 2160, height: 3840)
        XCTAssertEqual(scaled.focalPx, 1800, accuracy: 1e-12)
        XCTAssertEqual(scaled.centerPx.x, 1080, accuracy: 1e-12)
        XCTAssertEqual(scaled.centerPx.y, 1920, accuracy: 1e-12)
        XCTAssertEqual(scaled.distortion?.centerPx.x, 1096)
        XCTAssertEqual(scaled.distortion?.centerPx.y, 1904)
        XCTAssertEqual(scaled.distortion?.normPx, 2200)
        XCTAssertEqual(scaled.frameWidth, 2160)
        XCTAssertEqual(scaled.frameHeight, 3840)
        // World-space and dimensionless quantities must NOT move.
        XCTAssertEqual(scaled.rotation, camera.rotation)
        XCTAssertEqual(scaled.cameraCenterFt, camera.cameraCenterFt)
        XCTAssertEqual(scaled.distortion?.k1, camera.distortion?.k1)
    }

    func testScaledDefaultsAbsentNormPxBeforeScaling() throws {
        // court_model._distortion_params treats a missing/zero norm_px as
        // 1000.0; scaling has to materialize that or r^2 changes.
        var obj = try! JSONSerialization.jsonObject(with: sourceJSON()) as! [String: Any]
        obj["distortion"] = ["model": "division_k1", "k1": -0.05,
                             "center_px": [540.0, 960.0]]
        let camera = try CameraModel.fromJSON(
            try! JSONSerialization.data(withJSONObject: obj))
        let scaled = try camera.scaled(toWidth: 2160, height: 3840)
        XCTAssertEqual(scaled.distortion?.normPx, 2000)
    }

    /// Center, axis points and all four corners — off-axis is where a wrong
    /// principal point or distortion scale shows up first.
    private let probes: [SIMD2<Double>] = [
        SIMD2(540, 960), SIMD2(540, 12), SIMD2(17, 960), SIMD2(0, 0),
        SIMD2(1079, 0), SIMD2(0, 1919), SIMD2(1079, 1919),
        SIMD2(231, 1487), SIMD2(908, 344),
    ]

    /// The acceptance property: scaling the model and scaling the pixel are
    /// the same operation. Mirrors test_scale_camera_model_preserves_rays.
    func testScaledPreservesRays() throws {
        for withLens in [true, false] {
            let camera = try model(distortion: withLens)
            let scaled = try camera.scaled(toWidth: 2160, height: 3840)
            for px in probes {
                let (origin, dir) = camera.ray(camera.undistort(px))
                let (originS, dirS) = scaled.ray(scaled.undistort(px * 2.0))
                XCTAssertEqual(originS, origin)
                XCTAssertLessThan(simd_length(dirS - dir), 1e-9,
                                  "px \(px) lens=\(withLens)")
            }
        }
    }

    func testScaledRequiresKnownFrameSize() throws {
        let camera = try model(frameSize: nil)
        XCTAssertThrowsError(try camera.scaled(toWidth: 2160, height: 3840)) { error in
            XCTAssertEqual(error as? CameraModel.ScaleError, .unknownFrameSize)
        }
    }

    func testScaledRejectsAspectChange() throws {
        // 1080x1920 -> 2160x2880 is a different crop, not a resolution
        // change: no single factor can express it.
        let camera = try model()
        XCTAssertThrowsError(try camera.scaled(toWidth: 2160, height: 2880)) { error in
            guard case .aspectMismatch = error as? CameraModel.ScaleError else {
                return XCTFail("expected aspectMismatch, got \(error)")
            }
        }
    }

    func testScaledRejectsNonPositiveTarget() throws {
        let camera = try model()
        XCTAssertThrowsError(try camera.scaled(toWidth: 0, height: 0))
    }

    func testAdoptedForCaptureLandsInTheLivePixelSpace() throws {
        // Landscape source (1920x1080), matching capture's aspect: adoption
        // is a clean 2x scale to 3840x2160, not the portrait 1080x1920 fixture
        // every other test in this file uses, which now throws aspectMismatch
        // against a landscape capture target.
        let adopted = try model(frameSize: (1920, 1080)).adoptedForCapture()
        XCTAssertEqual(adopted.frameWidth, Double(CaptureSettings.frameWidth))
        XCTAssertEqual(adopted.frameHeight, Double(CaptureSettings.frameHeight))
        XCTAssertEqual(adopted.focalPx,
                       900 * Double(CaptureSettings.frameWidth) / 1920, accuracy: 1e-12)
    }

    func testAdoptedForCaptureRejectsAnUnknownSourceSpace() throws {
        // The live path must refuse rather than assume: a wrong-but-silent
        // line call is worse than no call.
        XCTAssertThrowsError(try model(frameSize: nil).adoptedForCapture())
    }
}
