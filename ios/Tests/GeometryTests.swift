import XCTest
@testable import SquashLineCalling

final class GeometryTests: XCTestCase {
    func testAspectFitLetterboxesTallContainer() {
        // 9:16 video in a 100x300 container: full width, centered vertically.
        let rect = Geometry.aspectFitRect(
            content: CGSize(width: 1080, height: 1920),
            container: CGSize(width: 100, height: 300))
        XCTAssertEqual(rect.origin.x, 0, accuracy: 0.01)
        XCTAssertEqual(rect.width, 100, accuracy: 0.01)
        XCTAssertEqual(rect.height, 100 * 1920 / 1080, accuracy: 0.01)
        XCTAssertEqual(rect.midY, 150, accuracy: 0.01)
    }

    func testOverlayRectFlipsVisionY() {
        // Vision origin is bottom-left; screen origin is top-left.
        let videoRect = CGRect(x: 0, y: 0, width: 100, height: 200)
        let vision = CGRect(x: 0.5, y: 0.0, width: 0.1, height: 0.1) // bottom of frame
        let mapped = Geometry.overlayRect(visionRect: vision, videoRect: videoRect)
        XCTAssertEqual(mapped.origin.x, 50, accuracy: 0.01)
        XCTAssertEqual(mapped.origin.y, 180, accuracy: 0.01)   // near the bottom on screen
        XCTAssertEqual(mapped.size, CGSize(width: 10, height: 20))
    }
}
