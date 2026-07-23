import XCTest
@testable import SquashLineCalling

final class SmokeTests: XCTestCase {
    func testConfigDefaultBaseIsHTTPS() {
        XCTAssertEqual(Config.baseURL.scheme, "https")
    }
}
