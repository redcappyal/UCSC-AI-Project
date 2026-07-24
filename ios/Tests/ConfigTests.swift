import XCTest
@testable import SquashLineCalling

final class ConfigTests: XCTestCase {
    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: Config.serverBaseKey)
        super.tearDown()
    }

    func testDefaultWhenNoOverride() {
        UserDefaults.standard.removeObject(forKey: Config.serverBaseKey)
        XCTAssertEqual(Config.baseURL.absoluteString, Config.defaultBase)
    }

    func testBareHostGetsHTTPScheme() {
        Config.setServerBase("192.168.1.20:5000")
        XCTAssertEqual(Config.baseURL.absoluteString, "http://192.168.1.20:5000")
    }

    func testExplicitSchemeKept() {
        Config.setServerBase("https://court.example.org")
        XCTAssertEqual(Config.baseURL.absoluteString, "https://court.example.org")
    }

    func testEmptyInputClearsOverride() {
        Config.setServerBase("192.168.1.20:5000")
        Config.setServerBase("   ")
        XCTAssertNil(UserDefaults.standard.string(forKey: Config.serverBaseKey))
        XCTAssertEqual(Config.baseURL.absoluteString, Config.defaultBase)
    }

    func testSchemeOnlyGarbageFallsBack() {
        UserDefaults.standard.set("http://", forKey: Config.serverBaseKey)
        XCTAssertEqual(Config.baseURL.absoluteString, Config.defaultBase)
    }
}
