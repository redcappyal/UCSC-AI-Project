import XCTest
@testable import SquashLineCalling

final class MultipartTests: XCTestCase {
    func testBodyStructure() {
        let body = Multipart.body(
            boundary: "BOUND", fields: [("video_id", "abc")],
            fileField: "video_file", filename: "clip.mp4",
            contentType: "video/mp4", fileData: Data("FILEBYTES".utf8))
        let text = String(decoding: body, as: UTF8.self)
        XCTAssertTrue(text.contains("--BOUND\r\n"))
        XCTAssertTrue(text.contains("name=\"video_id\"\r\n\r\nabc\r\n"))
        XCTAssertTrue(text.contains(
            "name=\"video_file\"; filename=\"clip.mp4\"\r\nContent-Type: video/mp4\r\n\r\nFILEBYTES\r\n"))
        XCTAssertTrue(text.hasSuffix("--BOUND--\r\n"))
    }

    func testFormURLEncoding() {
        let encoded = Multipart.formURLEncoded([
            ("calibration_json", #"{"a": 1}"#), ("start_time", "0")])
        XCTAssertEqual(encoded, "calibration_json=%7B%22a%22%3A%201%7D&start_time=0")
    }
}
