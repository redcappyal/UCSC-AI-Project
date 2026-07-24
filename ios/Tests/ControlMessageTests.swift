import XCTest
@testable import SquashLineCalling

final class ControlMessageTests: XCTestCase {
    private let hello = Hello(protoVersion: peerProtoVersion, appVersion: "1.0",
                              deviceModel: "iPhone15,2", nonce: 0xDEAD_BEEF,
                              frameW: 1080, frameH: 1920)

    func testRoundTripEveryCase() throws {
        let cases: [ControlMessage] = [
            .hello(hello),
            .role(.secondary),
            .calibration(profileID: "ucsc-left-fin", calibrationJSON: "{\"version\":2}"),
            .syncPing(SyncPing(pingID: 3, t1: 100.25)),
            .syncPong(SyncPong(pingID: 3, t1: 100.25, t2: 100.30, t3: 100.31)),
            .clapAnchor(hostTime: 4242.125),
            .record(action: "start", ptsNs: 9_000_000_000),
            .event(rallyID: 1, json: "{}"),
            .heartbeat(seq: 77),
            .sessionManifest(sessionID: "s-1", videoID: "v-abc"),
        ]
        for original in cases {
            let decoded = ControlMessage.decode(try ControlMessage.encode(original))
            XCTAssertEqual(decoded, original)
        }
    }

    func testMalformedReturnsNil() {
        XCTAssertNil(ControlMessage.decode(Data("not json".utf8)))
        XCTAssertNil(ControlMessage.decode(Data("{\"unknownCase\":{}}".utf8)))
    }

    func testPairingCodeIsSymmetricFourDigits() {
        var other = hello; other.nonce = 0x0000_1234
        let code = pairingCode(hello, other)
        XCTAssertEqual(code, pairingCode(other, hello))
        XCTAssertEqual(code.count, 4)
        XCTAssertTrue(code.allSatisfy(\.isNumber))
    }
}
