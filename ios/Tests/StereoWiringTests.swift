// ios/Tests/StereoWiringTests.swift
import XCTest
@testable import SquashLineCalling

final class StereoWiringTests: XCTestCase {
    func testCalibrationMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)

        var received: (String, String)?
        primary.onCalibration = { received = ($0, $1) }
        secondary.sendCalibration(profileID: "ucsc-right-fin",
                                  payloadJSON: "{\"focal_px\": 1600}")
        XCTAssertEqual(received?.0, "ucsc-right-fin")
        XCTAssertEqual(received?.1, "{\"focal_px\": 1600}")
    }

    func testSendCalibrationGatedOnPhase() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        var fired = false
        primary.onCalibration = { _, _ in fired = true }
        // Not started/paired: sending from an idle session must be a no-op.
        primary.sendCalibration(profileID: "x", payloadJSON: "{}")
        XCTAssertFalse(fired)
    }

    func testEventMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (UInt32, String)?
        secondary.onEvent = { received = ($0, $1) }
        primary.sendEvent(rallyID: 7, json: "{\"surface\":\"front_wall\"}")
        XCTAssertEqual(received?.0, 7)
        XCTAssertEqual(received?.1, "{\"surface\":\"front_wall\"}")
    }

    func testRecordMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (String, UInt64)?
        secondary.onRecord = { received = ($0, $1) }
        primary.sendRecord(action: "start", ptsNs: 1_234_567_890)
        XCTAssertEqual(received?.0, "start")
        XCTAssertEqual(received?.1, 1_234_567_890)
    }

    func testSessionManifestRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (String, String)?
        secondary.onSessionManifest = { received = ($0, $1) }
        primary.sendSessionManifest(sessionID: "S-1", videoID: "V-9")
        XCTAssertEqual(received?.0, "S-1")
        XCTAssertEqual(received?.1, "V-9")
    }

    func testSendRecordGatedOnPhase() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        var fired = false
        primary.onRecord = { _, _ in fired = true }
        primary.sendRecord(action: "start", ptsNs: 1)
        XCTAssertFalse(fired)
    }
}
