// ios/Tests/PeerSessionTests.swift
import XCTest
@testable import SquashLineCalling

final class PeerSessionTests: XCTestCase {
    private var pair: (LoopbackTransport, LoopbackTransport)!
    private var primary: PeerSession!
    private var secondary: PeerSession!

    override func setUp() {
        super.setUp()
        pair = LoopbackTransport.pair()
        primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
    }

    private func startBoth() { secondary.start(); primary.start() }

    func testHandshakeAssignsRolesAndMatchingCodes() {
        startBoth()
        XCTAssertEqual(primary.role, .primary)
        XCTAssertEqual(secondary.role, .secondary)
        guard case .confirming(let codeA) = primary.phase,
              case .confirming(let codeB) = secondary.phase else {
            return XCTFail("both sides must reach confirming, got \(primary.phase) / \(secondary.phase)")
        }
        XCTAssertEqual(codeA, codeB)
    }

    func testSyncBurstReachesReadyOnBothSides() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        XCTAssertEqual(secondary.phase, .ready)
        XCTAssertNotNil(primary.clockSync.estimate)
    }

    func testVersionMismatchFails() {
        // Corrupt the secondary's hello version by intercepting control frames.
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var hello)? = ControlMessage.decode(frame) else { return deliver(frame) }
            hello.protoVersion = 99
            deliver(try! ControlMessage.encode(.hello(hello)))
        }
        startBoth()
        guard case .failed = primary.phase else { return XCTFail("expected failed, got \(primary.phase)") }
    }

    func testHeartbeatLossDegradesAndRecoveryRestores() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        // Silence the link: hold every control frame from the secondary.
        pair.1.controlDeliveryHook = { _, _ in }
        for _ in 0..<50 { t += 0.5; primary.tick(now: t) }   // > heartbeatTimeout with no pongs/beats
        guard case .degraded = primary.phase else { return XCTFail("expected degraded, got \(primary.phase)") }
        pair.1.controlDeliveryHook = nil
        for _ in 0..<10 { t += 0.5; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
    }

    func testDetectionsFlowSecondaryToPrimary() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        primary.goLive(); secondary.goLive()
        var received: [[DetectionTuple]] = []
        primary.onRemoteDetections = { received.append($0) }
        let tuple = DetectionTuple(seq: 1, ptsNs: 42, x: 1, y: 2,
                                   conf: Float16(0.9), bboxH: Float16(10))
        secondary.sendDetections([tuple])
        XCTAssertEqual(received, [[tuple]])
    }

    func testClapAnchorSetsBothMappingsConsistently() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        primary.sendClapAnchor(localOnset: 10.0)
        secondary.sendClapAnchor(localOnset: 10.4)   // secondary clock is +0.4 ahead
        XCTAssertEqual(primary.clockSync.remoteToLocal(20.4) ?? -1, 20.0, accuracy: 1e-9)
        XCTAssertEqual(secondary.clockSync.remoteToLocal(20.0) ?? -1, 20.4, accuracy: 1e-9)
    }

    func testDoubleDisconnectStillRecoversToReady() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        pair.0.onStateChange?(.disconnected("blip 1"))
        pair.0.onStateChange?(.disconnected("blip 2"))
        guard case .degraded = primary.phase else { return XCTFail("expected degraded, got \(primary.phase)") }
        for _ in 0..<10 { t += 0.5; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
    }
}
