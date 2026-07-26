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

    // MARK: - endRally: the only way out of .live

    /// Drives both sides to `.ready`, leaving `t` where the caller can carry
    /// on ticking from. Factored out because every `endRally` test below needs
    /// the same 40-tick sync pass first.
    @discardableResult
    private func syncToReady() -> Double {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        return t
    }

    /// `goLive()`'s inverse has to be as guarded as `goLive()` itself: it may
    /// only ever undo a rally, never nudge a session that has not run one.
    /// `.failed` is covered by the same `default: break` arm as `.idle` here
    /// and is left to `testVersionMismatchFails` to reach.
    func testEndRallyIsANoOpOutsideLive() {
        XCTAssertEqual(primary.phase, .idle)
        primary.endRally()
        XCTAssertEqual(primary.phase, .idle, "a session that never paired must not move")

        startBoth()
        guard case .confirming = primary.phase else {
            return XCTFail("setup: expected confirming, got \(primary.phase)")
        }
        primary.endRally()
        guard case .confirming = primary.phase else {
            return XCTFail("endRally must not skip the code check, got \(primary.phase)")
        }

        primary.confirmPairing(); secondary.confirmPairing()
        XCTAssertEqual(primary.phase, .syncing)
        primary.endRally()
        XCTAssertEqual(primary.phase, .syncing, "endRally must not shortcut the clock sync")

        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        primary.endRally()
        XCTAssertEqual(primary.phase, .ready, "already ready — there is no rally to end")

        primary.end()
        XCTAssertEqual(primary.phase, .ended)
        primary.endRally()
        XCTAssertEqual(primary.phase, .ended, "a spent session must never be revived")
    }

    /// The dead end this closes: before `endRally` there was no transition out
    /// of `.live` at all — `tick`'s `.ready, .live` branch only ever leaves it
    /// to degrade — so the first rally made the session `.live` forever.
    func testEndRallyReturnsALiveSessionToReadyOnBothSides() {
        syncToReady()
        primary.goLive(); secondary.goLive()
        XCTAssertEqual(primary.phase, .live)
        XCTAssertEqual(secondary.phase, .live)

        primary.endRally(); secondary.endRally()
        XCTAssertEqual(primary.phase, .ready)
        XCTAssertEqual(secondary.phase, .ready)

        // `goLive()` only advances from `.ready`, so a second rally starting at
        // all is the proof that the session really came back — not just that
        // one enum value was overwritten.
        primary.goLive(); secondary.goLive()
        XCTAssertEqual(primary.phase, .live)
        XCTAssertEqual(secondary.phase, .live)
    }

    /// Every `PeerSession` sender gates on `.live || .ready`, and the
    /// `.sessionManifest` that carries this phone's uploaded video ID is sent
    /// *after* the rally ends — so returning the session to `.ready` must not
    /// be what silences it. Asserts all five senders from `.ready`, since they
    /// share one gate and the manifest is the one that actually runs
    /// post-rally.
    func testEverySenderStillReachesThePeerAfterTheRallyEnds() {
        syncToReady()
        primary.goLive(); secondary.goLive()
        primary.endRally(); secondary.endRally()
        XCTAssertEqual(primary.phase, .ready)

        var manifests: [(String, String)] = []
        secondary.onSessionManifest = { manifests.append(($0, $1)) }
        primary.sendSessionManifest(sessionID: "session-1", videoID: "run-7")
        XCTAssertEqual(manifests.count, 1, "the post-rally manifest must survive the return to .ready")
        XCTAssertEqual(manifests.first?.0, "session-1")
        XCTAssertEqual(manifests.first?.1, "run-7")

        var calibrations: [String] = []
        secondary.onCalibration = { _, payload in calibrations.append(payload) }
        primary.sendCalibration(profileID: "profile", payloadJSON: "{\"m\":1}")
        XCTAssertEqual(calibrations, ["{\"m\":1}"])

        var events: [String] = []
        secondary.onEvent = { _, json in events.append(json) }
        primary.sendEvent(rallyID: 0, json: "{\"call\":\"in\"}")
        XCTAssertEqual(events, ["{\"call\":\"in\"}"])

        var records: [String] = []
        secondary.onRecord = { action, _ in records.append(action) }
        primary.sendRecord(action: "start", ptsNs: 0)
        XCTAssertEqual(records, ["start"], "the next rally's own start has to go out from .ready")

        var received: [[DetectionTuple]] = []
        primary.onRemoteDetections = { received.append($0) }
        let tuple = DetectionTuple(seq: 1, ptsNs: 42, x: 1, y: 2,
                                   conf: Float16(0.9), bboxH: Float16(10))
        secondary.sendDetections([tuple])
        XCTAssertEqual(received, [[tuple]])
    }

    /// A rally can end while the link is down — that is exactly what the
    /// secondary's degraded-link STOP exists for. Two things have to hold:
    /// the phase stays honest about the link (§16 never silently downgrades),
    /// and the recovery `tick` performs must not restore `.live` for a rally
    /// that is already over, which is what `phaseBeforeDegraded` would
    /// otherwise still be holding.
    func testARallyEndingWhileDegradedRecoversToReadyNotLive() {
        var t = syncToReady()
        primary.goLive(); secondary.goLive()
        XCTAssertEqual(primary.phase, .live)

        // Silence the link mid-rally: hold every control frame the secondary
        // sends, so the primary degrades out of `.live`.
        pair.1.controlDeliveryHook = { _, _ in }
        for _ in 0..<50 { t += 0.5; primary.tick(now: t) }
        guard case .degraded = primary.phase else {
            return XCTFail("setup: expected degraded, got \(primary.phase)")
        }

        primary.endRally()
        guard case .degraded = primary.phase else {
            return XCTFail("endRally must not paper over a link that is still down, "
                           + "got \(primary.phase)")
        }

        pair.1.controlDeliveryHook = nil
        for _ in 0..<10 { t += 0.5; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready,
                       "recovery must land on ready — the rally it degraded out of is over")
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
