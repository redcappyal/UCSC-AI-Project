// ios/Tests/PairingModelTests.swift
import XCTest
@testable import SquashLineCalling

@MainActor
final class PairingModelTests: XCTestCase {
    /// Two sessions wired over loopback, plus the models under test.
    ///
    /// The transport pair is returned rather than stored because
    /// `LoopbackTransport.peer` is `weak` — drop either end and delivery goes
    /// silently dead, which looks exactly like a test that never handshakes.
    /// Built by a factory instead of `setUp()` so the class can stay
    /// `@MainActor` (`PairingModel` is main-isolated) without overriding a
    /// non-isolated `XCTestCase` member.
    private func makeRig() -> (
        pair: (LoopbackTransport, LoopbackTransport),
        primarySession: PeerSession, secondarySession: PeerSession,
        primary: PairingModel, secondary: PairingModel
    ) {
        let pair = LoopbackTransport.pair()
        let primarySession = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondarySession = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        return (pair, primarySession, secondarySession,
                PairingModel(session: primarySession),
                PairingModel(session: secondarySession))
    }

    // MARK: - Unpaired: the single-camera app

    func testUnpairedModelStaysIdleAndEveryControlIsANoOp() {
        let model = PairingModel(session: nil)
        XCTAssertEqual(model.step, .idle)
        XCTAssertEqual(model.statusLine, "Not paired")
        XCTAssertFalse(model.canConfirm)

        // Graceful degradation is a hard requirement: with no peer the app is
        // exactly today's single-camera app, so none of these may do anything.
        model.start(); model.confirm(); model.goLive(); model.end()
        model.pump(now: 1.0)

        XCTAssertEqual(model.step, .idle)
        XCTAssertEqual(model.statusLine, "Not paired")
    }

    // MARK: - Pairing

    func testConfirmStepCarriesTheSameFourDigitCodeOnBothSides() {
        let rig = makeRig()
        rig.secondary.start(); rig.primary.start()

        guard case .confirm(let codeA) = rig.primary.step,
              case .confirm(let codeB) = rig.secondary.step else {
            return XCTFail("both sides must reach confirm, got "
                           + "\(rig.primary.step) / \(rig.secondary.step)")
        }
        // The operators compare these across the court — they must agree.
        XCTAssertEqual(codeA, codeB)
        XCTAssertEqual(codeA.count, 4)
        XCTAssertTrue(codeA.allSatisfy(\.isNumber), "pair code must be digits, got \(codeA)")
        XCTAssertEqual(rig.primary.statusLine, "Codes match?")
        XCTAssertTrue(rig.primary.canConfirm)
    }

    func testConfirmingBothSidesReachesReadyWithAFiniteSyncFigure() {
        let rig = makeRig()
        rig.secondary.start(); rig.primary.start()
        rig.primary.confirm(); rig.secondary.confirm()

        var t = 0.0
        for _ in 0..<40 { t += 0.1; rig.primary.pump(now: t); rig.secondary.pump(now: t) }

        // Never compare `.ready` by value — syncMs is a float.
        guard case .ready(let syncMs) = rig.primary.step else {
            return XCTFail("expected ready, got \(rig.primary.step)")
        }
        XCTAssertTrue(syncMs.isFinite)
        XCTAssertGreaterThanOrEqual(syncMs, 0)
        // PeerSession only leaves .syncing under readyUncertainty (5 ms).
        XCTAssertLessThanOrEqual(syncMs, 5.0)
        XCTAssertTrue(rig.primary.statusLine.hasPrefix("Paired · sync ±"),
                      "got \(rig.primary.statusLine)")
        XCTAssertFalse(rig.primary.canConfirm)
    }

    func testGoLiveAdvancesOnlyFromReady() {
        let rig = makeRig()
        rig.secondary.start(); rig.primary.start()

        // Still confirming — goLive() must not skip the sync pass.
        rig.primary.goLive()
        XCTAssertNotEqual(rig.primary.step, .live)

        rig.primary.confirm(); rig.secondary.confirm()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; rig.primary.pump(now: t); rig.secondary.pump(now: t) }
        rig.primary.goLive()
        XCTAssertEqual(rig.primary.step, .live)
        // The link is still up mid-rally, so the row keeps reporting quality.
        XCTAssertTrue(rig.primary.statusLine.hasPrefix("Paired · sync ±"))
    }

    // MARK: - Honest failure states

    func testSilencedLinkDegradesAndSaysRecordingContinues() {
        let rig = makeRig()
        rig.secondary.start(); rig.primary.start()
        rig.primary.confirm(); rig.secondary.confirm()

        var t = 0.0
        for _ in 0..<40 { t += 0.1; rig.primary.pump(now: t); rig.secondary.pump(now: t) }

        // Hold every control frame the secondary sends: no pongs, no beats.
        rig.pair.1.controlDeliveryHook = { _, _ in }
        for _ in 0..<50 { t += 0.5; rig.primary.pump(now: t) }

        guard case .degraded = rig.primary.step else {
            return XCTFail("expected degraded, got \(rig.primary.step)")
        }
        // Never a silent downgrade, and the wording must say recording survives.
        XCTAssertTrue(rig.primary.statusLine.contains("still recording"),
                      "got \(rig.primary.statusLine)")
    }

    func testVersionMismatchSurfacesTheSessionReasonVerbatim() {
        let rig = makeRig()
        // Corrupt the secondary's hello version by intercepting control frames.
        rig.pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var hello)? = ControlMessage.decode(frame) else {
                return deliver(frame)
            }
            hello.protoVersion = 99
            deliver(try! ControlMessage.encode(.hello(hello)))
        }
        rig.secondary.start(); rig.primary.start()

        guard case .failed(let reason) = rig.primary.step else {
            return XCTFail("expected failed, got \(rig.primary.step)")
        }
        guard case .failed(let sessionReason) = rig.primarySession.phase else {
            return XCTFail("the session itself should have failed")
        }
        // §16: the link-status row shows the failure reason verbatim — no
        // rewriting, so a mount/orientation guard reads as itself on screen.
        XCTAssertEqual(reason, sessionReason)
        XCTAssertEqual(rig.primary.statusLine, sessionReason)
    }

    func testEndingReturnsToIdleSoRepairingIsOneTap() {
        let rig = makeRig()
        rig.secondary.start(); rig.primary.start()
        rig.primary.end()

        // §16: ending returns p-pair to its idle state, not all the way to Play.
        XCTAssertEqual(rig.primary.step, .idle)
        XCTAssertEqual(rig.primary.statusLine, "Not paired")
    }

    // MARK: - The pure mapping

    func testPhaseMappingCoversEveryStateInTheBlueprintTable() {
        // §16's p-pair table, row for row, without a live session.
        XCTAssertEqual(PairingModel.map(.idle, syncMs: 0).1, "Not paired")
        XCTAssertEqual(PairingModel.map(.searching, syncMs: 0).1,
                       "Looking for the other phone…")
        XCTAssertEqual(PairingModel.map(.confirming(code: "0421"), syncMs: 0).1,
                       "Codes match?")
        XCTAssertEqual(PairingModel.map(.syncing, syncMs: 0).1, "Syncing clocks…")
        XCTAssertEqual(PairingModel.map(.ready, syncMs: 1.44).1, "Paired · sync ±1.4 ms")
        XCTAssertEqual(PairingModel.map(.degraded("whatever"), syncMs: 0).1,
                       "Link lost — still recording")
        XCTAssertEqual(PairingModel.map(.ended, syncMs: 0).0, .idle)
        XCTAssertEqual(PairingModel.map(.failed("mount both phones the same way"), syncMs: 0).1,
                       "mount both phones the same way")
    }
}
