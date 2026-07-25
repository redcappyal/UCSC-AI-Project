// ios/Tests/CaptureOrientationTests.swift
import XCTest
@testable import SquashLineCalling

final class CaptureOrientationTests: XCTestCase {
    func testBothMountsShareOneUprightLandscapeFrameSpace() {
        // Identical dimensions for both mounts is the whole reason the wire
        // needs an explicit orientation field: width/height cannot tell a
        // landscape-left mount from a landscape-right one.
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let s = CaptureSettings.frameSize(for: orientation)
            XCTAssertEqual(s.width, CaptureSettings.sensorWidth, "\(orientation)")
            XCTAssertEqual(s.height, CaptureSettings.sensorHeight, "\(orientation)")
        }
    }

    func testRotationNormalizesEachMountUpright() {
        // Landscape-right matches the sensor's native readout; landscape-left
        // is that readout upside down, so it needs 180 to record upright.
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeRight), 0)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeLeft), 180)
    }

    func testFrameConstantsAreLandscape() {
        XCTAssertEqual(CaptureSettings.frameWidth, 3840)
        XCTAssertEqual(CaptureSettings.frameHeight, 2160)
    }

    func testMismatchedPeerFrameSizeIsRejected() {
        let pair = LoopbackTransport.pair()
        // Corrupt the incoming hello's frame size to simulate a peer in the
        // other orientation: same pixels, transposed — silently fatal for 3D.
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var h)? = ControlMessage.decode(frame) else { return deliver(frame) }
            (h.frameW, h.frameH) = (h.frameH, h.frameW)
            deliver(try! ControlMessage.encode(.hello(h)))
        }
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failure, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation")
                      || why.lowercased().contains("frame"), "unhelpful reason: \(why)")
    }

    /// The regression the guard existed for but could not have. Before this
    /// change both sessions advertised the same compile-time constants, so
    /// no pair of PeerSessions could disagree no matter how they were
    /// configured.
    func testOppositeLandscapeMountsAreRefused() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 },
                                  captureOrientation: .landscapeRight)
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 },
                                    captureOrientation: .landscapeLeft)
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failed on the primary, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation"), "unhelpful reason: \(why)")
        // Both sides must refuse: each runs its own guard against the other's
        // hello, and a one-sided refusal would leave the peer waiting.
        //
        // Incidental dependency: secondary only gets a chance to see
        // primary's hello (and thus reach .failed itself) because
        // primary.start() still calls transport.startInitiator() — and
        // handleTransportState still sends .hello on .connected — even
        // though primary's own phase is already .failed from the first,
        // synchronously-delivered exchange. Neither call is phase-guarded
        // today; that's production-equivalent (a failed session has nothing
        // left to protect), but it means this assertion also implicitly
        // relies on it. A future `guard internalPhase == .searching` before
        // sending .hello in handleTransportState would silently stop
        // secondary from ever receiving primary's hello, and this
        // assertion would then fail for a reason unrelated to whether the
        // orientation guard itself is correct.
        guard case .failed = secondary.phase else {
            return XCTFail("expected failed on the secondary, got \(secondary.phase)")
        }
    }

    func testMatchingMountsStillPair() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 },
                                  captureOrientation: .landscapeLeft)
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 },
                                    captureOrientation: .landscapeLeft)
        secondary.start(); primary.start()
        guard case .confirming = primary.phase else {
            return XCTFail("a matched pair must still pair, got \(primary.phase)")
        }
    }

    /// A peer on a build predating the field sends a hello with no
    /// captureOrientation key. It must decode (an Optional field is what makes
    /// that true) and then be refused with the orientation message — not
    /// dropped silently, and not reported as a version mismatch.
    func testLegacyPeerWithoutOrientationIsRefused() {
        let pair = LoopbackTransport.pair()
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var h)? = ControlMessage.decode(frame) else { return deliver(frame) }
            h.captureOrientation = nil     // JSONEncoder omits the key entirely
            deliver(try! ControlMessage.encode(.hello(h)))
        }
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failed, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation"), "unhelpful reason: \(why)")
        XCTAssertFalse(why.lowercased().contains("protocol version"),
                       "a legacy hello must not be reported as a version mismatch")
    }

    /// A hello with no orientation key must still DECODE. If this fails, the
    /// field was made non-optional and pairing with an older build will hang
    /// in .searching with no diagnostic at all.
    ///
    /// Built by encoding a nil mount rather than from a hand-written JSON
    /// literal: Swift synthesizes Codable for enums with unlabeled associated
    /// values under a "_0" key, so a literal guessed from the struct's field
    /// names would not decode and the test would pass for the wrong reason.
    /// JSONEncoder omits nil Optionals, so these bytes ARE a legacy peer's.
    func testHelloWithoutOrientationKeyStillDecodes() {
        let legacy = Hello(protoVersion: peerProtoVersion, appVersion: "dev",
                           deviceModel: "x", nonce: 7,
                           frameW: CaptureSettings.frameWidth,
                           frameH: CaptureSettings.frameHeight,
                           captureOrientation: nil)
        let data = try! ControlMessage.encode(.hello(legacy))
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("captureOrientation"),
                       "a nil mount must be absent from the wire, matching a legacy peer's bytes")
        guard case .hello(let decoded)? = ControlMessage.decode(data) else {
            return XCTFail("a legacy hello must decode, not drop")
        }
        XCTAssertNil(decoded.captureOrientation)
    }

    /// Not a version-bump tripwire in the ordinary sense: both sessions in
    /// every other test here build their hello from the same
    /// `peerProtoVersion` constant, so a bump to 2 would leave them matching
    /// each other and every other test would keep passing. What a bump
    /// actually breaks is invisible to this suite — a real peer still on
    /// version 1 would fail the protoVersion check in `handleControl`
    /// *before* the orientation guard ever runs, so it would be reported as
    /// a version mismatch instead of the specific, actionable orientation
    /// error. Pinned directly since nothing else here can catch that.
    func testProtoVersionStaysPinnedSoLegacyPeersReachTheOrientationGuard() {
        XCTAssertEqual(peerProtoVersion, 1,
                       "bumping this makes the orientation guard unreachable for legacy peers: the version check runs first")
    }

    /// `CaptureOrientation`'s raw values are wire format, not an internal
    /// implementation detail — pairing across app versions round-trips them
    /// through JSON. Renaming a case (e.g. `landscapeRight` →
    /// `landscapeStandard`) type-checks as a pure refactor but silently
    /// changes the encoded bytes, breaking any cross-version pair where one
    /// side has renamed and the other hasn't. Pin every case's raw value —
    /// pinning only one leaves the other exactly as breakable as no pin at
    /// all.
    ///
    /// Also pins the JSON *key*, `"captureOrientation"` itself, not just its
    /// value: a `CodingKeys` rename would break cross-version pairing just
    /// as silently. This is what anchors
    /// `testHelloWithoutOrientationKeyStillDecodes`'s negative assertion (the
    /// key is ABSENT for a nil mount) — that check alone can never fail,
    /// because under a renamed key "captureOrientation" is absent from the
    /// wire unconditionally, nil mount or not, and the assertion would keep
    /// passing for the wrong reason. Asserting here that the SAME literal
    /// key is PRESENT for a real, non-nil mount proves it is the actual key
    /// on the wire, so the other test's absence check means something.
    func testCaptureOrientationWireValuesAndKeyArePinned() {
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let hello = Hello(protoVersion: peerProtoVersion, appVersion: "dev", deviceModel: "x",
                              nonce: 1, frameW: CaptureSettings.frameWidth,
                              frameH: CaptureSettings.frameHeight,
                              captureOrientation: orientation)
            let json = String(decoding: try! ControlMessage.encode(.hello(hello)), as: UTF8.self)
            XCTAssertTrue(json.contains("captureOrientation"),
                         "a non-nil mount must put the key on the wire, anchoring " +
                         "testHelloWithoutOrientationKeyStillDecodes's absence check")
            XCTAssertTrue(json.contains(orientation.rawValue),
                         "the raw wire value for \(orientation) must stay " +
                         "\"\(orientation.rawValue)\" — renaming the case silently " +
                         "changes the bytes and breaks cross-version pairing")
        }
    }
}
