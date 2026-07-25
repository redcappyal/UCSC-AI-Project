// ios/Tests/LiveCallTests.swift
import XCTest
import simd
@testable import SquashLineCalling

final class LiveCallTests: XCTestCase {
    private func impact(call: String, confidence: String,
                        surface: String = "front_wall",
                        point: SIMD3<Double> = SIMD3(10, 0, 5)) -> StereoImpact {
        StereoImpact(tS: 1.0, surface: surface, pointFt: point, call: call,
                     marginFt: 1.0, confidence: confidence, snapDisagreementFt: nil)
    }

    // MARK: - §8.18's eight filled states

    func testHighConfidenceVerdictsCarryTheirCallWord() {
        for (call, word) in [("in", "IN"), ("out", "OUT"), ("down", "DOWN")] {
            let p = CallPresentation.from(impact(call: call, confidence: "high"))
            XCTAssertEqual(p.word, word)
            XCTAssertEqual(p.detail, "high confidence")
            XCTAssertTrue(p.isVerdict)
        }
    }

    func testOneViewIsStillAVerdictNotAGuess() {
        for (call, word) in [("in", "IN"), ("out", "OUT"), ("down", "DOWN")] {
            let p = CallPresentation.from(impact(call: call, confidence: "one_view"))
            XCTAssertEqual(p.word, word)
            XCTAssertEqual(p.detail, "one view")
            // One camera occluded is not the same as not knowing.
            XCTAssertTrue(p.isVerdict)
        }
    }

    func testFloorBounceIsNotALineVerdict() {
        let p = CallPresentation.from(impact(call: "bounce", confidence: "high",
                                             surface: "floor"))
        XCTAssertEqual(p.word, "NO CALL")
        XCTAssertEqual(p.detail, "floor bounce")
        XCTAssertFalse(p.isVerdict)
    }

    func testNoCallConfidenceReadsObstructed() {
        let p = CallPresentation.from(impact(call: "in", confidence: "no_call"))
        XCTAssertEqual(p.word, "NO CALL")
        XCTAssertEqual(p.detail, "obstructed")
        XCTAssertFalse(p.isVerdict)
    }

    /// The trap this whole component exists to avoid.
    ///
    /// `no_call` is a *confidence tier*, not a call — a no_call impact still
    /// carries a normal-looking `call` and `marginFt`. The `no_call` golden is
    /// `call: "in"`, `margin_ft: 3.487`, at a point 1.67 ft *behind* the front
    /// wall (raw, unsnapped triangulation). Anything that reads `.call`
    /// without gating on `.confidence` renders a confident IN from a
    /// physically impossible point.
    func testNoCallWinsOverTheCallItStillCarries() {
        let golden = impact(call: "in", confidence: "no_call",
                            point: SIMD3(13.63, -1.67, 5.07))
        let p = CallPresentation.from(golden)
        XCTAssertEqual(p.word, "NO CALL")
        XCTAssertNotEqual(p.word, "IN")
        XCTAssertFalse(p.isVerdict)
    }

    /// A floor impact that also failed to snap satisfies both no-call rules.
    /// Confidence wins: when nothing snapped we do not reliably know it was
    /// the floor either, so claiming "floor bounce" would assert geometry the
    /// engine did not resolve.
    func testUnresolvedFloorImpactReportsObstructedNotFloorBounce() {
        let p = CallPresentation.from(impact(call: "bounce", confidence: "no_call",
                                             surface: "floor"))
        XCTAssertEqual(p.word, "NO CALL")
        XCTAssertEqual(p.detail, "obstructed")
        XCTAssertFalse(p.isVerdict)
    }

    func testUnknownCallStringDegradesToNoCall() {
        // Relayed peer JSON is not trusted to stay in the vocabulary.
        let p = CallPresentation.from(impact(call: "wibble", confidence: "high"))
        XCTAssertEqual(p.word, "NO CALL")
        XCTAssertFalse(p.isVerdict)
    }

    func testEveryFilledBannerStateIsOneOfTheEightDESIGNAllows() {
        var seen = Set<String>()
        for call in ["in", "out", "down", "bounce"] {
            for confidence in ["high", "one_view", "no_call"] {
                let p = CallPresentation.from(impact(call: call, confidence: confidence))
                seen.insert("\(p.word) · \(p.detail)")
            }
        }
        // §8.18: 6 verdict combinations (3 words x 2 tiers) + 2 no-call reasons.
        let allowed: Set<String> = [
            "IN · high confidence", "IN · one view",
            "OUT · high confidence", "OUT · one view",
            "DOWN · high confidence", "DOWN · one view",
            "NO CALL · obstructed", "NO CALL · floor bounce",
        ]
        XCTAssertTrue(seen.isSubset(of: allowed),
                      "illegal banner states: \(seen.subtracting(allowed))")
    }

    // MARK: - §8.17 colour law

    func testDownTakesTheOutColourNotAThirdHue() {
        // A tin fault ends the rally the same way an out ball does.
        XCTAssertEqual(CallPresentation.from(impact(call: "down", confidence: "high")).color,
                       CallPresentation.from(impact(call: "out", confidence: "high")).color)
    }

    func testNoCallNeverBorrowsGreenOrRed() {
        let noCall = CallPresentation.from(impact(call: "in", confidence: "no_call"))
        XCTAssertEqual(noCall.color, Theme.unknown)
        XCTAssertNotEqual(noCall.color, Theme.inCall)
        XCTAssertNotEqual(noCall.color, Theme.outCall)
    }

    func testInIsTheOnlyGreenState() {
        XCTAssertEqual(CallPresentation.from(impact(call: "in", confidence: "high")).color,
                       Theme.inCall)
    }

    // MARK: - CallAnnouncer

    @MainActor
    func testAnnouncerIsSilentByDefault() {
        let announcer = CallAnnouncer()
        // The spec records audible calls as an undecided product choice, so
        // this must not quietly default on.
        XCTAssertFalse(announcer.isEnabled)
        announcer.announce(CallPresentation.from(impact(call: "out", confidence: "high")))
        XCTAssertNil(announcer.lastSpokenForTesting)
    }

    @MainActor
    func testAnnouncerNeverSpeaksANonVerdict() {
        let announcer = CallAnnouncer()
        announcer.isEnabled = true
        announcer.announce(CallPresentation.from(impact(call: "in", confidence: "no_call")))
        XCTAssertNil(announcer.lastSpokenForTesting)
        announcer.announce(CallPresentation.from(impact(call: "bounce", confidence: "high")))
        XCTAssertNil(announcer.lastSpokenForTesting)
    }

    @MainActor
    func testAnnouncerSpeaksTheVerdictWordWhenEnabled() {
        let announcer = CallAnnouncer()
        announcer.isEnabled = true
        announcer.announce(CallPresentation.from(impact(call: "out", confidence: "one_view")))
        XCTAssertEqual(announcer.lastSpokenForTesting, "OUT")
    }
}
