// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Sources/Live/LiveCallView.swift
import Foundation
import SwiftUI

/// Presentation-only mapping from an impact to what the screen says.
///
/// Pure and synchronous so the honest-states contract (DESIGN.md §8.18) is
/// unit-testable without standing up a view. The rendering rule that matters:
/// **gate on `confidence`, never on `call`**. `no_call` is a confidence tier,
/// not a call — a `no_call` impact still carries a normal-looking `call` and
/// `marginFt` from raw, unsnapped triangulation, and the goldens contain
/// exactly that case (`call: "in"`, `margin_ft: 3.487`, at a point behind the
/// front wall). Reading `.call` first renders a confident IN from a point that
/// cannot physically exist.
struct CallPresentation: Equatable {
    let word: String        // "IN" | "OUT" | "DOWN" | "NO CALL"
    let detail: String      // "high confidence" | "one view" | "obstructed" | "floor bounce"
    let isVerdict: Bool     // false => not a line call; the announcer stays quiet

    /// The single source of truth for verdict colour (§8.17).
    ///
    /// `isVerdict` alone cannot choose between green and red, and
    /// `MiniCourtView` (§8.10) must mark its impact with *the same* colour so
    /// the replay always agrees with the call that was made — so the mapping
    /// lives in exactly one place. Computed, so `Equatable` synthesis is
    /// unaffected and tests never have to compare `Color`s structurally.
    var color: Color {
        switch word {
        case "IN": return Theme.inCall
        // A tin fault ends the rally the same way an out ball does, so DOWN
        // takes the out colour rather than minting a third hue.
        case "OUT", "DOWN": return Theme.outCall
        // Never green or red: a no-call is explicitly not a verdict (§0.3).
        default: return Theme.unknown
        }
    }

    static func from(_ impact: StereoImpact) -> CallPresentation {
        // Only these two fields decide what the screen says — see
        // `from(call:confidence:)`, which is also what the secondary's relayed
        // JSON reaches, so both roles are gated by one piece of code.
        from(call: impact.call, confidence: impact.confidence)
    }

    /// The secondary's view of a call.
    ///
    /// The secondary owns no `StereoEngine` (`RecordModel.attachStereo` is
    /// primary-only), so the only thing it ever has is the JSON the primary
    /// put on the wire — `{surface, call, margin_ft, confidence, t_s}`, built
    /// in that same file's `engine.onEvent`. DESIGN.md §16's `p-live` states
    /// table draws no role distinction for the banner or the flash, and the
    /// spec's deliberate v1 narrowing is only about the mini-court (which
    /// needs the 3D track this payload genuinely does not carry) — so the two
    /// fields that decide the banner are read straight out of it and pushed
    /// through the *same* mapping the primary's own impact takes. No
    /// `StereoImpact` is reconstructed: the payload carries no geometry, and
    /// inventing a point to fill one in would be a lie that later readers
    /// could take seriously.
    ///
    /// `nil` — not a presentation — when the payload is not an object or is
    /// missing either field: a banner is a claim that a call happened, and
    /// there is nothing here to claim. An unrecognised *value* is a different
    /// case and deliberately does reach the mapping, which degrades it to
    /// `NO CALL · obstructed` (`verdictWord`'s `default`).
    static func fromRelayedEvent(json: String) -> CallPresentation? {
        guard let data = json.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let call = object["call"] as? String,
              let confidence = object["confidence"] as? String else { return nil }
        return from(call: call, confidence: confidence)
    }

    /// The one gate, for both roles and both sources of a call.
    ///
    /// Takes the two fields rather than a `StereoImpact` so the secondary's
    /// relayed JSON can reach it without a fabricated impact — the mapping
    /// never read anything else anyway.
    private static func from(call: String, confidence: String) -> CallPresentation {
        // Order is load-bearing. Confidence first: when nothing snapped we do
        // not reliably know the surface either, so "floor bounce" would assert
        // geometry the engine did not resolve. "obstructed" is the honest word.
        if confidence == "no_call" {
            return CallPresentation(word: "NO CALL", detail: "obstructed", isVerdict: false)
        }
        // A floor bounce is not a line verdict, whatever the confidence.
        if call == "bounce" {
            return CallPresentation(word: "NO CALL", detail: "floor bounce", isVerdict: false)
        }
        guard let word = verdictWord(call) else {
            // Impact JSON can arrive relayed from the peer; an unrecognised
            // call is a bug or a version skew, and either way it is not
            // something to render as a confident verdict.
            return CallPresentation(word: "NO CALL", detail: "obstructed", isVerdict: false)
        }
        // §8.18: verdicts always carry a confidence phrase, never a bare word.
        let detail = confidence == "high" ? "high confidence" : "one view"
        return CallPresentation(word: word, detail: detail, isVerdict: true)
    }

    private static func verdictWord(_ call: String) -> String? {
        switch call {
        case "in": return "IN"
        case "out": return "OUT"
        case "down": return "DOWN"
        default: return nil
        }
    }
}

/// §8.17 call flash — a full-stage colour wash with the verdict word.
///
/// Belongs on the **stage overlay layer**, never in the layout flow, so
/// appearing and clearing shifts nothing (§0.9). Kept separate from the banner
/// because DESIGN.md defines them as two components living in two different
/// places: the wash is transient and over the video, the banner is persistent
/// and beneath it.
struct CallFlashView: View {
    let presentation: CallPresentation?

    /// §10: the flash is skipped entirely under reduced motion — the end state
    /// appears directly. A state change, not an animation.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View { flash }

    private var flash: some View {
        ZStack {
            if let presentation {
                presentation.color.opacity(0.82)
                Text(presentation.word)
                    .font(.system(size: 44, weight: .bold))
                    .tracking(0.04 * 44)
                    .foregroundStyle(presentation.word == "IN" ? Theme.bg : Theme.text)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)          // display-only, like the analyzing scrim
        // §10 budgets the whole flash at ≤ 500 ms; the hold
        // (RecordModel.flashHoldS) takes the rest, so keep the ease short.
        .animation(reduceMotion ? nil : .easeOut(duration: 0.15), value: presentation)
    }
}

/// §8.18 call banner — the persistent one-line result under the stage.
///
/// Reserves its height at all times (the §8.7 reserved-space pattern) so a
/// call appearing or clearing never shifts the layout beneath it.
struct CallBannerView: View {
    let presentation: CallPresentation?

    var body: some View { banner }

    private var banner: some View {
        HStack(spacing: 8) {
            if let presentation {
                Text(presentation.word)
                    .font(.system(.headline).weight(.bold))
                    .foregroundStyle(presentation.color)
                Text("·").foregroundStyle(Theme.dim)
                // The confidence phrase stays dim and sentence case whatever
                // the word's colour, so it reads calmly under a red fill.
                Text(presentation.detail)
                    .font(.subheadline)
                    .foregroundStyle(Theme.dim)
            } else {
                Text("No call yet")
                    .font(.subheadline)
                    .foregroundStyle(Theme.dim)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 44)
        .padding(.horizontal, 14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(presentation == nil ? Color.clear : Theme.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(Theme.line,
                              style: StrokeStyle(lineWidth: 1,
                                                 dash: presentation == nil ? [4, 4] : []))
        )
    }
}

/// The plan's composed entry point: flash over banner, for any host that wants
/// both stacked rather than placed separately. `RecordView` places the two
/// halves itself — the flash on the stage overlay layer, the banner under it —
/// because that is where DESIGN.md puts them.
struct LiveCallView: View {
    let presentation: CallPresentation?

    init(presentation: CallPresentation?) {
        self.presentation = presentation
    }

    var body: some View {
        VStack(spacing: 0) {
            CallFlashView(presentation: presentation)
            CallBannerView(presentation: presentation)
        }
    }
}
