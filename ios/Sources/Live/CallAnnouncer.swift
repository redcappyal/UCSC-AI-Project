// ios/Sources/Live/CallAnnouncer.swift
import AVFoundation
import Foundation

/// Speaks the verdict word aloud, for the referee-less court.
///
/// **Defaults off.** The spec records "audible calls on or off" as an open
/// product question, and an app that starts shouting IN/OUT the first time it
/// pairs has answered that question on the user's behalf. Turning it on is a
/// decision someone has to make.
///
/// Never speaks a non-verdict: `NO CALL` is the app admitting it could not
/// see, and saying that out loud mid-rally is noise, not information.
@MainActor
final class CallAnnouncer {
    var isEnabled = false

    /// Last word actually handed to the synthesiser, so the gating is
    /// testable without a speech engine or an audio session.
    private(set) var lastSpokenForTesting: String?

    /// Retained deliberately: a synthesiser created inside `announce` would
    /// deallocate as the method returns and cut its own utterance off.
    private let synthesizer = AVSpeechSynthesizer()

    func announce(_ presentation: CallPresentation) {
        guard isEnabled, presentation.isVerdict else { return }
        lastSpokenForTesting = presentation.word
        let utterance = AVSpeechUtterance(string: presentation.word)
        // Calm and terse — the referee's voice (§14), not an announcer's.
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        synthesizer.speak(utterance)
    }
}
