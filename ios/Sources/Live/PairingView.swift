// ios/Sources/Live/PairingView.swift
import SwiftUI

/// `p-pair` — DESIGN.md §16's live-mode state table rendered.
///
/// A thin renderer over `LiveSessionModel`: `linkStatus`, `primaryTitle`, and
/// `primaryEnabled` already encode §16's whole state table, moved there
/// specifically so the table is assertable without standing up this view
/// (see `LiveSessionModelTests`). This view keeps no state logic of its own —
/// one primary action the whole way through (PAIR → CONFIRM → START RALLY,
/// §7), a `.link-status` row that speaks in words (§8.19), a `.pair-code`
/// card that reserves its footprint so nothing below it jumps when the code
/// arrives (§8.20, §0.9), and — idle state only — the §8.22 role segment.
struct PairingView: View {
    @ObservedObject var model: LiveSessionModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            VStack(spacing: 18) {
                linkStatus
                roleSegment
                pairCode
                Spacer(minLength: 0)
                #if DEBUG
                transportPicker
                #endif
                primary
            }
            .padding(.horizontal, 14)
            .padding(.top, 18)
            .padding(.bottom, 24)
        }
        .task { await model.prepare() }
        // §16's `p-pair` → `p-live` advance is deliberately NOT observed here
        // any more. It fires on `rally` becoming `.recording`, which happens
        // with no tap at all on the secondary (a remote record-start message),
        // and a rally can outlive this screen: this view can be popped while
        // the rally it started is still running, and an observer that has left
        // the hierarchy can never fire again. `PlayRootView` owns the whole
        // stack and drives `p-live` off `live.rally` directly — see its
        // `syncLiveStage()`. Nothing here decides navigation.
    }

    // MARK: - §8.19 link status

    /// 8 px `--dim` dot + an explicit sentence. The dot is decorative and
    /// carries no meaning on its own — a green "connected" dot would silently
    /// claim the verdict family (§0.3 reserves green/red for IN/OUT), so this
    /// component is built so that mistake cannot be made.
    private var linkStatus: some View {
        HStack(spacing: 8) {
            Circle().fill(Theme.dim).frame(width: 8, height: 8)
            Text(model.linkStatus)
                .font(.system(.subheadline).monospacedDigit())
                .foregroundStyle(Theme.dim)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        // Reserve two lines: the failure reason is shown verbatim and is
        // longer than "Not paired", and the row must not shove the card down.
        .frame(minHeight: 40, alignment: .leading)
    }

    // MARK: - §8.22 role segment

    /// §8.22's two-way segment, built native at the full 44 px minimum
    /// (§0.6) rather than inheriting the web control's 38 px debt. No
    /// default selection: two phones both defaulting to primary would both
    /// open a central and hang with nothing honest to show for it.
    ///
    /// Scoped to `p-pair`'s idle state only (§16) and hidden with `.opacity`,
    /// never a conditional, so nothing below it shifts when it disappears
    /// (§0.9) — it keeps its footprint the same way `.pair-code` does.
    /// `allowsHitTesting`/`accessibilityHidden` follow the same condition so
    /// an invisible pill can never register a tap or a VoiceOver swipe once
    /// pairing has started (by which point `LiveSessionModel.role`'s setter
    /// is a no-op anyway — belt and suspenders, not a correctness fix).
    private var roleSegment: some View {
        let isIdle = model.pairing.step == .idle
        return HStack(spacing: 6) {
            segmentHalf("THIS PHONE CALLS", role: .primary)
            segmentHalf("THIS PHONE ASSISTS", role: .secondary)
        }
        .opacity(isIdle ? 1 : 0)
        .allowsHitTesting(isIdle)
        .accessibilityHidden(!isIdle)
    }

    /// One independent pill per §8.22 — no shared capsule, fill, or border
    /// wrapping the pair. Unselected: 1 px `--line` border, transparent
    /// background, inherited `--text` label, weight 600. Selected:
    /// `--accent-bg` fill with `--accent-text`, border transparent, weight
    /// 700 — never green/red, because selection is not a verdict (§0.3).
    private func segmentHalf(_ title: String, role: PeerRole) -> some View {
        let selected = model.role == role
        return Button {
            model.role = role
        } label: {
            Text(title)
                .font(.system(.subheadline).weight(selected ? .bold : .semibold))
                .tracking(0.05 * 15)
                .foregroundStyle(selected ? Theme.accentText : Theme.text)
                .frame(maxWidth: .infinity, minHeight: 44)
                .background(selected ? Theme.accentBg : Color.clear, in: Capsule())
                .overlay(
                    Capsule().strokeBorder(selected ? Color.clear : Theme.line, lineWidth: 1)
                )
        }
        .accessibilityLabel(title)
    }

    // MARK: - §8.20 pair code

    /// Identical footprint filled or blank, so CONFIRM never jumps when the
    /// code appears or clears.
    private var pairCode: some View {
        VStack(spacing: 8) {
            Text("Codes match?")
                .font(.subheadline)
                .foregroundStyle(Theme.dim)
            Text(displayedCode ?? "————")
                .font(.system(size: 38, weight: .bold).monospacedDigit())
                .tracking(0.04 * 38)
                .foregroundStyle(displayedCode == nil ? Theme.dim : Theme.text)
            if displayedCode != nil {
                // A secondary text action, never a second filled primary (§7).
                // §0.6: the frame lives inside the label closure, not outside
                // the Button, so the full 44 pt is part of the tappable
                // region — see `LiveStageView.stop`'s comment for the same
                // bug class (framing outside the label leaves only the
                // Text's ~20 pt intrinsic height hit-testable). This is the
                // stranger's-phone escape hatch (§8.20), so it must actually
                // be reachable across its whole reserved height.
                Button { model.pairing.rejectCode() } label: {
                    Text("Codes don't match")
                        .font(.system(.subheadline).weight(.semibold))
                        .foregroundStyle(Theme.dim)
                        .frame(minHeight: 44)
                }
            } else {
                Color.clear.frame(height: 44)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(displayedCode == nil ? Color.clear : Theme.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Theme.line,
                              style: StrokeStyle(lineWidth: 1,
                                                 dash: displayedCode == nil ? [4, 4] : []))
        )
        .opacity(model.pairing.step == .idle ? 0 : 1)
    }

    private var displayedCode: String? {
        if case .confirm(let code) = model.pairing.step { return code }
        return nil
    }

    // MARK: - §7 the one primary

    private var primary: some View {
        // Navigation is not decided here — `PlayRootView.syncLiveStage()`
        // presents `p-live` off `live.rally`, whoever caused it and whichever
        // screen is on top. This action only ever advances the model's own
        // state machine.
        Button(action: { model.primaryTapped() }) {
            Text(model.primaryTitle)
                .font(.system(.headline).weight(.bold))
                .tracking(0.05 * 17)
                .foregroundStyle(Theme.accentText)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(Theme.accentBg, in: Capsule())
                .opacity(model.primaryEnabled ? 1 : 0.4)
        }
        .disabled(!model.primaryEnabled)
    }

    // MARK: - DEBUG

    #if DEBUG
    /// Same picker as the bench harness — the transport choice is still an
    /// open Phase 1 question, so both radios stay reachable. Bound straight
    /// to `model.transportName`: `LiveSessionModel.beginPairing()` reads
    /// that exact property to build the transport, so a view-local copy
    /// (the previous shape of this control, back when the view only knew
    /// `PairingModel`) would silently disagree with what pairing actually
    /// uses.
    private var transportPicker: some View {
        Picker("Transport", selection: $model.transportName) {
            Text("Bluetooth").tag("ble")
            Text("Wi-Fi P2P").tag("wifi-p2p")
        }
        .pickerStyle(.segmented)
        .disabled(model.pairing.step != .idle)
    }
    #endif
}
