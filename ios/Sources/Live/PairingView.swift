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
    /// Fires when `model.rally` becomes `.recording` — an automatic advance
    /// to `p-live` (§16), the same pattern as `p-analyze`'s auto-advance, not
    /// a tap outcome. Driven from `body`'s `.onChange(of: model.rally)`, which
    /// is what makes this correct for both roles: the primary's own START
    /// RALLY tap sets `rally` synchronously, and the secondary's `rally`
    /// flips the same way the moment a remote record-start message lands,
    /// with no tap on this phone at all.
    var onGoLive: () -> Void = {}

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
        // §16: p-pair → p-live is an automatic advance on `rally` becoming
        // `.recording`, not a side effect of whichever tap happened to cause
        // it. Coupling this to the button (a prior shape of this file did)
        // breaks on the secondary: `rally` can flip to `.recording` from a
        // remote message while the secondary's own screen is still on
        // Confirm, and the secondary's *next* tap — an ordinary CONFIRM,
        // unrelated to the rally starting — would then be misread as the
        // trigger. Observing `model.rally` itself fires at the actual moment
        // it changes, for either role, independent of any tap. `onChange`
        // only fires on a real change of the `Equatable` value, so this
        // cannot double-fire while `rally` sits at `.recording`, and a second
        // rally (a later `.recording` reached after leaving it) fires again
        // the same way.
        .onChange(of: model.rally) { _, newValue in
            if newValue == .recording { onGoLive() }
        }
    }

    // MARK: - §8.19 link status

    /// 8 px `--dim` dot + an explicit sentence. The dot is decorative and
    /// carries no meaning on its own — a green "connected" dot would silently
    /// claim the verdict family (§0.3 reserves green/red for IN/OUT), so this
    /// component is built so that mistake cannot be made.
    private var linkStatus: some View {
        HStack(spacing: 9) {
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
                Button("Codes don't match") { model.pairing.rejectCode() }
                    .font(.system(.subheadline).weight(.semibold))
                    .foregroundStyle(Theme.dim)
                    .frame(minHeight: 44)
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
        // Navigation is not decided here — see the `.onChange(of: model.
        // rally)` observation in `body` and `onGoLive`'s doc comment above.
        // This action only ever advances the model's own state machine.
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
