// ios/Sources/Live/PairingView.swift
import SwiftUI

/// `p-pair` — DESIGN.md §16's live-mode state table rendered.
///
/// One primary action the whole way through (PAIR → CONFIRM → START RALLY,
/// §7), a `.link-status` row that speaks in words (§8.19), and a `.pair-code`
/// card that reserves its footprint so nothing below it jumps when the code
/// arrives (§8.20, §0.9).
struct PairingView: View {
    @ObservedObject var model: PairingModel
    /// Tapping START RALLY hands off to `p-live` — the host owns that
    /// transition, the same way `p-analyze` auto-advances.
    var onGoLive: () -> Void = {}

    /// Declared unconditionally even though only the DEBUG picker reads it:
    /// a stored property inside `#if DEBUG` would give the struct a different
    /// memberwise initializer in release than in debug, and every call site
    /// would have to fork with it.
    @State private var transportName = "ble"

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            VStack(spacing: 18) {
                linkStatus
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
    }

    // MARK: - §8.19 link status

    /// 8 px `--dim` dot + an explicit sentence. The dot is decorative and
    /// carries no meaning on its own — a green "connected" dot would silently
    /// claim the verdict family (§0.3 reserves green/red for IN/OUT), so this
    /// component is built so that mistake cannot be made.
    private var linkStatus: some View {
        HStack(spacing: 9) {
            Circle().fill(Theme.dim).frame(width: 8, height: 8)
            Text(model.statusLine)
                .font(.system(.subheadline).monospacedDigit())
                .foregroundStyle(Theme.dim)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        // Reserve two lines: the failure reason is shown verbatim and is
        // longer than "Not paired", and the row must not shove the card down.
        .frame(minHeight: 40, alignment: .leading)
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
                Button("Codes don't match") { model.rejectCode() }
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
        .opacity(model.step == .idle ? 0 : 1)
    }

    private var displayedCode: String? {
        if case .confirm(let code) = model.step { return code }
        return nil
    }

    // MARK: - §7 the one primary

    private var primary: some View {
        Button(action: primaryAction) {
            Text(primaryTitle)
                .font(.system(.headline).weight(.bold))
                .tracking(0.05 * 17)
                .foregroundStyle(Theme.accentText)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(Theme.accentBg, in: Capsule())
                .opacity(primaryEnabled ? 1 : 0.4)
        }
        .disabled(!primaryEnabled)
    }

    /// PAIR → CONFIRM → START RALLY, exactly the §16 table's Primary column.
    private var primaryTitle: String {
        switch model.step {
        case .idle, .searching, .failed: return "PAIR"
        case .confirm, .syncing:         return "CONFIRM"
        case .ready:                     return "START RALLY"
        case .live:                      return "RALLY LIVE"
        case .degraded:                  return "START RALLY"
        }
    }

    private var primaryEnabled: Bool {
        switch model.step {
        // Searching has no peer yet to act on; syncing needs no user action.
        case .searching, .syncing, .live: return false
        case .idle, .failed:              return true
        case .confirm:                    return model.canConfirm
        case .ready:                      return true
        // Degraded leaves recording untouched, so the primary is unaffected.
        case .degraded:                   return false
        }
    }

    private func primaryAction() {
        switch model.step {
        case .idle, .failed:  model.start()
        case .confirm:        model.confirm()
        case .ready:          model.goLive(); onGoLive()
        case .searching, .syncing, .live, .degraded: break
        }
    }

    // MARK: - DEBUG

    #if DEBUG
    /// Same picker as the bench harness — the transport choice is still an
    /// open Phase 1 question, so both radios stay reachable.
    private var transportPicker: some View {
        Picker("Transport", selection: $transportName) {
            Text("Bluetooth").tag("ble")
            Text("Wi-Fi P2P").tag("wifi-p2p")
        }
        .pickerStyle(.segmented)
        .disabled(model.step != .idle)
    }
    #endif
}
