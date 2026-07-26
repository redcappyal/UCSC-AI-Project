// ios/Sources/Live/PairingModel.swift
import Foundation

/// Presentation state for `p-pair` — DESIGN.md §16's live-mode state table.
///
/// A pure mapping from `PeerSession.Phase` to the one step the screen shows,
/// plus the `.link-status` sentence for it (§8.19). No networking of its own:
/// the session owns the link, this owns the words.
///
/// `session == nil` is the unpaired single-camera app. Every control is then a
/// no-op, which is what makes pairing *strictly additive* — a hard requirement,
/// not a design preference.
@MainActor
final class PairingModel: ObservableObject {
    enum Step: Equatable {
        case idle, searching, confirm(code: String), syncing
        case ready(syncMs: Double), live, degraded(String), failed(String)
    }

    @Published private(set) var step: Step = .idle
    @Published private(set) var statusLine: String = PairingModel.notPairedLine

    var canConfirm: Bool { if case .confirm = step { return true }; return false }

    /// True only in `.ready` — `p-pair`'s primary reads START RALLY there and
    /// hands off to `p-live`.
    var canGoLive: Bool { if case .ready = step { return true }; return false }

    /// `PeerSession` is single-use: `start()` only moves `.idle → .searching`,
    /// and the production transports latch themselves torn-down. Once a
    /// session is spent, PAIR has nothing left to do — and §7's one primary
    /// must never advertise a tap that cannot fire. Reviving the screen needs
    /// a fresh session via `onSessionEnded`.
    ///
    /// Note this cannot be decided from `step` alone: `map` folds `.ended`
    /// into `.idle`, so the view cannot tell a spent session from a fresh one.
    var canPair: Bool {
        guard let session else { return false }
        switch session.phase {
        case .ended, .failed: return false
        default: return true
        }
    }

    private let session: PeerSession?

    init(session: PeerSession?) {
        self.session = session
        syncFromSession()
    }

    // MARK: - Controls
    //
    // Each mirrors one `PeerSession` entry point and then republishes. The
    // session applies its own guards (`confirmPairing` is a no-op outside
    // `.confirming`, `goLive` outside `.ready`), so these deliberately do not
    // re-check: one authority for the state machine, not two.

    func start() {
        guard let session else { return }
        session.start()
        syncFromSession()
    }

    func confirm() {
        guard let session else { return }
        session.confirmPairing()
        syncFromSession()
    }

    func goLive() {
        guard let session else { return }
        session.goLive()
        syncFromSession()
    }

    func end() {
        guard let session else { return }
        session.end()
        syncFromSession()
        onSessionEnded?()
    }

    /// The §8.20 "Codes don't match" path — the wrong-phone / stranger's-phone
    /// case, reachable without waiting for a timeout.
    func rejectCode() { end() }

    /// Fired once `end()` has torn the session down.
    ///
    /// `PeerSession` has no reset: `start()` only moves `.idle → .searching`,
    /// so it will not revive an `.ended` session. Getting back to Searching
    /// therefore needs a *fresh* session, which only the owner can build — this
    /// is that seam. With no owner attached, ending lands on `.idle`, one PAIR
    /// tap short of §16's "returns to Searching".
    var onSessionEnded: (() -> Void)?

    /// Owner-pumped, like `StereoEngine.processIfDue` — `RecordModel`'s 0.05 s
    /// peer pump is the natural driver.
    ///
    /// Advances the session's timers, then republishes. The republish is not
    /// optional bookkeeping: `phase` is `PeerSession`'s *only* `@Published`
    /// member and `clockSync` is not observable at all, so without this the
    /// sync figure would never reach the screen.
    func pump(now: TimeInterval) {
        guard let session else { return }
        session.tick(now: now)
        syncFromSession()
    }

    /// Republish without advancing the session's timers. `RecordModel.attachPeer`
    /// already ticks the peer at 20 Hz; a second `pump` here would tick it twice.
    func refresh() { syncFromSession() }

    // MARK: - Mapping

    static let notPairedLine = "Not paired"

    /// Phase → (step, link-status sentence). Static and pure so §16's table can
    /// be asserted row for row without standing up a transport.
    static func map(_ phase: PeerSession.Phase, syncMs: Double) -> (Step, String) {
        switch phase {
        case .idle:
            return (.idle, notPairedLine)
        case .searching:
            return (.searching, "Looking for the other phone…")
        case .confirming(let code):
            return (.confirm(code: code), "Codes match?")
        case .syncing:
            return (.syncing, "Syncing clocks…")
        case .ready:
            return (.ready(syncMs: syncMs), pairedLine(syncMs))
        case .live:
            // The link is still up mid-rally, so keep reporting its quality
            // rather than blanking the row the moment the rally starts.
            return (.live, pairedLine(syncMs))
        case .degraded(let reason):
            // §16: never a silent downgrade — and recording is unaffected, so
            // the sentence says so. The reason rides along on the step for
            // logging without shouting radio jargon at the operator.
            return (.degraded(reason), "Link lost — still recording")
        case .ended:
            // §16: ending returns `p-pair` to its idle state, not all the way
            // to Play, so re-pairing is one tap away.
            return (.idle, notPairedLine)
        case .failed(let message):
            // §16: the row shows the session's reason *verbatim* — the
            // capture-orientation guard has to read as itself on screen.
            return (.failed(message), message)
        }
    }

    private static func pairedLine(_ syncMs: Double) -> String {
        // §8.19: the figure is tabular in the view; one decimal is the
        // precision the ≤ 2 ms budget is argued in.
        String(format: "Paired · sync ±%.1f ms", syncMs)
    }

    private func syncFromSession() {
        guard let session else { return }
        let (nextStep, nextStatus) = Self.map(session.phase, syncMs: Self.syncMs(session))
        if nextStep != step { step = nextStep }
        if nextStatus != statusLine { statusLine = nextStatus }
    }

    private static func syncMs(_ session: PeerSession) -> Double {
        // ClockSync works in seconds (`PeerSession.readyUncertainty` is 0.005,
        // i.e. the 5 ms gate); the screen speaks milliseconds.
        (session.clockSync.estimate?.uncertainty ?? 0) * 1000
    }
}
