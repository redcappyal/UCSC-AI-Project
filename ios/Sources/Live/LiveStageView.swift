// ios/Sources/Live/LiveStageView.swift
import SwiftUI

/// §16 `p-live` — the record stage plus the call flash, the honest-state
/// banner, and the post-rally mini-court. Pushed from `p-pair` automatically
/// once `LiveSessionModel.rally` becomes `.recording` (see `PairingView`'s
/// `onGoLive`, wired in `PlayRootView`).
///
/// The mini-court is primary-only in v1: the secondary sees relayed event
/// JSON, which carries no 3D track (spec, "Deliberate v1 narrowing"). Its
/// footprint is reserved either way so nothing shifts when a call lands
/// (§0.9) — see `miniCourt` below for how that also covers the secondary.
///
/// This is also where the §16 dead end closes: before this screen existed,
/// a started rally had no way to end — `PairingView`'s primary is disabled
/// once live ("RALLY LIVE"), and `showsLocalStop` had no consumer. Both
/// roles' STOP here call `LiveSessionModel.stopRally()` only, never
/// `RecordModel` directly — see `LiveSessionModel.showsStop` for the
/// per-role rule (published there, not computed in this view).
struct LiveStageView: View {
    @ObservedObject var record: RecordModel
    @ObservedObject var live: LiveSessionModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: record.camera.session).ignoresSafeArea()
            OverlayView(trail: record.trail).ignoresSafeArea()

            // §8.17: the flash is a stage overlay, never in the layout flow,
            // so a call appearing or clearing shifts nothing below it — the
            // same placement `RecordView` uses.
            CallFlashView(presentation: record.flashPresentation)
                .allowsHitTesting(false)
                .ignoresSafeArea()

            VStack(spacing: 10) {
                linkStatus
                Spacer(minLength: 0)
                miniCourt
                // §8.18: persistent, beneath the stage, height always
                // reserved — it renders its blank state before any call.
                CallBannerView(presentation: record.livePresentation)
                    .padding(.horizontal, 14)
                stop
            }
            .padding(.bottom, 24)
        }
        .task { await record.startCamera() }
    }

    // MARK: - §8.19 link status

    /// Same recipe `PairingView` uses for the identical DESIGN.md component
    /// (§8.19: an 8 px `--dim` dot, decorative only, plus an explicit
    /// sentence) — §16 lists this row on `p-live` too ("persistent — a lost
    /// link stays visible mid-rally"), so it is the same component, not a
    /// second visual language for the same information.
    private var linkStatus: some View {
        HStack(spacing: 9) {
            Circle().fill(Theme.dim).frame(width: 8, height: 8)
            Text(live.linkStatus)
                .font(.system(.subheadline).monospacedDigit())
                .foregroundStyle(Theme.dim)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .frame(minHeight: 40, alignment: .leading)
        .padding(.horizontal, 14)
    }

    // MARK: - §8.10 post-rally mini-court

    /// Never visible before a call (§16), and its footprint is reserved
    /// either way so appearing shifts nothing.
    ///
    /// The third condition — `!record.liveTrack.isEmpty` — is what actually
    /// keeps this primary-only: `RecordModel.liveTrack` is set only inside
    /// the primary's `StereoEngine.onEvent` closure (`attachStereo`); the
    /// secondary's mirror of `.event` (`peer?.onEvent`, same file) only
    /// appends to `stereoEvents` and never touches `liveTrack`/`liveImpact`/
    /// `livePresentation`. So on the secondary this branch can never be
    /// taken — no role check needed here, the same reserved blank the
    /// pre-call state already shows is what the secondary sees for the
    /// whole rally, which is exactly §16's "Ready (before a rally)" row
    /// ("hidden — reserved footprint"). Deliberately not given secondary-
    /// specific explanatory copy: DESIGN.md's §16 table draws no such
    /// distinction for this footprint, and the surrounding `.link-status` /
    /// `.call-banner` rows are the honest-state readout for both roles —
    /// adding a second, mini-court-only explanation here would be new UI
    /// language the spec doesn't call for.
    @ViewBuilder private var miniCourt: some View {
        if record.livePresentation != nil, record.flashPresentation == nil,
           !record.liveTrack.isEmpty {
            MiniCourtView(track: record.liveTrack, impact: record.liveImpact)
                .frame(height: 180)
                .padding(.horizontal, 14)
        } else {
            Color.clear.frame(height: 180)
        }
    }

    // MARK: - STOP (closes the §16 dead end)

    /// `live.showsStop` is the whole rule (§16's dead-end closing): the
    /// primary can always end a rally it started — DESIGN.md's plan text,
    /// "The primary stops; `.record("stop", ptsNs)` stops the secondary" —
    /// while the secondary has no general STOP of its own and only gains a
    /// local one while the link is degraded, because otherwise a dropped
    /// link leaves it recording 4K60 with nothing that can ever end it. Kept
    /// on the model, not computed here, so it is assertable without a view
    /// (`LiveSessionModelTests`) and this view carries no state logic of its
    /// own. Calls `live.stopRally()` only — never `RecordModel` directly.
    @ViewBuilder private var stop: some View {
        if live.showsStop {
            Button("STOP") { live.stopRally() }
                .font(.system(.headline).weight(.bold))
                .tracking(0.05 * 17)
                .foregroundStyle(Theme.accentText)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(Theme.accentBg, in: Capsule())
                .padding(.horizontal, 14)
        } else {
            // §0.9: reserve the button's footprint so its appearance —
            // e.g. the secondary's degraded-link safety valve arriving —
            // shifts nothing above it.
            Color.clear.frame(height: 48).padding(.horizontal, 14)
        }
    }
}
