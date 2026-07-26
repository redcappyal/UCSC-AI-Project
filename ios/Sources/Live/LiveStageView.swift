// ios/Sources/Live/LiveStageView.swift
import SwiftUI

/// §16 `p-live` — the record stage plus the call flash, the honest-state
/// banner, and the post-rally mini-court. Presented automatically, for the
/// whole time `LiveSessionModel.rally` is `.recording`, by
/// `PlayRootView.syncLiveStage()` — from wherever the user was standing, and
/// whether or not `p-pair` is still in the stack.
///
/// The mini-court is primary-only in v1: the secondary sees relayed event
/// JSON, which carries no 3D track (spec, "Deliberate v1 narrowing"). Its
/// footprint is reserved either way so nothing shifts when a call lands
/// (§0.9) — see `miniCourt` below for how that also covers the secondary.
/// The *banner* is not role-split: the secondary mirrors the primary's
/// relayed call through the same `CallPresentation` gate (see
/// `RecordModel.attachStereo`'s secondary-side `onEvent`).
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
        // While the rally is running, STOP is the *only* way off this screen.
        // A back tap (or the interactive edge swipe, which goes with the
        // button) would pop the one screen that can end the rally, leaving the
        // camera recording 4K60 with no control anywhere that can stop it —
        // `p-pair`'s primary reads a disabled "RALLY LIVE", and `RecordView`
        // correctly refuses to touch a `.live`-owned recording. It comes back
        // the moment the rally leaves `.recording`, which is also the moment
        // `PlayRootView` pops this screen itself.
        .navigationBarBackButtonHidden(live.rally == .recording)
        // Configuration backstop only (`RecordModel.cameraStarted`): the
        // camera is really configured when `p-pair` is entered, because a
        // rally can start before this screen exists — see `PlayRootView`'s
        // `.pair` destination. Not a no-op, though: like every other appearance
        // in the Play stack this re-seeds the capture mount, and it is the one
        // caller that can fire with a recording already running, which is what
        // `startCamera`'s `isRecording` guard and the re-seed's own place on
        // the recording transition chain are there for.
        .task { await record.startCamera() }
    }

    // MARK: - §8.19 link status

    /// Same recipe `PairingView` uses for the identical DESIGN.md component
    /// (§8.19: an 8 px `--dim` dot, decorative only, plus an explicit
    /// sentence) — §16 lists this row on `p-live` too ("persistent — a lost
    /// link stays visible mid-rally"), so it is the same component, not a
    /// second visual language for the same information.
    private var linkStatus: some View {
        HStack(spacing: 8) {
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
    /// the primary's `StereoEngine.onEvent` closure (`attachStereo`). The
    /// secondary's mirror of `.event` (`peer?.onEvent`, same file) does set
    /// `livePresentation` and flash — the banner is not role-split — but
    /// deliberately leaves `liveTrack`/`liveImpact` alone, because the relayed
    /// payload carries no 3D geometry and there would be nothing to draw. So
    /// on the secondary this branch can never be taken — no role check needed
    /// here, and what it sees instead is the same reserved blank any phone
    /// shows before its first call, which is exactly §16's "Ready (before a
    /// rally)" row ("hidden — reserved footprint"). Deliberately not given
    /// secondary-specific explanatory copy: DESIGN.md's §16 table draws no
    /// such distinction for this footprint, and the `.link-status` /
    /// `.call-banner` rows around it are the honest-state readout for both
    /// roles — adding a second, mini-court-only explanation here would be new
    /// UI language the spec doesn't call for.
    @ViewBuilder private var miniCourt: some View {
        if record.livePresentation != nil, record.flashPresentation == nil,
           !record.liveTrack.isEmpty {
            MiniCourtView(track: record.liveTrack, impact: record.liveImpact)
                .frame(height: MiniCourtView.reservedHeight)
                .padding(.horizontal, 14)
        } else {
            // The same constant both branches, from the view itself, so the
            // reservation cannot drift from what actually renders (§0.9). The
            // old hardcoded 180 was ~52 pt short of `MiniCourtView`'s natural
            // height, and its canvases were fixed, so the side-elevation panel
            // drew straight over the call banner below.
            Color.clear.frame(height: MiniCourtView.reservedHeight)
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
            // §0.6: the capsule must be the tappable area, not just the word.
            // Styling it *outside* the `Button` (an earlier shape of this
            // file) leaves the hit region at the label's intrinsic ~50x21 pt
            // `Text` size, painted inside a 48 pt capsule that is mostly dead
            // — on the one control that ends a live rally. Everything visual
            // therefore goes inside the label closure, exactly as
            // `PairingView.primary` and `RecordView.recordButton` do.
            Button { live.stopRally() } label: {
                Text("STOP")
                    .font(.system(.headline).weight(.bold))
                    .tracking(0.05 * 17)
                    .foregroundStyle(Theme.accentText)
                    .frame(maxWidth: .infinity, minHeight: 48)
                    .background(Theme.accentBg, in: Capsule())
            }
            .padding(.horizontal, 14)
        } else {
            // §0.9: reserve the button's footprint so its appearance —
            // e.g. the secondary's degraded-link safety valve arriving —
            // shifts nothing above it.
            Color.clear.frame(height: 48).padding(.horizontal, 14)
        }
    }
}
