// ios/Sources/Record/OrientationLock.swift
import UIKit

/// Which tab is showing, and therefore which orientations the app permits.
enum RootTab: Hashable, CaseIterable {
    // `.play` has no tab item since 2026-07-28 (the tab bar is the four web
    // section roots); the case stays because the capture stack it keys —
    // this file's masks and pins — is hidden, not deleted.
    case play, load, matches, coach, progress

    /// The tab the app opens into. `RootTabView`'s initial `@State` and
    /// `OrientationPolicy`'s seeded mask both read this one literal, so the
    /// two can never drift apart — a hand-copied literal in each place is
    /// what let the launch-orientation bug happen the first time.
    static let launch: RootTab = .matches
}

/// The app's supported-orientation policy as pure functions.
///
/// Capture is always landscape (`CaptureSettings`), so the Play tab is
/// landscape too — not just while recording. The camera preview is live while
/// the operator aims the phone in its back-wall mount, which is exactly when a
/// sideways preview costs the most. The web tabs (Dashboard, Analysis,
/// Training, Progress) are portrait mobile web UI and keep rotating freely.
enum OrientationLock {
    /// Orientations permitted while `tab` is showing, before capture pins.
    static func mask(for tab: RootTab) -> UIInterfaceOrientationMask {
        switch tab {
        case .play: return .landscape
        case .load, .matches, .coach, .progress: return .all
        }
    }

    /// Called by `RecordModel.applyRecording`'s start path once the mount is
    /// re-resolved at record start. Narrows to the single mount the recording
    /// commits to, so that the device will not be able to flip to the other
    /// landscape mid-recording and leave the orientation advertised in
    /// `Hello` describing a mount that is no longer real.
    static func pinnedMask(for orientation: CaptureSettings.CaptureOrientation) -> UIInterfaceOrientationMask {
        switch orientation {
        case .landscapeRight: return .landscapeRight
        case .landscapeLeft: return .landscapeLeft
        }
    }

    /// Called wherever an interface orientation needs mapping to a mount —
    /// in practice from one place, `RecordModel.interfaceMount()`, the default
    /// behind `RecordModel`'s injectable mount resolver. Both of that model's
    /// readers go through it: `startCamera`'s unpinned seed (once before
    /// `configure()`, and again on every later Play appearance) and
    /// `applyRecording`'s start path, the resolution that actually gets pinned.
    /// The mount an interface orientation implies, or nil when the interface is
    /// portrait — portrait is not a capture mode, so callers fall back to a
    /// default (or, for `applyRecording`, refuse to start) rather than
    /// inventing a mount.
    static func captureOrientation(for interface: UIInterfaceOrientation) -> CaptureSettings.CaptureOrientation? {
        switch interface {
        case .landscapeRight: return .landscapeRight
        case .landscapeLeft: return .landscapeLeft
        default: return nil
        }
    }
}

/// Holds the mask the app delegate serves. UIKit asks the delegate on every
/// rotation decision, so mutating this and then telling UIKit to re-ask is
/// what actually moves the device.
@MainActor
final class OrientationPolicy {
    static let shared = OrientationPolicy()
    /// Seeded to `RootTab.launch`'s mask: UIKit asks the delegate for the
    /// launch orientation at scene connection, before any SwiftUI
    /// `onAppear` can run, so a default of `.all` would bring the app up
    /// portrait.
    private(set) var mask: UIInterfaceOrientationMask = OrientationLock.mask(for: .launch)
    /// The mount `RecordModel.applyRecording`'s start path pinned for the
    /// running recording, or nil before any recording has pinned one — or
    /// again after `releaseCapturePin` clears it once that recording stops.
    /// `applyForTab` consults this so a tab round trip back to Play, mid-
    /// recording, re-asserts the pin instead of the Play tab's wider resting
    /// `.landscape` mask — nothing stops the recording on tab exit, so
    /// without this the device could flip mid-recording the moment the
    /// operator glances at another tab and back.
    private(set) var capturePin: UIInterfaceOrientationMask?
    /// The tab `applyForTab` was last called for — i.e. the one showing.
    /// Seeded to `RootTab.launch` for the same reason `mask` is: UIKit asks
    /// before `RootTabView`'s first `onChange` can run. Read by
    /// `releaseCapturePin`, which can now fire while a tab other than Play is
    /// showing (see there).
    private(set) var currentTab: RootTab = .launch

    /// Two-tier scene lookup: prefer the foreground-active scene, but a
    /// transient system interruption (an alert, Control Center) can leave
    /// every scene `.foregroundInactive` for a moment, and returning nothing
    /// there would leave a caller with no scene to act on until something
    /// else happened to trigger another query. Falling back to any
    /// qualifying scene still reaches one.
    ///
    /// Shared by `apply` (which re-asks UIKit for a rotation) and by
    /// `RecordModel.interfaceMount()`, the production default behind that
    /// model's mount resolver — which is the single reader for both of its
    /// orientation resolutions, `startCamera`'s unpinned per-appearance seed
    /// and `applyRecording`'s start-path re-resolution, the one that actually
    /// gets pinned. One lookup, one `requiresKeyWindow` knob, rather than
    /// several call sites that could quietly diverge and disagree about which
    /// scene is "the" active one — a second, divergent lookup was already a
    /// review finding on this branch.
    ///
    /// The callers need different guarantees from it, which is what the
    /// parameter is for. `apply` dereferences `scene.keyWindow` directly, so
    /// it needs `requiresKeyWindow: true` (the default) — a caller can then
    /// act on the result unconditionally instead of adding its own further
    /// guard. The mount resolutions only read `scene.interfaceOrientation`,
    /// and that property is readable on a foreground-active scene with no key
    /// window yet; requiring one there would needlessly fall through to a
    /// fallback for a scene that could otherwise have resolved the real
    /// mount, so they pass `requiresKeyWindow: false`.
    static func activeWindowScene(requiresKeyWindow: Bool = true) -> UIWindowScene? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        func qualifies(_ scene: UIWindowScene) -> Bool {
            !requiresKeyWindow || scene.keyWindow != nil
        }
        return scenes.first(where: { $0.activationState == .foregroundActive && qualifies($0) })
            ?? scenes.first(where: qualifies)
    }

    /// Sets `mask` and tells UIKit to re-ask the delegate for it, which is
    /// what actually moves the device. UIKit does not poll, so skipping the
    /// re-ask would leave the device on the stale orientation until
    /// something else happened to trigger another query.
    func apply(_ newMask: UIInterfaceOrientationMask) {
        mask = newMask
        guard let scene = Self.activeWindowScene() else { return }
        scene.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: newMask))
    }

    /// Called by `RecordModel.applyRecording`'s start path once the mount is
    /// re-resolved at record start (not by `startCamera`, whose per-appearance
    /// seeds are deliberately unpinned — see `CameraController.orientation`),
    /// and only
    /// AFTER `camera.startRecording()` has actually succeeded — pinning first
    /// and having the recording then fail to start would strand the operator
    /// locked to this mount with no running recording to ever release it
    /// from. Pins the mask and remembers it as `capturePin`, so a later tab
    /// round trip back to Play (`applyForTab`) re-asserts this exact mount
    /// instead of the wider `.landscape` the Play tab permits at rest.
    func pinForCapture(_ pinnedMask: UIInterfaceOrientationMask) {
        capturePin = pinnedMask
        apply(pinnedMask)
    }

    /// Called by `RecordModel.applyRecording`'s stop path once a recording
    /// finishes, cleanly or not — that is the point the operator is meant to
    /// be able to re-mount before the next rally. Clears `capturePin` AND
    /// re-widens the live mask right away: clearing the stored value alone
    /// would leave `mask` at whichever mount `pinForCapture` last applied,
    /// since nothing re-asks UIKit until something calls `apply`.
    ///
    /// It widens to **the showing tab's** mask, not unconditionally to Play's.
    /// This used to widen straight to `OrientationLock.mask(for: .play)`,
    /// which was correct only while the single production caller was the
    /// record button — an control that exists solely on Play, so Play was
    /// necessarily showing. That is no longer true: the live layer stops a
    /// a recording started while the tab was away
    /// (`LiveSessionModel.handleRemoteRecord`), with no tap on this phone at
    /// all, and a pin survives a mid-recording excursion to Matches or Coach.
    /// Widening to `.landscape` there would leave a portrait web tab locked
    /// landscape until the next tab change. Routing through `applyForTab`
    /// keeps the one rule — the mask is a function of the showing tab, minus
    /// any live capture pin — in one place.
    func releaseCapturePin() {
        capturePin = nil
        applyForTab(currentTab)
    }

    /// Called by `RootTabView` on every tab change. For `.play`, prefers a
    /// live `capturePin` over the tab's resting mask, so a pin set by an
    /// in-progress recording (`RecordModel.applyRecording`'s start path)
    /// survives a round trip through another tab. This one is load-bearing,
    /// not just a fast path: the Play stack's `.task`s (`startCamera`, from
    /// `RecordView`'s `.task`) does re-fire on return to Play, and
    /// each of those re-entries deliberately re-seeds the capture mount — but
    /// none of them pins anything. Pinning moved to record start, precisely so
    /// Play stays at both-landscape while the operator is only framing; and
    /// the re-seed itself refuses to run while a recording is in flight
    /// (`RecordModel.startCamera`'s `isRecording` guard, plus the re-seed's
    /// place on the recording transition chain). If a
    /// recording is running when the operator glances at another tab and
    /// back, this `capturePin` check is the only thing that re-narrows Play
    /// to the recording's mount; without it, returning to Play would fall
    /// back to the wider resting `.landscape` mask while a recording already
    /// committed to one mount keeps writing. `OrientationLock.mask(for:)`
    /// itself stays pure and capture-agnostic; this is the capture-aware
    /// layer on top of it.
    func applyForTab(_ tab: RootTab) {
        currentTab = tab
        switch tab {
        case .play: apply(capturePin ?? OrientationLock.mask(for: .play))
        case .load, .matches, .coach, .progress: apply(OrientationLock.mask(for: tab))
        }
    }
}

/// Serves `OrientationPolicy.shared.mask`. SwiftUI has no orientation-lock
/// API, so the app delegate is the only hook UIKit consults.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        OrientationPolicy.shared.mask
    }
}
