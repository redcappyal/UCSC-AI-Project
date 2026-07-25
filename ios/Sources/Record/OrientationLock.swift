// ios/Sources/Record/OrientationLock.swift
import UIKit

/// Which tab is showing, and therefore which orientations the app permits.
enum RootTab: Hashable, CaseIterable {
    case play, matches, coach

    /// The tab the app opens into. `RootTabView`'s initial `@State` and
    /// `OrientationPolicy`'s seeded mask both read this one literal, so the
    /// two can never drift apart — a hand-copied literal in each place is
    /// what let the launch-orientation bug happen the first time.
    static let launch: RootTab = .play
}

/// The app's supported-orientation policy as pure functions.
///
/// Capture is always landscape (`CaptureSettings`), so the Play tab is
/// landscape too — not just while recording. The camera preview is live while
/// the operator aims the phone in its back-wall mount, which is exactly when a
/// sideways preview costs the most. Matches and Coach are portrait mobile web
/// UI and keep rotating freely.
enum OrientationLock {
    /// Orientations permitted while `tab` is showing, before capture pins.
    static func mask(for tab: RootTab) -> UIInterfaceOrientationMask {
        switch tab {
        case .play: return .landscape
        case .matches, .coach: return .all
        }
    }

    /// Called by `RecordModel.startCamera` once the mount is resolved.
    /// Narrows to the single mount capture resolves at configure time, so
    /// that the device will not be able to flip to the other landscape
    /// mid-session and leave the orientation advertised in `Hello`
    /// describing a mount that is no longer real.
    static func pinnedMask(for orientation: CaptureSettings.CaptureOrientation) -> UIInterfaceOrientationMask {
        switch orientation {
        case .landscapeRight: return .landscapeRight
        case .landscapeLeft: return .landscapeLeft
        }
    }

    /// Called by `RecordModel.startCamera` once the interface orientation is
    /// known. The mount an interface orientation implies, or nil when the
    /// interface is portrait — portrait is not a capture mode, so the caller
    /// falls back to a default rather than inventing a mount.
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
    /// The mount `RecordModel.startCamera` pinned for the running capture
    /// session, or nil before any session has pinned one — nothing in
    /// production ever clears it back to nil, since there is no
    /// capture-session teardown yet (see `releaseCapturePin`). `applyForTab`
    /// consults this so a tab round trip back to Play re-asserts the pin
    /// instead of the Play tab's wider resting `.landscape` mask — nothing
    /// stops the capture session on tab exit, so without this the device
    /// could flip mid-session the moment the operator glances at another tab
    /// and back.
    private(set) var capturePin: UIInterfaceOrientationMask?

    /// Two-tier scene lookup: prefer the foreground-active scene, but a
    /// transient system interruption (an alert, Control Center) can leave
    /// every scene `.foregroundInactive` for a moment, and returning nothing
    /// there would leave a caller with no scene to act on until something
    /// else happened to trigger another query. Falling back to any
    /// qualifying scene still reaches one.
    ///
    /// Shared by `apply` (which re-asks UIKit for a rotation) and
    /// `RecordModel.startCamera` (which reads the scene's
    /// `interfaceOrientation` to resolve the mount) so the two can never pick
    /// different scenes and disagree about which one is "the" active scene —
    /// one lookup, one `requiresKeyWindow` knob, rather than two call sites
    /// that could quietly diverge.
    ///
    /// The two callers need different guarantees from it, which is what the
    /// parameter is for. `apply` dereferences `scene.keyWindow` directly, so
    /// it needs `requiresKeyWindow: true` (the default) — a caller can then
    /// act on the result unconditionally instead of adding its own further
    /// guard. `startCamera` only reads `scene.interfaceOrientation` to
    /// resolve the capture mount, and that property is readable on a
    /// foreground-active scene with no key window yet; requiring one there
    /// would needlessly fall through to the `.landscapeRight` literal
    /// fallback for a scene that could otherwise have resolved the real
    /// mount, so it passes `requiresKeyWindow: false`.
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

    /// Called once by `RecordModel.startCamera` when a capture session
    /// starts. Pins the mask and remembers it as `capturePin`, so a later
    /// tab round trip back to Play (`applyForTab`) re-asserts this exact
    /// mount instead of the wider `.landscape` the Play tab permits at rest.
    func pinForCapture(_ pinnedMask: UIInterfaceOrientationMask) {
        capturePin = pinnedMask
        apply(pinnedMask)
    }

    /// No production caller yet — there is no `stopCamera` or other teardown
    /// in `RecordModel` that stops a running capture session, so a pin set at
    /// configure time is meant to outlive every tab change for the app's
    /// lifetime. This exists for the teardown hook `RecordModel` does not yet
    /// have, and is exercised directly by `OrientationLockTests`.
    func releaseCapturePin() {
        capturePin = nil
    }

    /// Called by `RootTabView` on every tab change. For `.play`, prefers a
    /// live `capturePin` over the tab's resting mask, so the pin a running
    /// capture session set survives a round trip through another tab. This is
    /// still worth keeping even though `RecordView`'s `.task` (and so
    /// `startCamera`/`pinForCapture`) does re-fire on return to Play — it is
    /// cancelled when Play disappears and restarts when it reappears: the
    /// pin here makes the mask correct immediately, on the tab-change
    /// callback, rather than waiting on the async re-run of `startCamera` to
    /// get there. `OrientationLock.mask(for:)` itself stays pure and
    /// capture-agnostic; this is the capture-aware layer on top of it.
    func applyForTab(_ tab: RootTab) {
        switch tab {
        case .play: apply(capturePin ?? OrientationLock.mask(for: .play))
        case .matches, .coach: apply(OrientationLock.mask(for: tab))
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
