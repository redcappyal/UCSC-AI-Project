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

    func apply(_ newMask: UIInterfaceOrientationMask) {
        mask = newMask
        // Prefer the foreground-active scene, but a transient system
        // interruption (an alert, Control Center) can leave every scene
        // `.foregroundInactive` for a moment. Returning early there would
        // update `mask` without ever telling UIKit to re-ask, and UIKit does
        // not poll — it would stay on the stale orientation until something
        // else (a rotation, a tab switch) happened to trigger another query.
        // Falling back to any scene with a key window still reaches UIKit.
        // Both predicates require a key window so the call below is never
        // partial: a `.foregroundActive` scene with a nil `keyWindow` would
        // otherwise optional-chain away `setNeedsUpdateOfSupportedInterfaceOrientations()`
        // and leave only `requestGeometryUpdate` running.
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        guard let scene = scenes.first(where: { $0.activationState == .foregroundActive && $0.keyWindow != nil })
            ?? scenes.first(where: { $0.keyWindow != nil }) else { return }
        scene.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: newMask))
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
