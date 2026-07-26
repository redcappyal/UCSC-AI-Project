import SwiftUI

@main
struct SquashApp: App {
    // SwiftUI cannot lock orientation on its own; the delegate is the hook
    // UIKit consults. See OrientationLock.swift.
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .preferredColorScheme(.dark)
        }
    }
}
