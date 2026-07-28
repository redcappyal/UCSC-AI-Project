import SwiftUI

struct RootTabView: View {
    // Observed so the web tabs are torn down and reloaded (via .id) when
    // the server override changes; WebScreen deliberately never reloads.
    @AppStorage(Config.serverBaseKey) private var serverBase = ""
    // Bound selection rather than per-tab onAppear/onDisappear: SwiftUI does
    // not guarantee their ordering across a tab switch, and the mask must be
    // a function of which tab is showing, not of which callback ran last.
    @State private var tab: RootTab = .launch

    var body: some View {
        TabView(selection: $tab) {
            // Play is `RecordView` directly again. It was wrapped in
            // `PlayRootView`'s NavigationStack only to offer a second hero
            // card — the two-camera live match — which is archived
            // (archive/stereo/README.md). With one destination there is no
            // stack to push onto, so the stack went with it and the
            // server-settings gear went back to being an overlay on
            // `RecordView` (where it lived before the live entry point).
            //
            // The per-tab orientation mask below stays exactly as it is: it
            // came from the landscape-only capture work, not the live work,
            // and the tab is still what the mask is a function of.
            RecordView()
                .tabItem { Label("Play", systemImage: "record.circle") }
                .tag(RootTab.play)
            // Labels track the web app's section roots (design-lab IA,
            // DESIGN.md §8.3): the `matches`/`coach` cases and their #tab
            // fragments are the stable internal names those roots kept.
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=matches&shell=1")!)
                .id("matches-\(serverBase)")
                .tabItem { Label("Analysis", systemImage: "play.rectangle") }
                .tag(RootTab.matches)
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=coach&shell=1")!)
                .id("coach-\(serverBase)")
                .tabItem { Label("Training", systemImage: "target") }
                .tag(RootTab.coach)
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=progress&shell=1")!)
                .id("progress-\(serverBase)")
                .tabItem { Label("Progress", systemImage: "chart.line.uptrend.xyaxis") }
                .tag(RootTab.progress)
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
        .onChange(of: tab, initial: true) { _, newTab in
            OrientationPolicy.shared.applyForTab(newTab)
        }
    }
}
