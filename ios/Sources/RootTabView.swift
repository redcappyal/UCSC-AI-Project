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
            // PlayRootView is a NavigationStack of its own now, so the
            // server-settings gear and (DEBUG) peer-bench button live on its
            // toolbar instead of as overlays bolted onto a bare RecordView.
            // The orientation lock moved with it: the tab is what the mask is
            // a function of, and PlayRootView — not RecordView — is the Play
            // tab now. RecordView is one destination inside its stack, so
            // leaving the lock attached there would have left the Play root
            // and `p-pair` free to rotate portrait while the operator is
            // framing the shot in a landscape mount.
            PlayRootView()
                .tabItem { Label("Play", systemImage: "record.circle") }
                .tag(RootTab.play)
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=matches&shell=1")!)
                .id("matches-\(serverBase)")
                .tabItem { Label("Matches", systemImage: "square.stack") }
                .tag(RootTab.matches)
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=coach&shell=1")!)
                .id("coach-\(serverBase)")
                .tabItem { Label("Coach", systemImage: "figure.tennis") }
                .tag(RootTab.coach)
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
        .onChange(of: tab, initial: true) { _, newTab in
            OrientationPolicy.shared.applyForTab(newTab)
        }
    }
}
