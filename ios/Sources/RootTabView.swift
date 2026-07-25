import SwiftUI

struct RootTabView: View {
    // Observed so the web tabs are torn down and reloaded (via .id) when
    // the server override changes; WebScreen deliberately never reloads.
    @AppStorage(Config.serverBaseKey) private var serverBase = ""

    var body: some View {
        TabView {
            // PlayRootView is a NavigationStack of its own now, so the
            // server-settings gear and (DEBUG) peer-bench button live on its
            // toolbar instead of as overlays bolted onto a bare RecordView.
            PlayRootView()
                .tabItem { Label("Play", systemImage: "record.circle") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=matches&shell=1")!)
                .id("matches-\(serverBase)")
                .tabItem { Label("Matches", systemImage: "square.stack") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=coach&shell=1")!)
                .id("coach-\(serverBase)")
                .tabItem { Label("Coach", systemImage: "figure.tennis") }
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
    }
}
