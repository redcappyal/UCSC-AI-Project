import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            RecordView()
                .tabItem { Label("Play", systemImage: "record.circle") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=matches&shell=1")!)
                .tabItem { Label("Matches", systemImage: "square.stack") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=coach&shell=1")!)
                .tabItem { Label("Coach", systemImage: "figure.tennis") }
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
    }
}
