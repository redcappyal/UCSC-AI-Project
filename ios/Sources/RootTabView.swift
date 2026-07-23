import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            Text("Record").tabItem { Label("Play", systemImage: "record.circle") }
            Text("Matches").tabItem { Label("Matches", systemImage: "square.stack") }
            Text("Coach").tabItem { Label("Coach", systemImage: "figure.tennis") }
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
    }
}
