import SwiftUI

struct RootTabView: View {
    #if DEBUG
    // Play tab has no NavigationStack of its own (RecordView is a bare
    // ZStack), so the toolbar/NavigationLink hook the brief prefers has
    // nowhere to attach — sheet-from-overlay-button fallback per the brief.
    @State private var showPeerBench = false
    #endif

    var body: some View {
        TabView {
            RecordView()
                #if DEBUG
                .overlay(alignment: .topTrailing) {
                    Button { showPeerBench = true } label: {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                            .foregroundStyle(Theme.dim)
                            .padding(10)
                            .background(Theme.surface, in: Circle())
                    }
                    .padding(.top, 8)
                    .padding(.trailing, 8)
                }
                .sheet(isPresented: $showPeerBench) {
                    NavigationStack { PeerBenchView() }
                }
                #endif
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
