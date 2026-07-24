import SwiftUI

struct RootTabView: View {
    #if DEBUG
    // Play tab has no NavigationStack of its own (RecordView is a bare
    // ZStack), so the toolbar/NavigationLink hook the brief prefers has
    // nowhere to attach — sheet-from-overlay-button fallback per the brief.
    @State private var showPeerBench = false
    #endif
    @State private var showServerSettings = false
    // Observed so the web tabs are torn down and reloaded (via .id) when
    // the server override changes; WebScreen deliberately never reloads.
    @AppStorage(Config.serverBaseKey) private var serverBase = ""

    var body: some View {
        TabView {
            RecordView()
                .overlay(alignment: .topLeading) {
                    Button { showServerSettings = true } label: {
                        Image(systemName: "gearshape")
                            .foregroundStyle(Theme.dim)
                            .padding(10)
                            .background(Theme.surface, in: Circle())
                    }
                    .accessibilityLabel("Server settings")
                    .padding(.top, 8)
                    .padding(.leading, 8)
                }
                .sheet(isPresented: $showServerSettings) {
                    ServerSettingsView()
                }
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
