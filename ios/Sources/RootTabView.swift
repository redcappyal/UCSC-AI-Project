import SwiftUI

struct RootTabView: View {
    // Observed so the web tabs are torn down and reloaded (via .id) when
    // the server override changes; WebScreen deliberately never reloads.
    @AppStorage(Config.serverBaseKey) private var serverBase = ""
    // Bound selection rather than per-tab onAppear/onDisappear: SwiftUI does
    // not guarantee their ordering across a tab switch, and the mask must be
    // a function of which tab is showing, not of which callback ran last.
    @State private var tab: RootTab = .launch
    @State private var showServerSettings = false

    var body: some View {
        TabView(selection: $tab) {
            // The native Play (record) tab was removed 2026-07-28: the tab bar
            // is the three web section roots only. `RecordView` and the whole
            // capture stack stay in the target, just unreachable, until the
            // auto-calibrating capture flow is ready to earn the slot back.
            //
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
        // The server-settings gear lived as an overlay on `RecordView`; with
        // the Play tab hidden it moves here so a fresh install can still point
        // the app at a pipeline. Bottom-trailing, lifted above the tab bar:
        // both top corners belong to the web header (back/home left, theme
        // toggle right), and the web's own bottom docks are center-aligned.
        .overlay(alignment: .bottomTrailing) {
            Button { showServerSettings = true } label: {
                Image(systemName: "gearshape")
                    .foregroundStyle(Theme.dim)
                    .padding(10)
                    .background(Theme.surface, in: Circle())
            }
            .accessibilityLabel("Server settings")
            .padding(.trailing, 12)
            .padding(.bottom, 60)
        }
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        .onChange(of: tab, initial: true) { _, newTab in
            OrientationPolicy.shared.applyForTab(newTab)
        }
    }
}
