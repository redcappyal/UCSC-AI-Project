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
    /// The tab whose page has navigated below its section root, if any. That
    /// screen hides the tab bar so the page's back chevron is its only exit.
    @State private var detailTab: RootTab?

    var body: some View {
        TabView(selection: $tab) {
            // The native Play (record) tab was removed 2026-07-28: the tab bar
            // is the web section roots only. `RecordView` and the whole capture
            // stack stay in the target, just unreachable, until the
            // auto-calibrating capture flow is ready to earn the slot back.
            //
            // The four items and their order mirror the web nav dock
            // (DESIGN.md §8.3) one for one — the same menu, drawn in the
            // platform's own chrome. Icon-only like the dock: `tabItem` takes a
            // bare `Image`, since a `Label` there always draws its title on
            // iPhone. The name survives for VoiceOver as the image's
            // accessibility label, the native counterpart of the dock buttons'
            // `aria-label`. The `load`/`matches`/`coach` cases and their #tab
            // fragments are the stable internal names those roots kept.
            webTab(.load, "load", "house", "Dashboard")
            webTab(.matches, "matches", "play.rectangle", "Analysis")
            webTab(.coach, "coach", "target", "Training")
            webTab(.progress, "progress", "chart.line.uptrend.xyaxis", "Progress")
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
        // The server-settings gear lived as an overlay on `RecordView`; with
        // the Play tab hidden it moves here so a fresh install can still point
        // the app at a pipeline. Bottom-trailing, lifted above the tab bar:
        // both top corners belong to the web header (back/home left, theme
        // toggle right), and the web's own bottom docks are center-aligned.
        // It is shell chrome, so it leaves with the tab bar on a detail screen.
        .overlay(alignment: .bottomTrailing) {
            if detailTab == nil {
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
        }
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        .onChange(of: tab, initial: true) { _, newTab in
            OrientationPolicy.shared.applyForTab(newTab)
        }
    }

    /// One section root: its webview, its icon, and the tab-bar rule that a
    /// page which has gone deeper than its root owns the whole screen.
    private func webTab(_ tab: RootTab, _ path: String,
                        _ systemImage: String, _ name: String) -> some View {
        WebScreen(url: Self.rootURL(tab: path),
                  onSectionRootChange: { isRoot in setSectionRoot(tab, isRoot) })
            .id("\(path)-\(serverBase)")
            .toolbar(detailTab == tab ? .hidden : .visible, for: .tabBar)
            .tabItem { Self.icon(systemImage, name) }
            .tag(tab)
    }

    /// Compared against `tab` rather than assigned blindly: a root report from
    /// some other tab must not clear a detail this one is still showing.
    private func setSectionRoot(_ tab: RootTab, _ isRoot: Bool) {
        if isRoot {
            if detailTab == tab { detailTab = nil }
        } else {
            detailTab = tab
        }
    }

    /// One icon-only tab item, named for assistive tech only.
    private static func icon(_ systemImage: String, _ name: String) -> some View {
        Image(systemName: systemImage).accessibilityLabel(name)
    }

    /// A section root inside the shell. `shell=1` is a *query* parameter, not
    /// part of the fragment: the page strips its own hash before reloading
    /// (`openRunReview`), so a hash-borne flag would be dropped on that reload
    /// and the web nav dock would come back under this tab bar.
    private static func rootURL(tab: String) -> URL {
        URL(string: Config.baseURL.absoluteString + "/?shell=1#tab=" + tab)!
    }
}
