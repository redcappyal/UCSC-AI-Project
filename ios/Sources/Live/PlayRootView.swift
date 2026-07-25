// ios/Sources/Live/PlayRootView.swift
import SwiftUI

/// §16 `p-load` — the Play section root.
///
/// Owns `RecordModel`, which the live layer only borrows: DESIGN.md §16 makes
/// "pairing adds capability, never gates it" a hard requirement, so a defect in
/// `LiveSessionModel` must not be able to reach plain recording.
///
/// Two hero cards, not the web's three — "Judge a clip" is a file input with no
/// native equivalent, so "Record a clip" takes the accent slot (§3.2).
struct PlayRootView: View {
    @StateObject private var record = RecordModel()
    @StateObject private var live = LiveSessionModel()
    @State private var showServerSettings = false
    #if DEBUG
    @State private var showPeerBench = false
    #endif

    // Both hero cards are plainly navigable. The shared-`RecordModel` hazard
    // is handled where the camera actually lives — `RecordModel`'s one
    // serialized, owner-tagged `setRecording(_:owner:)` funnel — not by
    // gating navigation from here. Gating entry was the wrong boundary three
    // ways: `.disabled` cannot reach a stage already pushed (a remote
    // "record" starts a rally on the secondary phone with no tap on it at
    // all), reading `live.rally` to decide whether plain recording is
    // reachable is the coupling DESIGN.md §16 forbids, and the gate opened
    // before the camera stopped.

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.bg.ignoresSafeArea()
                VStack(spacing: 12) {
                    NavigationLink { RecordView(model: record) } label: {
                        heroCard("Record a clip", "Film a rally with this phone's camera",
                                 systemImage: "video", accent: true)
                    }

                    NavigationLink {
                        PairingView(model: live)
                    } label: {
                        heroCard("Live match", "Record and call in real time",
                                 systemImage: "dot.radiowaves.left.and.right", accent: false)
                    }
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.top, 18)
            }
            .navigationTitle("Play")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showServerSettings = true } label: {
                        Image(systemName: "gearshape").foregroundStyle(Theme.dim)
                    }
                    .accessibilityLabel("Server settings")
                }
                #if DEBUG
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showPeerBench = true } label: {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                            .foregroundStyle(Theme.dim)
                    }
                    // §15: icon-only buttons carry a label, same as the gear.
                    .accessibilityLabel("Peer link bench")
                }
                #endif
            }
        }
        .task { live.bind(record: record) }
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        #if DEBUG
        .sheet(isPresented: $showPeerBench) { NavigationStack { PeerBenchView() } }
        #endif
    }

    /// §8.15: radius 8 (§4.4's card token — 14 is reserved for the verdict
    /// box), full width, `min-height:72`, padding 14, row gap 12, 28 px line
    /// icon. Accent = `--accent-bg` fill with *all* ink `--accent-text`;
    /// surface = `--surface` fill, `1px --line` border, `--text` title,
    /// `--dim` icon and description.
    private func heroCard(_ title: String, _ subtitle: String,
                          systemImage: String, accent: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 28))
                .foregroundStyle(accent ? Theme.accentText : Theme.dim)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(.headline).weight(.bold))
                    .foregroundStyle(accent ? Theme.accentText : Theme.text)
                Text(subtitle).font(.footnote)
                    // No `opacity(0.7)` on the accent card: §8.15 says all of
                    // its ink is `--accent-text`, and an alpha step is not a
                    // token.
                    .foregroundStyle(accent ? Theme.accentText : Theme.dim)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .frame(minHeight: 72)
        .background(accent ? Theme.accentBg : Theme.surface,
                    in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            if !accent {
                RoundedRectangle(cornerRadius: 8).stroke(Theme.line, lineWidth: 1)
            }
        }
    }
}
