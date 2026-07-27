// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
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
///
/// It also owns the whole Play stack as data (`path`), because `p-live` must be
/// on screen whenever a rally is actually running — see `syncLiveStage()`.
struct PlayRootView: View {
    @StateObject private var record = RecordModel()
    @StateObject private var live = LiveSessionModel()

    /// The Play stack, as data.
    ///
    /// `p-live` used to be pushed by `PairingView` reporting its own
    /// `rally == .recording` (`onGoLive`). That put the trigger inside a view
    /// that can be popped while the state it guards is still live: one back
    /// tap on `p-live` cleared the flag, `p-pair` was left showing a disabled
    /// "RALLY LIVE" primary, and because `rally` never changed again its
    /// `.onChange` could never re-fire — a 4K60 recording with no STOP
    /// anywhere. The secondary had it worse: a remote record-start flips its
    /// `rally` with no tap at all, so if it had already backed out of `p-pair`
    /// there was no observer in the hierarchy to notice.
    ///
    /// So the trigger lives here instead, on the object that outlives every
    /// push, and is driven by `live.rally` alone — not by which screen the
    /// user happens to be on, and not by whether `p-pair` still exists.
    @State private var path: [PlayRoute] = []
    @State private var showServerSettings = false
    #if DEBUG
    @State private var showPeerBench = false
    #endif

    /// Deliberately a value type in a `NavigationStack(path:)`, not a
    /// `.navigationDestination(isPresented:)` bound to a `Bool`: an
    /// `isPresented` binding is written from both ends (SwiftUI clears it on a
    /// pop, this view sets it from `rally`), which is exactly the shape that
    /// can fight itself. The array is written from one place only —
    /// `syncLiveStage()` — and every write is idempotent, so it settles.
    private enum PlayRoute: Hashable { case record, pair, live }

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
        NavigationStack(path: $path) {
            ZStack {
                Theme.bg.ignoresSafeArea()
                VStack(spacing: 12) {
                    NavigationLink(value: PlayRoute.record) {
                        heroCard("Record a clip", "Film a rally with this phone's camera",
                                 systemImage: "video", accent: true)
                    }

                    // §16: `p-pair` → `p-live` is an automatic advance on
                    // `rally` becoming `.recording`, for either role — never a
                    // second tap target here, and no longer this card's
                    // business at all (see `path` and `syncLiveStage()`).
                    NavigationLink(value: PlayRoute.pair) {
                        heroCard("Live match", "Record and call in real time",
                                 systemImage: "dot.radiowaves.left.and.right", accent: false)
                    }
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.top, 18)
            }
            .navigationTitle("Play")
            .navigationDestination(for: PlayRoute.self) { route in
                switch route {
                case .record:
                    RecordView(model: record)
                case .pair:
                    // The camera is configured when `p-pair` is entered, not
                    // when `p-live` appears: `startRally()` issues the
                    // recording start synchronously from the tap on `p-pair`,
                    // so a camera first configured on `p-live` would open its
                    // AVAssetWriter against a session that was never started,
                    // then reconfigure it (removing and re-adding every input
                    // and output) mid-rally, and meter the court exposure
                    // *during* the rally instead of before it. `startCamera()`
                    // configures at most once (`cameraStarted`), so `p-live`'s
                    // own call is left in place as a backstop and does no
                    // reconfiguration — it only re-seeds the capture mount,
                    // which every appearance in this stack deliberately does.
                    PairingView(model: live)
                        .task { await record.startCamera() }
                case .live:
                    LiveStageView(record: record, live: live)
                }
            }
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
        // Reconcile on appearance as well as on change. `.onChange` only fires
        // on a transition this view was actually evaluated for, and Play is one
        // of three tabs: `onAppear` runs again every time this tab is
        // re-selected, so a rally that began while the user was on Matches or
        // Coach still finds its stage on the way back. (`.task` would not — it
        // runs once for the view's lifetime, and TabView keeps tabs alive.)
        // Idempotent, so the overlap with `.onChange` costs nothing.
        .onAppear { syncLiveStage() }
        .onChange(of: live.rally) { _, _ in syncLiveStage() }
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        #if DEBUG
        .sheet(isPresented: $showPeerBench) { NavigationStack { PeerBenchView() } }
        #endif
    }

    /// The one writer of `.live` in `path`: `p-live` is on top exactly while a
    /// rally is running, wherever the user was when it started (Play root,
    /// `p-record`, or `p-pair` — all three simply gain one more level).
    ///
    /// Settles rather than oscillates, by construction:
    ///  - both branches are guarded on what `path` already ends with, so
    ///    repeated calls (this runs from `.onAppear` *and* every `rally`
    ///    change) are no-ops after the first;
    ///  - nothing else ever appends or removes `.live`, and while the rally is
    ///    running `LiveStageView` hides the back button, so SwiftUI cannot pop
    ///    it out from under this rule and leave the two disagreeing;
    ///  - the pop happens on the first non-`.recording` state (`.submitting`),
    ///    and the later `.submitted`/`.failed`/`.idle` transitions find
    ///    `path.last != .live` and do nothing. One push, one pop, per rally.
    private func syncLiveStage() {
        if live.rally == .recording {
            if path.last != .live { path.append(.live) }
        } else if path.last == .live {
            path.removeLast()
        }
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
