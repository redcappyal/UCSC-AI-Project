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

    // MARK: - Shared-`RecordModel` hazard (see task-9 report)
    //
    // `RecordView`'s record button calls `RecordModel.toggleRecording()`
    // directly. `LiveSessionModel` never does — every call it makes is
    // routed through `toggleRecordingChained(ifRecordingIs:)`, which
    // re-reads `record.isRecording` at execution time so a queued toggle can
    // never invert. That chain protects call *ordering* within the live
    // layer; it has no way to know about a toggle that happened outside it
    // entirely. Sharing one `RecordModel` between both stages means a plain
    // recording started from `RecordView` is invisible to the live layer's
    // guards (and vice versa): `startRally()`'s `ifRecordingIs: false` guard
    // would see an already-true `isRecording` it did not cause, skip its own
    // toggle, and `reconcileRallyAfterStartAttempt()` would read that
    // pre-existing `true` as "the start succeeded" — after which
    // `stopRally()` would stop and submit that stray clip as the rally
    // footage. The mirror case is just as real: stopping a live rally's
    // recording via `RecordView`'s raw button would hand that footage to the
    // plain judge flow (`ResultsView`, via the shared `finishedClip`)
    // instead of the paired-upload path, while `LiveSessionModel` is left
    // believing a recording it no longer owns is still running.
    //
    // Fixed here, not in either model: block the navigation link into
    // whichever stage did not cause the state currently in flight. Neither
    // check strands the user — each can still return to the stage that
    // *does* own the in-flight recording to resolve it (stop it from
    // `RecordView`, or drive the rally to completion from the live side).

    /// True only when a recording is running that the live layer did not
    /// start — `live.rally == .recording` is what the live layer's own chain
    /// caused, so excluding it here is what keeps this from also blocking a
    /// return to Live while its *own* rally is in progress.
    private var liveBlockedByPlainRecording: Bool {
        record.isRecording && live.rally != .recording
    }

    /// A live rally in flight uses `RecordModel` too; entering the record
    /// stage while one is recording would expose it to the raw button above.
    private var recordBlockedByLiveRally: Bool {
        live.rally == .recording
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.bg.ignoresSafeArea()
                VStack(spacing: 12) {
                    NavigationLink { RecordView(model: record) } label: {
                        heroCard("Record a clip", "Film a rally with this phone's camera",
                                 systemImage: "video", accent: true)
                    }
                    .disabled(recordBlockedByLiveRally)
                    .opacity(recordBlockedByLiveRally ? 0.4 : 1)

                    NavigationLink {
                        PairingView(model: live)
                    } label: {
                        heroCard("Live match", "Record and call in real time",
                                 systemImage: "dot.radiowaves.left.and.right", accent: false)
                    }
                    .disabled(liveBlockedByPlainRecording)
                    .opacity(liveBlockedByPlainRecording ? 0.4 : 1)
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

    private func heroCard(_ title: String, _ subtitle: String,
                          systemImage: String, accent: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(accent ? Theme.accentText : Theme.text)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(.headline).weight(.bold))
                    .foregroundStyle(accent ? Theme.accentText : Theme.text)
                Text(subtitle).font(.footnote)
                    .foregroundStyle(accent ? Theme.accentText.opacity(0.7) : Theme.dim)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .frame(minHeight: 72)
        // §4.4 radius scale: 8px is the card token (14px is reserved for the
        // verdict box) — the one deliberate deviation from the brief's draft
        // snippet, corrected to match DESIGN.md rather than drift from it.
        .background(accent ? Theme.accentBg : Theme.surface,
                    in: RoundedRectangle(cornerRadius: 8))
    }
}
