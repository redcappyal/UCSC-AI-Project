import SwiftUI

struct ResultsView: View {
    let clip: FinishedClip
    @StateObject private var submission = RunSubmission()
    @State private var showFullReview = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.bg.ignoresSafeArea()
                content
            }
            .navigationTitle("Rally")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }.tint(Theme.accentBg)
                }
            }
        }
        .task { await submission.submit(videoURL: clip.url, duration: clip.duration) }
        .fullScreenCover(isPresented: $showFullReview) {
            if let runID = submission.completedRunID {
                WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#run=\(runID)&shell=1")!,
                          showsClose: true)
            }
        }
    }

    @ViewBuilder private var content: some View {
        switch submission.phase {
        case .idle, .fetchingCalibration:
            progress("Fetching court calibration…")
        case .uploading:
            progress("Uploading rally…")
        case .tracking(let percent, let message):
            VStack(spacing: 12) {
                ProgressView(value: percent, total: 100).tint(Theme.accentBg)
                Text(message).font(.footnote).foregroundStyle(Theme.dim)
            }
            .padding(24)
        case .failed(let message):
            VStack(spacing: 16) {
                Text(message)
                    .foregroundStyle(Theme.text)
                    .multilineTextAlignment(.center)
                Button("Try again") {
                    Task { await submission.submit(videoURL: clip.url,
                                                   duration: clip.duration) }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentBg).foregroundStyle(Theme.accentText)
            }
            .padding(24)
        case .complete(let job):
            completeList(job)
        }
    }

    private func progress(_ label: String) -> some View {
        VStack(spacing: 12) {
            ProgressView().tint(Theme.accentBg)
            Text(label).font(.footnote).foregroundStyle(Theme.dim)
        }
    }

    private func completeList(_ job: JobStatus) -> some View {
        let hits = (job.hits ?? []).frontWall
        return ScrollView {
            VStack(spacing: 10) {
                if hits.isEmpty {
                    Text("No front-wall hits detected in this rally.")
                        .foregroundStyle(Theme.dim)
                        .padding(.top, 40)
                }
                ForEach(Array(hits.enumerated()), id: \.element.id) { index, hit in
                    HStack {
                        Text("Hit \(index + 1)")
                            .foregroundStyle(Theme.text)
                            .font(.body.weight(.semibold))
                        Text(String(format: "%.1fs", hit.timestampSeconds))
                            .foregroundStyle(Theme.dim).font(.footnote)
                        Spacer()
                        callChip(hit)
                    }
                    .padding(14)
                    .background(Theme.surface,
                                in: RoundedRectangle(cornerRadius: 12))
                }
                Button("Open full review") { showFullReview = true }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accentBg).foregroundStyle(Theme.accentText)
                    .padding(.top, 8)
            }
            .padding(16)
        }
    }

    private func callChip(_ hit: Hit) -> some View {
        let call = hit.call ?? "UNKNOWN"
        let color: Color = call == "IN" ? Theme.inCall
            : call == "OUT" ? Theme.outCall : Theme.unknown
        let margin = hit.marginPx.map {
            String(format: " %+.1f px", $0)
        } ?? ""
        return Text(call + margin)
            .font(.footnote.weight(.bold))
            .foregroundStyle(.black)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(color, in: Capsule())
    }
}
