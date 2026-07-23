import SwiftUI

struct RecordView: View {
    @StateObject private var model = RecordModel()

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: model.camera.session).ignoresSafeArea()
            OverlayView(trail: model.trail).ignoresSafeArea()

            VStack {
                if model.detectorMissing {
                    Text("Ball model missing — overlay disabled")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.dim)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
                if let errorText = model.errorText {
                    Text(errorText)
                        .font(.footnote)
                        .foregroundStyle(Theme.text)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
                Spacer()
                recordControls
            }
        }
        .task { await model.startCamera() }
        .sheet(item: $model.finishedClip) { clip in
            ResultsView(clip: clip)
        }
    }

    private var recordControls: some View {
        VStack(spacing: 12) {
            if model.isRecording, let start = model.recordingStartedAt {
                Text(start, style: .timer)
                    .font(.system(.title3, design: .monospaced).weight(.semibold))
                    .foregroundStyle(Theme.text)
            }
            Button {
                Task { await model.toggleRecording() }
            } label: {
                ZStack {
                    Circle().stroke(Theme.text, lineWidth: 4).frame(width: 76, height: 76)
                    if model.isRecording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Theme.outCall).frame(width: 32, height: 32)
                    } else {
                        Circle().fill(Theme.accentBg).frame(width: 62, height: 62)
                    }
                }
            }
            .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
        }
        .padding(.bottom, 24)
    }
}
