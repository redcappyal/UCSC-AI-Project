import SwiftUI

struct RecordView: View {
    // Owned here again: `PlayRootView` held it only so the archived live stage
    // could borrow the same camera (archive/stereo/README.md). One consumer,
    // one owner.
    @StateObject private var model = RecordModel()
    @State private var showServerSettings = false

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: model.camera.session).ignoresSafeArea()
            OverlayView(trail: model.trail).ignoresSafeArea()

            VStack {
                if model.detectorKind == RecordModel.DetectorKind.synthetic {
                    // A fixture must never be mistaken for a real detection.
                    Text("Synthetic detector — not a real ball")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.dim)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
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
                if let exposureNote = model.exposureNote {
                    Text(exposureNote)
                        .font(.footnote)
                        .foregroundStyle(Theme.dim)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
                Spacer()
                recordControls
            }
        }
        // The server-settings gear rode on `PlayRootView`'s toolbar while the
        // Play tab was a NavigationStack. This screen is the Play tab again,
        // and it is a bare ZStack, so the gear is an overlay button — exactly
        // where it lived before the live entry point existed.
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
        .sheet(isPresented: $showServerSettings) { ServerSettingsView() }
        .task { await model.startCamera() }
        .sheet(item: $model.singleCameraClip) { clip in
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
            recordButton
        }
        .padding(.bottom, 24)
    }

    private var recordButton: some View {
        Button {
            // Absolute intent, read at tap time, funnelled through the model:
            // the tap says "start" or "stop", never "flip whatever it is now".
            let shouldRecord = !model.isRecording
            Task { _ = await model.setRecording(shouldRecord) }
        } label: {
            ZStack {
                Circle().stroke(Theme.text, lineWidth: 4).frame(width: 76, height: 76)
                if model.isRecording {
                    // Yellow, not red: DESIGN.md reserves red for OUT
                    // verdicts ("never a red record dot", §8.16).
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Theme.accentBg).frame(width: 32, height: 32)
                } else {
                    Circle().fill(Theme.accentBg).frame(width: 62, height: 62)
                }
            }
        }
        .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
    }
}
