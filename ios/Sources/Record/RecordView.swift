import SwiftUI

struct RecordView: View {
    // Owned by `PlayRootView` now, not here — the live stage borrows the same
    // instance via `LiveSessionModel.bind(record:)`, so it can no longer be a
    // private @StateObject. Nothing else below reads anything but `model`.
    @ObservedObject var model: RecordModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: model.camera.session).ignoresSafeArea()
            OverlayView(trail: model.trail).ignoresSafeArea()

            // §8.17: the flash is a stage overlay, never in the layout flow,
            // so a call appearing or clearing shifts nothing below it.
            CallFlashView(presentation: model.flashPresentation)
                .allowsHitTesting(false)
                .ignoresSafeArea()

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
                // §8.18: persistent, beneath the stage, height always
                // reserved — it renders its blank state before any call.
                CallBannerView(presentation: model.livePresentation)
                    .padding(.horizontal, 14)
                    .padding(.bottom, 8)
                recordControls
            }
        }
        #if DEBUG
        // The plan put this beside PlayRootView's peer-bench button, but the
        // demo cube overlays the live camera stage itself — it belongs here
        // regardless of who owns `model` now. Same chip styling, offset so
        // it clears where the bench button sits one screen up.
        .overlay(alignment: .topTrailing) {
            Button {
                model.startStereoDemo(localModelJSON: StereoDemo.localModelJSON,
                                      remoteModelJSON: StereoDemo.remoteModelJSON)
            } label: {
                Image(systemName: "cube.transparent")
                    .foregroundStyle(Theme.dim)
                    .padding(10)
                    .background(Theme.surface, in: Circle())
            }
            .accessibilityLabel("Run stereo demo")
            .padding(.top, 60)
            .padding(.trailing, 8)
        }
        #endif
        .task { await model.startCamera() }
        // `singleCameraClip`, not the raw clip: a rally the live layer
        // recorded on this same camera must never open the plain judge flow.
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
            // The camera is shared with the live layer, so this stage can be
            // standing in front of a rally it did not start — on the
            // secondary phone that happens with no tap here at all, since a
            // remote "record" starts the local recording. `canSetRecording`
            // is the model's own rule for what this owner may do, asked here
            // so the screen cannot offer a stop the model would refuse.
            if model.canSetRecording(!model.isRecording, owner: .single) {
                recordButton
            } else {
                // §0.3: never let appearance alone carry the meaning, and
                // §13: say something specific and actionable. Sized to the
                // button it replaces so nothing above it moves (§0.9).
                Text("The live match is using this camera — end the rally there first.")
                    .font(.footnote)
                    .foregroundStyle(Theme.dim)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .frame(minHeight: 76)
            }
        }
        .padding(.bottom, 24)
    }

    private var recordButton: some View {
        Button {
            // Absolute intent, read at tap time, funnelled through the model:
            // the tap says "start" or "stop", never "flip whatever it is now".
            let shouldRecord = !model.isRecording
            Task { _ = await model.setRecording(shouldRecord, owner: .single) }
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
