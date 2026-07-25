import SwiftUI

/// Live ball marker + short fading trail over the camera preview.
/// Capture frame geometry follows `orientation` (CaptureSettings.frameSize);
/// the preview letterboxes with resizeAspect, so map through the same
/// aspect-fit rect.
struct OverlayView: View {
    let trail: [BallObservation]   // oldest first, newest last
    /// Defaults to `.portrait` — today's handheld behaviour, unchanged.
    var orientation: CaptureSettings.CaptureOrientation = .portrait

    var contentSize: CGSize {
        let size = CaptureSettings.frameSize(for: orientation)
        return CGSize(width: size.width, height: size.height)
    }

    var body: some View {
        GeometryReader { proxy in
            let videoRect = Geometry.aspectFitRect(
                content: contentSize, container: proxy.size)
            Canvas { context, _ in
                for (index, observation) in trail.enumerated() {
                    let rect = Geometry.overlayRect(
                        visionRect: observation.rect, videoRect: videoRect)
                    let radius = max(6, rect.width / 2)
                    let circle = CGRect(
                        x: rect.midX - radius, y: rect.midY - radius,
                        width: radius * 2, height: radius * 2)
                    let age = Double(index + 1) / Double(trail.count)   // newest -> 1
                    if index == trail.count - 1 {
                        context.stroke(Path(ellipseIn: circle),
                                       with: .color(Theme.accentBg), lineWidth: 3)
                    } else {
                        context.fill(Path(ellipseIn: circle.insetBy(
                            dx: radius * 0.6, dy: radius * 0.6)),
                            with: .color(Theme.accentBg.opacity(0.15 + 0.5 * age)))
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }
}
