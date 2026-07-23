import CoreGraphics

enum Geometry {
    /// Where aspect-fit content lands inside a container (mirrors
    /// AVCaptureVideoPreviewLayer's .resizeAspect).
    static func aspectFitRect(content: CGSize, container: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0 else { return .zero }
        let scale = min(container.width / content.width,
                        container.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(x: (container.width - size.width) / 2,
                      y: (container.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    /// Vision-normalized rect (origin bottom-left) -> screen rect inside the
    /// aspect-fit video area (origin top-left).
    static func overlayRect(visionRect: CGRect, videoRect: CGRect) -> CGRect {
        CGRect(
            x: videoRect.minX + visionRect.minX * videoRect.width,
            y: videoRect.minY + (1 - visionRect.minY - visionRect.height) * videoRect.height,
            width: visionRect.width * videoRect.width,
            height: visionRect.height * videoRect.height)
    }
}
