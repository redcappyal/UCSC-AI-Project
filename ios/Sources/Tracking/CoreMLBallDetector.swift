import CoreML
import CoreVideo
import Foundation
import Vision

/// Runs the bundled YOLO Core ML model (ios/Model/BallDetector.mlpackage,
/// exported with nms=True so Vision yields VNRecognizedObjectObservation).
/// init fails soft when the model is absent so the app still builds/runs
/// before the training workstream lands — RecordView shows a badge instead.
final class CoreMLBallDetector: BallDetecting {
    static let modelName = "BallDetector"
    static let confidenceThreshold: Float = 0.25   // tracking_common parity

    private let model: VNCoreMLModel

    init?() {
        guard let url = Bundle.main.url(forResource: Self.modelName,
                                        withExtension: "mlmodelc") else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // let Core ML place it on the ANE
        guard let coreml = try? MLModel(contentsOf: url, configuration: configuration),
              let vnModel = try? VNCoreMLModel(for: coreml) else { return nil }
        self.model = vnModel
    }

    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill   // matches YOLO letterbox-free export
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
        let best = (request.results as? [VNRecognizedObjectObservation])?
            .filter { $0.confidence >= Self.confidenceThreshold }
            .max(by: { $0.confidence < $1.confidence })
        guard let best else { return nil }
        return BallObservation(timestamp: timestamp,
                               rect: best.boundingBox,
                               confidence: best.confidence)
    }
}
