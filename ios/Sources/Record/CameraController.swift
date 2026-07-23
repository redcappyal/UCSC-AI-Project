import AVFoundation
import Foundation

final class CameraController: NSObject {
    enum CameraError: LocalizedError {
        case permissionDenied, configurationFailed, notRecording, recordingEmpty

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Camera or microphone access was denied. Enable both in Settings."
            case .configurationFailed: return "The camera could not be configured."
            case .notRecording: return "No recording is in progress."
            case .recordingEmpty:
                return "Recording stopped before any video was captured. Try again."
            }
        }
    }

    let session = AVCaptureSession()
    /// Every video frame, on the output queue. RecordView wires this to
    /// BallTracker.process.
    var onVideoSample: ((CVPixelBuffer, TimeInterval) -> Void)?

    private let sessionQueue = DispatchQueue(label: "slc.camera.session")
    // One queue for BOTH outputs: writer state below is queue-confined to it.
    private let outputQueue = DispatchQueue(label: "slc.camera.output")

    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()

    private var writer: AVAssetWriter?
    private var writerVideo: AVAssetWriterInput?
    private var writerAudio: AVAssetWriterInput?
    private var writerSessionStarted = false
    private var outputURL: URL?

    func configure() async throws {
        let camera = await AVCaptureDevice.requestAccess(for: .video)
        let microphone = await AVCaptureDevice.requestAccess(for: .audio)
        guard camera && microphone else { throw CameraError.permissionDenied }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do { try self.configureSession(); continuation.resume() }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let cameraInput = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(cameraInput) else {
            throw CameraError.configurationFailed
        }
        session.addInput(cameraInput)

        if let microphone = AVCaptureDevice.default(for: .audio),
           let microphoneInput = try? AVCaptureDeviceInput(device: microphone),
           session.canAddInput(microphoneInput) {
            session.addInput(microphoneInput)   // audio rescue needs the track
        }

        videoOutput.videoSettings =
            [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        guard session.canAddOutput(videoOutput) else { throw CameraError.configurationFailed }
        session.addOutput(videoOutput)

        if session.canAddOutput(audioOutput) {
            audioOutput.setSampleBufferDelegate(self, queue: outputQueue)
            session.addOutput(audioOutput)
        }

        // Portrait upright to match the locked UI orientation.
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
    }

    func start() {
        sessionQueue.async {
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    func startRecording() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rally-\(Int(Date().timeIntervalSince1970)).mp4")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 1080,     // portrait: rotated 1920x1080
            AVVideoHeightKey: 1920,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 12_000_000],
        ])
        video.expectsMediaDataInRealTime = true

        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 96_000,
        ])
        audio.expectsMediaDataInRealTime = true

        guard writer.canAdd(video), writer.canAdd(audio) else {
            throw CameraError.configurationFailed
        }
        writer.add(video)
        writer.add(audio)
        guard writer.startWriting() else {
            throw writer.error ?? CameraError.configurationFailed
        }

        outputQueue.sync {
            self.writer = writer
            self.writerVideo = video
            self.writerAudio = audio
            self.writerSessionStarted = false
            self.outputURL = url
        }
    }

    func stopRecording() async throws -> URL {
        let (writer, video, audio, url, sessionStarted) = outputQueue.sync {
            let state = (self.writer, self.writerVideo, self.writerAudio,
                         self.outputURL, self.writerSessionStarted)
            self.writer = nil
            self.writerVideo = nil
            self.writerAudio = nil
            self.outputURL = nil
            return state
        }
        guard let writer, let url else { throw CameraError.notRecording }
        guard sessionStarted else {
            // No frame ever reached the writer (instant stop / stalled
            // session): finishWriting would fail with an opaque -11800,
            // so cancel and clean up instead.
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: url)
            throw CameraError.recordingEmpty
        }
        video?.markAsFinished()
        audio?.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw writer.error ?? CameraError.configurationFailed
        }
        return url
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate,
                            AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if output === videoOutput,
           let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            onVideoSample?(pixelBuffer, CMTimeGetSeconds(timestamp))
        }

        guard let writer else { return }
        if output === videoOutput {
            if !writerSessionStarted {
                writer.startSession(atSourceTime: timestamp)
                writerSessionStarted = true
            }
            if let input = writerVideo, input.isReadyForMoreMediaData {
                input.append(sampleBuffer)
            }
        } else if writerSessionStarted {
            if let input = writerAudio, input.isReadyForMoreMediaData {
                input.append(sampleBuffer)
            }
        }
    }
}
