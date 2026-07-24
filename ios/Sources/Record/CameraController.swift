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
    /// Every audio sample buffer, on the output queue. The pairing clap
    /// detector subscribes here; nil costs nothing.
    var onAudioSample: ((CMSampleBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "slc.camera.session")
    // One queue for BOTH outputs: writer state below is queue-confined to it.
    private let outputQueue = DispatchQueue(label: "slc.camera.output")

    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()
    private var videoDevice: AVCaptureDevice?

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
        // .inputPriority hands format choice to us: any real preset would
        // stomp the activeFormat set below.
        session.sessionPreset = .inputPriority

        // Ultrawide for court coverage from the back-wall mount. It is the
        // slower lens (typically f/2.4 against the main camera's f/1.6), so
        // it costs roughly 1.3 stops — which is exactly why the exposure
        // solve in lockForCourt() has a fallback shutter. Devices without an
        // ultrawide fall back to the main camera rather than failing.
        guard let camera = AVCaptureDevice.default(.builtInUltraWideCamera,
                                                   for: .video, position: .back)
                ?? AVCaptureDevice.default(.builtInWideAngleCamera,
                                           for: .video, position: .back),
              let cameraInput = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(cameraInput) else {
            throw CameraError.configurationFailed
        }
        session.addInput(cameraInput)
        videoDevice = camera
        try applyCaptureFormat(to: camera)

        if let microphone = AVCaptureDevice.default(for: .audio),
           let microphoneInput = try? AVCaptureDeviceInput(device: microphone),
           session.canAddInput(microphoneInput) {
            session.addInput(microphoneInput)   // audio rescue needs the track
        }

        // Native biplanar YUV, NOT BGRA. At 4K60 a BGRA conversion is ~33 MB
        // per frame and ~2 GB/s of pointless bandwidth; Vision and the HEVC
        // encoder both consume 420f directly, so nothing downstream needs it.
        let native = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        if videoOutput.availableVideoPixelFormatTypes.contains(native) {
            videoOutput.videoSettings =
                [kCVPixelBufferPixelFormatTypeKey as String: native]
        }
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        guard session.canAddOutput(videoOutput) else { throw CameraError.configurationFailed }
        session.addOutput(videoOutput)

        if session.canAddOutput(audioOutput) {
            audioOutput.setSampleBufferDelegate(self, queue: outputQueue)
            session.addOutput(audioOutput)
        }

        if let connection = videoOutput.connection(with: .video) {
            // Portrait upright to match the locked UI orientation.
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            // Stabilisation OFF, and this one is not negotiable: OIS and EIS
            // both warp the frame per-frame, which would silently invalidate
            // the fixed floor homography and the 15-correspondence camera
            // solve. A calibrated camera has to stay geometrically rigid.
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }
    }

    /// Pins the sensor to 4K60 (see CaptureSettings). Throws only when the
    /// device offers no usable video format at all.
    private func applyCaptureFormat(to device: AVCaptureDevice) throws {
        let specs = device.formats.map { format -> CaptureSettings.FormatSpec in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return CaptureSettings.FormatSpec(
                width: dimensions.width,
                height: dimensions.height,
                maxFrameRate: format.videoSupportedFrameRateRanges
                    .map(\.maxFrameRate).max() ?? 0)
        }
        guard let index = CaptureSettings.bestFormatIndex(specs) else {
            throw CameraError.configurationFailed
        }

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        device.activeFormat = device.formats[index]
        // Pin min AND max: leaving the range open lets the device drop frame
        // rate on its own in dim light, which is the one automatic behaviour
        // that would quietly halve our trajectory sampling mid-rally.
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(CaptureSettings.frameRate))
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration
    }

    /// Meters the court once, then freezes exposure, white balance and focus.
    ///
    /// Call after `start()`, with the camera pointed at the court from its
    /// mounted position — this is the "set per court" step. Returns the solve
    /// so the caller can surface an underexposed court instead of discovering
    /// it in the footage.
    @discardableResult
    func lockForCourt() async throws -> CaptureSettings.ExposureSolution {
        guard let device = videoDevice else { throw CameraError.configurationFailed }

        // Let the meter converge before freezing whatever it found. The first
        // wait is unconditional: called straight after start(), no frame has
        // reached the sensor yet, so isAdjustingExposure is still false and
        // the poll below would fall through on stale defaults.
        try? await Task.sleep(nanoseconds: 400_000_000)
        var settleAttempts = 0
        while device.isAdjustingExposure || device.isAdjustingWhiteBalance {
            guard settleAttempts < 20 else { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
            settleAttempts += 1
        }

        let solution = CaptureSettings.solveExposure(
            meteredDuration: device.exposureDuration,
            meteredISO: device.iso,
            deviceMinISO: device.activeFormat.minISO,
            deviceMaxISO: device.activeFormat.maxISO)

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        if device.isExposureModeSupported(.custom) {
            let duration = CMTimeClampToRange(
                solution.duration,
                range: CMTimeRange(start: device.activeFormat.minExposureDuration,
                                   end: device.activeFormat.maxExposureDuration))
            // Explicit completionHandler: in an async context the bare call
            // resolves to the awaitable overload, which would suspend here
            // while still holding the device configuration lock.
            device.setExposureModeCustom(duration: duration, iso: solution.iso,
                                         completionHandler: nil)
        }
        // Locking freezes the gains the meter just settled on — per court,
        // per lighting rig, exactly as intended.
        if device.isWhiteBalanceModeSupported(.locked) {
            device.whiteBalanceMode = .locked
        }
        // The mount does not move and neither does the court, so autofocus
        // has nothing to contribute and everything to hunt for. Ultrawides
        // with fixed focus report this unsupported; nothing to do there.
        if device.isFocusModeSupported(.locked) {
            device.focusMode = .locked
        }
        return solution
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
            AVVideoCodecKey: CaptureSettings.videoCodec,
            AVVideoWidthKey: CaptureSettings.frameWidth,    // portrait: rotated 4K
            AVVideoHeightKey: CaptureSettings.frameHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: CaptureSettings.videoBitRate,
                // Rate control guesses badly at 4K without being told the
                // cadence it is encoding.
                AVVideoExpectedSourceFrameRateKey: Int(CaptureSettings.frameRate),
            ],
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

        if output === audioOutput {
            onAudioSample?(sampleBuffer)
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
