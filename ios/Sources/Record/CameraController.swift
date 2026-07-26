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
    /// Which way the phone sits in its mount — the source of truth for both
    /// the video connection's rotation and the asset writer's dimensions
    /// below. `updateOrientation(_:)` below is the ONLY writer, always on
    /// `sessionQueue` — `RecordModel` never assigns this property directly,
    /// so there is one write path, not a direct write from the main actor
    /// racing a queued one.
    ///
    /// `RecordModel.startCamera` calls `updateOrientation(_:)` with a guess
    /// resolved from the interface orientation — once before `configure()`
    /// runs, and again on every later Play appearance — purely so the preview
    /// and the court-exposure meter have something sensible to work with, and
    /// so the mount cannot go stale after the operator re-mounts the phone.
    /// That value is deliberately NOT pinned, which is what lets the Play tab
    /// stay at both-landscape (`.landscape`) through the whole framing
    /// window. `RecordModel.applyRecording`'s start path is what actually
    /// commits to a mount: it re-resolves the interface orientation at
    /// record start and calls `updateOrientation(_:)` again — this has to
    /// land before `startRecording()` sizes the asset writer, so it comes
    /// first — but the orientation mask is only pinned, and this model's own
    /// `captureOrientation` only assigned, AFTER `startRecording()` actually
    /// succeeds; a failed start reverts this property back rather than leave
    /// it describing a recording that never began. Either caller can also
    /// leave this at its PREVIOUS value rather than the one it asked for, if
    /// the connection can't support the requested angle — see
    /// `updateOrientation(_:)`'s return value for when and why.
    var orientation: CaptureSettings.CaptureOrientation = .landscapeRight
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

    /// Repeatable: every call leaves the session configured from scratch.
    ///
    /// That matters because most of this method's throws happen *after* the
    /// session has already been mutated — `applyCaptureFormat` throws with the
    /// camera input added, the video-output guard throws with the same, and
    /// `lockForCourt()` throws with the session fully configured and running.
    /// `RecordModel.startCamera()` clears its own `cameraStarted` flag on any
    /// of those so the user can retry, and the model now outlives the view, so
    /// popping and re-pushing the record stage re-runs its `.task` and really
    /// does retry. Without the teardown below, that retry hits
    /// `session.canAddInput(...) == false` and throws `configurationFailed`
    /// forever — the camera would be bricked for the rest of the launch.
    ///
    /// Tearing down first rather than tracking "already configured": a
    /// half-configured session is the case that actually needs handling, and
    /// that state has to be *undone*, not skipped. One rule, no second flag to
    /// keep in step with `RecordModel.cameraStarted`.
    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
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
            // 0 for landscape-right (the sensor's native readout) or 180 for
            // landscape-left (that readout upside down) — see
            // CaptureSettings.rotationAngle(for:).
            let angle = CaptureSettings.rotationAngle(for: orientation)
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
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

    /// Re-resolves `orientation`, applying the matching rotation to the live
    /// video connection when one already exists — the counterpart to
    /// `configureSession()`'s one-time `videoRotationAngle` set above, needed
    /// now that the mount can be re-resolved on every Play appearance
    /// (`RecordModel.startCamera`) and again at record start
    /// (`RecordModel.applyRecording`) rather than only once, before
    /// `configure()` ever runs. This is also the ONLY writer of `orientation`
    /// (see its doc) — `RecordModel.startCamera`'s mount seeds go through here
    /// too, not a direct assignment, so every write is `sessionQueue`-confined
    /// rather than a mix of that and the main actor.
    ///
    /// Two outcomes, both deliberate:
    /// - **No live connection yet** (`configure()` hasn't run — this is
    ///   `startCamera`'s first, pre-configure seed): there is nothing to apply
    ///   or diverge from, so this simply records `newOrientation` and returns
    ///   `true`. `configureSession()` reads `orientation` to set the
    ///   connection's INITIAL rotation once it runs. `startCamera`'s later
    ///   per-appearance re-seeds take the connection branch below instead,
    ///   like `applyRecording` does.
    /// - **A connection exists:** `isVideoRotationAngleSupported` is checked
    ///   FIRST, not as a courtesy. Setting an unsupported `videoRotationAngle`
    ///   directly is a programmer error AVFoundation raises on — the opposite
    ///   of a silent no-op — so this guard is what keeps that from crashing a
    ///   connection that genuinely cannot rotate to the requested angle. If
    ///   unsupported, `orientation` is left at its PREVIOUS value and this
    ///   returns `false`: updating it while the connection kept the old
    ///   rotation would be exactly the divergence this whole change exists to
    ///   prevent. If supported, the connection is rotated, `orientation` is
    ///   updated to match, and this returns `true`.
    ///
    /// AVFoundation allows setting `videoRotationAngle` on a running
    /// connection at any time — no `beginConfiguration()`/
    /// `commitConfiguration()` bracket needed, unlike adding or removing
    /// inputs and outputs — but the connection is still session state, so
    /// the mutation goes through `sessionQueue` like every other touch of
    /// `session` in this file.
    ///
    /// Callers that need the connection correct before their next step
    /// (`applyRecording` awaits this, checks the `Bool`, and only then calls
    /// `startRecording()`, which reads `orientation` synchronously to size
    /// the asset writer — the mask is pinned later still, only once that
    /// recording actually starts) can await it and act on the result: it only
    /// returns once the change has actually applied on `sessionQueue` — or
    /// definitively failed to — not merely been scheduled.
    @discardableResult
    func updateOrientation(_ newOrientation: CaptureSettings.CaptureOrientation) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            sessionQueue.async {
                guard let connection = self.videoOutput.connection(with: .video) else {
                    self.orientation = newOrientation
                    continuation.resume(returning: true)
                    return
                }
                let angle = CaptureSettings.rotationAngle(for: newOrientation)
                guard connection.isVideoRotationAngleSupported(angle) else {
                    continuation.resume(returning: false)
                    return
                }
                connection.videoRotationAngle = angle
                self.orientation = newOrientation
                continuation.resume(returning: true)
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

        // Same geometry for both mounts (CaptureSettings.frameSize(for:)):
        // rotationAngle(for:) above already normalizes the frame upright
        // either way, so the writer always sees 4K landscape.
        let frameSize = CaptureSettings.frameSize(for: orientation)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: CaptureSettings.videoCodec,
            AVVideoWidthKey: frameSize.width,
            AVVideoHeightKey: frameSize.height,
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
