import Foundation

/// Record-to-results pipeline: latest calibration -> upload -> start track ->
/// poll until complete/failed. Mirrors the web app's runTrackBtn flow.
@MainActor
final class RunSubmission: ObservableObject {
    enum Phase: Equatable {
        case idle
        case fetchingCalibration
        case uploading
        case tracking(progress: Double, message: String)
        case complete(JobStatus)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    private let api: APIClientProtocol
    private let pollInterval: Duration

    init(api: APIClientProtocol = APIClient(), pollInterval: Duration = .seconds(1)) {
        self.api = api
        self.pollInterval = pollInterval
    }

    var completedRunID: String? {
        if case .complete(let job) = phase { return job.runID }
        return nil
    }

    func submit(videoURL: URL, duration: Double) async {
        do {
            phase = .fetchingCalibration
            let calibration = try await api.latestCalibration()

            phase = .uploading
            let upload = try await api.upload(videoURL: videoURL)
            let clipDuration = upload.duration ?? duration

            var job = try await api.startTrack(
                videoID: upload.videoID,
                calibrationJSON: calibration.calibrationJSON,
                duration: clipDuration)

            while job.status == "queued" || job.status == "running" {
                phase = .tracking(progress: job.progress ?? 0,
                                  message: job.message ?? "Analyzing…")
                try await Task.sleep(for: pollInterval)
                guard let runID = job.runID else { throw APIError.badResponse }
                job = try await api.trackStatus(runID: runID)
            }

            if job.status == "complete" {
                phase = .complete(job)
            } else {
                phase = .failed(job.error ?? "Tracking failed.")
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }
}
