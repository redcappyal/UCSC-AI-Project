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
    private let pollTimeout: Duration

    /// `pollTimeout` bounds the tracking poll below. It is not a guess about
    /// how long a clip *should* take; it is the point past which continuing to
    /// wait is worse than admitting we do not know.
    ///
    /// Why a cap has to exist at all: a job whose worker died still answers
    /// `"running"` forever. The HTTP layer is perfectly healthy, so nothing
    /// throws, and an uncapped loop never leaves `submit`, so the sheet the
    /// caller is presenting never resolves and only killing the app recovers.
    ///
    /// Why 60 minutes, for a 4K60 rally clip:
    ///
    /// * `job_runner.TRACKING_JOB_SEMAPHORE` is `Semaphore(1)` — the server
    ///   tracks exactly one clip at a time, and `run_tracking_job` only sets
    ///   `status="running"` after acquiring it, so a clip can sit `queued`
    ///   behind whatever else is already in flight.
    /// * The clip is tracked at `frame_stride` 1: every frame at the full
    ///   `inference_width`, and `job_runner` skips the refine pass entirely at
    ///   stride 1. That is roughly 4x the strided coarse-then-refine path the
    ///   original 20 minutes was sized against, which is where 60 comes from —
    ///   the same sizing, times the stride, rounded down to a round number.
    /// * The local WASB backend tiles at *native* resolution and ignores
    ///   `inference_width` (MODEL.md §6), so wall clock scales with the source
    ///   pixels, not with 960. A 4K clip is ~4x a 1080p one through that
    ///   backend and can still outlast this cap while perfectly healthy.
    /// * The same client already grants a single request on the same clip 15
    ///   minutes (`APIClient`'s `timeoutIntervalForResource`, for the upload).
    ///   A poll bound below that would be the client contradicting itself.
    ///
    /// The asymmetry is what settles it. Tripping early is cheap and
    /// recoverable: the run keeps going server-side and still lands in
    /// Matches. Not tripping at all is a permanent app-level dead end. So this
    /// is set generously and treated as a backstop, not a deadline anyone
    /// should meet.
    init(api: APIClientProtocol = APIClient(), pollInterval: Duration = .seconds(1),
         pollTimeout: Duration = .seconds(60 * 60)) {
        self.api = api
        self.pollInterval = pollInterval
        self.pollTimeout = pollTimeout
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
            // Server duration can be 0.0 when OpenCV fails to read frame_count;
            // fall back to the measured recording duration then.
            let serverDuration = upload.duration ?? 0
            let clipDuration = serverDuration > 0 ? serverDuration : duration

            var job = try await api.startTrack(
                videoID: upload.videoID,
                calibrationJSON: calibration.calibrationJSON,
                duration: clipDuration)

            // Measured from the first poll, not from `submit`'s entry: the
            // upload above can legitimately take minutes on a 4K60 clip and
            // already has `URLSession`'s own resource timeout over it, so
            // charging it against the tracking bound would shorten the bound by
            // an unrelated amount.
            //
            // `ContinuousClock`, not `Date`: this must not be movable by a
            // clock adjustment mid-rally, and it must keep counting while the
            // phone is asleep in someone's pocket between rallies.
            let deadline = ContinuousClock.now + pollTimeout
            while job.status == "queued" || job.status == "running" {
                phase = .tracking(progress: job.progress ?? 0,
                                  message: job.message ?? "Analyzing…")
                try await Task.sleep(for: pollInterval)
                // Checked after the sleep and before the next status call, so
                // the bound is enforced even against a server that answers
                // instantly and forever. Throwing (rather than breaking) routes
                // it through the same `catch` every other failure uses, so the
                // caller sees one shape of outcome: `.failed(message)`.
                guard ContinuousClock.now < deadline else {
                    throw APIError.trackingTimedOut(pollTimeout)
                }
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
