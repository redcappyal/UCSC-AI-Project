import AVFoundation
import CoreMedia
import Foundation

/// The court-capture contract, pinned in one place.
///
/// Every value here is deliberately fixed rather than left to the device's
/// automatic behaviour. The recording serves two masters — it feeds the live
/// tracker AND it becomes training footage — and both want the same thing:
/// no free variables. Auto-exposure hunting as a player in dark kit crosses
/// frame is a nuisance dimension the detector would otherwise have to spend
/// labelled examples absorbing.
///
/// Rationale for the numbers, so nobody "optimises" them back:
///
/// - **4K, not 1080p.** The detector downsamples to 960 either way
///   (inference_engine.resize_frame_for_inference), so 4K costs today's
///   pipeline nothing. It buys the archive: at the front wall the ball is
///   ~6 px at 1080p and ~12 px at 4K, and eval_set/BASELINE-2026-07-23.md
///   has 65 of 71 missed bounces on wall hits. Pixels not sampled now can
///   never be recovered by a future model; frames can at least be
///   interpolated. Apple encodes 4K60 at ~400 MB/min against ~350 MB/min for
///   1080p120, so this is ~14% more bytes for 4x the pixels.
/// - **60 fps, not 120.** High-frame-rate modes use a faster, noisier, often
///   cropped sensor readout, and we are already ISO-limited on a squash
///   court. 60 stays on the main readout path and leaves thermal headroom
///   for a full match.
/// - **1/1000 shutter.** A ~28 m/s ball travels 230 mm in 1/120 s — a ~34 px
///   streak of a ~6 px ball at the front wall. At 1/1000 it travels 28 mm,
///   so the streak is about the ball's own size. Blur, not resolution, is
///   what hides far-wall contacts.
enum CaptureSettings {
    /// Sensor-space capture dimensions (landscape, as the format reports).
    static let sensorWidth = 3840
    static let sensorHeight = 2160

    /// Landscape frame geometry — the space every consumer works in. Overlay
    /// mapping, peer detection tuples and the asset writer all key off these.
    ///
    /// Capture is always landscape: the phone sits on its side in a back-wall
    /// mount, where the sensor's readout is already the right way round.
    /// Both mounts (see `CaptureOrientation`) produce these same dimensions,
    /// because `rotationAngle(for:)` normalizes the frame upright either way.
    static let frameWidth = sensorWidth
    static let frameHeight = sensorHeight

    /// Which way the phone sits in its mount. The cases differ only in which
    /// end the lens is at, which is what decides the rotation needed to bring
    /// the frame upright.
    ///
    /// The two phones in a paired session MUST agree. Both cases yield the
    /// same `frameSize(for:)`, so dimensions alone cannot tell them apart —
    /// the mount travels explicitly in `Hello.captureOrientation` and
    /// `PeerSession` enforces the match at handshake (see its `.hello`
    /// handler). Mixing mounts is refused rather than absorbed, so the
    /// operator learns the mount is wrong instead of the archive inheriting
    /// it.
    enum CaptureOrientation: String, Codable, Equatable, CaseIterable {
        case landscapeRight, landscapeLeft
    }

    /// Rotation applied to the video connection so the recorded frame comes
    /// out absolute-upright whichever way the phone is mounted: 0 for
    /// landscape-right (the sensor's native readout), 180 for landscape-left
    /// (that same readout upside down).
    ///
    /// Normalizing here rather than downstream is what keeps solo footage —
    /// which is also the training archive — the right way up when there is no
    /// peer to catch a wrong mount.
    static func rotationAngle(for orientation: CaptureOrientation) -> CGFloat {
        switch orientation {
        case .landscapeRight: return 0
        case .landscapeLeft: return 180
        }
    }

    /// Frame geometry the video connection and asset writer actually produce
    /// once `rotationAngle(for:)` has been applied. Identical for both mounts
    /// by design — that is what normalization buys.
    static func frameSize(for orientation: CaptureOrientation) -> (width: Int, height: Int) {
        (frameWidth, frameHeight)
    }

    static let frameRate: Double = 60

    /// Freezes the ball. See the type doc for the motion-blur arithmetic.
    static let targetShutter = CMTime(value: 1, timescale: 1000)

    /// Given up only when the court is too dim to hold `targetShutter` under
    /// the ISO ceiling. Still an ~8 px streak versus ~34 px wide open, so it
    /// keeps most of the win and buys a full stop.
    static let fallbackShutter = CMTime(value: 1, timescale: 500)

    /// Noise ceiling we would rather not cross. Sensor noise costs the
    /// detector far less than motion blur does, but past roughly this point
    /// the footage stops being pleasant to review. Clamped further by
    /// whatever the device reports.
    static let preferredMaxISO: Float = 2500

    /// HEVC. Roughly half the bytes of H.264 at matching quality, which is
    /// what makes 4K60 affordable to keep.
    static let videoCodec = AVVideoCodecType.hevc

    /// ~53 Mbps, matching Apple's own 4K60 rate (~400 MB/min).
    static let videoBitRate = 53_000_000

    // MARK: format selection

    /// The measurable part of an `AVCaptureDevice.Format`, so the choice can
    /// be tested without a camera (formats cannot be constructed).
    struct FormatSpec: Equatable {
        let width: Int32
        let height: Int32
        let maxFrameRate: Double
    }

    /// Index of the format to run, or nil when the device offers none.
    ///
    /// Frame rate is the hard requirement and resolution is the soft one: a
    /// 4K30 format loses to 1080p60, because trajectory density is what the
    /// ballistic fit needs and 30 fps quantises impact timing to 0.93 m of
    /// ball travel. Above 4K we stop wanting pixels — stills-sized formats
    /// blow the encoder and ANE budget for detail the detector discards.
    static func bestFormatIndex(_ specs: [FormatSpec]) -> Int? {
        guard !specs.isEmpty else { return nil }
        let atFrameRate = specs.indices.filter { specs[$0].maxFrameRate >= frameRate }
        let pool = atFrameRate.isEmpty ? Array(specs.indices) : atFrameRate
        return pool.min { rank(specs[$0]) < rank(specs[$1]) }
    }

    /// Sorts under-budget formats ahead of over-budget ones, then by distance
    /// from the 4K budget — so under budget maximises pixels and over budget
    /// minimises the overshoot.
    private static func rank(_ spec: FormatSpec) -> (Int, Int) {
        let pixels = Int(spec.width) * Int(spec.height)
        let budget = sensorWidth * sensorHeight
        return pixels <= budget ? (0, budget - pixels) : (1, pixels - budget)
    }

    // MARK: exposure

    struct ExposureSolution: Equatable {
        let duration: CMTime
        let iso: Float
        /// True when even `fallbackShutter` could not hold the metered
        /// exposure inside the ISO ceiling — the court is genuinely too dark
        /// and the footage will come out under. Surface it rather than
        /// silently shooting a black frame.
        let underexposed: Bool
    }

    /// Converts a metered auto-exposure reading into locked manual settings
    /// at (or near) `targetShutter`.
    ///
    /// Exposure is proportional to duration x ISO at fixed aperture, so
    /// shortening the shutter by some factor means multiplying ISO by the
    /// same factor to land on the same brightness. We spend shutter last:
    /// noise is recoverable information, blur is not.
    static func solveExposure(meteredDuration: CMTime, meteredISO: Float,
                              deviceMinISO: Float, deviceMaxISO: Float) -> ExposureSolution {
        let ceiling = min(preferredMaxISO, deviceMaxISO)
        let metered = meteredDuration.seconds
        guard metered > 0, meteredISO > 0, ceiling > 0 else {
            // Meter never converged; take the safer shutter and say so.
            return ExposureSolution(duration: fallbackShutter, iso: ceiling,
                                    underexposed: true)
        }

        for shutter in [targetShutter, fallbackShutter] {
            let needed = meteredISO * Float(metered / shutter.seconds)
            if needed <= ceiling {
                return ExposureSolution(duration: shutter,
                                        iso: max(needed, deviceMinISO),
                                        underexposed: false)
            }
        }
        return ExposureSolution(duration: fallbackShutter, iso: ceiling,
                                underexposed: true)
    }

    /// One-line readout for the operator setting up a court. Worth showing
    /// even when nothing is wrong: it confirms the lock actually took, and
    /// the resolved shutter/ISO is what tells you whether the venue has
    /// enough light before you shoot a whole match.
    static func summary(for solution: ExposureSolution) -> String {
        let shutter = Int((1 / solution.duration.seconds).rounded())
        let iso = Int(solution.iso.rounded())
        guard solution.underexposed else { return "Locked 1/\(shutter) · ISO \(iso)" }
        return "Dim court — locked 1/\(shutter) · ISO \(iso). Add light if the ball smears."
    }
}
