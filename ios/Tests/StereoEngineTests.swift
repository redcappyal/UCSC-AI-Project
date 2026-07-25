import XCTest
import simd
@testable import SquashLineCalling

final class StereoEngineTests: XCTestCase {
    private var goldens: [String: Any] = [:]
    private var left: CameraModel!
    private var right: CameraModel!

    override func setUp() {
        super.setUp()
        let url = Bundle(for: Self.self)
            .url(forResource: "stereo_goldens", withExtension: "json")!
        goldens = try! JSONSerialization.jsonObject(
            with: try! Data(contentsOf: url)) as! [String: Any]
        let cameras = goldens["cameras"] as! [String: Any]
        func decode(_ key: String) -> CameraModel {
            try! CameraModel.fromJSON(
                try! JSONSerialization.data(withJSONObject: cameras[key]!))
        }
        left = decode("left"); right = decode("right")
    }

    private func caseNamed(_ name: String) -> [String: Any] {
        (goldens["trajectories"] as! [[String: Any]]).first { ($0["name"] as! String) == name }!
    }

    private static let goldenData: [String: Any] = {
        let url = Bundle(for: StereoEngineTests.self)
            .url(forResource: "stereo_goldens", withExtension: "json")!
        return try! JSONSerialization.jsonObject(
            with: try! Data(contentsOf: url)) as! [String: Any]
    }()

    /// Pixel/timestamp samples for the "clean" golden trajectory, keyed to
    /// whichever eye `camera` is: `left`'s camera-center x is 9.0 in the
    /// fixture (`right`'s is 12.0), and that pairing is how the fixture ties
    /// `samples_a` to the left eye and `samples_b` to the right one. Lifts
    /// the sample construction `testCleanTrajectoryEmitsGoldenImpact` used to
    /// build inline, so both tests share one source instead of duplicating it.
    private static func goldenSamples(for camera: CameraModel) -> [TrackSample] {
        let cameras = goldenData["cameras"] as! [String: Any]
        let leftCenter = (cameras["left"] as! [String: Any])["camera_center_ft"] as! [Double]
        let key = camera.cameraCenterFt == SIMD3(leftCenter[0], leftCenter[1], leftCenter[2])
            ? "samples_a" : "samples_b"
        let case_ = (goldenData["trajectories"] as! [[String: Any]])
            .first { ($0["name"] as! String) == "clean" }!
        return (case_[key] as! [[String: Any]]).map { s in
            let px = s["px"] as! [Double]
            return TrackSample(tS: s["t_s"] as! Double, px: SIMD2(px[0], px[1]))
        }
    }

    /// Feed the clean golden trajectory through the live paths and expect the
    /// same impact the batch authority produced.
    func testCleanTrajectoryEmitsGoldenImpact() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })   // same clock in tests
        var events: [StereoEvent] = []
        engine.onEvent = { events.append($0) }

        let samplesA = Self.goldenSamples(for: left)
        let samplesB = Self.goldenSamples(for: right)
        for s in samplesA {
            engine.addLocalPixel(s.px, tS: s.tS)
        }
        let tuples = samplesB.enumerated().map { index, s in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64(s.tS * 1_000_000_000),
                           x: Float(s.px.x),
                           y: Float(s.px.y),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(tuples)
        engine.processIfDue(now: 100.0)
        engine.flushForTesting()

        let expected = (case_["impacts"] as! [[String: Any]]).first!
        guard case .impact(let impact, _)? = events.first else {
            return XCTFail("no impact emitted")
        }
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(impact.surface, expected["surface"] as! String)
        XCTAssertEqual(impact.call, expected["call"] as! String)
        XCTAssertEqual(impact.confidence, expected["confidence"] as! String)
        XCTAssertEqual(impact.tS, expected["t_s"] as! Double, accuracy: 0.02)
    }

    /// Re-processing must not re-emit the same impact (dedupe), and a second
    /// processIfDue inside the interval must be a no-op.
    func testImpactEmittedExactlyOnce() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })
        var count = 0
        engine.onEvent = { _ in count += 1 }
        for s in case_["samples_a"] as! [[String: Any]] {
            engine.addLocalPixel(SIMD2((s["px"] as! [Double])[0], (s["px"] as! [Double])[1]),
                                 tS: s["t_s"] as! Double)
        }
        let tuples = (case_["samples_b"] as! [[String: Any]]).enumerated().map { index, s in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64((s["t_s"] as! Double) * 1_000_000_000),
                           x: Float((s["px"] as! [Double])[0]),
                           y: Float((s["px"] as! [Double])[1]),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(tuples)
        engine.processIfDue(now: 100.0)
        engine.processIfDue(now: 100.1)   // inside interval: no-op
        engine.processIfDue(now: 100.5)   // recompute: dedupe must hold
        engine.flushForTesting()
        XCTAssertEqual(count, 1)
    }

    /// Between two real (non-no-op) `processIfDue` calls, extend both sides'
    /// tails with a synthetic continuation so `tHi` — and therefore the
    /// resampled timeline grid handed to `StereoTrack.detectImpacts` on the
    /// recompute — actually grows (unlike the untouched-data recompute in
    /// `testImpactEmittedExactlyOnce`, where every call operates on the
    /// identical `tLo`/`tHi`/track). `tLo` is pinned by `.first` on an
    /// append-only buffer, so it can't move without an out-of-order insert;
    /// growing `tHi` already forces a real recompute — a longer track, a
    /// larger `count`, more surfaces×indices scanned by `detectImpacts` —
    /// while the already-emitted impact must still be recognized as a dupe
    /// (surface + within `emitDedupeS`) rather than re-emitted because the
    /// underlying track object is now a different shape.
    ///
    /// The continuation extrapolates each side LINEARLY from its own last
    /// two real samples (constant velocity, matching position *and*
    /// velocity at the join) rather than replaying a chunk of real samples
    /// verbatim or holding the last position: "clean"'s tail is the ball
    /// genuinely still descending toward the floor when the real data ends
    /// (planeDistance for "floor" is signed height z, strictly decreasing
    /// through the whole tail), so anything spliced on that isn't
    /// velocity-matched — a repeated earlier chunk (position jumps
    /// backwards), a held/frozen point (velocity jumps to zero) — creates a
    /// discontinuity at the join that a windowed quadratic fit turns into a
    /// spurious local extremum right at the boundary, which registers as a
    /// bogus SECOND "floor" impact and would defeat the point of this test
    /// (grid-shift dedupe of the one real impact, not "no extra impacts are
    /// invented by the test's own splice"). A velocity-matched linear
    /// continuation keeps z monotonically decreasing with no reversal, so
    /// `detectImpacts`'s local-minimum test (dist decreasing before,
    /// increasing after) never re-fires anywhere in the synthetic tail.
    func testDedupeSurvivesGridShift() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })
        var count = 0
        engine.onEvent = { _ in count += 1 }
        let samplesA = case_["samples_a"] as! [[String: Any]]
        let samplesB = case_["samples_b"] as! [[String: Any]]
        for s in samplesA {
            engine.addLocalPixel(SIMD2((s["px"] as! [Double])[0], (s["px"] as! [Double])[1]),
                                 tS: s["t_s"] as! Double)
        }
        let tuples = samplesB.enumerated().map { index, s in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64((s["t_s"] as! Double) * 1_000_000_000),
                           x: Float((s["px"] as! [Double])[0]),
                           y: Float((s["px"] as! [Double])[1]),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(tuples)
        engine.processIfDue(now: 100.0)   // first real process: emits the impact

        // Constant-velocity extrapolation from each side's own last two real
        // samples — 10 more points per side at the same cadence, continuing
        // (not jumping) past the tail. Pushes tHi (and the resampled grid)
        // later with no position/velocity discontinuity at the join.
        func extrapolate(_ samples: [[String: Any]]) -> [(t: Double, px: [Double])] {
            let last = samples[samples.count - 1]
            let prev = samples[samples.count - 2]
            let lastT = last["t_s"] as! Double
            let dt = lastT - (prev["t_s"] as! Double)
            let lastPx = last["px"] as! [Double]
            let prevPx = prev["px"] as! [Double]
            let vel = [(lastPx[0] - prevPx[0]) / dt, (lastPx[1] - prevPx[1]) / dt]
            return (1...10).map { i in
                let dtStep = Double(i) * dt
                return (t: lastT + dtStep,
                        px: [lastPx[0] + vel[0] * dtStep, lastPx[1] + vel[1] * dtStep])
            }
        }
        for point in extrapolate(samplesA) {
            engine.addLocalPixel(SIMD2(point.px[0], point.px[1]), tS: point.t)
        }
        let baseSeq = UInt32(samplesB.count)
        let extraTuples = extrapolate(samplesB).enumerated().map { offset, point in
            DetectionTuple(seq: baseSeq + UInt32(offset),
                           ptsNs: UInt64(point.t * 1_000_000_000),
                           x: Float(point.px[0]), y: Float(point.px[1]),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(extraTuples)

        engine.processIfDue(now: 100.5)   // second real process: grid shifted, dedupe must hold
        engine.flushForTesting()
        XCTAssertEqual(count, 1)
    }

    func testRemoteTuplesWithUnmappableClockAreDropped() {
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { _ in nil })
        engine.addRemote([DetectionTuple(seq: 0, ptsNs: 1, x: 1, y: 1,
                                         conf: Float16(1), bboxH: Float16(1))])
        engine.processIfDue(now: 1.0)
        engine.flushForTesting()
        // Nothing to assert beyond "no crash, no event": guard via onEvent.
        var fired = false
        engine.onEvent = { _ in fired = true }
        engine.processIfDue(now: 2.0)
        engine.flushForTesting()
        XCTAssertFalse(fired)
    }

    func testAnalyzeReturnsTheTrackThatProducedTheImpacts() {
        // Reuses this class's existing `left` / `right` golden camera models and
        // the same sample construction `testCleanTrajectoryEmitsGoldenImpact`
        // uses. `analyze` must agree with `detectImpacts` exactly — it is the
        // same computation, surfaced, not a second one.
        let samplesLeft = Self.goldenSamples(for: left)
        let samplesRight = Self.goldenSamples(for: right)
        let timeline = stride(from: 0.0, to: 1.0, by: 1.0 / 240.0).map { $0 }

        let viaDetect = StereoTrack.detectImpacts(left, samplesLeft,
                                                  right, samplesRight,
                                                  timelineS: timeline)
        let analyzed = StereoTrack.analyze(left, samplesLeft,
                                           right, samplesRight,
                                           timelineS: timeline)
        XCTAssertEqual(analyzed.impacts, viaDetect)
        XCTAssertFalse(analyzed.track.isEmpty)
    }
}
