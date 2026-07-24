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

    /// Feed the clean golden trajectory through the live paths and expect the
    /// same impact the batch authority produced.
    func testCleanTrajectoryEmitsGoldenImpact() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })   // same clock in tests
        var events: [StereoEvent] = []
        engine.onEvent = { events.append($0) }

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
        engine.processIfDue(now: 100.0)
        engine.flushForTesting()

        let expected = (case_["impacts"] as! [[String: Any]]).first!
        guard case .impact(let impact)? = events.first else {
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
}
