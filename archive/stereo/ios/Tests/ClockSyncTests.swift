// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Tests/ClockSyncTests.swift
import XCTest
@testable import SquashLineCalling

final class ClockSyncTests: XCTestCase {
    /// Simulate: remote clock = local + 0.5 s exactly. Symmetric 20 ms RTT.
    private func addSymmetricSamples(_ sync: ClockSync, count: Int, offset: Double = 0.5) {
        for i in 0..<count {
            let t1 = Double(i)                       // local send
            let t2 = t1 + 0.010 + offset             // remote recv (10 ms up-leg)
            let t3 = t2 + 0.001                      // remote turnaround
            let t4 = t1 + 0.021                      // local recv (10 ms down-leg)
            sync.addSample(t1: t1, t2: t2, t3: t3, t4: t4)
        }
    }

    func testNilBeforeFiveSamples() {
        let sync = ClockSync()
        addSymmetricSamples(sync, count: 4)
        XCTAssertNil(sync.estimate)
    }

    func testRecoversTrueOffsetWithinUncertainty() {
        let sync = ClockSync()
        addSymmetricSamples(sync, count: 20)
        let estimate = try! XCTUnwrap(sync.estimate)
        XCTAssertEqual(estimate.offset, 0.5, accuracy: 1e-9)
        XCTAssertLessThanOrEqual(abs(estimate.offset - 0.5), estimate.uncertainty)
    }

    /// AWDL stall: a few samples have +150 ms on one leg. Min-RTT filtering
    /// must reject them; a mean would be pulled by ~ half the stall.
    func testStallOutliersAreRejected() {
        let sync = ClockSync()
        addSymmetricSamples(sync, count: 10)
        for i in 0..<10 {   // asymmetric stalls: up-leg +150 ms
            let t1 = 100.0 + Double(i)
            let t2 = t1 + 0.160 + 0.5
            let t3 = t2 + 0.001
            let t4 = t1 + 0.171
            sync.addSample(t1: t1, t2: t2, t3: t3, t4: t4)
        }
        let estimate = try! XCTUnwrap(sync.estimate)
        XCTAssertEqual(estimate.offset, 0.5, accuracy: 0.001)
    }

    func testAsymmetricPathBiasIsBoundedByHalfMinRTT() {
        let sync = ClockSync()
        for i in 0..<20 {   // 5 ms up, 15 ms down: true bias = −5 ms
            let t1 = Double(i)
            let t2 = t1 + 0.005 + 0.5
            let t3 = t2 + 0.001
            let t4 = t1 + 0.021
            sync.addSample(t1: t1, t2: t2, t3: t3, t4: t4)
        }
        let estimate = try! XCTUnwrap(sync.estimate)
        XCTAssertLessThanOrEqual(abs(estimate.offset - 0.5), estimate.uncertainty)
    }

    func testAnchorOverridesNetworkBias() {
        let sync = ClockSync()
        for i in 0..<20 {   // asymmetric network: biased estimate
            let t1 = Double(i)
            sync.addSample(t1: t1, t2: t1 + 0.005 + 0.5, t3: t1 + 0.006 + 0.5, t4: t1 + 0.021)
        }
        sync.applyAnchor(localEventTime: 50.0, remoteEventTime: 50.5)   // truth: 0.5
        let estimate = try! XCTUnwrap(sync.estimate)
        XCTAssertEqual(estimate.offset, 0.5, accuracy: 1e-9)
        XCTAssertEqual(estimate.uncertainty, 0.0005, accuracy: 1e-9)
        XCTAssertEqual(sync.remoteToLocal(51.0), 50.5)
    }
}
