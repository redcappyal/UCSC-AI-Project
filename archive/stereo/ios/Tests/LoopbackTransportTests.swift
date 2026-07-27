// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import XCTest
@testable import SquashLineCalling

final class LoopbackTransportTests: XCTestCase {
    func testControlAndDatagramReachPeer() {
        let (a, b) = LoopbackTransport.pair()
        var bControl: [Data] = []; var aDatagrams: [Data] = []
        b.onControl = { bControl.append($0) }
        a.onDatagram = { aDatagrams.append($0) }
        a.startInitiator(); b.startResponder()
        a.sendControl(Data([1])); b.sendDatagram(Data([2]))
        XCTAssertEqual(bControl, [Data([1])])
        XCTAssertEqual(aDatagrams, [Data([2])])
    }

    func testDeliveryHookCanDelayAndDrop() {
        let (a, b) = LoopbackTransport.pair()
        var received: [Data] = []
        b.onDatagram = { received.append($0) }
        var held: [(Data, (Data) -> Void)] = []
        a.datagramDeliveryHook = { payload, deliver in held.append((payload, deliver)) }
        a.startInitiator(); b.startResponder()
        a.sendDatagram(Data([9])); a.sendDatagram(Data([8]))
        XCTAssertTrue(received.isEmpty)          // held, not delivered
        held[1].1(held[1].0)                      // deliver out of order; drop the first
        XCTAssertEqual(received, [Data([8])])
    }

    func testStateTransitionsOnStartAndStop() {
        let (a, _) = LoopbackTransport.pair()
        var states: [TransportState] = []
        a.onStateChange = { states.append($0) }
        a.startInitiator()
        a.stop()
        XCTAssertEqual(states, [.connected, .disconnected("stopped")])
    }

    func testPairDoesNotLeakWhenReferencesDrop() {
        weak var weakA: LoopbackTransport?
        weak var weakB: LoopbackTransport?
        autoreleasepool {
            let (a, b) = LoopbackTransport.pair()
            weakA = a; weakB = b
            a.sendControl(Data([1]))   // exercise delivery before release
        }
        XCTAssertNil(weakA)
        XCTAssertNil(weakB)
    }
}
