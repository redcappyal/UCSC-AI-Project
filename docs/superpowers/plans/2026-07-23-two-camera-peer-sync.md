# Two-Camera Peer Layer + Clock Sync (Plan A: spec Phases 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two iPhones pair on-court over an abstracted transport (BLE and peer-to-peer
Wi-Fi implementations), sync clocks to a measured uncertainty (NTP-style min-RTT +
clap-at-the-T acoustic anchor), stream ball detections secondary → primary, and produce
the bench report that decides the primary transport (spec's Phase 1–2 selection gate).

**Architecture:** A `PeerTransport` protocol hides BLE vs Wi-Fi behind two channels
(reliable control frames, lossy datagrams). `PeerSession` owns the pairing state machine,
version handshake, heartbeats, and sync rounds; `ClockSync` is a pure estimator fed
timestamp quadruples; `ClapDetector` is a pure onset finder fed PCM. Everything
radio-independent is unit-tested over a `LoopbackTransport`; radio code is verified by an
on-hardware bench procedure documented in `ios/PEER.md`.

**Tech Stack:** Swift 5.9, GCD + ObservableObject (match existing codebase style — no
async/await in new code except where the codebase already uses it), CoreBluetooth,
Network.framework, XCTest.

## Global Constraints

- iOS deployment target **17.0**, Swift **5.9**, `TARGETED_DEVICE_FAMILY: "1"` (ios/project.yml).
- **No MultipeerConnectivity** anywhere (spec decision log #5).
- Capture geometry in this plan is today's **portrait 1080×1920**; the wire format is
  dimension-agnostic (`hello` carries `frameW`/`frameH`) so Phase 4's landscape switch is
  a data change, not a protocol change.
- Peer protocol version constant `peerProtoVersion = 1`; version mismatch is a terminal
  `failed` state, never best-effort parsing (spec Component 1).
- Timestamps in peer code are **seconds on the capture host clock** (`CMClockGetTime(CMClockGetHostTimeClock())`),
  the same domain as `CameraController`'s `onVideoSample` PTS. Never `Date()`.
- New UI in this plan is **`#if DEBUG` only** (PeerBenchView). Shipped-UI screens are
  Phase 4 and out of scope; DESIGN.md is therefore untouched by this plan.
- Server code (`app.py`, `job_runner.py`, …) is untouched by this plan.
- All new Swift files live under `ios/Sources/Peer/` (plus one hook in
  `CameraController.swift`, one in `RecordModel.swift`); tests in `ios/Tests/`.
- Test command (macOS + Xcode): `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`.
  If Xcode/simulator is unavailable in the working sandbox, the fallback gate is
  parse-only: `xcrun -sdk iphonesimulator swiftc -parse <changed .swift files>` — and the
  full test run moves to the user-owned checklist in `ios/PEER.md` (same convention as
  the iOS-migration plan).
- Commit after every task (not every step) with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

---

### Task 1: FrameCodec — length-prefixed control framing

Both transports deliver the reliable control channel as arbitrary byte chunks (TCP
segments / BLE writes). `FrameCodec` turns a stream of chunks into whole frames.

**Files:**
- Create: `ios/Sources/Peer/FrameCodec.swift`
- Test: `ios/Tests/FrameCodecTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces: `FrameCodec.encode(_ payload: Data) -> Data` (static);
  `final class FrameReassembler { func ingest(_ chunk: Data) -> [Data] }` — returns zero
  or more complete payloads per chunk. Max frame 1 MiB → oversize throws is NOT used;
  reassembler drops the connection contract instead: `ingest` returns `nil` on protocol
  violation via `func ingest(_ chunk: Data) throws -> [Data]` with
  `FrameCodecError.oversizeFrame`.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/FrameCodecTests.swift
import XCTest
@testable import SquashLineCalling

final class FrameCodecTests: XCTestCase {
    func testRoundTripSingleFrame() throws {
        let payload = Data("hello".utf8)
        let wire = FrameCodec.encode(payload)
        let reassembler = FrameReassembler()
        XCTAssertEqual(try reassembler.ingest(wire), [payload])
    }

    func testSplitAcrossChunksAndCoalesced() throws {
        let a = Data("alpha".utf8), b = Data("bravo-long-payload".utf8)
        var wire = FrameCodec.encode(a); wire.append(FrameCodec.encode(b))
        let reassembler = FrameReassembler()
        // Feed one byte at a time: frames must pop out exactly twice, in order.
        var frames: [Data] = []
        for byte in wire { frames.append(contentsOf: try reassembler.ingest(Data([byte]))) }
        XCTAssertEqual(frames, [a, b])
    }

    func testEmptyPayloadAllowed() throws {
        let reassembler = FrameReassembler()
        XCTAssertEqual(try reassembler.ingest(FrameCodec.encode(Data())), [Data()])
    }

    func testOversizeFrameThrows() {
        var header = withUnsafeBytes(of: UInt32(2_000_000).littleEndian) { Data($0) }
        header.append(Data(repeating: 0, count: 8))
        let reassembler = FrameReassembler()
        XCTAssertThrowsError(try reassembler.ingest(header)) { error in
            XCTAssertEqual(error as? FrameCodecError, .oversizeFrame)
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15' -only-testing:SquashLineCallingTests/FrameCodecTests`
Expected: build FAILS with "cannot find 'FrameCodec' in scope".
(Sandbox fallback: `xcrun -sdk iphonesimulator swiftc -parse ios/Tests/FrameCodecTests.swift` parses; the missing-type failure is deferred to the user-run gate.)

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/FrameCodec.swift
import Foundation

enum FrameCodecError: Error, Equatable { case oversizeFrame }

/// u32 little-endian payload length + payload. Shared by the TCP control
/// stream and the BLE control characteristic.
enum FrameCodec {
    static let maxFrameLength = 1 << 20   // 1 MiB: calibration JSON fits with margin

    static func encode(_ payload: Data) -> Data {
        var out = withUnsafeBytes(of: UInt32(payload.count).littleEndian) { Data($0) }
        out.append(payload)
        return out
    }
}

/// Stateful: feed chunks in arrival order, get complete payloads out.
final class FrameReassembler {
    private var buffer = Data()

    func ingest(_ chunk: Data) throws -> [Data] {
        buffer.append(chunk)
        var frames: [Data] = []
        while buffer.count >= 4 {
            let length = buffer.prefix(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
            guard length <= FrameCodec.maxFrameLength else { throw FrameCodecError.oversizeFrame }
            let total = 4 + Int(length)
            guard buffer.count >= total else { break }
            frames.append(Data(buffer[buffer.startIndex + 4 ..< buffer.startIndex + total]))
            buffer.removeFirst(total)
        }
        return frames
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2. Expected: `FrameCodecTests` 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/FrameCodec.swift ios/Tests/FrameCodecTests.swift
git commit -m "feat(peer): length-prefixed control frame codec"
```

---

### Task 2: DetectionTuple binary codec + batching

The lossy datagram channel carries batches of ball detections (spec wire format, 24 B
per tuple).

**Files:**
- Create: `ios/Sources/Peer/DetectionTuple.swift`
- Test: `ios/Tests/DetectionTupleTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```swift
  struct DetectionTuple: Equatable {
      var seq: UInt32
      var ptsNs: UInt64        // sender host clock, nanoseconds
      var x: Float             // pixels in sender frame (frameW/frameH from hello)
      var y: Float             // pixels; doubles as rolling-shutter row
      var conf: Float16
      var bboxH: Float16       // bbox height, pixels
  }
  enum DetectionBatch {
      static func encode(_ tuples: [DetectionTuple]) -> Data
      static func decode(_ data: Data) -> [DetectionTuple]?   // nil = malformed
  }
  ```
  Wire: `[0x01][count: UInt16 LE][count × 24 B tuples]`. Type byte 0x01 reserves the
  datagram namespace for future message kinds.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/DetectionTupleTests.swift
import XCTest
@testable import SquashLineCalling

final class DetectionTupleTests: XCTestCase {
    private func sample(_ seq: UInt32) -> DetectionTuple {
        DetectionTuple(seq: seq, ptsNs: 123_456_789_000, x: 540.5, y: 960.25,
                       conf: Float16(0.87), bboxH: Float16(22.0))
    }

    func testEncodedSizeIsHeaderPlus24PerTuple() {
        XCTAssertEqual(DetectionBatch.encode([sample(1), sample(2)]).count, 3 + 48)
    }

    func testRoundTrip() {
        let tuples = [sample(7), sample(8), sample(9)]
        XCTAssertEqual(DetectionBatch.decode(DetectionBatch.encode(tuples)), tuples)
    }

    func testEmptyBatchRoundTrips() {
        XCTAssertEqual(DetectionBatch.decode(DetectionBatch.encode([])), [])
    }

    func testMalformedReturnsNil() {
        XCTAssertNil(DetectionBatch.decode(Data([0x01, 5, 0, 1, 2])))   // truncated
        XCTAssertNil(DetectionBatch.decode(Data([0x02, 0, 0])))          // unknown type
        XCTAssertNil(DetectionBatch.decode(Data()))                       // empty
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/DetectionTupleTests` (full command as Task 1).
Expected: build FAILS with "cannot find 'DetectionTuple' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/DetectionTuple.swift
import Foundation

struct DetectionTuple: Equatable {
    var seq: UInt32
    var ptsNs: UInt64
    var x: Float
    var y: Float
    var conf: Float16
    var bboxH: Float16
}

enum DetectionBatch {
    static let typeByte: UInt8 = 0x01
    static let tupleSize = 24

    static func encode(_ tuples: [DetectionTuple]) -> Data {
        var out = Data([typeByte])
        out.append(withUnsafeBytes(of: UInt16(tuples.count).littleEndian) { Data($0) })
        for t in tuples {
            out.append(withUnsafeBytes(of: t.seq.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.ptsNs.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.x.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.y.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.conf.bitPattern.littleEndian) { Data($0) })
            out.append(withUnsafeBytes(of: t.bboxH.bitPattern.littleEndian) { Data($0) })
        }
        return out
    }

    static func decode(_ data: Data) -> [DetectionTuple]? {
        guard data.count >= 3, data[data.startIndex] == typeByte else { return nil }
        let body = data.dropFirst(1)
        let count = Int(body.prefix(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian)
        let tuplesData = body.dropFirst(2)
        guard tuplesData.count == count * tupleSize else { return nil }
        var tuples: [DetectionTuple] = []
        tuples.reserveCapacity(count)
        var offset = tuplesData.startIndex
        func read<T: FixedWidthInteger>(_: T.Type) -> T {
            let value = tuplesData[offset ..< offset + MemoryLayout<T>.size]
                .withUnsafeBytes { $0.loadUnaligned(as: T.self) }
            offset += MemoryLayout<T>.size
            return T(littleEndian: value)
        }
        for _ in 0..<count {
            tuples.append(DetectionTuple(
                seq: read(UInt32.self),
                ptsNs: read(UInt64.self),
                x: Float(bitPattern: read(UInt32.self)),
                y: Float(bitPattern: read(UInt32.self)),
                conf: Float16(bitPattern: read(UInt16.self)),
                bboxH: Float16(bitPattern: read(UInt16.self))))
        }
        return tuples
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `DetectionTupleTests` 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/DetectionTuple.swift ios/Tests/DetectionTupleTests.swift
git commit -m "feat(peer): detection tuple binary codec and batching"
```

---

### Task 3: ControlMessage JSON protocol

**Files:**
- Create: `ios/Sources/Peer/ControlMessage.swift`
- Test: `ios/Tests/ControlMessageTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces (exact names later tasks rely on):
  ```swift
  let peerProtoVersion = 1
  enum PeerRole: String, Codable { case primary, secondary }
  struct Hello: Codable, Equatable {
      var protoVersion: Int
      var appVersion: String
      var deviceModel: String
      var nonce: UInt32          // pairing-code input
      var frameW: Int            // 1080 today (portrait)
      var frameH: Int            // 1920 today
  }
  struct SyncPing: Codable, Equatable { var pingID: UInt32; var t1: Double }
  struct SyncPong: Codable, Equatable { var pingID: UInt32; var t1: Double; var t2: Double; var t3: Double }
  enum ControlMessage: Codable, Equatable {
      case hello(Hello)
      case role(PeerRole)                       // primary assigns; payload = RECEIVER's role
      case calibration(profileID: String, calibrationJSON: String)
      case syncPing(SyncPing)
      case syncPong(SyncPong)
      case clapAnchor(hostTime: Double)         // sender's clap onset, host-clock seconds
      case record(action: String, ptsNs: UInt64)   // action: "start" | "stop"
      case event(rallyID: UInt32, json: String)    // primary → secondary rendered calls (Phase 3 fills json)
      case heartbeat(seq: UInt32)
      case sessionManifest(sessionID: String, videoID: String)
      static func encode(_ message: ControlMessage) throws -> Data
      static func decode(_ data: Data) -> ControlMessage?     // nil = malformed/unknown
  }
  func pairingCode(_ a: Hello, _ b: Hello) -> String           // 4 digits, symmetric
  ```
  Encoding: Swift-synthesized Codable for enums with associated values (Swift 5.9),
  `JSONEncoder` with `.sortedKeys` for stable bytes. Both ends run the same app version
  in practice; cross-version safety is the hello gate, not schema tolerance.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/ControlMessageTests.swift
import XCTest
@testable import SquashLineCalling

final class ControlMessageTests: XCTestCase {
    private let hello = Hello(protoVersion: peerProtoVersion, appVersion: "1.0",
                              deviceModel: "iPhone15,2", nonce: 0xDEAD_BEEF,
                              frameW: 1080, frameH: 1920)

    func testRoundTripEveryCase() throws {
        let cases: [ControlMessage] = [
            .hello(hello),
            .role(.secondary),
            .calibration(profileID: "ucsc-left-fin", calibrationJSON: "{\"version\":2}"),
            .syncPing(SyncPing(pingID: 3, t1: 100.25)),
            .syncPong(SyncPong(pingID: 3, t1: 100.25, t2: 100.30, t3: 100.31)),
            .clapAnchor(hostTime: 4242.125),
            .record(action: "start", ptsNs: 9_000_000_000),
            .event(rallyID: 1, json: "{}"),
            .heartbeat(seq: 77),
            .sessionManifest(sessionID: "s-1", videoID: "v-abc"),
        ]
        for original in cases {
            let decoded = ControlMessage.decode(try ControlMessage.encode(original))
            XCTAssertEqual(decoded, original)
        }
    }

    func testMalformedReturnsNil() {
        XCTAssertNil(ControlMessage.decode(Data("not json".utf8)))
        XCTAssertNil(ControlMessage.decode(Data("{\"unknownCase\":{}}".utf8)))
    }

    func testPairingCodeIsSymmetricFourDigits() {
        var other = hello; other.nonce = 0x0000_1234
        let code = pairingCode(hello, other)
        XCTAssertEqual(code, pairingCode(other, hello))
        XCTAssertEqual(code.count, 4)
        XCTAssertTrue(code.allSatisfy(\.isNumber))
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/ControlMessageTests`
Expected: build FAILS with "cannot find 'Hello' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/ControlMessage.swift
import Foundation

let peerProtoVersion = 1

enum PeerRole: String, Codable { case primary, secondary }

struct Hello: Codable, Equatable {
    var protoVersion: Int
    var appVersion: String
    var deviceModel: String
    var nonce: UInt32
    var frameW: Int
    var frameH: Int
}

struct SyncPing: Codable, Equatable { var pingID: UInt32; var t1: Double }
struct SyncPong: Codable, Equatable { var pingID: UInt32; var t1: Double; var t2: Double; var t3: Double }

enum ControlMessage: Codable, Equatable {
    case hello(Hello)
    case role(PeerRole)
    case calibration(profileID: String, calibrationJSON: String)
    case syncPing(SyncPing)
    case syncPong(SyncPong)
    case clapAnchor(hostTime: Double)
    case record(action: String, ptsNs: UInt64)
    case event(rallyID: UInt32, json: String)
    case heartbeat(seq: UInt32)
    case sessionManifest(sessionID: String, videoID: String)

    static func encode(_ message: ControlMessage) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(message)
    }

    static func decode(_ data: Data) -> ControlMessage? {
        try? JSONDecoder().decode(ControlMessage.self, from: data)
    }
}

/// Same 4 digits on both screens; users visually confirm. XOR makes it
/// order-independent.
func pairingCode(_ a: Hello, _ b: Hello) -> String {
    String(format: "%04d", (a.nonce ^ b.nonce) % 10000)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `ControlMessageTests` 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/ControlMessage.swift ios/Tests/ControlMessageTests.swift
git commit -m "feat(peer): control message protocol with version gate and pairing code"
```

---

### Task 4: PeerTransport protocol + LoopbackTransport test double

**Files:**
- Create: `ios/Sources/Peer/PeerTransport.swift`
- Create: `ios/Sources/Peer/LoopbackTransport.swift` (app target: PeerBenchView reuses it in DEBUG; it is ~40 lines and has no UIKit dependency)
- Test: `ios/Tests/LoopbackTransportTests.swift`

**Interfaces:**
- Consumes: `FrameCodec`/`FrameReassembler` (Task 1) — inside concrete transports, not the protocol.
- Produces:
  ```swift
  enum TransportState: Equatable { case idle, searching, connected, disconnected(String) }
  protocol PeerTransport: AnyObject {
      var name: String { get }                            // "loopback" | "ble" | "wifi-p2p"
      var onControl: ((Data) -> Void)? { get set }        // whole frames, in order
      var onDatagram: ((Data) -> Void)? { get set }       // whole datagrams, lossy
      var onStateChange: ((TransportState) -> Void)? { get set }
      func startInitiator()                               // browse/scan + connect
      func startResponder()                               // advertise/listen
      func sendControl(_ frame: Data)                     // reliable, ordered
      func sendDatagram(_ datagram: Data)                 // best-effort
      func stop()
  }
  /// In-memory pair with injectable per-direction delay and datagram drop,
  /// for state-machine and sync tests. `LoopbackTransport.pair()` returns
  /// two connected ends; delivery is synchronous unless `deliver` is
  /// overridden (tests inject delays by scheduling on their own queue).
  final class LoopbackTransport: PeerTransport {
      static func pair() -> (LoopbackTransport, LoopbackTransport)
      var controlDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?   // default: immediate
      var datagramDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?
  }
  ```
  Delivery hooks receive `(payload, deliverToPeer)`; tests add latency by delaying the
  `deliverToPeer` call. This is how sync tests simulate asymmetric RTT and AWDL stalls.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/LoopbackTransportTests.swift
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
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/LoopbackTransportTests`
Expected: build FAILS with "cannot find 'LoopbackTransport' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/PeerTransport.swift
import Foundation

enum TransportState: Equatable { case idle, searching, connected, disconnected(String) }

protocol PeerTransport: AnyObject {
    var name: String { get }
    var onControl: ((Data) -> Void)? { get set }
    var onDatagram: ((Data) -> Void)? { get set }
    var onStateChange: ((TransportState) -> Void)? { get set }
    func startInitiator()
    func startResponder()
    func sendControl(_ frame: Data)
    func sendDatagram(_ datagram: Data)
    func stop()
}
```

```swift
// ios/Sources/Peer/LoopbackTransport.swift
import Foundation

/// In-memory transport pair for tests and DEBUG benches. Synchronous
/// delivery by default; hooks let tests delay, reorder, or drop.
final class LoopbackTransport: PeerTransport {
    let name = "loopback"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?
    var controlDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?
    var datagramDeliveryHook: ((Data, @escaping (Data) -> Void) -> Void)?

    private weak var peer: LoopbackTransport?

    static func pair() -> (LoopbackTransport, LoopbackTransport) {
        let a = LoopbackTransport(), b = LoopbackTransport()
        a.peer = b; b.peer = a
        // Hold strong references through captured closures so `weak var peer`
        // stays alive as long as either end is.
        a.retainedPeer = b; b.retainedPeer = a
        return (a, b)
    }
    private var retainedPeer: LoopbackTransport?

    func startInitiator() { onStateChange?(.connected) }
    func startResponder() { onStateChange?(.connected) }

    func sendControl(_ frame: Data) {
        let deliver: (Data) -> Void = { [weak peer] in peer?.onControl?($0) }
        if let hook = controlDeliveryHook { hook(frame, deliver) } else { deliver(frame) }
    }

    func sendDatagram(_ datagram: Data) {
        let deliver: (Data) -> Void = { [weak peer] in peer?.onDatagram?($0) }
        if let hook = datagramDeliveryHook { hook(datagram, deliver) } else { deliver(datagram) }
    }

    func stop() { onStateChange?(.disconnected("stopped")) }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `LoopbackTransportTests` 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/PeerTransport.swift ios/Sources/Peer/LoopbackTransport.swift ios/Tests/LoopbackTransportTests.swift
git commit -m "feat(peer): transport protocol and loopback test double"
```

---

### Task 5: ClockSync estimator (pure math)

NTP-style offset from timestamp quadruples, min-RTT filtered, with clap-anchor bias and
a conservative uncertainty bound. Deterministic: no clock reads inside.

**Files:**
- Create: `ios/Sources/Peer/ClockSync.swift`
- Test: `ios/Tests/ClockSyncTests.swift`

**Interfaces:**
- Consumes: `SyncPong` (Task 3) field names only (the estimator takes raw doubles).
- Produces:
  ```swift
  struct ClockSyncEstimate: Equatable {
      var offset: Double            // seconds: remoteClock − localClock
      var uncertainty: Double       // seconds, conservative bound
      var sampleCount: Int
  }
  final class ClockSync {
      static func hostNow() -> TimeInterval    // CMClockGetTime(CMClockGetHostTimeClock()) seconds
      func addSample(t1: Double, t2: Double, t3: Double, t4: Double)
      func applyAnchor(localEventTime: Double, remoteEventTime: Double)
      var estimate: ClockSyncEstimate?         // nil until ≥ 5 samples
      func remoteToLocal(_ tRemote: Double) -> Double?
  }
  ```
  Algorithm: per sample, `rtt = (t4−t1)−(t3−t2)`, `theta = ((t2−t1)+(t3−t4))/2`. Keep a
  rolling window (last 200 samples); the estimate is the median `theta` of the 5 samples
  with smallest `rtt`; `uncertainty = max(minRTT/2, spread of those thetas)`. An anchor
  replaces the bias: after `applyAnchor`, `offset = remoteEventTime − localEventTime`
  and `uncertainty = 0.5 ms` floor (acoustic resolution + geometry), while network
  samples continue to track drift relative to the anchored offset via the median-theta
  delta. Drift handling in Plan A is re-estimation from the rolling window (re-sync
  cadence is PeerSession's job, Task 6); an explicit slope model is deferred until the
  bench shows it is needed (YAGNI — 20 ppm over a 30 s re-sync window is 0.6 ms, inside
  budget).

- [ ] **Step 1: Write the failing tests**

```swift
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/ClockSyncTests`
Expected: build FAILS with "cannot find 'ClockSync' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/ClockSync.swift
import CoreMedia
import Foundation

struct ClockSyncEstimate: Equatable {
    var offset: Double
    var uncertainty: Double
    var sampleCount: Int
}

/// Pure estimator: feed NTP-style quadruples, read an offset. All times are
/// seconds on each device's capture host clock. "remote − local" sign.
final class ClockSync {
    static let minimumSamples = 5
    static let windowSize = 200
    static let anchorUncertainty = 0.0005   // 0.5 ms: acoustic resolution + T geometry

    private struct Sample { var rtt: Double; var theta: Double }
    private var samples: [Sample] = []
    private var anchor: Double?              // authoritative offset from clap

    static func hostNow() -> TimeInterval {
        CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
    }

    func addSample(t1: Double, t2: Double, t3: Double, t4: Double) {
        let rtt = (t4 - t1) - (t3 - t2)
        guard rtt >= 0 else { return }       // clock weirdness: discard
        samples.append(Sample(rtt: rtt, theta: ((t2 - t1) + (t3 - t4)) / 2))
        if samples.count > ClockSync.windowSize { samples.removeFirst() }
    }

    func applyAnchor(localEventTime: Double, remoteEventTime: Double) {
        anchor = remoteEventTime - localEventTime
    }

    var estimate: ClockSyncEstimate? {
        if let anchor {
            return ClockSyncEstimate(offset: anchor,
                                     uncertainty: ClockSync.anchorUncertainty,
                                     sampleCount: samples.count)
        }
        guard samples.count >= ClockSync.minimumSamples else { return nil }
        let best = samples.sorted { $0.rtt < $1.rtt }.prefix(5)
        let thetas = best.map(\.theta).sorted()
        let median = thetas[thetas.count / 2]
        let spread = (thetas.last ?? 0) - (thetas.first ?? 0)
        let minRTT = best.first.map(\.rtt) ?? 0
        return ClockSyncEstimate(offset: median,
                                 uncertainty: max(minRTT / 2, spread),
                                 sampleCount: samples.count)
    }

    func remoteToLocal(_ tRemote: Double) -> Double? {
        guard let estimate else { return nil }
        return tRemote - estimate.offset
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `ClockSyncTests` 5/5 PASS.
Note on `testAsymmetricPathBiasIsBoundedByHalfMinRTT`: the estimator cannot detect pure
asymmetry (theory), so the assertion is that `uncertainty` honestly covers it — minRTT/2
= 10 ms ≥ the 5 ms bias. That honesty is what the clap anchor then fixes.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/ClockSync.swift ios/Tests/ClockSyncTests.swift
git commit -m "feat(peer): min-RTT clock sync estimator with acoustic anchor"
```

---

### Task 6: PeerSession state machine + sync rounds

The orchestrator: handshake, roles, pairing code, heartbeats, sync cadence, detection
fan-in/out. UI observes `phase` (mirrors the `RunSubmission.Phase` pattern).

**Files:**
- Create: `ios/Sources/Peer/PeerSession.swift`
- Test: `ios/Tests/PeerSessionTests.swift`

**Interfaces:**
- Consumes: `PeerTransport`/`LoopbackTransport` (Task 4), `ControlMessage`/`Hello`/`pairingCode` (Task 3), `ClockSync` (Task 5), `DetectionBatch` (Task 2), `FrameCodec` — control frames are already whole per `PeerTransport`, so PeerSession does NOT re-frame (framing lives inside concrete transports).
- Produces:
  ```swift
  final class PeerSession: ObservableObject {
      enum Phase: Equatable {
          case idle, searching, confirming(code: String), syncing, ready, live,
               degraded(String), ended, failed(String)
      }
      @Published private(set) var phase: Phase
      private(set) var role: PeerRole?
      let clockSync: ClockSync
      var peerHello: Hello?                                  // frameW/frameH for decoding
      var onRemoteDetections: (([DetectionTuple]) -> Void)?  // primary side, delivery queue = transport's
      init(transport: PeerTransport, isInitiator: Bool,
           now: @escaping () -> TimeInterval = ClockSync.hostNow,
           heartbeatTimeout: TimeInterval = 3.0)
      func start()                       // → searching; transport start{Initiator|Responder}
      func confirmPairing()              // user tapped "codes match" → syncing
      func goLive()                      // ready → live
      func end()
      func sendDetections(_ tuples: [DetectionTuple])        // secondary side
      func sendClapAnchor(localOnset: TimeInterval)          // both sides call with own onset
      // Test/scheduler seam: PeerSession never spins its own timers; the
      // owner pumps it. RecordModel (Task 9) pumps from its existing queue;
      // tests pump manually.
      func tick(now: TimeInterval)       // drives sync pings (burst of 30 while syncing,
                                         // 1 every 10 s in ready/live) and heartbeat/timeout checks
  }
  ```
  Protocol flow: both send `hello` on transport `.connected`. Initiator = `primary` by
  definition; on valid hello (version match) primary sends `.role(.secondary)`; both
  enter `.confirming(code:)` with `pairingCode(mine, theirs)`. `confirmPairing()` (from
  BOTH devices — each side gates on its own user tap; the other side's readiness is
  implied by its sync pongs) → `.syncing`; primary runs a 30-ping burst via `tick`;
  when `clockSync.estimate.uncertainty ≤ 0.005` → send `.heartbeat(seq: 0)` and enter
  `.ready`. Secondary enters `.ready` when it sees the first heartbeat. Version
  mismatch → `.failed("protocol version mismatch — update both apps")`. Missing
  heartbeats past `heartbeatTimeout` in ready/live → `.degraded("link lost")`; a
  resumed heartbeat returns to the prior phase. Clap: each side detects its own onset
  and sends `.clapAnchor`; PRIMARY computes `applyAnchor(localEventTime: myOnset,
  remoteEventTime: theirOnset)`; secondary mirrors with swapped arguments so both ends
  hold the same mapping (signs opposite, consistent with each side's "remote").

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/PeerSessionTests.swift
import XCTest
@testable import SquashLineCalling

final class PeerSessionTests: XCTestCase {
    private var pair: (LoopbackTransport, LoopbackTransport)!
    private var primary: PeerSession!
    private var secondary: PeerSession!

    override func setUp() {
        super.setUp()
        pair = LoopbackTransport.pair()
        primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
    }

    private func startBoth() { secondary.start(); primary.start() }

    func testHandshakeAssignsRolesAndMatchingCodes() {
        startBoth()
        XCTAssertEqual(primary.role, .primary)
        XCTAssertEqual(secondary.role, .secondary)
        guard case .confirming(let codeA) = primary.phase,
              case .confirming(let codeB) = secondary.phase else {
            return XCTFail("both sides must reach confirming, got \(primary.phase) / \(secondary.phase)")
        }
        XCTAssertEqual(codeA, codeB)
    }

    func testSyncBurstReachesReadyOnBothSides() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        XCTAssertEqual(secondary.phase, .ready)
        XCTAssertNotNil(primary.clockSync.estimate)
    }

    func testVersionMismatchFails() {
        // Corrupt the secondary's hello version by intercepting control frames.
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var hello)? = ControlMessage.decode(frame) else { return deliver(frame) }
            hello.protoVersion = 99
            deliver(try! ControlMessage.encode(.hello(hello)))
        }
        startBoth()
        guard case .failed = primary.phase else { return XCTFail("expected failed, got \(primary.phase)") }
    }

    func testHeartbeatLossDegradesAndRecoveryRestores() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
        // Silence the link: hold every control frame from the secondary.
        pair.1.controlDeliveryHook = { _, _ in }
        for _ in 0..<50 { t += 0.5; primary.tick(now: t) }   // > heartbeatTimeout with no pongs/beats
        guard case .degraded = primary.phase else { return XCTFail("expected degraded, got \(primary.phase)") }
        pair.1.controlDeliveryHook = nil
        for _ in 0..<10 { t += 0.5; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)
    }

    func testDetectionsFlowSecondaryToPrimary() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        primary.goLive(); secondary.goLive()
        var received: [[DetectionTuple]] = []
        primary.onRemoteDetections = { received.append($0) }
        let tuple = DetectionTuple(seq: 1, ptsNs: 42, x: 1, y: 2,
                                   conf: Float16(0.9), bboxH: Float16(10))
        secondary.sendDetections([tuple])
        XCTAssertEqual(received, [[tuple]])
    }

    func testClapAnchorSetsBothMappingsConsistently() {
        startBoth()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        primary.sendClapAnchor(localOnset: 10.0)
        secondary.sendClapAnchor(localOnset: 10.4)   // secondary clock is +0.4 ahead
        XCTAssertEqual(primary.clockSync.remoteToLocal(20.4) ?? -1, 20.0, accuracy: 1e-9)
        XCTAssertEqual(secondary.clockSync.remoteToLocal(20.0) ?? -1, 20.4, accuracy: 1e-9)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/PeerSessionTests`
Expected: build FAILS with "cannot find 'PeerSession' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/PeerSession.swift
import Foundation

/// Owns the pairing lifecycle over an abstract transport. Threading: all
/// callbacks arrive on the transport's delivery context and are processed
/// inline (transports guarantee serial delivery); @Published writes hop to
/// main. `tick` must be called from one consistent queue.
final class PeerSession: ObservableObject {
    enum Phase: Equatable {
        case idle, searching, confirming(code: String), syncing, ready, live,
             degraded(String), ended, failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    private(set) var role: PeerRole?
    let clockSync = ClockSync()
    var peerHello: Hello?
    var onRemoteDetections: (([DetectionTuple]) -> Void)?

    private let transport: PeerTransport
    private let isInitiator: Bool
    private let now: () -> TimeInterval
    private let heartbeatTimeout: TimeInterval

    private var myHello: Hello
    private var pendingPings: [UInt32: Double] = [:]   // pingID → t1
    private var nextPingID: UInt32 = 0
    private var nextHeartbeatSeq: UInt32 = 0
    private var burstPingsRemaining = 0
    private var lastPingAt: TimeInterval = -.infinity
    private var lastPeerActivityAt: TimeInterval = 0
    private var phaseBeforeDegraded: Phase?
    private var myClapOnset: Double?
    private var peerClapOnset: Double?

    static let syncBurstCount = 30
    static let burstInterval = 0.05        // 20 Hz during the burst
    static let steadyInterval = 10.0       // re-sync cadence in ready/live
    static let readyUncertainty = 0.005    // 5 ms gate to leave syncing

    init(transport: PeerTransport, isInitiator: Bool,
         now: @escaping () -> TimeInterval = ClockSync.hostNow,
         heartbeatTimeout: TimeInterval = 3.0) {
        self.transport = transport
        self.isInitiator = isInitiator
        self.now = now
        self.heartbeatTimeout = heartbeatTimeout
        self.myHello = Hello(protoVersion: peerProtoVersion,
                             appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                             deviceModel: ProcessInfo.processInfo.hostName,
                             nonce: UInt32.random(in: .min ... .max),
                             frameW: 1080, frameH: 1920)
        transport.onControl = { [weak self] in self?.handleControl($0) }
        transport.onDatagram = { [weak self] in self?.handleDatagram($0) }
        transport.onStateChange = { [weak self] in self?.handleTransportState($0) }
    }

    // MARK: lifecycle

    func start() {
        // A synchronously-delivered peer hello (loopback tests; fast radios)
        // can move us to .confirming before our own start() runs — never
        // regress an established phase back to .searching.
        if phase == .idle { setPhase(.searching) }
        isInitiator ? transport.startInitiator() : transport.startResponder()
    }

    func confirmPairing() {
        guard case .confirming = phase else { return }
        setPhase(.syncing)
        burstPingsRemaining = PeerSession.syncBurstCount
    }

    func goLive() { if phase == .ready { setPhase(.live) } }

    func end() {
        setPhase(.ended)
        transport.stop()
    }

    // MARK: outbound

    func sendDetections(_ tuples: [DetectionTuple]) {
        guard phase == .live || phase == .ready else { return }
        transport.sendDatagram(DetectionBatch.encode(tuples))
    }

    func sendClapAnchor(localOnset: TimeInterval) {
        myClapOnset = localOnset
        sendControl(.clapAnchor(hostTime: localOnset))
        tryApplyAnchor()
    }

    /// Pump from the owner's queue. Drives pings + timeout detection.
    func tick(now t: TimeInterval) {
        switch phase {
        case .syncing:
            if burstPingsRemaining > 0, t - lastPingAt >= PeerSession.burstInterval {
                sendPing(at: t); burstPingsRemaining -= 1
            }
            if let estimate = clockSync.estimate, estimate.uncertainty <= PeerSession.readyUncertainty {
                sendControl(.heartbeat(seq: nextHeartbeatSeq)); nextHeartbeatSeq += 1
                lastPeerActivityAt = t
                setPhase(.ready)
            }
        case .ready, .live:
            if t - lastPingAt >= PeerSession.steadyInterval { sendPing(at: t) }
            if t - lastPeerActivityAt > heartbeatTimeout {
                phaseBeforeDegraded = phase
                setPhase(.degraded("link lost"))
            } else if t - lastPeerActivityAt > heartbeatTimeout / 2 {
                sendControl(.heartbeat(seq: nextHeartbeatSeq)); nextHeartbeatSeq += 1
            }
        case .degraded:
            sendControl(.heartbeat(seq: nextHeartbeatSeq)); nextHeartbeatSeq += 1
            if t - lastPeerActivityAt <= heartbeatTimeout {
                setPhase(phaseBeforeDegraded ?? .ready)
            }
        default: break
        }
    }

    // MARK: inbound

    private func handleTransportState(_ state: TransportState) {
        switch state {
        case .connected:
            sendControl(.hello(myHello))
        case .disconnected(let reason):
            if phase != .ended, phase != .idle {
                phaseBeforeDegraded = phase
                setPhase(.degraded(reason))
            }
        case .idle, .searching: break
        }
    }

    private func handleControl(_ frame: Data) {
        guard let message = ControlMessage.decode(frame) else { return }
        lastPeerActivityAt = now()
        switch message {
        case .hello(let theirs):
            guard theirs.protoVersion == peerProtoVersion else {
                setPhase(.failed("protocol version mismatch — update both apps"))
                transport.stop()
                return
            }
            peerHello = theirs
            role = isInitiator ? .primary : .secondary
            if isInitiator { sendControl(.role(.secondary)) }
            setPhase(.confirming(code: pairingCode(myHello, theirs)))
        case .role(let assigned):
            role = assigned
        case .syncPing(let ping):
            let t2 = now()
            sendControl(.syncPong(SyncPong(pingID: ping.pingID, t1: ping.t1, t2: t2, t3: now())))
            // Symmetric sync: the responder also learns the offset from its own pings.
            if case .syncing = phase, burstPingsRemaining == 0 {
                burstPingsRemaining = PeerSession.syncBurstCount
            }
        case .syncPong(let pong):
            guard pendingPings.removeValue(forKey: pong.pingID) != nil else { return }
            clockSync.addSample(t1: pong.t1, t2: pong.t2, t3: pong.t3, t4: now())
        case .clapAnchor(let hostTime):
            peerClapOnset = hostTime
            tryApplyAnchor()
        case .heartbeat:
            if phase == .syncing, clockSync.estimate != nil { setPhase(.ready) }
        case .calibration, .record, .event, .sessionManifest:
            break   // consumed by Phase 3/4/5 code; parsing is already validated
        }
    }

    private func handleDatagram(_ datagram: Data) {
        guard let tuples = DetectionBatch.decode(datagram), !tuples.isEmpty else { return }
        onRemoteDetections?(tuples)
    }

    // MARK: helpers

    private func sendPing(at t: TimeInterval) {
        lastPingAt = t
        let ping = SyncPing(pingID: nextPingID, t1: now())
        pendingPings[ping.pingID] = ping.t1
        nextPingID += 1
        sendControl(.syncPing(ping))
    }

    private func tryApplyAnchor() {
        guard let mine = myClapOnset, let theirs = peerClapOnset else { return }
        clockSync.applyAnchor(localEventTime: mine, remoteEventTime: theirs)
    }

    private func sendControl(_ message: ControlMessage) {
        guard let data = try? ControlMessage.encode(message) else { return }
        transport.sendControl(data)
    }

    private func setPhase(_ newPhase: Phase) {
        if Thread.isMainThread { phase = newPhase }
        else { DispatchQueue.main.sync { self.phase = newPhase } }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `PeerSessionTests` 6/6 PASS. If
`testSyncBurstReachesReadyOnBothSides` hangs on the secondary side, check that the
responder's burst is started by the first incoming `syncPing` (see `case .syncPing`).

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/PeerSession.swift ios/Tests/PeerSessionTests.swift
git commit -m "feat(peer): pairing state machine with sync rounds and degradation"
```

---

### Task 7: ClapDetector + CameraController audio hook

Pure onset detection on PCM samples; a thin CMSampleBuffer adapter; a new
`onAudioSample` hook so peers can hear the clap. (Audio buffers already flow through
`captureOutput` — today they only feed the writer.)

**Files:**
- Create: `ios/Sources/Peer/ClapDetector.swift`
- Modify: `ios/Sources/Record/CameraController.swift:22-23` (add hook), `:170-194` (invoke in audio branch)
- Test: `ios/Tests/ClapDetectorTests.swift`

**Interfaces:**
- Consumes: `CameraController` audio delegate path (existing).
- Produces:
  ```swift
  final class ClapDetector {
      /// Pure core. Returns onset time (seconds, caller's clock) if a clap
      /// onset occurs in this chunk. `startTime` is the host-clock time of
      /// samples[0]. Stateful across calls (rolling noise floor, refractory).
      func process(samples: [Float], startTime: Double, sampleRate: Double) -> Double?
      /// Adapter: extracts mono Float samples + PTS from a CMSampleBuffer
      /// (Int16 and Float32 LPCM supported) and calls the core.
      func process(sampleBuffer: CMSampleBuffer) -> Double?
      var onClap: ((Double) -> Void)?   // convenience: fired by the adapter
  }
  ```
  Algorithm: 5 ms RMS windows; rolling noise floor = EMA of window RMS (alpha 0.05,
  only updated when not triggering); trigger when window RMS > max(8 × noise floor,
  0.05 absolute); onset = time of first sample in the triggering window whose |sample|
  exceeds 4 × noise floor; 500 ms refractory so one clap fires once.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/ClapDetectorTests.swift
import XCTest
@testable import SquashLineCalling

final class ClapDetectorTests: XCTestCase {
    private let rate = 44_100.0

    private func quiet(_ seconds: Double) -> [Float] {
        (0..<Int(seconds * rate)).map { _ in Float.random(in: -0.004...0.004) }
    }

    /// Sharp transient at a known sample index.
    private func withClap(at second: Double, total: Double) -> [Float] {
        var samples = quiet(total)
        let start = Int(second * rate)
        for i in 0..<Int(0.02 * rate) {   // 20 ms burst, decaying
            samples[start + i] = 0.8 * Float(pow(0.9995, Double(i))) * (i.isMultiple(of: 2) ? 1 : -1)
        }
        return samples
    }

    func testQuietAudioNeverTriggers() {
        let detector = ClapDetector()
        XCTAssertNil(detector.process(samples: quiet(2.0), startTime: 100.0, sampleRate: rate))
    }

    func testClapOnsetWithinTwoMilliseconds() {
        let detector = ClapDetector()
        _ = detector.process(samples: quiet(1.0), startTime: 100.0, sampleRate: rate)   // learn floor
        let onset = detector.process(samples: withClap(at: 0.5, total: 1.0),
                                     startTime: 101.0, sampleRate: rate)
        XCTAssertNotNil(onset)
        XCTAssertEqual(onset ?? -1, 101.5, accuracy: 0.002)
    }

    func testRefractorySuppressesDoubleFire() {
        let detector = ClapDetector()
        _ = detector.process(samples: quiet(1.0), startTime: 100.0, sampleRate: rate)
        var samples = withClap(at: 0.1, total: 1.0)
        let echo = withClap(at: 0.3, total: 1.0)   // second transient inside 500 ms
        for i in 0..<samples.count { samples[i] = max(samples[i], echo[i]) }
        var onsets: [Double] = []
        // Feed in 100 ms chunks like the capture path would.
        let chunk = Int(0.1 * rate)
        for (index, start) in stride(from: 0, to: samples.count, by: chunk).enumerated() {
            let slice = Array(samples[start ..< min(start + chunk, samples.count)])
            if let t = detector.process(samples: slice,
                                        startTime: 101.0 + Double(index) * 0.1,
                                        sampleRate: rate) { onsets.append(t) }
        }
        XCTAssertEqual(onsets.count, 1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/ClapDetectorTests`
Expected: build FAILS with "cannot find 'ClapDetector' in scope".

- [ ] **Step 3: Implement the pure core + adapter**

```swift
// ios/Sources/Peer/ClapDetector.swift
import Accelerate
import CoreMedia
import Foundation

/// Onset detector for the pairing clap. Pure core (`process(samples:...)`)
/// is unit-tested; the CMSampleBuffer adapter is exercised on hardware.
final class ClapDetector {
    var onClap: ((Double) -> Void)?

    private var noiseFloor: Float = 0.01
    private var lastOnset: Double = -.infinity
    private let windowSeconds = 0.005
    private let refractorySeconds = 0.5
    private let floorAlpha: Float = 0.05

    func process(samples: [Float], startTime: Double, sampleRate: Double) -> Double? {
        let window = max(1, Int(windowSeconds * sampleRate))
        var index = 0
        while index < samples.count {
            let upper = min(index + window, samples.count)
            let slice = Array(samples[index..<upper])
            var rms: Float = 0
            vDSP_rmsqv(slice, 1, &rms, vDSP_Length(slice.count))
            let windowStart = startTime + Double(index) / sampleRate
            let triggers = rms > max(8 * noiseFloor, 0.05)
            if triggers, windowStart - lastOnset > refractorySeconds {
                let gate = 4 * noiseFloor
                let offsetInWindow = slice.firstIndex { abs($0) > gate } ?? 0
                let onset = windowStart + Double(offsetInWindow) / sampleRate
                lastOnset = onset
                return onset
            }
            if !triggers {
                noiseFloor += floorAlpha * (rms - noiseFloor)
            }
            index = upper
        }
        return nil
    }

    func process(sampleBuffer: CMSampleBuffer) -> Double? {
        guard let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        guard CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
                                          totalLengthOut: &length, dataPointerOut: &pointer) == noErr,
              let raw = pointer else { return nil }

        let channels = max(1, Int(asbd.mChannelsPerFrame))
        var mono: [Float]
        if asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0 {
            let floats = UnsafeRawPointer(raw).assumingMemoryBound(to: Float.self)
            let frameCount = length / MemoryLayout<Float>.size / channels
            mono = (0..<frameCount).map { floats[$0 * channels] }
        } else {
            let ints = UnsafeRawPointer(raw).assumingMemoryBound(to: Int16.self)
            let frameCount = length / MemoryLayout<Int16>.size / channels
            mono = (0..<frameCount).map { Float(ints[$0 * channels]) / Float(Int16.max) }
        }
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        let onset = process(samples: mono, startTime: pts, sampleRate: asbd.mSampleRate)
        if let onset { onClap?(onset) }
        return onset
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `ClapDetectorTests` 3/3 PASS.

- [ ] **Step 5: Add the audio hook to CameraController**

In `ios/Sources/Record/CameraController.swift`, below the existing `onVideoSample`
declaration (line 23), add:

```swift
    /// Every audio sample buffer, on the output queue. The pairing clap
    /// detector subscribes here; nil costs nothing.
    var onAudioSample: ((CMSampleBuffer) -> Void)?
```

In `captureOutput(_:didOutput:from:)`, the audio path currently starts at the
`guard let writer` line. Audio must reach the hook even when not recording, so insert
BEFORE `guard let writer else { return }`:

```swift
        if output === audioOutput {
            onAudioSample?(sampleBuffer)
        }
```

(The subsequent writer block is unchanged; it independently re-checks `output`.)

- [ ] **Step 6: Verify the full suite still passes and commit**

Run: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`
Expected: all existing + new tests PASS (no regression in `SmokeTests`).

```bash
git add ios/Sources/Peer/ClapDetector.swift ios/Sources/Record/CameraController.swift ios/Tests/ClapDetectorTests.swift
git commit -m "feat(peer): clap onset detector and camera audio hook"
```

---

### Task 8: BLETransport

CoreBluetooth implementation: initiator = central, responder = peripheral. One service,
two characteristics. Control frames chunked to the negotiated write length and
reassembled with `FrameReassembler`; datagrams are single writes (a detection batch of
4 tuples = 99 B, under every MTU ≥ 102 iOS negotiates in practice — oversize datagrams
are dropped, honoring the lossy contract).

**Files:**
- Create: `ios/Sources/Peer/BLETransport.swift`
- Modify: `ios/project.yml:23-33` (add `NSBluetoothAlwaysUsageDescription` to `info.properties`)
- Test: `ios/Tests/BLEChunkerTests.swift` (pure chunking logic only; radio paths are bench-verified via `ios/PEER.md`, Task 11)

**Interfaces:**
- Consumes: `PeerTransport` (Task 4), `FrameCodec`/`FrameReassembler` (Task 1).
- Produces: `final class BLETransport: NSObject, PeerTransport` with
  `init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.ble"))`; plus the
  extracted pure helper used by tests:
  ```swift
  enum BLEChunker {
      /// Splits an encoded control frame into ≤ maxWriteLength chunks.
      static func chunks(_ frame: Data, maxWriteLength: Int) -> [Data]
  }
  ```
  UUIDs (fixed, generated once for this app):
  - Service: `8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A11`
  - Control characteristic (write + indicate): `8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A12`
  - Datagram characteristic (writeWithoutResponse + notify): `8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A13`

- [ ] **Step 1: Write the failing chunker test**

```swift
// ios/Tests/BLEChunkerTests.swift
import XCTest
@testable import SquashLineCalling

final class BLEChunkerTests: XCTestCase {
    func testChunksReassembleToOriginalFrame() throws {
        let payload = Data((0..<1000).map { UInt8($0 % 256) })
        let frame = FrameCodec.encode(payload)
        let chunks = BLEChunker.chunks(frame, maxWriteLength: 180)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 180 })
        XCTAssertEqual(chunks.count, Int(ceil(Double(frame.count) / 180.0)))
        let reassembler = FrameReassembler()
        var frames: [Data] = []
        for chunk in chunks { frames.append(contentsOf: try reassembler.ingest(chunk)) }
        XCTAssertEqual(frames, [payload])
    }

    func testSmallFrameIsSingleChunk() {
        XCTAssertEqual(BLEChunker.chunks(Data([1, 2, 3]), maxWriteLength: 180).count, 1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/BLEChunkerTests`
Expected: build FAILS with "cannot find 'BLEChunker' in scope".

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Peer/BLETransport.swift
import CoreBluetooth
import Foundation

enum BLEChunker {
    static func chunks(_ frame: Data, maxWriteLength: Int) -> [Data] {
        stride(from: 0, to: frame.count, by: maxWriteLength).map {
            frame.subdata(in: frame.startIndex + $0 ..< frame.startIndex + min($0 + maxWriteLength, frame.count))
        }
    }
}

/// Initiator = CBCentralManager (scans, connects, writes control with
/// response / datagrams without response). Responder = CBPeripheralManager
/// (advertises, receives writes, sends via indicate/notify). All delegate
/// work on `queue`; callbacks fire on `queue`.
final class BLETransport: NSObject, PeerTransport {
    static let serviceUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A11")
    static let controlUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A12")
    static let datagramUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A13")

    let name = "ble"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?

    private let queue: DispatchQueue
    private let reassembler = FrameReassembler()

    // Central (initiator) state
    private var central: CBCentralManager?
    private var remotePeripheral: CBPeripheral?
    private var remoteControl: CBCharacteristic?
    private var remoteDatagram: CBCharacteristic?

    // Peripheral (responder) state
    private var peripheralManager: CBPeripheralManager?
    private var localControl: CBMutableCharacteristic?
    private var localDatagram: CBMutableCharacteristic?
    private var subscribedCentral: CBCentral?

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.ble")) {
        self.queue = queue
        super.init()
    }

    func startInitiator() {
        onStateChange?(.searching)
        central = CBCentralManager(delegate: self, queue: queue)
    }

    func startResponder() {
        onStateChange?(.searching)
        peripheralManager = CBPeripheralManager(delegate: self, queue: queue)
    }

    func sendControl(_ frame: Data) {
        let encoded = FrameCodec.encode(frame)
        if let peripheral = remotePeripheral, let characteristic = remoteControl {
            let maxLength = peripheral.maximumWriteValueLength(for: .withResponse)
            for chunk in BLEChunker.chunks(encoded, maxWriteLength: maxLength) {
                peripheral.writeValue(chunk, for: characteristic, type: .withResponse)
            }
        } else if let manager = peripheralManager, let characteristic = localControl {
            let maxLength = subscribedCentral?.maximumUpdateValueLength ?? 180
            for chunk in BLEChunker.chunks(encoded, maxWriteLength: maxLength) {
                _ = manager.updateValue(chunk, for: characteristic, onSubscribedCentrals: nil)
            }
        }
    }

    func sendDatagram(_ datagram: Data) {
        if let peripheral = remotePeripheral, let characteristic = remoteDatagram {
            guard datagram.count <= peripheral.maximumWriteValueLength(for: .withoutResponse) else { return }
            peripheral.writeValue(datagram, for: characteristic, type: .withoutResponse)
        } else if let manager = peripheralManager, let characteristic = localDatagram {
            guard datagram.count <= (subscribedCentral?.maximumUpdateValueLength ?? 180) else { return }
            _ = manager.updateValue(datagram, for: characteristic, onSubscribedCentrals: nil)
        }
    }

    func stop() {
        if let remotePeripheral { central?.cancelPeripheralConnection(remotePeripheral) }
        central?.stopScan()
        peripheralManager?.stopAdvertising()
        central = nil; peripheralManager = nil
        onStateChange?(.disconnected("stopped"))
    }

    private func ingestControlChunk(_ chunk: Data) {
        guard let frames = try? reassembler.ingest(chunk) else {
            onStateChange?(.disconnected("control stream corrupted"))
            return
        }
        for frame in frames { onControl?(frame) }
    }
}

extension BLETransport: CBCentralManagerDelegate, CBPeripheralDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        guard central.state == .poweredOn else { return }
        central.scanForPeripherals(withServices: [Self.serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        remotePeripheral = peripheral
        peripheral.delegate = self
        central.stopScan()
        central.connect(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.discoverServices([Self.serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        onStateChange?(.disconnected(error?.localizedDescription ?? "peer disconnected"))
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let service = peripheral.services?.first(where: { $0.uuid == Self.serviceUUID }) else { return }
        peripheral.discoverCharacteristics([Self.controlUUID, Self.datagramUUID], for: service)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        for characteristic in service.characteristics ?? [] {
            if characteristic.uuid == Self.controlUUID {
                remoteControl = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            } else if characteristic.uuid == Self.datagramUUID {
                remoteDatagram = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            }
        }
        if remoteControl != nil, remoteDatagram != nil { onStateChange?(.connected) }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let value = characteristic.value else { return }
        if characteristic.uuid == Self.controlUUID { ingestControlChunk(value) }
        else if characteristic.uuid == Self.datagramUUID { onDatagram?(value) }
    }
}

extension BLETransport: CBPeripheralManagerDelegate {
    func peripheralManagerDidUpdateState(_ manager: CBPeripheralManager) {
        guard manager.state == .poweredOn else { return }
        let control = CBMutableCharacteristic(type: Self.controlUUID,
                                              properties: [.write, .indicate],
                                              value: nil, permissions: [.writeable])
        let datagram = CBMutableCharacteristic(type: Self.datagramUUID,
                                               properties: [.writeWithoutResponse, .notify],
                                               value: nil, permissions: [.writeable])
        localControl = control; localDatagram = datagram
        let service = CBMutableService(type: Self.serviceUUID, primary: true)
        service.characteristics = [control, datagram]
        manager.add(service)
        manager.startAdvertising([CBAdvertisementDataServiceUUIDsKey: [Self.serviceUUID]])
    }

    func peripheralManager(_ manager: CBPeripheralManager, central: CBCentral,
                           didSubscribeTo characteristic: CBCharacteristic) {
        subscribedCentral = central
        if characteristic.uuid == Self.datagramUUID { onStateChange?(.connected) }
    }

    func peripheralManager(_ manager: CBPeripheralManager,
                           didReceiveWrite requests: [CBATTRequest]) {
        for request in requests {
            guard let value = request.value else { continue }
            if request.characteristic.uuid == Self.controlUUID {
                ingestControlChunk(value)
                manager.respond(to: request, withResult: .success)
            } else if request.characteristic.uuid == Self.datagramUUID {
                onDatagram?(value)
            }
        }
    }
}
```

- [ ] **Step 4: Add the Bluetooth usage description**

In `ios/project.yml`, inside `targets.SquashLineCalling.info.properties` (alongside
`NSCameraUsageDescription`), add:

```yaml
        NSBluetoothAlwaysUsageDescription: Pairs with the second court camera to sync line calls.
```

- [ ] **Step 5: Run tests + parse gate and commit**

Run: `... -only-testing:SquashLineCallingTests/BLEChunkerTests` — expected 2/2 PASS —
then the full suite. Radio paths compile but are exercised on hardware (Task 11 bench).

```bash
git add ios/Sources/Peer/BLETransport.swift ios/Tests/BLEChunkerTests.swift ios/project.yml
git commit -m "feat(peer): CoreBluetooth transport (central/peripheral, chunked control)"
```

---

### Task 9: WiFiP2PTransport

Network.framework implementation: Bonjour-advertised TCP control listener + UDP
datagram listener on the responder; browser + two outbound connections on the
initiator. `includePeerToPeer = true` enables AWDL with no infrastructure Wi-Fi.

**Files:**
- Create: `ios/Sources/Peer/WiFiP2PTransport.swift`
- Modify: `ios/project.yml:23-33` (add `NSLocalNetworkUsageDescription`, `NSBonjourServices`)
- Test: compile + hardware bench only (all pure logic — framing, batching — is already covered by Tasks 1–2 tests; this class is wiring)

**Interfaces:**
- Consumes: `PeerTransport` (Task 4), `FrameCodec`/`FrameReassembler` (Task 1).
- Produces: `final class WiFiP2PTransport: PeerTransport` with
  `init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.wifi"))`.
  Bonjour service types: control `_crosscourt._tcp`, datagram advertised via a
  `udpPort` line sent as the FIRST control frame (before any `ControlMessage` — the
  frame payload is `UDP:<port>` ASCII; `PeerSession` never sees it because the
  transport consumes it). Sequence: responder listens TCP (Bonjour) + UDP (ephemeral);
  initiator browses, connects TCP, receives `UDP:<port>`, opens UDP to the same host,
  sends `UDP:<port>` of its own UDP listener; `.connected` fires on each side when its
  UDP path is set up. This keeps `PeerTransport` symmetric for `PeerSession`.

- [ ] **Step 1: Implement**

```swift
// ios/Sources/Peer/WiFiP2PTransport.swift
import Foundation
import Network

/// Peer-to-peer Wi-Fi (AWDL) transport. Responder: Bonjour TCP listener +
/// ephemeral UDP listener. Initiator: Bonjour browser + TCP and UDP
/// connections. The first control frame each way is the private
/// "UDP:<port>" announcement; everything after is opaque to this class.
final class WiFiP2PTransport: PeerTransport {
    static let bonjourType = "_crosscourt._tcp"

    let name = "wifi-p2p"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?

    private let queue: DispatchQueue
    private let reassembler = FrameReassembler()
    private var tcpListener: NWListener?
    private var udpListener: NWListener?
    private var browser: NWBrowser?
    private var control: NWConnection?
    private var datagramOut: NWConnection?
    private var datagramIn: NWConnection?
    private var announcedLocalUDPPort = false

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.wifi")) {
        self.queue = queue
    }

    private static func p2pParameters(_ base: NWParameters) -> NWParameters {
        base.includePeerToPeer = true
        return base
    }

    // MARK: start

    func startResponder() {
        onStateChange?(.searching)
        startUDPListener()
        do {
            let listener = try NWListener(using: Self.p2pParameters(.tcp))
            listener.service = NWListener.Service(name: UUID().uuidString, type: Self.bonjourType)
            listener.newConnectionHandler = { [weak self] connection in
                self?.adoptControl(connection)
            }
            listener.start(queue: queue)
            tcpListener = listener
        } catch {
            onStateChange?(.disconnected("listener failed: \(error.localizedDescription)"))
        }
    }

    func startInitiator() {
        onStateChange?(.searching)
        startUDPListener()
        let browser = NWBrowser(for: .bonjour(type: Self.bonjourType, domain: nil),
                                using: Self.p2pParameters(.tcp))
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self, self.control == nil, let first = results.first else { return }
            self.browser?.cancel()
            self.adoptControl(NWConnection(to: first.endpoint, using: Self.p2pParameters(.tcp)))
        }
        browser.start(queue: queue)
        self.browser = browser
    }

    private func adoptControl(_ connection: NWConnection) {
        control = connection
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready: self.announceUDPPortIfNeeded()
            case .failed(let error): self.onStateChange?(.disconnected(error.localizedDescription))
            case .cancelled: self.onStateChange?(.disconnected("cancelled"))
            default: break
            }
        }
        receiveControlLoop(connection)
        connection.start(queue: queue)
    }

    private func startUDPListener() {
        do {
            let listener = try NWListener(using: Self.p2pParameters(.udp))
            listener.newConnectionHandler = { [weak self] connection in
                self?.datagramIn = connection
                self?.receiveDatagramLoop(connection)
                connection.start(queue: self?.queue ?? .main)
            }
            listener.start(queue: queue)
            udpListener = listener
        } catch {
            onStateChange?(.disconnected("udp listener failed: \(error.localizedDescription)"))
        }
    }

    private func announceUDPPortIfNeeded() {
        guard !announcedLocalUDPPort, let port = udpListener?.port else { return }
        announcedLocalUDPPort = true
        rawSendControl(FrameCodec.encode(Data("UDP:\(port.rawValue)".utf8)))
    }

    // MARK: receive

    private func receiveControlLoop(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { return }
            if let data { self.ingestControlChunk(data) }
            if done || error != nil {
                self.onStateChange?(.disconnected(error?.localizedDescription ?? "control closed"))
            } else {
                self.receiveControlLoop(connection)
            }
        }
    }

    private func ingestControlChunk(_ chunk: Data) {
        guard let frames = try? reassembler.ingest(chunk) else {
            onStateChange?(.disconnected("control stream corrupted")); return
        }
        for frame in frames {
            if frame.starts(with: Data("UDP:".utf8)) {
                openDatagramPath(portFrame: frame)
            } else {
                onControl?(frame)
            }
        }
    }

    private func openDatagramPath(portFrame: Data) {
        guard let text = String(data: portFrame, encoding: .utf8),
              let rawPort = UInt16(text.dropFirst(4)),
              let port = NWEndpoint.Port(rawValue: rawPort),
              let control = control,
              case .hostPort(let host, _)? = control.currentPath?.remoteEndpoint else { return }
        let connection = NWConnection(host: host, port: port, using: Self.p2pParameters(.udp))
        connection.stateUpdateHandler = { [weak self] state in
            if case .ready = state { self?.onStateChange?(.connected) }
        }
        connection.start(queue: queue)
        datagramOut = connection
    }

    private func receiveDatagramLoop(_ connection: NWConnection) {
        connection.receiveMessage { [weak self] data, _, _, error in
            guard let self else { return }
            if let data { self.onDatagram?(data) }
            if error == nil { self.receiveDatagramLoop(connection) }
        }
    }

    // MARK: send

    func sendControl(_ frame: Data) { rawSendControl(FrameCodec.encode(frame)) }

    private func rawSendControl(_ encoded: Data) {
        control?.send(content: encoded, completion: .contentProcessed { _ in })
    }

    func sendDatagram(_ datagram: Data) {
        datagramOut?.send(content: datagram, completion: .contentProcessed { _ in })
    }

    func stop() {
        browser?.cancel(); tcpListener?.cancel(); udpListener?.cancel()
        control?.cancel(); datagramOut?.cancel(); datagramIn?.cancel()
        onStateChange?(.disconnected("stopped"))
    }
}
```

- [ ] **Step 2: Add local-network plist entries**

In `ios/project.yml`, inside `targets.SquashLineCalling.info.properties`, add:

```yaml
        NSLocalNetworkUsageDescription: Finds and pairs with the second court camera on this court.
        NSBonjourServices: [_crosscourt._tcp]
```

Do NOT add the `com.apple.developer.networking.multicast` entitlement to project.yml —
it requires manual Apple approval and would break Automatic signing for every build
until granted. The contingency (browse failing with `NoAuth`/-65555 on hardware) and
the request procedure are documented in `ios/PEER.md` (Task 11).

- [ ] **Step 3: Build + full suite**

Run: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`
Expected: builds clean; all tests PASS (no new unit tests in this task — wiring only).
Sandbox fallback: `xcrun -sdk iphonesimulator swiftc -parse ios/Sources/Peer/WiFiP2PTransport.swift`.

- [ ] **Step 4: Commit**

```bash
git add ios/Sources/Peer/WiFiP2PTransport.swift ios/project.yml
git commit -m "feat(peer): peer-to-peer Wi-Fi transport over Network.framework"
```

---

### Task 10: Detection streaming wiring in RecordModel

Secondary streams its `BallTracker` observations to the primary; primary buffers remote
detections for Phase 3. Batching: flush every 2 tuples or 30 ms, whichever first (BLE
interval alignment; negligible on Wi-Fi).

**Files:**
- Modify: `ios/Sources/Record/RecordModel.swift` (add `attachPeer`, batcher, remote store)
- Test: `ios/Tests/DetectionStreamTests.swift`

**Interfaces:**
- Consumes: `PeerSession` (Task 6), `BallTracker.subscribe` (existing), `BallObservation` (existing), `DetectionTuple` (Task 2).
- Produces:
  ```swift
  // In RecordModel:
  func attachPeer(_ peer: PeerSession)
  let remoteDetections = RemoteDetectionStore()      // primary side, Phase 3 reads this
  // New file-scope helpers in RecordModel.swift:
  final class RemoteDetectionStore {                  // thread-safe ring of remote tuples
      func append(_ tuples: [DetectionTuple])
      var recent: [DetectionTuple] { get }            // last 900
  }
  enum DetectionMapper {
      /// Vision-normalized (bottom-left) rect → pixel tuple in the LOCAL
      /// frame (frameW/frameH). y flips to top-left row for rolling shutter.
      static func tuple(seq: UInt32, observation: BallObservation,
                        frameW: Int, frameH: Int) -> DetectionTuple
  }
  ```

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/DetectionStreamTests.swift
import XCTest
@testable import SquashLineCalling

final class DetectionStreamTests: XCTestCase {
    func testMapperConvertsNormalizedRectToPixels() {
        let observation = BallObservation(
            timestamp: 1.5,
            rect: CGRect(x: 0.4, y: 0.7, width: 0.1, height: 0.05),  // Vision bottom-left
            confidence: 0.9)
        let tuple = DetectionMapper.tuple(seq: 3, observation: observation,
                                          frameW: 1080, frameH: 1920)
        XCTAssertEqual(tuple.seq, 3)
        XCTAssertEqual(tuple.ptsNs, 1_500_000_000)
        XCTAssertEqual(tuple.x, 0.45 * 1080, accuracy: 0.01)          // midX
        // Vision midY 0.725 from bottom → row from top = (1 − 0.725) × 1920
        XCTAssertEqual(tuple.y, (1 - 0.725) * 1920, accuracy: 0.01)
        XCTAssertEqual(Float(tuple.bboxH), Float(0.05 * 1920), accuracy: 0.1)
    }

    func testRemoteStoreKeepsNewest900() {
        let store = RemoteDetectionStore()
        let tuples = (0..<1000).map {
            DetectionTuple(seq: UInt32($0), ptsNs: UInt64($0), x: 0, y: 0,
                           conf: Float16(1), bboxH: Float16(1))
        }
        store.append(tuples)
        XCTAssertEqual(store.recent.count, 900)
        XCTAssertEqual(store.recent.first?.seq, 100)
        XCTAssertEqual(store.recent.last?.seq, 999)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/DetectionStreamTests`
Expected: build FAILS with "cannot find 'DetectionMapper' in scope".

- [ ] **Step 3: Implement**

First read `ios/Sources/Record/RecordModel.swift` in full (it owns the
tracker/camera wiring and the inference queue). Add at file scope:

```swift
// Appended to ios/Sources/Record/RecordModel.swift (file scope)

final class RemoteDetectionStore {
    private let lock = NSLock()
    private var buffer = RingBuffer<DetectionTuple>(capacity: BallTracker.bufferCapacity)

    func append(_ tuples: [DetectionTuple]) {
        lock.lock(); defer { lock.unlock() }
        for tuple in tuples { buffer.append(tuple) }
    }

    var recent: [DetectionTuple] {
        lock.lock(); defer { lock.unlock() }
        return buffer.elements
    }
}

enum DetectionMapper {
    static func tuple(seq: UInt32, observation: BallObservation,
                      frameW: Int, frameH: Int) -> DetectionTuple {
        DetectionTuple(
            seq: seq,
            ptsNs: UInt64(observation.timestamp * 1_000_000_000),
            x: Float(observation.rect.midX) * Float(frameW),
            y: Float(1 - observation.rect.midY) * Float(frameH),
            conf: Float16(observation.confidence),
            bboxH: Float16(Float(observation.rect.height) * Float(frameH)))
    }
}
```

Then inside the `RecordModel` class add (follow its existing property style):

```swift
    // MARK: peer streaming

    let remoteDetections = RemoteDetectionStore()
    private var peer: PeerSession?
    private var peerPumpTimer: Timer?
    private var nextDetectionSeq: UInt32 = 0
    private var pendingTuples: [DetectionTuple] = []
    private var lastFlushAt: TimeInterval = 0
    private let peerFrameW = 1080, peerFrameH = 1920   // matches Hello until Phase 4

    /// Wire a paired session. Safe to call once, after init. Subscriber runs
    /// on the main queue (BallTracker's fan-out queue). The timer pump keeps
    /// heartbeats/sync alive even when no ball is detected — a primary with
    /// zero local detections must still tick.
    func attachPeer(_ peer: PeerSession) {
        self.peer = peer
        peer.onRemoteDetections = { [weak self] tuples in
            self?.remoteDetections.append(tuples)
        }
        peerPumpTimer?.invalidate()
        peerPumpTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak peer] _ in
            peer?.tick(now: ClockSync.hostNow())
        }
        tracker.subscribe { [weak self] observation in
            guard let self, let peer = self.peer, peer.role == .secondary else { return }
            let tuple = DetectionMapper.tuple(seq: self.nextDetectionSeq,
                                              observation: observation,
                                              frameW: self.peerFrameW, frameH: self.peerFrameH)
            self.nextDetectionSeq += 1
            self.pendingTuples.append(tuple)
            let now = observation.timestamp
            if self.pendingTuples.count >= 2 || now - self.lastFlushAt >= 0.030 {
                peer.sendDetections(self.pendingTuples)
                self.pendingTuples.removeAll(keepingCapacity: true)
                self.lastFlushAt = now
            }
        }
    }
```

(If `tracker` is named differently in RecordModel, use the actual property name — do
not rename existing members. If no `tracker` property is exposed, add
`private let tracker: BallTracker` was already there per current init; adjust only the
new code.)

- [ ] **Step 4: Run tests to verify they pass, then the full suite**

Run: `... -only-testing:SquashLineCallingTests/DetectionStreamTests` — 2/2 PASS —
then the full suite for regressions.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Record/RecordModel.swift ios/Tests/DetectionStreamTests.swift
git commit -m "feat(peer): stream secondary detections to primary with batching"
```

---

### Task 11: PeerBenchView (DEBUG), BenchReport, and ios/PEER.md runbook

The measurement deliverable: a DEBUG-only screen that runs the pairing flow on a chosen
transport, drives a timed bench, and exports the JSON report the selection gate needs.

**Files:**
- Create: `ios/Sources/Peer/BenchReport.swift`
- Create: `ios/Sources/Peer/PeerBenchView.swift`
- Modify: `ios/Sources/RootTabView.swift` (DEBUG-only entry point; read the file first and add the smallest hook that its structure allows — a toolbar/dev button on the Play tab gated by `#if DEBUG`)
- Create: `ios/PEER.md`
- Test: `ios/Tests/BenchReportTests.swift`

**Interfaces:**
- Consumes: `PeerSession`, `ClockSync`, both transports, `ClapDetector`, `CameraController.onAudioSample`.
- Produces:
  ```swift
  struct BenchSample: Codable { var at: Double; var rttMs: Double }
  struct BenchReport: Codable {
      var transport: String
      var startedAt: Date
      var durationS: Double
      var rttMedianMs: Double
      var rttP95Ms: Double
      var rttMaxMs: Double
      var datagramsSent: Int
      var datagramsReceived: Int
      var lossPercent: Double
      var offsetMs: Double
      var offsetUncertaintyMs: Double
      var clapDeltaMs: Double?          // |anchor − network estimate|
      var thermalStates: [String]       // sampled each 30 s via ProcessInfo
      static func build(transport: String, startedAt: Date, durationS: Double,
                        rtts: [Double], sent: Int, received: Int,
                        estimate: ClockSyncEstimate?, anchorDelta: Double?,
                        thermal: [String]) -> BenchReport
  }
  ```
  `build` is pure (percentiles computed inside) → unit-testable. The view is thin glue.

- [ ] **Step 1: Write the failing test for the pure report builder**

```swift
// ios/Tests/BenchReportTests.swift
import XCTest
@testable import SquashLineCalling

final class BenchReportTests: XCTestCase {
    func testPercentilesAndLoss() {
        let rtts = (1...100).map { Double($0) }   // 1..100 ms
        let report = BenchReport.build(
            transport: "loopback", startedAt: Date(timeIntervalSince1970: 0),
            durationS: 60, rtts: rtts, sent: 200, received: 150,
            estimate: ClockSyncEstimate(offset: 0.0021, uncertainty: 0.0009, sampleCount: 40),
            anchorDelta: 0.0004, thermal: ["nominal"])
        XCTAssertEqual(report.rttMedianMs, 50.5, accuracy: 1.0)
        XCTAssertEqual(report.rttP95Ms, 95.0, accuracy: 1.0)
        XCTAssertEqual(report.rttMaxMs, 100.0)
        XCTAssertEqual(report.lossPercent, 25.0)
        XCTAssertEqual(report.offsetMs, 2.1, accuracy: 1e-9)
        XCTAssertEqual(report.clapDeltaMs ?? -1, 0.4, accuracy: 1e-9)
    }

    func testEmptyRTTsProduceZeros() {
        let report = BenchReport.build(transport: "ble", startedAt: Date(), durationS: 0,
                                       rtts: [], sent: 0, received: 0,
                                       estimate: nil, anchorDelta: nil, thermal: [])
        XCTAssertEqual(report.rttMedianMs, 0)
        XCTAssertEqual(report.lossPercent, 0)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -only-testing:SquashLineCallingTests/BenchReportTests`
Expected: build FAILS with "cannot find 'BenchReport' in scope".

- [ ] **Step 3: Implement BenchReport**

```swift
// ios/Sources/Peer/BenchReport.swift
import Foundation

struct BenchSample: Codable { var at: Double; var rttMs: Double }

struct BenchReport: Codable {
    var transport: String
    var startedAt: Date
    var durationS: Double
    var rttMedianMs: Double
    var rttP95Ms: Double
    var rttMaxMs: Double
    var datagramsSent: Int
    var datagramsReceived: Int
    var lossPercent: Double
    var offsetMs: Double
    var offsetUncertaintyMs: Double
    var clapDeltaMs: Double?
    var thermalStates: [String]

    static func build(transport: String, startedAt: Date, durationS: Double,
                      rtts: [Double], sent: Int, received: Int,
                      estimate: ClockSyncEstimate?, anchorDelta: Double?,
                      thermal: [String]) -> BenchReport {
        let sorted = rtts.sorted()
        func percentile(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let index = min(sorted.count - 1, Int(Double(sorted.count) * p))
            return sorted[index]
        }
        let median = sorted.isEmpty ? 0 :
            (sorted.count.isMultiple(of: 2)
                ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
                : sorted[sorted.count / 2])
        return BenchReport(
            transport: transport, startedAt: startedAt, durationS: durationS,
            rttMedianMs: median, rttP95Ms: percentile(0.95), rttMaxMs: sorted.last ?? 0,
            datagramsSent: sent, datagramsReceived: received,
            lossPercent: sent == 0 ? 0 : Double(sent - received) / Double(sent) * 100,
            offsetMs: (estimate?.offset ?? 0) * 1000,
            offsetUncertaintyMs: (estimate?.uncertainty ?? 0) * 1000,
            clapDeltaMs: anchorDelta.map { $0 * 1000 },
            thermalStates: thermal)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: `BenchReportTests` 2/2 PASS.

- [ ] **Step 5: Implement PeerBenchView (DEBUG-only glue)**

```swift
// ios/Sources/Peer/PeerBenchView.swift
#if DEBUG
import SwiftUI

/// Dev-only bench harness: pick transport + role, pair, watch live stats,
/// run a timed bench, share the JSON report. Not part of shipped UI
/// (DESIGN.md phases arrive in spec Phase 4).
struct PeerBenchView: View {
    @StateObject private var model = PeerBenchModel()

    var body: some View {
        List {
            Section("Setup") {
                Picker("Transport", selection: $model.transportName) {
                    Text("Bluetooth").tag("ble")
                    Text("Wi-Fi P2P").tag("wifi-p2p")
                }
                Picker("Role", selection: $model.isInitiator) {
                    Text("Primary (initiator)").tag(true)
                    Text("Secondary").tag(false)
                }
                Button(model.running ? "Stop" : "Start pairing") { model.toggle() }
            }
            Section("Status") {
                Text("Phase: \(model.phaseText)")
                Text("Offset: \(model.offsetText)")
                Text("RTT median/p95/max: \(model.rttText)")
                Text("Datagram loss: \(model.lossText)")
                Text("Thermal: \(model.thermalText)")
            }
            Section("Bench") {
                Button("Run 60 s datagram bench") { model.runBench() }
                    .disabled(!model.canBench)
                Button("Arm clap anchor") { model.armClap() }
                    .disabled(!model.canBench)
                if let url = model.reportURL {
                    ShareLink(item: url) { Text("Share report JSON") }
                }
            }
        }
        .navigationTitle("Peer Bench")
    }
}

/// Owns a PeerSession over the chosen transport, pumps `tick` on a timer,
/// counts echoed datagrams for RTT/loss, and writes BenchReport JSON to
/// the Documents directory. ~150 lines of glue; no protocol logic —
/// everything measurable is delegated to PeerSession/ClockSync/BenchReport.
final class PeerBenchModel: ObservableObject {
    @Published var transportName = "ble"
    @Published var isInitiator = true
    @Published var running = false
    @Published var phaseText = "idle"
    @Published var offsetText = "—"
    @Published var rttText = "—"
    @Published var lossText = "—"
    @Published var thermalText = "nominal"
    @Published var reportURL: URL?
    var canBench: Bool { running && (phaseText.hasPrefix("ready") || phaseText.hasPrefix("live")) }

    private var session: PeerSession?
    private var transport: PeerTransport?
    private var timer: Timer?
    private var rtts: [Double] = []
    private var sent = 0, received = 0
    private var benchStart: Date?
    private var thermal: [String] = []
    private let clapDetector = ClapDetector()
    private let camera = CameraController()   // audio-only use for the clap

    func toggle() { running ? stop() : start() }

    func start() {
        let transport: PeerTransport = transportName == "ble" ? BLETransport() : WiFiP2PTransport()
        let session = PeerSession(transport: transport, isInitiator: isInitiator)
        self.transport = transport; self.session = session
        session.onRemoteDetections = { [weak self] tuples in
            guard let self else { return }
            self.received += tuples.count
            // Echo bench: initiator sends, responder reflects, initiator times.
            if !self.isInitiator { session.sendDetections(tuples) }
            else {
                let now = ClockSync.hostNow()
                for tuple in tuples {
                    self.rtts.append((now - Double(tuple.ptsNs) / 1e9) * 1000)
                }
            }
        }
        session.start()
        running = true
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.pump()
        }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        session?.end(); session = nil; transport = nil
        running = false
    }

    private func pump() {
        guard let session else { return }
        session.tick(now: ClockSync.hostNow())
        phaseText = "\(session.phase)"
        if let estimate = session.clockSync.estimate {
            offsetText = String(format: "%.2f ms ± %.2f ms",
                                estimate.offset * 1000, estimate.uncertainty * 1000)
        }
        if session.phase == .ready || session.phase == .live {
            session.goLive()
        }
        if let benchStart, session.phase == .live {
            // 100 Hz synthetic tuples while the bench runs.
            let tuple = DetectionTuple(seq: UInt32(sent), ptsNs: UInt64(ClockSync.hostNow() * 1e9),
                                       x: 0, y: 0, conf: Float16(1), bboxH: Float16(1))
            session.sendDetections([tuple]); sent += 1
            let sorted = rtts.sorted()
            if !sorted.isEmpty {
                rttText = String(format: "%.0f / %.0f / %.0f ms",
                                 sorted[sorted.count / 2],
                                 sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))],
                                 sorted.last!)
            }
            lossText = String(format: "%.1f %%",
                              sent == 0 ? 0 : Double(sent - received) / Double(sent) * 100)
            if Date().timeIntervalSince(benchStart) >= 60 { finishBench() }
        }
        let state = ProcessInfo.processInfo.thermalState
        thermalText = ["nominal", "fair", "serious", "critical"][state.rawValue]
    }

    func runBench() {
        rtts.removeAll(); sent = 0; received = 0
        thermal.removeAll(); benchStart = Date()
    }

    func armClap() {
        camera.onAudioSample = { [weak self] buffer in
            guard let self, let onset = self.clapDetector.process(sampleBuffer: buffer) else { return }
            self.session?.sendClapAnchor(localOnset: onset)
            DispatchQueue.main.async { self.camera.onAudioSample = nil }
        }
        Task { try? await camera.configure(); camera.start() }
    }

    private func finishBench() {
        defer { benchStart = nil }
        guard let session, let startedAt = benchStart else { return }
        thermal.append(thermalText)
        let report = BenchReport.build(
            transport: transport?.name ?? "?", startedAt: startedAt, durationS: 60,
            rtts: rtts, sent: sent, received: received,
            estimate: session.clockSync.estimate,
            anchorDelta: nil,   // filled manually: compare pre/post-anchor offsets in the report pair
            thermal: thermal)
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("peer-bench-\(Int(startedAt.timeIntervalSince1970)).json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? (try? encoder.encode(report))?.write(to: url)
        reportURL = url
    }
}
#endif
```

- [ ] **Step 6: Add the DEBUG entry point**

Read `ios/Sources/RootTabView.swift` first. Add, in the Play tab's view builder (exact
placement per the file's structure — smallest possible hook):

```swift
#if DEBUG
            .toolbar {
                NavigationLink("Peer bench") { PeerBenchView() }
            }
#endif
```

If the Play tab has no NavigationStack, wrap the bench link as a `.sheet` presented
from a small overlay button instead — keep it `#if DEBUG` either way.

- [ ] **Step 7: Write ios/PEER.md**

```markdown
# Peer layer (two-camera) — bench & ops runbook

## What exists (Plan A)
`Sources/Peer/`: FrameCodec, DetectionTuple, ControlMessage, PeerTransport
(+ Loopback/BLE/WiFiP2P), ClockSync, ClapDetector, PeerSession, BenchReport,
PeerBenchView (DEBUG only). Secondary streams ball detections to primary;
clocks sync via min-RTT NTP + clap anchor.

## Transport selection gate (spec Phase 1)
On BOTH court phones, DEBUG build, mounted on the fins:
1. Play tab → Peer bench. Phone A: Primary + Bluetooth. Phone B: Secondary + Bluetooth.
2. Start pairing → confirm codes → wait for `ready`.
3. Run 60 s datagram bench (players rallying on court for realism).
4. Share both report JSONs (AirDrop) into `hardware/bench/` in the repo.
5. Repeat with Wi-Fi P2P.
6. Fill the table below in a PR; pick the primary transport.

| Metric (60 s, on mounts, rally in progress) | BLE | Wi-Fi P2P |
|---|---|---|
| RTT median / p95 / max (ms) |  |  |
| Datagram loss % |  |  |
| Offset uncertainty (ms) |  |  |
| Clap-anchor delta vs network offset (ms) |  |  |
| Thermal state after 10 min live |  |  |

Selection rule of thumb: loss < 2 % and p95 RTT < 100 ms are both fine for
marking UX; the tiebreaker is offset uncertainty (sync quality) then battery.

## Clap anchor procedure
Stand at the T (equidistant from both fins — this is what cancels the
speed-of-sound bias). Arm clap on both phones, one loud clap. Each phone
detects its own onset and exchanges it; the offset estimate switches to the
anchored value (uncertainty drops to 0.5 ms).

## Sync validation (spec Phase 2 gate, ≤ 2 ms budget)
Clap TWICE, ~30 s apart, from the T. The first arms the anchor; the second
is measured against the anchored mapping: both phones log the second clap's
onset; `|primary_onset − remoteToLocal(secondary_onset)|` is the end-to-end
sync error. Record it in the bench table. Target ≤ 2 ms.

## Wi-Fi P2P contingency: multicast entitlement
If NWBrowser fails with NoAuth / error -65555 on hardware, Bonjour browsing
on this OS requires `com.apple.developer.networking.multicast`:
request it at https://developer.apple.com/contact/request/networking-multicast
(manual Apple approval, takes days–weeks). After approval, add to project.yml:
    entitlements:
      path: Generated/SquashLineCalling.entitlements
      properties:
        com.apple.developer.networking.multicast: true
Do not add it before approval — Automatic signing fails for everyone.

## Known limits (by design, Plan A)
- Peer link does not survive backgrounding; keep both apps foregrounded.
- Portrait 1080×1920 capture (landscape lands with spec Phase 4).
- Detections flow one way (secondary → primary); events flow back in Phase 3.
```

- [ ] **Step 8: Full suite + commit**

Run: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`
Expected: all tests PASS; app target builds with PeerBenchView under DEBUG.

```bash
git add ios/Sources/Peer/BenchReport.swift ios/Sources/Peer/PeerBenchView.swift ios/Sources/RootTabView.swift ios/PEER.md ios/Tests/BenchReportTests.swift
git commit -m "feat(peer): DEBUG bench harness, report export, and PEER.md runbook"
```

---

## User-owned checklist (needs Xcode + two iPhones + the court)

The plan's code gates run in CI/simulator. These need you:

- [ ] Run the full test suite once in Xcode (`xcodebuild test ...`) if the sandbox only ran parse gates.
- [ ] Transport selection bench on the mounts (PEER.md table) — decides BLE vs Wi-Fi P2P.
- [ ] Double-clap sync validation — asserts the ≤ 2 ms Phase 2 budget on real hardware.
- [ ] If Wi-Fi P2P hits NoAuth: file the multicast entitlement request (PEER.md).
- [ ] 10-minute live thermal check on the mounts (bench screen shows thermal state).
- [ ] Drop a real `BallDetector.mlpackage` into `ios/Model/` (MODEL.md) so the secondary has actual detections to stream.

## Self-review notes

- **Spec coverage (Phases 1–2):** transport abstraction + both transports (Tasks 4, 8, 9),
  pairing state machine + code confirm (Task 6), NTP min-RTT sync + re-sync cadence
  (Tasks 5–6), clap anchor incl. equidistant-T procedure (Tasks 7, 11), detection
  stream with batching (Tasks 2, 10), bench + selection gate + entitlement contingency
  (Task 11 + PEER.md). Rolling-shutter row correction and PTS-semantics measurement are
  spec Phase 2 *bench items*, not code: captured in PEER.md's validation steps and the
  user checklist; the correction formula itself is consumed by Phase 3's StereoEngine
  (Plan B) — `DetectionTuple.y` already carries the row so no wire change will be needed.
- **Known simplification:** `ClockSync` drift is handled by rolling re-estimation, not an
  explicit slope model (justified in Task 5; bench validates).
- **Type consistency check:** `PeerSession.sendDetections` ↔ `RecordModel.attachPeer`
  batching; `DetectionBatch.decode` returns `[DetectionTuple]?` and `PeerSession` guards
  empty; `ClockSyncEstimate` field names match `BenchReport.build` usage; `Hello.frameW/frameH`
  match `DetectionMapper` args. Checked.
