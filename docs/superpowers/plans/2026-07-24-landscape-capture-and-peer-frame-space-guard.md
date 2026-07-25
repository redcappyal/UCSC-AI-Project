# Landscape Capture and a Peer Frame-Space Guard That Can Fail — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make capture actually run in landscape, and make the peer frame-space guard
compare two values that can genuinely differ — so a mismatched pair fails instead of
triangulating transposed coordinates into confident, wrong line calls.

**Architecture:** `CaptureSettings` becomes landscape-only with two mount cases that share
one frame size; the physical mount travels in `Hello.captureOrientation` and the guard
compares against `myHello` rather than against globals; a new `OrientationLock` supplies
the per-tab orientation mask; `RecordModel` resolves the mount when capture configures and
pins the mask there so the advertised value cannot go stale.

**Tech Stack:** Swift 5.9, SwiftUI, UIKit (orientation only), AVFoundation, XCTest,
XcodeGen.

## Global Constraints

- **Branch:** `claude/capture-orientation-guard`, already created off `claude/phase4-ui`.
- **⚠️ This plan CANNOT be executed or verified on Windows.** The repo's dev machine is
  win32; `xcodegen` and `xcodebuild` do not exist there. Every verification step below
  requires macOS with Xcode. If you are on Windows you may write the code but you MUST NOT
  claim any step passed — record it as unverified and stop at Task 5.
- **No macOS CI exists.** `.github/workflows/tests.yml` is pytest only, so nothing here is
  covered by CI. Local verification is the only verification.
- **Never edit `ios/SquashLineCalling.xcodeproj`** — it is generated and gitignored. Edit
  `ios/project.yml` and re-run `xcodegen generate`.
- **`peerProtoVersion` stays at `1`.** Bumping it makes the version check subsume every
  case the frame check exists for, killing the guard being fixed here. Do not change it.
- **`Hello.captureOrientation` must stay `Optional`.** A required field makes a legacy
  peer's hello throw in `JSONDecoder`; `ControlMessage.decode` returns `nil` and
  `handleControl` drops the frame with no branch, so pairing hangs in `.searching` with no
  diagnostic.
- **Python tests are unaffected** and must still pass: `.venv/bin/python -m pytest tests/ -q`.

**Full Swift suite** (macOS only):

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

**Single test** (macOS only), substituting the class/method:

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/CaptureOrientationTests/testOppositeLandscapeMountsAreRefused
```

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ios/Sources/Record/CaptureSettings.swift` | Modify — landscape geometry, two mount cases, 0/180 normalization | 1 |
| `ios/Sources/Record/OverlayView.swift` | Modify — drop the `orientation` parameter | 1 |
| `ios/Sources/Record/CameraController.swift` | Modify — default `orientation` to `.landscapeRight` | 1 |
| `ios/Tests/CaptureOrientationTests.swift` | Modify — geometry tests, then guard tests | 1, 2 |
| `ios/CAPTURE.md` | Modify — portrait is no longer a capture mode | 1 |
| `ios/Sources/Peer/ControlMessage.swift` | Modify — `Hello.captureOrientation` | 2 |
| `ios/Sources/Peer/PeerSession.swift` | Modify — injectable orientation, `myHello` guard, `advertisedHello` | 2 |
| `ios/PEER.md` | Modify — replace the "guard does not catch that" warning | 2 |
| `ios/Sources/Record/OrientationLock.swift` | **Create** — pure orientation policy + the mask applier | 3 |
| `ios/Tests/OrientationLockTests.swift` | **Create** — pure policy tests | 3 |
| `ios/Sources/SquashApp.swift` | Modify — `@UIApplicationDelegateAdaptor` | 3 |
| `ios/Sources/RootTabView.swift` | Modify — `selection` binding drives the mask | 3 |
| `ios/project.yml` | Modify — permit landscape-left | 3 |
| `ios/Sources/Record/RecordModel.swift` | Modify — resolve + pin orientation, `detectionFrameSize` | 4 |
| `ios/Tests/LiveWiringTests.swift` | Modify — frame-space-matches-hello invariant | 4 |

---

### Task 1: Capture geometry becomes landscape

Removing `.portrait` from the enum breaks three call sites at compile time
(`OverlayView`'s default, `CameraController`'s default, and the old geometry test), so all
of them move together — this task is not complete until the target builds.

**Files:**
- Modify: `ios/Sources/Record/CaptureSettings.swift:40-74`
- Modify: `ios/Sources/Record/OverlayView.swift:6-15`
- Modify: `ios/Sources/Record/CameraController.swift:20-25`
- Modify: `ios/CAPTURE.md:9`
- Test: `ios/Tests/CaptureOrientationTests.swift:6-19`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `CaptureSettings.CaptureOrientation` with cases `.landscapeRight`,
  `.landscapeLeft`, conforming to `String, Codable, Equatable, CaseIterable`;
  `CaptureSettings.frameWidth: Int` == 3840; `CaptureSettings.frameHeight: Int` == 2160;
  `CaptureSettings.frameSize(for: CaptureOrientation) -> (width: Int, height: Int)`;
  `CaptureSettings.rotationAngle(for: CaptureOrientation) -> CGFloat`.

- [ ] **Step 1: Rewrite the geometry tests to expect landscape**

Replace the first two test methods in `ios/Tests/CaptureOrientationTests.swift` (keep
`testMismatchedPeerFrameSizeIsRejected` exactly as it is — it still passes and still
guards a real case):

```swift
    func testBothMountsShareOneUprightLandscapeFrameSpace() {
        // Identical dimensions for both mounts is the whole reason the wire
        // needs an explicit orientation field: width/height cannot tell a
        // landscape-left mount from a landscape-right one.
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let s = CaptureSettings.frameSize(for: orientation)
            XCTAssertEqual(s.width, CaptureSettings.sensorWidth, "\(orientation)")
            XCTAssertEqual(s.height, CaptureSettings.sensorHeight, "\(orientation)")
        }
    }

    func testRotationNormalizesEachMountUpright() {
        // Landscape-right matches the sensor's native readout; landscape-left
        // is that readout upside down, so it needs 180 to record upright.
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeRight), 0)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeLeft), 180)
    }

    func testFrameConstantsAreLandscape() {
        XCTAssertEqual(CaptureSettings.frameWidth, 3840)
        XCTAssertEqual(CaptureSettings.frameHeight, 2160)
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/CaptureOrientationTests
```

Expected: FAIL to compile — `type 'CaptureSettings.CaptureOrientation' has no member 'allCases'` and `'landscapeLeft'`.

- [ ] **Step 3: Rewrite the orientation block in CaptureSettings**

In `ios/Sources/Record/CaptureSettings.swift`, replace lines 40-74 (from the
`/// Portrait frame geometry` comment through the closing brace of `frameSize(for:)`) with:

```swift
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
```

- [ ] **Step 4: Fix the two call sites that no longer compile**

In `ios/Sources/Record/CameraController.swift`, replace lines 20-25:

```swift
    let session = AVCaptureSession()
    /// Which way the phone sits in its mount — set before `configure()` runs,
    /// since it drives both the video connection's rotation and the asset
    /// writer's dimensions below. `RecordModel.startCamera` resolves it from
    /// the interface orientation and pins it there.
    var orientation: CaptureSettings.CaptureOrientation = .landscapeRight
```

In `ios/Sources/Record/OverlayView.swift`, replace lines 3-15 (the doc comment through
`contentSize`) with:

```swift
/// Live ball marker + short fading trail over the camera preview.
/// Capture is always landscape and both mounts normalize to one upright
/// frame space (CaptureSettings), so there is nothing to branch on here; the
/// preview letterboxes with resizeAspect, so map through the same
/// aspect-fit rect.
struct OverlayView: View {
    let trail: [BallObservation]   // oldest first, newest last

    var contentSize: CGSize {
        CGSize(width: CaptureSettings.frameWidth, height: CaptureSettings.frameHeight)
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

Expected: PASS, whole suite. `RecordView`'s `OverlayView(trail:)` call site already omits
`orientation`, so it needs no change.

- [ ] **Step 6: Update CAPTURE.md**

In `ios/CAPTURE.md`, replace line 9:

```markdown
| Resolution | 3840x2160 landscape, both mounts | `applyCaptureFormat` via `bestFormatIndex` |
```

Then add immediately below the table:

```markdown
Capture is always landscape. The phone mounts on its side, and
`CaptureSettings.rotationAngle(for:)` applies 0 or 180 so the recorded frame is upright
whichever end the lens is at. Portrait is not a capture mode.
```

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/Record/CaptureSettings.swift ios/Sources/Record/OverlayView.swift ios/Sources/Record/CameraController.swift ios/Tests/CaptureOrientationTests.swift ios/CAPTURE.md && git commit -m "feat(capture): landscape-only geometry with both mounts normalized upright"
```

---

### Task 2: The guard compares against what this phone advertised

**Files:**
- Modify: `ios/Sources/Peer/ControlMessage.swift:7-14`
- Modify: `ios/Sources/Peer/PeerSession.swift:20-22, 95-109, 245-256`
- Modify: `ios/PEER.md:56-70`
- Test: `ios/Tests/CaptureOrientationTests.swift`

> **Note on test placement:** the spec put these in `PeerSessionTests`. They go in
> `CaptureOrientationTests` instead, because the existing sibling
> `testMismatchedPeerFrameSizeIsRejected` already lives there — tests that change together
> live together. That existing test stays untouched and must keep passing: it corrupts
> `frameW/frameH` in transit via `controlDeliveryHook`, which is still a genuine mismatch
> and still a case worth guarding. It is also why the claim "no such test can be written"
> needs a caveat: the guard *could* be provoked by rewriting the wire; what was impossible
> was two differently-**configured** sessions disagreeing.

**Interfaces:**
- Consumes: `CaptureSettings.CaptureOrientation`, `CaptureSettings.frameSize(for:)` (Task 1).
- Produces: `Hello.captureOrientation: CaptureSettings.CaptureOrientation?`;
  `PeerSession.init(transport:isInitiator:now:heartbeatTimeout:captureOrientation:)` with
  `captureOrientation` defaulting to `.landscapeRight`; `PeerSession.advertisedHello: Hello`.

- [ ] **Step 1: Write the failing tests**

Append to `ios/Tests/CaptureOrientationTests.swift`:

```swift
    /// The regression the guard existed for but could not have. Before this
    /// change both sessions advertised the same compile-time portrait
    /// constants, so no pair of PeerSessions could disagree no matter how
    /// they were configured.
    func testOppositeLandscapeMountsAreRefused() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 },
                                  captureOrientation: .landscapeRight)
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 },
                                    captureOrientation: .landscapeLeft)
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failed on the primary, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation"), "unhelpful reason: \(why)")
        // Both sides must refuse: each runs its own guard against the other's
        // hello, and a one-sided refusal would leave the peer waiting.
        guard case .failed = secondary.phase else {
            return XCTFail("expected failed on the secondary, got \(secondary.phase)")
        }
    }

    func testMatchingMountsStillPair() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 },
                                  captureOrientation: .landscapeLeft)
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 },
                                    captureOrientation: .landscapeLeft)
        secondary.start(); primary.start()
        guard case .confirming = primary.phase else {
            return XCTFail("a matched pair must still pair, got \(primary.phase)")
        }
    }

    /// A peer on a build predating the field sends a hello with no
    /// captureOrientation key. It must decode (an Optional field is what makes
    /// that true) and then be refused with the orientation message — not
    /// dropped silently, and not reported as a version mismatch.
    func testLegacyPeerWithoutOrientationIsRefused() {
        let pair = LoopbackTransport.pair()
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var h)? = ControlMessage.decode(frame) else { return deliver(frame) }
            h.captureOrientation = nil     // JSONEncoder omits the key entirely
            deliver(try! ControlMessage.encode(.hello(h)))
        }
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failed, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation"), "unhelpful reason: \(why)")
        XCTAssertFalse(why.lowercased().contains("protocol version"),
                       "a legacy hello must not be reported as a version mismatch")
    }

    /// A hello with no orientation key must still DECODE. If this fails, the
    /// field was made non-optional and pairing with an older build will hang
    /// in .searching with no diagnostic at all.
    ///
    /// Built by encoding a nil mount rather than from a hand-written JSON
    /// literal: Swift synthesizes Codable for enums with unlabeled associated
    /// values under a "_0" key, so a literal guessed from the struct's field
    /// names would not decode and the test would pass for the wrong reason.
    /// JSONEncoder omits nil Optionals, so these bytes ARE a legacy peer's.
    func testHelloWithoutOrientationKeyStillDecodes() {
        let legacy = Hello(protoVersion: peerProtoVersion, appVersion: "dev",
                           deviceModel: "x", nonce: 7,
                           frameW: CaptureSettings.frameWidth,
                           frameH: CaptureSettings.frameHeight,
                           captureOrientation: nil)
        let data = try! ControlMessage.encode(.hello(legacy))
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("captureOrientation"),
                       "a nil mount must be absent from the wire, matching a legacy peer's bytes")
        guard case .hello(let decoded)? = ControlMessage.decode(data) else {
            return XCTFail("a legacy hello must decode, not drop")
        }
        XCTAssertNil(decoded.captureOrientation)
    }
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/CaptureOrientationTests
```

Expected: FAIL to compile — `extra argument 'captureOrientation' in call` and `value of type 'Hello' has no member 'captureOrientation'`.

- [ ] **Step 3: Add the wire field**

In `ios/Sources/Peer/ControlMessage.swift`, replace the `Hello` struct:

```swift
struct Hello: Codable, Equatable {
    var protoVersion: Int
    var appVersion: String
    var deviceModel: String
    var nonce: UInt32
    var frameW: Int
    var frameH: Int
    /// Which way the phone is mounted. Both mounts share one frame size, so
    /// frameW/frameH cannot distinguish them — this can.
    ///
    /// Optional is load-bearing, not laziness. A required field makes a hello
    /// from a build predating it THROW in JSONDecoder; ControlMessage.decode
    /// returns nil and handleControl guards that away with no branch, so the
    /// frame is dropped silently and pairing hangs in .searching with no
    /// diagnostic. Optional decodes cleanly and nil reads as "legacy peer,
    /// portrait capture" — a mismatch against any landscape session, refused
    /// with a message that tells the operator what to do.
    var captureOrientation: CaptureSettings.CaptureOrientation?
}
```

- [ ] **Step 4: Make the advertised orientation injectable and the guard self-referential**

In `ios/Sources/Peer/PeerSession.swift`, add below the `peerHello` property (line ~22):

```swift
    /// What this device advertises. Exposed so callers can assert their
    /// detection frame space matches the space the peer will read it in —
    /// the two disagreeing is precisely the bug this file used to have.
    var advertisedHello: Hello { stateLock.lock(); defer { stateLock.unlock() }; return myHello }
```

Replace the `init` signature and `myHello` construction (lines ~95-106):

```swift
    init(transport: PeerTransport, isInitiator: Bool,
         now: @escaping () -> TimeInterval = ClockSync.hostNow,
         heartbeatTimeout: TimeInterval = 3.0,
         captureOrientation: CaptureSettings.CaptureOrientation = .landscapeRight) {
        self.transport = transport
        self.isInitiator = isInitiator
        self.now = now
        self.heartbeatTimeout = heartbeatTimeout
        // Advertise the space this session actually captures in. Reading the
        // CaptureSettings statics here instead is what made the guard below
        // inert: both sides then advertise the same compile-time constants
        // regardless of how they are really capturing.
        let frame = CaptureSettings.frameSize(for: captureOrientation)
        self.myHello = Hello(protoVersion: peerProtoVersion,
                             appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                             deviceModel: ProcessInfo.processInfo.hostName,
                             nonce: UInt32.random(in: .min ... .max),
                             frameW: frame.width,
                             frameH: frame.height,
                             captureOrientation: captureOrientation)
```

Replace the frame-space guard (lines ~245-256):

```swift
            // Detection tuples are expressed in the advertised frame space.
            // Compared against myHello, NOT against the CaptureSettings
            // globals: the advertised value and the compared value are then
            // the same value and cannot drift apart. Comparing against the
            // globals is what made this guard a tautology — both sides built
            // their hello from those same constants, so it could never fail.
            //
            // Orientation is compared explicitly because both mounts share
            // one frame size; a nil orientation is a peer predating the field
            // (portrait capture), which is a genuine mismatch.
            guard theirs.frameW == myHello.frameW,
                  theirs.frameH == myHello.frameH,
                  theirs.captureOrientation == myHello.captureOrientation else {
                setPhase(.failed("peer camera orientation doesn't match this phone — mount both phones the same way and reconnect"))
                transport.stop()
                return
            }
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

Expected: PASS, whole suite. `PeerBenchView.swift:92` still compiles unchanged — it takes
the `.landscapeRight` default.

- [ ] **Step 6: Correct PEER.md**

In `ios/PEER.md`, replace the capture bullet and the whole `⚠️` bullet (lines ~56-70) with:

```markdown
- Capture is 4K60 landscape (3840x2160) in both mounts; `CaptureSettings.rotationAngle`
  normalizes landscape-left upright with 180. **Both phones must run the same mount** —
  `PeerSession` enforces it at handshake by comparing the peer's advertised frame size
  AND `captureOrientation` against its own `myHello`, and refuses the pair otherwise.
  A peer on a build predating the `captureOrientation` field advertises none; that reads
  as legacy portrait capture and is refused with the same message. `peerProtoVersion`
  deliberately stays at 1 so those peers reach this specific, actionable error instead of
  a generic version mismatch.
- When the production pairing path is wired, `PeerSession` must be constructed with the
  session's resolved `captureOrientation` (`RecordModel.captureOrientation`), not the
  `.landscapeRight` default.
```

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/Peer/ControlMessage.swift ios/Sources/Peer/PeerSession.swift ios/Tests/CaptureOrientationTests.swift ios/PEER.md && git commit -m "fix(peer): compare the peer's frame space against what this phone advertised"
```

---

### Task 3: Per-tab orientation policy

**Files:**
- Create: `ios/Sources/Record/OrientationLock.swift`
- Create: `ios/Tests/OrientationLockTests.swift`
- Modify: `ios/Sources/SquashApp.swift`
- Modify: `ios/Sources/RootTabView.swift:15-17, 44-53`
- Modify: `ios/project.yml:44-46`

**Interfaces:**
- Consumes: `CaptureSettings.CaptureOrientation` (Task 1).
- Produces: `RootTab` enum with `.play`, `.matches`, `.coach`;
  `OrientationLock.mask(for: RootTab) -> UIInterfaceOrientationMask`;
  `OrientationLock.pinnedMask(for: CaptureSettings.CaptureOrientation) -> UIInterfaceOrientationMask`;
  `OrientationLock.captureOrientation(for: UIInterfaceOrientation) -> CaptureSettings.CaptureOrientation?`;
  `OrientationPolicy.shared.apply(_ mask: UIInterfaceOrientationMask)`.

> **Note on where this lives:** the spec put `captureOrientation(for:)` on `CaptureSettings`.
> It goes on `OrientationLock` instead so `CaptureSettings` stays UIKit-free — it is an
> AVFoundation-domain type and importing UIKit there to map one enum would invert the
> dependency. Same behaviour, cleaner boundary.

- [ ] **Step 1: Write the failing tests**

Create `ios/Tests/OrientationLockTests.swift`:

```swift
// ios/Tests/OrientationLockTests.swift
import XCTest
import UIKit
@testable import SquashLineCalling

final class OrientationLockTests: XCTestCase {
    func testPlayTabIsLandscapeOnly() {
        // The preview is live while the operator aims the phone in its mount,
        // so a sideways preview is worst exactly where it matters most.
        XCTAssertEqual(OrientationLock.mask(for: .play), .landscape)
    }

    func testWebTabsRotateFreely() {
        // Matches and Coach are portrait mobile web UI.
        XCTAssertEqual(OrientationLock.mask(for: .matches), .all)
        XCTAssertEqual(OrientationLock.mask(for: .coach), .all)
    }

    func testPinnedMaskNarrowsToTheResolvedMount() {
        XCTAssertEqual(OrientationLock.pinnedMask(for: .landscapeRight), .landscapeRight)
        XCTAssertEqual(OrientationLock.pinnedMask(for: .landscapeLeft), .landscapeLeft)
    }

    /// Pinning must only ever narrow what the Play tab already permits — a
    /// pinned mask outside it would ask UIKit to rotate somewhere forbidden.
    func testPinnedMaskIsAlwaysASubsetOfThePlayMask() {
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            XCTAssertTrue(OrientationLock.mask(for: .play)
                .contains(OrientationLock.pinnedMask(for: orientation)), "\(orientation)")
        }
    }

    func testLandscapeInterfaceMapsToTheMatchingMount() {
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeRight), .landscapeRight)
        XCTAssertEqual(OrientationLock.captureOrientation(for: .landscapeLeft), .landscapeLeft)
    }

    func testPortraitInterfaceHasNoMount() {
        // Portrait is not a capture mode; callers fall back rather than guess.
        XCTAssertNil(OrientationLock.captureOrientation(for: .portrait))
        XCTAssertNil(OrientationLock.captureOrientation(for: .portraitUpsideDown))
        XCTAssertNil(OrientationLock.captureOrientation(for: .unknown))
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/OrientationLockTests
```

Expected: FAIL to compile — `cannot find 'OrientationLock' in scope`.

- [ ] **Step 3: Create OrientationLock**

Create `ios/Sources/Record/OrientationLock.swift`:

```swift
// ios/Sources/Record/OrientationLock.swift
import UIKit

/// Which tab is showing, and therefore which orientations the app permits.
enum RootTab: Hashable, CaseIterable { case play, matches, coach }

/// The app's supported-orientation policy as pure functions.
///
/// Capture is always landscape (`CaptureSettings`), so the Play tab is
/// landscape too — not just while recording. The camera preview is live while
/// the operator aims the phone in its back-wall mount, which is exactly when a
/// sideways preview costs the most. Matches and Coach are portrait mobile web
/// UI and keep rotating freely.
enum OrientationLock {
    /// Orientations permitted while `tab` is showing, before capture pins.
    static func mask(for tab: RootTab) -> UIInterfaceOrientationMask {
        switch tab {
        case .play: return .landscape
        case .matches, .coach: return .all
        }
    }

    /// Narrowed to the single mount capture resolved at configure time, so the
    /// device cannot flip to the other landscape mid-session and leave the
    /// orientation advertised in `Hello` describing a mount that is no longer
    /// real.
    static func pinnedMask(for orientation: CaptureSettings.CaptureOrientation) -> UIInterfaceOrientationMask {
        switch orientation {
        case .landscapeRight: return .landscapeRight
        case .landscapeLeft: return .landscapeLeft
        }
    }

    /// The mount an interface orientation implies, or nil when the interface
    /// is portrait — portrait is not a capture mode, so callers fall back to a
    /// default rather than inventing a mount.
    static func captureOrientation(for interface: UIInterfaceOrientation) -> CaptureSettings.CaptureOrientation? {
        switch interface {
        case .landscapeRight: return .landscapeRight
        case .landscapeLeft: return .landscapeLeft
        default: return nil
        }
    }
}

/// Holds the mask the app delegate serves. UIKit asks the delegate on every
/// rotation decision, so mutating this and then telling UIKit to re-ask is
/// what actually moves the device.
final class OrientationPolicy {
    static let shared = OrientationPolicy()
    private(set) var mask: UIInterfaceOrientationMask = .all

    func apply(_ newMask: UIInterfaceOrientationMask) {
        mask = newMask
        guard let scene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first(where: { $0.activationState == .foregroundActive }) else { return }
        scene.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: newMask))
    }
}

/// Serves `OrientationPolicy.shared.mask`. SwiftUI has no orientation-lock
/// API, so the app delegate is the only hook UIKit consults.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        OrientationPolicy.shared.mask
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/OrientationLockTests
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Register the delegate and drive the mask from tab selection**

Replace `ios/Sources/SquashApp.swift` entirely:

```swift
import SwiftUI

@main
struct SquashApp: App {
    // SwiftUI cannot lock orientation on its own; the delegate is the hook
    // UIKit consults. See OrientationLock.swift.
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .preferredColorScheme(.dark)
        }
    }
}
```

In `ios/Sources/RootTabView.swift`, add below the `@AppStorage` line (~line 11):

```swift
    // Bound selection rather than per-tab onAppear/onDisappear: SwiftUI does
    // not guarantee their ordering across a tab switch, and the mask must be
    // a function of which tab is showing, not of which callback ran last.
    @State private var tab: RootTab = .play
```

Change `TabView {` to `TabView(selection: $tab) {`, add `.tag(RootTab.play)` to the
`RecordView()` branch (after its `.tabItem`), `.tag(RootTab.matches)` and
`.tag(RootTab.coach)` to the two `WebScreen` branches, and add to the `TabView`'s modifier
chain beside `.tint(...)`:

```swift
        .onAppear { OrientationPolicy.shared.apply(OrientationLock.mask(for: tab)) }
        .onChange(of: tab) { _, newTab in
            OrientationPolicy.shared.apply(OrientationLock.mask(for: newTab))
        }
```

- [ ] **Step 6: Permit landscape-left in the Info.plist**

In `ios/project.yml`, replace the `UISupportedInterfaceOrientations` block and its comment
(lines ~43-46):

```yaml
        # Superset of every orientation any tab allows; OrientationLock does
        # the per-tab narrowing at runtime. Portrait stays because the Matches
        # and Coach webviews are portrait mobile UI — a runtime mask can only
        # restrict what this list already permits, never extend it.
        UISupportedInterfaceOrientations:
          [UIInterfaceOrientationPortrait, UIInterfaceOrientationLandscapeLeft,
           UIInterfaceOrientationLandscapeRight]
```

- [ ] **Step 7: Run the full suite and commit**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

Expected: PASS, whole suite.

```bash
git add ios/Sources/Record/OrientationLock.swift ios/Tests/OrientationLockTests.swift ios/Sources/SquashApp.swift ios/Sources/RootTabView.swift ios/project.yml && git commit -m "feat(capture): lock the Play tab to landscape via a per-tab orientation mask"
```

---

### Task 4: RecordModel resolves and pins the mount

**Files:**
- Modify: `ios/Sources/Record/RecordModel.swift:1, 76-80, 154, 164, 334-346`
- Test: `ios/Tests/LiveWiringTests.swift`

**Interfaces:**
- Consumes: `OrientationLock.captureOrientation(for:)`, `OrientationLock.pinnedMask(for:)`,
  `OrientationPolicy.shared.apply(_:)` (Task 3); `PeerSession.advertisedHello` (Task 2);
  `CaptureSettings.frameSize(for:)` (Task 1).
- Produces: `RecordModel.captureOrientation: CaptureSettings.CaptureOrientation` (published,
  private setter); `RecordModel.detectionFrameSize: (width: Int, height: Int)`;
  `RecordModel.init(detector:captureOrientation:)` with both parameters defaulted.

- [ ] **Step 1: Write the failing test**

Append to `ios/Tests/LiveWiringTests.swift`, inside `final class LiveWiringTests`:

```swift
    /// The invariant the original bug broke: RecordModel labelled every
    /// detection tuple with hardcoded portrait constants while PeerSession
    /// advertised its own. Whatever orientation a session resolves, the space
    /// tuples are expressed in must equal the space the peer is told to read
    /// them in.
    func testDetectionFrameSpaceMatchesTheAdvertisedHello() {
        for orientation in CaptureSettings.CaptureOrientation.allCases {
            let model = RecordModel(detector: nil, captureOrientation: orientation)
            let (transport, _) = LoopbackTransport.pair()
            let session = PeerSession(transport: transport, isInitiator: true, now: { 0 },
                                      captureOrientation: model.captureOrientation)
            XCTAssertEqual(model.detectionFrameSize.width,
                           session.advertisedHello.frameW, "\(orientation)")
            XCTAssertEqual(model.detectionFrameSize.height,
                           session.advertisedHello.frameH, "\(orientation)")
        }
    }

    func testCaptureOrientationDefaultsToLandscapeRight() {
        XCTAssertEqual(RecordModel(detector: nil).captureOrientation, .landscapeRight)
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd ios && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1' -only-testing:SquashLineCallingTests/LiveWiringTests
```

Expected: FAIL to compile — `value of type 'RecordModel' has no member 'detectionFrameSize'`.

- [ ] **Step 3: Replace the hardcoded frame constants**

In `ios/Sources/Record/RecordModel.swift`, change line 1 from `import Foundation` to:

```swift
import Foundation
import UIKit
```

Replace lines 76-80 (the `peerFrameW`/`peerFrameH` comment and both `let`s):

```swift
    /// The mount this session capture-locked to, resolved when the camera
    /// configures and pinned there (see `startCamera`). Everything that labels
    /// a pixel space reads this.
    @Published private(set) var captureOrientation: CaptureSettings.CaptureOrientation = .landscapeRight

    /// The pixel space local detections are expressed in — the same space the
    /// peer is told to read them in via `Hello`. Derived from the resolved
    /// mount rather than from a constant, which is exactly what went wrong
    /// before: hardcoded portrait constants labelled every tuple in a session
    /// that was capturing landscape.
    var detectionFrameSize: (width: Int, height: Int) {
        CaptureSettings.frameSize(for: captureOrientation)
    }

Then widen `init` (line ~35) so tests can choose a mount without a test-only mutator on the
production type. Change the signature and set the property as the first statement in the
body, leaving the rest of `init` exactly as it is:

```swift
    init(detector: BallDetecting? = CoreMLBallDetector(),
         captureOrientation: CaptureSettings.CaptureOrientation = .landscapeRight) {
        // Production overwrites this in `startCamera` once the interface
        // orientation is known; the parameter exists so tests can pick a mount
        // without a live window scene.
        self.captureOrientation = captureOrientation
```

The five existing `RecordModel(detector:)` call sites in `LiveWiringTests` and
`SyntheticDetectorTests` keep compiling unchanged — the new parameter is defaulted.
```

Replace the two consumers. Line ~154 becomes:

```swift
                let tuple = DetectionMapper.tuple(seq: self.nextDetectionSeq,
                                                  observation: observation,
                                                  frameW: self.detectionFrameSize.width,
                                                  frameH: self.detectionFrameSize.height)
```

Line ~164 becomes:

```swift
                engine.addLocalObservation(observation,
                                           frameW: self.detectionFrameSize.width,
                                           frameH: self.detectionFrameSize.height)
```

- [ ] **Step 4: Resolve and pin the mount when capture configures**

Replace `startCamera()` (lines ~334-346):

```swift
    func startCamera() async {
        do {
            // Resolve the mount from the interface orientation, which the Play
            // tab has already constrained to landscape, then pin the mask
            // there. Pinning is what keeps the orientation advertised in Hello
            // true for the session's whole life: without it a mid-session flip
            // to the other landscape would leave the peer holding a mount
            // description that is no longer real.
            let interface = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first { $0.activationState == .foregroundActive }?
                .interfaceOrientation ?? .landscapeRight
            captureOrientation = OrientationLock.captureOrientation(for: interface) ?? .landscapeRight
            camera.orientation = captureOrientation
            OrientationPolicy.shared.apply(OrientationLock.pinnedMask(for: captureOrientation))

            try await camera.configure()
            camera.start()
            // Meter the court once from the mounted position, then freeze
            // exposure/WB/focus for the session. Everything after this point
            // is shot under identical conditions, which is what keeps the
            // footage usable as training data.
            exposureNote = CaptureSettings.summary(for: try await camera.lockForCourt())
        } catch {
            errorText = error.localizedDescription
        }
    }
```

- [ ] **Step 5: Run the full suite to verify it passes**

```bash
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

Expected: PASS, whole suite.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Record/RecordModel.swift ios/Tests/LiveWiringTests.swift && git commit -m "fix(record): label detection tuples with the mount the session actually captured in"
```

---

### Task 5: Verify the two claims tests cannot reach

The spec flagged these as asserted rather than measured. They are the reason this task
exists as its own gate: both are UIKit/AVFoundation runtime behaviour that no unit test
can assert, and shipping on the assumption they hold is how the original bug happened.

**Files:** none (verification only; fixes land as follow-up commits if anything fails).

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a recorded verification result.

- [ ] **Step 1: Verify the Play tab actually locks to landscape**

Run the app in the simulator. On the Play tab, rotate the simulator (⌘←/⌘→) through all
four orientations.

Expected: the UI stays landscape and never shows portrait. Rotating between
landscape-left and landscape-right is permitted **before** the camera configures.

- [ ] **Step 2: Verify the pin holds after capture configures**

Stay on the Play tab until the exposure note appears (capture has configured), then try to
rotate to the other landscape.

Expected: the UI does NOT rotate — the mask has narrowed to the resolved mount.

- [ ] **Step 3: Verify the web tabs still rotate**

Switch to Matches, then Coach. Rotate through all four orientations.

Expected: both tabs rotate freely including portrait. Switch back to Play and confirm it
returns to landscape.

- [ ] **Step 4: Verify 180 normalization on a device (REQUIRES HARDWARE)**

The simulator has no camera, so this step needs a real iPhone. Mount or hold the phone in
**landscape-left**, record a short clip, then play it back.

Expected: the recorded video is upright, not upside down. If it IS upside down, the
`rotationAngle` mapping in `CaptureSettings` is inverted — swap the 0 and 180 cases and
re-run Task 1's tests, which assert the mapping and will need updating together.

- [ ] **Step 5: Record the outcome honestly**

Append to `ios/CAPTURE.md` a short "Verified" line stating the date, the device model, and
which of steps 1-4 actually passed. If step 4 could not be run for lack of hardware, say
so explicitly rather than omitting it — an unverified normalization claim is exactly the
kind of thing this whole change exists to stop.

- [ ] **Step 6: Commit**

```bash
git add ios/CAPTURE.md && git commit -m "docs(capture): record what landscape verification actually covered"
```

---

## Done when

- The full Swift suite passes on macOS.
- `.venv/bin/python -m pytest tests/ -q` still passes (259 tests) — nothing here touches Python.
- Two `PeerSession`s configured with opposite mounts both land in `.failed`.
- A legacy hello with no `captureOrientation` key decodes and is refused with the
  orientation message, not a version mismatch.
- `RecordModel.detectionFrameSize` equals `PeerSession.advertisedHello`'s frame size for
  every mount.
- Task 5's verification outcome is recorded, including anything that could not be run.

## Explicitly NOT done by this plan

Wiring `RecordModel.attachPeer` to a production caller. `PeerSession` is still constructed
only in `PeerBenchView` (DEBUG). This plan makes the guard correct and the mount real; it
does not ship two-phone live pairing. PEER.md gains the note that the production path must
pass `RecordModel.captureOrientation` when that work happens.
