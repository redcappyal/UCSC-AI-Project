# Phase 4: Live Experience (p-pair, p-live, replay, landscape) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the finished stereo engine into something a player can actually use on
court: pair two phones, see and hear line calls live, review the rally in 3D — with the
whole chain demoable on one device before the ball model exists.

**Architecture:** The engine is done and untouched by this plan. Phase 4 is the layer
above it: a `PairingModel` driving a new `p-pair` screen off `PeerSession`'s existing
phases, a `LiveCallView` rendering `StereoImpact` events per DESIGN.md §17.3's
pre-authorised grammar, a post-rally mini-court replay, and the landscape capture the
mount requires. A DEBUG `SyntheticBallDetector` drives the full path with no model and
no second phone, so every screen is verifiable during the two-day model gap.

**Tech Stack:** SwiftUI (iOS 17, Swift 5.9), XCTest, existing `Theme`/`CaptureSettings`;
DESIGN.md for all visual decisions.

## Global Constraints

- **DESIGN.md is binding and must be updated in the same change** (CLAUDE.md). §17.1: a
  new screen is a new phase + reused §8 components; **a new component requires its own
  §8 subsection written first**. §17.2: a new colour needs a token + light override + a
  §5.2 family assignment.
- **§17.3 pre-authorises the live grammar** — *"a live call is a full-stage
  verdict-colored flash + uppercase word, not a new visual language."* Use it; do not
  invent one.
- §0.2 one accent (`#ffd60a`). §0.3 colour = meaning: **green/red are verdict-only** —
  connection state must NOT be a naive green dot; use `Theme.dim`/`Theme.unknown` plus a
  text label. §0.4 capsules/cards only. §0.6 44 pt targets (48 pt primary). §0.7
  uppercase controls, sentence-case guidance. §0.8 tabular numerals. **§0.9 no layout
  shift** — reserve space for the call banner. §0.11 respect the shell, no new nav
  chrome. §7 one primary action per phase.
- **Graceful degradation is a hard requirement**: with no peer connected the app behaves
  exactly as today's single-camera app. Pairing strictly adds capability.
- **Recording never depends on the link.** A dropped link shows a visible banner and
  keeps recording locally — never a silent downgrade.
- **Honest states only**: `called (high)` / `called (one-view)` / `no-call (obstructed)`.
  Never render a guess as a confident call.
- Frame space: every live consumer works in `CaptureSettings.frameWidth × frameHeight`.
  Camera models arrive at arbitrary solve resolutions and are adopted via the existing
  `adoptedForCapture()`; when adoption fails the session must not go live (already
  implemented — do not weaken it).
- New Swift files under `ios/Sources/Live/`; tests in `ios/Tests/`. Never `Date()` in
  timing paths — use `ClockSync.hostNow()`.
- Baselines to hold: **Python 271**, **Swift 102**. Test command:
  `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'`
- **Simulator cannot run the real capture path** (4K60 + locked exposure needs hardware).
  Everything in this plan must therefore be testable via injected state, not live camera.
- Commit per task with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Branch: `claude/phase4-ui`.

## File Structure

- `DESIGN.md` (Task 1) — new §8 subsections + `p-pair`/`p-live` phase entries.
- `ios/Sources/Live/SyntheticBallDetector.swift` (Task 2, DEBUG) — model-free detection source.
- `ios/Sources/Live/PairingModel.swift` (Task 4) — observable pairing state machine.
- `ios/Sources/Live/PairingView.swift` (Task 4) — the `p-pair` screen.
- `ios/Sources/Live/LiveCallView.swift` (Task 5) — verdict flash + honest-state banner.
- `ios/Sources/Live/CallAnnouncer.swift` (Task 5) — audible call, default off.
- `ios/Sources/Live/MiniCourtView.swift` (Task 7) — post-rally 3D replay.
- Modified: `ios/project.yml` + `CameraController` + `OverlayView` (Task 3, landscape),
  `RecordModel` + `RecordView` + `RootTabView` (Task 6, wiring).

---

### Task 1: DESIGN.md — components and phases for the live surfaces

§17.1 requires the design entries to exist **before** the components. This task is
documentation only, and it gates every UI task after it.

**Files:** Modify `DESIGN.md`.

**Interfaces:** Produces the names later tasks must implement against: `.call-flash`,
`.call-banner`, `.link-status`, `.pair-code`, `.mini-court`, phases `p-pair`, `p-live`.

- [ ] **Step 1: Read the surrounding conventions**

Read §8 (component library), §13 (states table), §16 (screen blueprints), §17
(extension process), and the existing `p-live` placeholder (~line 699). Match the
existing subsection format exactly — each §8 entry documents purpose, anatomy, tokens
used, states, and touch-target sizes.

- [ ] **Step 2: Add the §8 component subsections**

Add, in §8's existing style:

- **Call flash** (`.call-flash`) — full-stage verdict-coloured wash, uppercase word
  (`IN` / `OUT` / `DOWN` / `NO CALL`), 
  `Theme.inCall`/`Theme.outCall` for verdicts, `Theme.unknown` for no-call. Duration
  ≤ 500 ms per §18. Occupies the stage; does not displace layout (§0.9).
- **Call banner** (`.call-banner`) — persistent one-line result under the stage carrying
  the honest state: `IN · high confidence` / `IN · one view` / `NO CALL · obstructed`.
  Confidence is always a text label, never colour alone (§0.3).
- **Link status** (`.link-status`) — pairing/link state row. **Must not use green/red**
  (§0.3 reserves them for verdicts): use `--dim` text plus an explicit word
  (`Paired` / `Syncing` / `Link lost`).
- **Pair code** (`.pair-code`) — 4-digit confirmation code, tabular numerals (§0.8),
  large enough to read across a court.
- **Mini court** (`.mini-court`) — post-rally 3D trajectory over a court outline;
  extends §8.10's court-visualisation family rather than inventing a new one.

- [ ] **Step 3: Add the phase entries**

Add `p-pair` as a real phase (it does not exist today) and promote `p-live` from
placeholder to a real blueprint, each following §16's format: purpose, single primary
action (proxied to the header pill per §3.4), the components used, and the state table
per §13. `p-pair`'s primary is `PAIR`; `p-live`'s primary is `START RALLY`.

- [ ] **Step 4: Note the deliberate deviation, if any**

If any entry needs to bend a §0 rule, state it explicitly in the same edit with the
reason — CLAUDE.md forbids silent drift. If nothing bends, say so in the commit body.

- [ ] **Step 5: Commit**

```bash
git add DESIGN.md
git commit -m "design: live-call, link-status and mini-court components; p-pair and p-live phases"
```

---

### Task 2: SyntheticBallDetector — model-free detection source (DEBUG)

The gate that makes the next two days productive. `BallTrackerTests.ScriptedDetector` is
the proven template; this promotes the idea into an app-buildable DEBUG source that
generates a *plausible moving ball* rather than a canned array.

**Files:**
- Create: `ios/Sources/Live/SyntheticBallDetector.swift`
- Modify: `ios/Sources/Record/RecordModel.swift` (add `detectorKind`)
- Test: `ios/Tests/SyntheticDetectorTests.swift`

**Interfaces:**
- Consumes: `BallDetecting` (`detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation?`), `BallObservation(timestamp:rect:confidence:)` (Vision-normalized, **bottom-left origin**).
- Produces:
  ```swift
  #if DEBUG
  /// Deterministic ballistic arc in Vision-normalized space. Ignores the pixel
  /// buffer entirely — it exists so the live path can be exercised before the
  /// Core ML model lands.
  final class SyntheticBallDetector: BallDetecting {
      init(period: TimeInterval = 2.0, ballSize: Double = 0.012)
      func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation?
  }
  #endif
  ```
  Motion: `u` sweeps 0.1→0.9 linearly over `period`; `v` follows a gravity arc
  `0.8 - 4·h·(t/period)·(1 - t/period)` clamped to `[0.02, 0.98]`. Confidence fixed 0.9.
  Deterministic in `timestamp` alone (no stored phase, no RNG) so tests are exact.
- Produces on `RecordModel`:
  ```swift
  enum DetectorKind: Equatable { case none, synthetic, model }
  @Published private(set) var detectorKind: DetectorKind
  ```
  `detectorMissing` stays as-is (other code depends on it). `detectorKind` exists because
  `detectorMissing` cannot distinguish "no model" from "synthetic stand-in", and the UI
  must never imply a real detection when it is a fixture.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/SyntheticDetectorTests.swift
import XCTest
@testable import SquashLineCalling

#if DEBUG
final class SyntheticDetectorTests: XCTestCase {
    private func buffer() -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &pb)
        return pb!
    }

    func testProducesObservationAtEveryTimestamp() {
        let d = SyntheticBallDetector()
        for t in stride(from: 0.0, through: 2.0, by: 0.25) {
            XCTAssertNotNil(d.detect(buffer(), timestamp: t), "no observation at \(t)")
        }
    }

    func testDeterministicForTheSameTimestamp() {
        let a = SyntheticBallDetector().detect(buffer(), timestamp: 0.7)
        let b = SyntheticBallDetector().detect(buffer(), timestamp: 0.7)
        XCTAssertEqual(a, b)
    }

    func testStaysInsideTheNormalizedFrame() {
        let d = SyntheticBallDetector()
        for t in stride(from: 0.0, through: 6.0, by: 0.05) {
            let r = d.detect(buffer(), timestamp: t)!.rect
            XCTAssertGreaterThanOrEqual(r.minX, 0); XCTAssertLessThanOrEqual(r.maxX, 1)
            XCTAssertGreaterThanOrEqual(r.minY, 0); XCTAssertLessThanOrEqual(r.maxY, 1)
        }
    }

    func testSweepsAcrossTheFrameAndArcs() {
        let d = SyntheticBallDetector(period: 2.0)
        let start = d.detect(buffer(), timestamp: 0.0)!.rect.midX
        let mid   = d.detect(buffer(), timestamp: 1.0)!.rect
        let end   = d.detect(buffer(), timestamp: 1.99)!.rect.midX
        XCTAssertLessThan(start, end, "ball should travel across frame")
        // Vision origin is bottom-left, so the arc's apex is a LARGER v.
        XCTAssertGreaterThan(mid.midY, d.detect(buffer(), timestamp: 0.0)!.rect.midY)
    }

    func testDetectorKindReportsSynthetic() {
        let model = RecordModel(detector: SyntheticBallDetector())
        XCTAssertEqual(model.detectorKind, .synthetic)
        XCTAssertFalse(model.detectorMissing)
    }

    func testDetectorKindReportsNoneWhenNil() {
        XCTAssertEqual(RecordModel(detector: nil).detectorKind, .none)
    }
}
#endif
```

- [ ] **Step 2: Run to verify failure** — `cannot find 'SyntheticBallDetector'`.

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Live/SyntheticBallDetector.swift
#if DEBUG
import CoreVideo
import Foundation

/// A deterministic ballistic arc in Vision-normalized space (origin
/// bottom-left). Ignores the pixel buffer: it exists so the pairing, live-call
/// and replay surfaces can be exercised before the Core ML model exists.
/// Never compiled into a release build.
final class SyntheticBallDetector: BallDetecting {
    private let period: TimeInterval
    private let ballSize: Double

    init(period: TimeInterval = 2.0, ballSize: Double = 0.012) {
        self.period = max(0.1, period)
        self.ballSize = ballSize
    }

    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        let phase = (timestamp.truncatingRemainder(dividingBy: period)) / period
        let u = 0.1 + 0.8 * phase
        // Apex mid-flight; Vision's y grows upward, so the arc peaks high.
        let v = min(0.98, max(0.02, 0.2 + 4.0 * 0.6 * phase * (1.0 - phase)))
        let rect = CGRect(x: u - ballSize / 2, y: v - ballSize / 2,
                          width: ballSize, height: ballSize)
        return BallObservation(timestamp: timestamp, rect: rect, confidence: 0.9)
    }
}
#endif
```

In `RecordModel`, add next to `detectorMissing`:

```swift
    enum DetectorKind: Equatable { case none, synthetic, model }

    /// `detectorMissing` only knows nil-vs-non-nil, so it cannot tell a real
    /// model from the DEBUG stand-in. The UI must never imply a real detection
    /// when the source is synthetic.
    @Published private(set) var detectorKind: DetectorKind
```

and set it in `init(detector:)` — `.none` when nil; `.synthetic` when the detector is a
`SyntheticBallDetector` (guard the check with `#if DEBUG`); `.model` otherwise.

- [ ] **Step 4: Run tests** — 6/6, then full suite (expect 108).

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Live/SyntheticBallDetector.swift ios/Sources/Record/RecordModel.swift ios/Tests/SyntheticDetectorTests.swift
git commit -m "feat(live): synthetic ball detector for model-free live-path work"
```

---

### Task 3: Landscape capture for mounted sessions

The mount holds the phone landscape; the app is portrait-locked. This is a real capture
change, not a relayout. Keep portrait working — handheld recording still uses it.

**Files:** Modify `ios/project.yml`, `ios/Sources/Record/CameraController.swift`,
`ios/Sources/Record/OverlayView.swift`, `ios/Sources/Record/CaptureSettings.swift`.
Test: `ios/Tests/CaptureOrientationTests.swift`.

**Interfaces:**
- Produces on `CaptureSettings`:
  ```swift
  enum CaptureOrientation { case portrait, landscapeRight }
  /// Rotation applied to the video connection: 90 for portrait, 0 for landscape.
  static func rotationAngle(for: CaptureOrientation) -> CGFloat
  /// Frame geometry in the orientation's own space.
  static func frameSize(for: CaptureOrientation) -> (width: Int, height: Int)
  ```
  `frameSize(for: .portrait)` = `(2160, 3840)` (today's values, unchanged);
  `.landscapeRight` = `(3840, 2160)`.
- `CameraController` gains `var orientation: CaptureOrientation = .portrait`, applied
  where `videoRotationAngle = 90` is currently hardcoded, and to the asset writer's
  width/height keys.
- `OverlayView.contentSize` becomes a function of the active orientation instead of a
  static.

**Landmine — do not skip:** the peer `Hello` advertises `CaptureSettings.frameWidth/
Height`, and detection tuples are in that space. **Two phones in different orientations
would silently produce garbage 3D.** `PeerSession` must reject a peer whose advertised
frame size disagrees with the local one, with a `failed` reason naming it. Add that here.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/CaptureOrientationTests.swift
import XCTest
@testable import SquashLineCalling

final class CaptureOrientationTests: XCTestCase {
    func testPortraitMatchesTodaysGeometry() {
        let s = CaptureSettings.frameSize(for: .portrait)
        XCTAssertEqual(s.width, CaptureSettings.sensorHeight)
        XCTAssertEqual(s.height, CaptureSettings.sensorWidth)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .portrait), 90)
    }

    func testLandscapeIsSensorNativeAndUnrotated() {
        let s = CaptureSettings.frameSize(for: .landscapeRight)
        XCTAssertEqual(s.width, CaptureSettings.sensorWidth)
        XCTAssertEqual(s.height, CaptureSettings.sensorHeight)
        XCTAssertEqual(CaptureSettings.rotationAngle(for: .landscapeRight), 0)
    }

    func testMismatchedPeerFrameSizeIsRejected() {
        let pair = LoopbackTransport.pair()
        // Corrupt the incoming hello's frame size to simulate a peer in the
        // other orientation: same pixels, transposed — silently fatal for 3D.
        pair.1.controlDeliveryHook = { frame, deliver in
            guard case .hello(var h)? = ControlMessage.decode(frame) else { return deliver(frame) }
            (h.frameW, h.frameH) = (h.frameH, h.frameW)
            deliver(try! ControlMessage.encode(.hello(h)))
        }
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        guard case .failed(let why) = primary.phase else {
            return XCTFail("expected failure, got \(primary.phase)")
        }
        XCTAssertTrue(why.lowercased().contains("orientation")
                      || why.lowercased().contains("frame"), "unhelpful reason: \(why)")
    }
}
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — add the `CaptureOrientation` API; thread `orientation`
through `CameraController` (rotation angle + writer dimensions); make
`OverlayView.contentSize` orientation-derived; in `PeerSession`'s `.hello` handler,
after the version check, compare `theirs.frameW/frameH` to the local
`CaptureSettings.frameWidth/Height` and `setPhase(.failed(...))` on mismatch with a
message naming orientation. In `ios/project.yml` add landscape-right to
`UISupportedInterfaceOrientations` (keep portrait; the ASC 90474 `UIRequiresFullScreen`
note stays).

- [ ] **Step 4: Run tests** (expect 111) — and confirm the existing `PeerSessionTests`
still pass, since the hello path changed.

- [ ] **Step 5: Commit**

```bash
git add ios/project.yml ios/Sources/Record/CaptureSettings.swift ios/Sources/Record/CameraController.swift ios/Sources/Record/OverlayView.swift ios/Sources/Peer/PeerSession.swift ios/Tests/CaptureOrientationTests.swift
git commit -m "feat(capture): landscape orientation for mounted sessions, with a peer frame-space guard"
```

---

### Task 4: PairingModel + p-pair screen

**Files:** Create `ios/Sources/Live/PairingModel.swift`, `ios/Sources/Live/PairingView.swift`.
Test: `ios/Tests/PairingModelTests.swift`.

**Interfaces:**
- Consumes: `PeerSession` (`phase`, `start()`, `confirmPairing()`, `goLive()`, `end()`, `tick(now:)`, `clockSync`), `BLETransport`/`WiFiP2PTransport`.
- Produces:
  ```swift
  @MainActor
  final class PairingModel: ObservableObject {
      enum Step: Equatable {
          case idle, searching, confirm(code: String), syncing
          case ready(syncMs: Double), live, degraded(String), failed(String)
      }
      @Published private(set) var step: Step = .idle
      @Published private(set) var statusLine: String = "Not paired"
      var canConfirm: Bool { if case .confirm = step { return true }; return false }
      init(session: PeerSession?)                 // nil = unpaired single-camera mode
      func start(); func confirm(); func goLive(); func end()
      func pump(now: TimeInterval)                // maps session.phase -> Step
  }
  ```
  `statusLine` is the §8 link-status text — words, never colour alone: `Not paired`,
  `Looking for the other phone…`, `Codes match?`, `Syncing clocks…`,
  `Paired · sync ±1.4 ms`, `Link lost — still recording`, plus the failure reason.
  `.ready` carries `syncMs` from `clockSync.estimate?.uncertainty * 1000` so the screen
  can show sync quality (tabular numerals, §0.8).

- [ ] **Step 1: Write the failing tests** — drive two `PairingModel`s over
`LoopbackTransport` exactly as `PeerSessionTests` does: assert `.confirm` carries equal
4-digit codes on both sides; that `confirm()` on both advances to `.ready` with a finite
`syncMs`; that a silenced link moves to `.degraded` with a status line containing "still
recording"; that a version-mismatch peer lands in `.failed`; and that
`PairingModel(session: nil).step == .idle` with `statusLine == "Not paired"`.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `PairingModel`** — a pure mapping from `PeerSession.Phase` to
`Step` plus the status strings. No new networking. `pump(now:)` calls
`session.tick(now:)` then re-reads `phase`.

- [ ] **Step 4: Implement `PairingView`** — per the DESIGN.md entries from Task 1: link
status row, pair-code display when confirming, single primary action (`PAIR` →
`CONFIRM` → `START RALLY`) proxied per §3.4, 48 pt primary, uppercase labels. No green
dot. Include the DEBUG transport picker (BLE / Wi-Fi) reusing `PeerBenchView`'s pattern.

- [ ] **Step 5: Run tests** (expect ~117) and commit

```bash
git add ios/Sources/Live/PairingModel.swift ios/Sources/Live/PairingView.swift ios/Tests/PairingModelTests.swift
git commit -m "feat(live): pairing model and p-pair screen"
```

---

### Task 5: Live call rendering + optional announcement

**Files:** Create `ios/Sources/Live/LiveCallView.swift`, `ios/Sources/Live/CallAnnouncer.swift`.
Test: `ios/Tests/LiveCallTests.swift`.

**Interfaces:**
- Consumes: `StereoImpact` (`tS`, `surface`, `pointFt`, `call`, `marginFt`, `confidence`, `snapDisagreementFt`), `Theme`.
- Produces:
  ```swift
  /// Presentation-only mapping from an impact to what the screen says.
  /// Pure and synchronous so it is unit-testable without any view.
  struct CallPresentation: Equatable {
      let word: String          // "IN" | "OUT" | "DOWN" | "NO CALL"
      let detail: String        // "high confidence" | "one view" | "obstructed"
      let isVerdict: Bool       // false => use Theme.unknown, not green/red
      static func from(_ impact: StereoImpact) -> CallPresentation
  }
  struct LiveCallView: View { init(presentation: CallPresentation?) }
  @MainActor final class CallAnnouncer {
      var isEnabled: Bool           // default FALSE — spec leaves this open
      func announce(_ p: CallPresentation)
  }
  ```
  Mapping rules: `confidence == "no_call"` → word `NO CALL`, detail `obstructed`,
  `isVerdict false`. `confidence == "one_view"` → the call word, detail `one view`,
  `isVerdict true`. `high` → call word, `high confidence`. `call == "bounce"` renders as
  `NO CALL`/`floor bounce` with `isVerdict false` — a floor bounce is not a line verdict.
- `CallAnnouncer` uses `AVSpeechSynthesizer`, defaults **off** (the spec records this as
  an undecided product choice — do not silently default it on), and never speaks a
  non-verdict.

- [ ] **Step 1: Write the failing tests** — a `CallPresentation.from` case per
confidence tier and for `bounce`, asserting word/detail/isVerdict; and that the
announcer stays silent when disabled and for non-verdicts.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `LiveCallView` is the §8 call-flash + call-banner pair:
full-stage colour wash (`Theme.inCall`/`Theme.outCall`/`Theme.unknown`) with the
uppercase word, ≤ 500 ms (§18), and a **fixed-height** banner beneath so nothing shifts
when a call appears (§0.9). Respect `UIAccessibility.isReduceMotionEnabled` — no flash
animation when set, just the state change.

- [ ] **Step 4: Run tests** (expect ~123) and commit

```bash
git add ios/Sources/Live/LiveCallView.swift ios/Sources/Live/CallAnnouncer.swift ios/Tests/LiveCallTests.swift
git commit -m "feat(live): verdict flash, honest-state banner, and opt-in announcements"
```

---

### Task 6: Wire the live path together

`attachPeer`/`attachStereo` still have **no callers**. This task gives them one and makes
the screens real — behind a DEBUG demo entry so it is exercisable without a second phone.

**Files:** Modify `ios/Sources/Record/RecordModel.swift`, `ios/Sources/Record/RecordView.swift`,
`ios/Sources/RootTabView.swift`. Test: `ios/Tests/LiveWiringTests.swift`.

**Interfaces:**
- Produces on `RecordModel`:
  ```swift
  @Published private(set) var livePresentation: CallPresentation?   // nil = nothing to show
  /// Builds a StereoEngine directly from two camera models and drives it with
  /// synthetic detections — no peer, no model, one device. DEBUG only.
  #if DEBUG
  func startStereoDemo(localModelJSON: String, remoteModelJSON: String)
  #endif
  ```
  The existing `stereoEvents` JSON list stays (the bench reads it). `livePresentation` is
  set on the main queue from the engine's `onEvent`, cleared after the flash window.
- `RootTabView` gains a DEBUG "Stereo demo" entry beside the existing peer-bench button,
  using the same `#if DEBUG` sheet pattern.

- [ ] **Step 1: Write the failing tests** — using the two camera models from
`stereo_goldens.json` (available to the **test** target), assert that feeding
`startStereoDemo` a golden trajectory produces a `livePresentation` whose word/detail
match the golden impact's call and confidence; and that `livePresentation` is nil before
any event. Reuse `StereoEngineTests`' fixture-loading helper.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `startStereoDemo` constructs `StereoEngine` directly
(mirroring `StereoEngineTests`, **not** spoofing a `PeerSession` handshake), feeds it
the synthetic trajectory on the existing pump timer, and maps `onEvent` →
`CallPresentation` → `livePresentation` with a main hop. In `RecordView`, overlay
`LiveCallView(presentation: model.livePresentation)` on the stage and show the
`detectorKind == .synthetic` notice so a fixture is never mistaken for a real detection.

- [ ] **Step 4: Run tests** (expect ~126) and commit

```bash
git add ios/Sources/Record/RecordModel.swift ios/Sources/Record/RecordView.swift ios/Sources/RootTabView.swift ios/Tests/LiveWiringTests.swift
git commit -m "feat(live): wire pairing, engine and call rendering into the record path"
```

---

### Task 7: Post-rally mini-court replay

**Files:** Create `ios/Sources/Live/MiniCourtView.swift`. Test: `ios/Tests/MiniCourtTests.swift`.

**Interfaces:**
- Consumes: `TrackPoint3D` (`tS`, `pointFt`, `gapFt`), `StereoMath` court constants
  (`courtWidthFt` 21, `courtLengthFt` 32, `outLineHeightFt` 15, `tinTopHeightFt` 19/12).
- Produces:
  ```swift
  /// Court-feet -> unit-square projection for the replay. Pure, so the
  /// geometry is testable without rendering.
  enum CourtProjection {
      /// Side elevation: x = depth along court (y_ft), y = height (z_ft).
      static func sideElevation(_ p: SIMD3<Double>) -> CGPoint
      /// Plan view from above: x = width (x_ft), y = depth (y_ft).
      static func planView(_ p: SIMD3<Double>) -> CGPoint
  }
  struct MiniCourtView: View { init(track: [TrackPoint3D], impact: StereoImpact?) }
  ```
  Both projections return points in `[0,1]²` with the court's own extremes mapping to the
  edges. Impact marker uses the verdict colour; the trajectory uses `Theme.dim`. Extends
  §8.10's court family per Task 1's DESIGN.md entry.

- [ ] **Step 1: Write the failing tests** — assert the four court corners map to the unit
square's corners in `planView`; that floor (`z=0`) and the front-wall out line
(`z=15`) map to the expected `y` in `sideElevation`; and that a point outside the court
clamps rather than escaping `[0,1]`.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `CourtProjection` plus a `Canvas`-based `MiniCourtView`
drawing the court outline, the trajectory polyline, and the impact marker. No animation
beyond §10's motion budget.

- [ ] **Step 4: Run tests** (expect ~130) and commit

```bash
git add ios/Sources/Live/MiniCourtView.swift ios/Tests/MiniCourtTests.swift
git commit -m "feat(live): post-rally mini-court replay"
```

---

### Task 8: Verification pass

**Files:** Modify `ios/PEER.md` (live-path runbook section). No new source.

- [ ] **Step 1:** Run the full suite; record the number.
- [ ] **Step 2:** Use the `/verify` skill to check `p-pair` and `p-live` at a
  390 × 844 phone viewport in **both themes** (§0.12). Capture screenshots.
- [ ] **Step 3:** Drive the DEBUG stereo demo end to end in the simulator; confirm the
  call flash, the honest-state banner, the synthetic-source notice, and that nothing in
  the layout shifts when a call appears (§0.9).
- [ ] **Step 4:** Re-read the §0 rules and §18's never-do list against the new screens;
  fix anything that drifted, or document the deviation in DESIGN.md.
- [ ] **Step 5:** Add a "Live path" section to `ios/PEER.md` covering the two-phone
  bring-up and what still needs hardware (real camera, ANE model, on-court pairing).
- [ ] **Step 6: Commit.**

---

## User-owned checklist (needs hardware)

- Two phones on the mounts: real pairing over BLE and Wi-Fi, landscape, on court.
- A 4K calibration per phone (or a scaled older one — the adoption path handles it).
- The trained YOLOX model + ANE performance report before live calls mean anything.
- Decide the two open product questions: **audible calls default on or off**, and
  whether serve-depth `estimated` calls render as calls or analytics-only.

## Self-review notes

- **Spec coverage:** p-pair (Task 4), p-live (Tasks 5–6), mini-court replay (Task 7),
  landscape capture (Task 3), DESIGN.md additions (Task 1, gating per §17.1). Audible
  calls are built but default off, since the spec records that choice as open.
- **Deliberately deferred:** live video preview between phones, player tracking, and
  shot classification are spec non-goals.
- **Biggest risk:** Task 3's peer frame-space guard. Two phones in different orientations
  produce transposed pixel spaces that triangulate to plausible-looking nonsense. The
  guard is cheap; discovering it on court is not.
- **Type consistency check:** `CallPresentation` is produced in Task 5 and consumed in
  Task 6; `DetectorKind` in Task 2 is consumed by Task 6's UI notice; `CourtProjection`
  is self-contained in Task 7. `StereoImpact` field names match `StereoTrack.swift`.
