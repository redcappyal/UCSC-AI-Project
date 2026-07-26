# Landscape Capture and a Peer Frame-Space Guard That Can Fail — Design

**Date:** 2026-07-24
**Status:** Approved in conversation (Ian), pending written-spec review
**Branch:** `claude/capture-orientation-guard`, off `claude/phase4-ui`
**Supersedes:** the orientation half of 968128e ("feat(capture): landscape orientation for
mounted sessions, with a peer frame-space guard")

## Problem

The peer frame-space guard added by 968128e cannot detect the mismatch it was written to
prevent, and the landscape capture it was written to protect is unreachable.

`PeerSession` guards an incoming hello with
`theirs.frameW == CaptureSettings.frameWidth, theirs.frameH == CaptureSettings.frameHeight`
(PeerSession.swift:251-253), and builds its *own* hello from those same two statics
(PeerSession.swift:104-105). Those statics are fixed at compile time to the portrait
geometry — `frameWidth = sensorHeight` (2160), `frameHeight = sensorWidth` (3840)
(CaptureSettings.swift:45-46). The orientation-dependent geometry lives in
`frameSize(for:)` (CaptureSettings.swift:71-72), which the handshake never calls.

So every phone advertises the portrait constants no matter how it is capturing, and the
comparison is a value against itself. Two landscape phones pass the guard while streaming
detections in a 3840×2160 space labelled 2160×3840; a portrait/landscape pair also passes.
Either way stereo triangulates transposed coordinates into confident, silently wrong
IN/OUT calls — exactly what CaptureSettings.swift:50-54 claims PeerSession enforces
against. `RecordModel.peerFrameW/peerFrameH` (RecordModel.swift:79-80) read the same
statics, so `DetectionMapper.tuple(...)` and `StereoEngine.addLocalObservation(...)` label
every tuple portrait in every session.

**Found while investigating, and larger than the above:** nothing in the codebase ever
assigns `CameraController.orientation`. It is a `var` defaulting to `.portrait`
(CameraController.swift:25) and `git grep "orientation\s*=" ios/Sources/` returns nothing.
`RecordView` also constructs `OverlayView(trail:)` without an orientation, so the overlay
defaults to portrait too. `CaptureOrientationTests` passes green because it tests
`frameSize(for:)` and `rotationAngle(for:)` as pure arithmetic — never that anything calls
them with `.landscapeRight`. Landscape capture has never run.

That is why no regression test could be written for the guard: there is no second
orientation in production for the advertised value to vary to. The lock and the guard fix
are one change, not two.

`ios/PEER.md:61-70` already documents the guard as broken rather than claiming it works;
this design makes the docs true instead of apologetic.

## Decision log (rulings that shaped this design)

1. **Capture is always landscape** (Ian). Portrait capture is dropped, not kept as a
   handheld alternative. `CaptureSettings.frameWidth/frameHeight` become the landscape
   geometry (3840×2160) and portrait stops being a capture mode.
2. **The landscape lock covers the whole Play tab** (Ian), not just the recording window.
   The camera preview is live while the operator aims the phone in its back-wall mount,
   and that is precisely when a sideways preview is most costly. Matches and Coach are
   webviews built as portrait mobile UI and keep rotating freely.
3. **Both landscape orientations are allowed, and the physical orientation is advertised
   in the hello** (Ian). Landscape-left and landscape-right produce identical 3840×2160
   dimensions, so dimensions alone cannot distinguish them — an explicit field is the only
   thing that can. The guard refuses an opposite-facing pair rather than absorbing it.
4. **Capture output is normalized to absolute-upright** (Ian). `rotationAngle(for:)`
   returns 0 or 180 so a recorded frame is upright however the phone sits in the mount.
   Without this, a solo operator mounting landscape-left records upside-down footage with
   no peer to catch it — and that footage is also the training archive. Normalization
   makes the guard's job operator discipline (mount both phones alike) rather than
   geometric necessity, which is the intended reading, not an accident.
5. **The supported-orientation mask pins on capture start** (Ian). Both landscapes are
   selectable while framing; once the camera configures, the mask narrows to the resolved
   orientation so the device cannot flip out from under a live session and leave the
   advertised value stale. No live reconciliation state is needed.
6. **`peerProtoVersion` is NOT bumped.** See "Wire compatibility" — bumping it would
   replace two accurate, actionable errors with one generic one.

## Goals

- The frame-space guard compares two values that can genuinely differ, and a regression
  test proves it fails on a mismatched pair.
- Landscape capture actually runs, with the Play tab locked to landscape.
- Detection tuples are labelled with the pixel space they are actually expressed in.
- No silent failure mode is introduced to replace the one removed.

## Non-goals

- Wiring `attachPeer` to a production caller. `PeerSession` is still constructed only in
  `PeerBenchView` (DEBUG) and `RecordModel.attachPeer` has no production call site;
  completing the live pairing path is separate work and is not started here.
- Any change to stereo math, calibration, or the Python pipeline.
- Changing what the Matches/Coach webviews do.

## Design

### 1. Where the session's orientation lives

New `ios/Sources/Record/OrientationLock.swift` holds the policy as pure functions;
`RecordModel` owns the resolved value and is the single source every consumer reads.

```
OrientationLock.mask(for: RootTab) -> UIInterfaceOrientationMask
    .play            -> .landscape        (both landscape orientations)
    .matches, .coach -> .all
OrientationLock.pinnedMask(for: CaptureOrientation) -> UIInterfaceOrientationMask
    .landscapeRight  -> .landscapeRight
    .landscapeLeft   -> .landscapeLeft
```

`CaptureSettings.captureOrientation(for: UIInterfaceOrientation) -> CaptureOrientation?`
maps an interface orientation to a capture orientation, `nil` for portrait.

Plumbing:

- `RootTabView` gains a `selection` binding and a `RootTab` enum, so the mask is a
  function of the selected tab rather than of `onAppear`/`onDisappear` ordering — which
  SwiftUI does not guarantee across tab switches. Deterministic and testable.
- `SquashApp` gains an `AppDelegate` via `@UIApplicationDelegateAdaptor` (it has none
  today) to serve `application(_:supportedInterfaceOrientationsFor:)` from a small
  observable holder, paired with `setNeedsUpdateOfSupportedInterfaceOrientations()` and
  `UIWindowScene.requestGeometryUpdate` (both available at the iOS 17 deployment target).
- `RecordModel` publishes `captureOrientation`, resolved when `startCamera()` configures
  the session and pinned there per decision 5. It sets `camera.orientation`, derives
  `peerFrameW/H` from `CaptureSettings.frameSize(for:)`, and supplies the value handed to
  `PeerSession.init`.

### 2. CaptureSettings

- `CaptureOrientation` becomes `{ landscapeLeft, landscapeRight }`. Portrait is gone.
- `frameWidth = sensorWidth` (3840), `frameHeight = sensorHeight` (2160). Landscape is now
  the geometry, and the existing doc comment about portrait being "the space every
  consumer works in" is rewritten rather than left to rot.
- `frameSize(for:)` returns (3840, 2160) for **both** cases. This is deliberate and is
  precisely why decision 3 needs an explicit wire field: the dimensions cannot tell the two
  mounts apart.
- `rotationAngle(for:)` returns `0` for `.landscapeRight` and `180` for `.landscapeLeft`,
  normalizing recorded output to absolute-upright (decision 4).
- `CameraController` keeps its `orientation` var — now actually assigned — and continues
  to feed both the video-connection rotation (CameraController.swift:110) and the asset
  writer's frame size (CameraController.swift:225).
- `OverlayView`'s `orientation` parameter is removed: after normalization there is one
  frame space, so the overlay has nothing to branch on. This also fixes the existing
  RecordView call site that silently defaulted it to portrait.

### 3. The wire protocol and the guard

`Hello` gains one field:

```swift
var captureOrientation: CaptureSettings.CaptureOrientation?
```

Optional is load-bearing, not laziness — see "Wire compatibility".

`PeerSession.init` takes `captureOrientation:` (defaulted), builds `myHello` from
`CaptureSettings.frameSize(for:)` plus the orientation, and the guard compares against
**`myHello`** rather than against globals:

```swift
guard theirs.frameW == myHello.frameW,
      theirs.frameH == myHello.frameH,
      theirs.captureOrientation == myHello.captureOrientation else {
    setPhase(.failed("peer camera orientation doesn't match this phone — mount both phones the same way and reconnect"))
    transport.stop()
    return
}
```

Routing the comparison through `myHello` is the structural fix: the advertised value and
the compared value are now the same value, so they cannot drift apart again regardless of
how orientation is later decided. A future change to where orientation comes from cannot
silently re-break the guard.

`RecordModel.peerFrameW/peerFrameH` become computed from the resolved orientation via the
same `frameSize(for:)`, so `DetectionMapper.tuple(...)` and
`StereoEngine.addLocalObservation(frameW:frameH:)` label tuples with the space they are
actually in.

### 4. Wire compatibility (why `peerProtoVersion` stays at 1)

`Hello` is `Codable`, encoded as JSON by `ControlMessage.encode`.

- **New field must be optional.** A required field would make an old peer's hello (which
  lacks the key) throw in `JSONDecoder`; `ControlMessage.decode` returns `Optional` and
  `handleControl` guards `nil` away with no branch — so the message is *silently dropped*
  and pairing hangs in `.searching` with no diagnostic. Optional decodes cleanly, and
  `nil` reads as "peer predates the field" → legacy portrait → rejected with the useful
  message.
- **Old build receiving a new hello already behaves correctly.** `JSONDecoder` ignores
  unknown keys, so an old build decodes a new hello, sees 3840×2160 against its own
  portrait statics, and its otherwise-inert guard fires — because across versions those
  statics genuinely describe its own capture. The tautology only hides same-version
  mismatches.
- **Therefore do not bump `peerProtoVersion`.** The version check runs *first*
  (PeerSession.swift:241); bumping it would reject old peers as a version mismatch before
  the frame check is ever reached, re-killing the guard the moment it is fixed and
  downgrading a specific, actionable message ("mount both phones the same way") to a
  generic one. Two ordered guards are only both reachable if the first can pass while the
  second fails.

### 5. Tests (`ios/Tests/`)

The regression test the guard has been missing:

- `PeerSessionTests`: two `PeerSession`s over `LoopbackTransport`, one `.landscapeLeft`
  and one `.landscapeRight`, both reach `.failed`. This is writable only because the
  advertised value is now injectable and can vary.
- `PeerSessionTests`: a hello with `captureOrientation == nil` (legacy peer) is rejected
  with the orientation message, not dropped and not reported as a version mismatch.
- `PeerSessionTests`: a matched pair still reaches `.confirming` — the guard must not
  become a blanket refusal.
- `CaptureOrientationTests`: `frameSize(for:)` is (3840, 2160) for both cases;
  `rotationAngle(for:)` is 0 / 180; the `UIInterfaceOrientation` mapping returns `nil` for
  portrait.
- New `OrientationLockTests`: `mask(for:)` per tab and `pinnedMask(for:)` per orientation.

All are pure or loopback-driven and need no camera. This matters because the repo has no
macOS CI job — `.github/workflows/tests.yml` is pytest only — so the Swift suite runs
locally:

```
cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling \
  -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'
```

### 6. Docs to update in the same change

- `ios/PEER.md:55-70` — replace the "⚠️ the frame-space guard does not currently catch
  that" limitation with what the guard now enforces, including the legacy-peer behaviour.
- `ios/CAPTURE.md` — portrait is no longer a capture mode; record the landscape geometry
  and the normalization rule.
- `ios/project.yml` — add `UIInterfaceOrientationLandscapeLeft`, and **keep**
  `UIInterfaceOrientationPortrait`. Decision 2 requires Matches and Coach to rotate
  freely, which the app-level list must permit; the plist stays a superset of everything
  any tab allows, and `OrientationLock`'s mask does the per-tab narrowing at runtime. A
  mask can only restrict what the plist already permits, never extend it.

## Risks

- **The mask/pinning code is the least testable part.** `mask(for:)` and
  `pinnedMask(for:)` are pure and covered, but the `requestGeometryUpdate` call itself is
  UIKit behaviour that unit tests cannot assert. It needs a manual simulator check on the
  Play tab in both landscapes plus a tab switch to Matches.
- **`UIRequiresFullScreen: true` and `TARGETED_DEVICE_FAMILY: "1"`** are already set, so
  the iPad multitasking all-orientations requirement (ASC 90474) does not apply. Adding
  landscape-left does not change that, but the plist edit should be re-verified against
  an actual archive before a TestFlight upload.
- **Normalization is asserted, not yet measured.** That `rotationAngle` 180 produces a
  genuinely upright frame on a landscape-left mount is a claim about AVFoundation
  behaviour; it is checked on device during implementation, not taken on faith from this
  document.

## Out of scope, explicitly

Completing the production pairing path (`attachPeer` has no caller;
`PeerSession` is DEBUG-bench-only). This design makes the guard correct and the
orientation real; it does not ship two-phone live pairing.
