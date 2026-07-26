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
3. Primary phone: Run 60 s datagram bench (120 Hz) — originates synthetic traffic
   batched 2 tuples/datagram at 60 Hz (120 Hz tuple rate, matching production's
   batched streaming) and measures RTT/loss. Secondary phone: also press Run — its
   report captures sync offset, thermal, and reflected-echo count. Rally on court
   during the 60 s for realism.
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
- Capture is 4K60 landscape (3840x2160) in both mounts; `CaptureSettings.rotationAngle`
  normalizes landscape-left upright with 180. **Both phones must run the same mount** —
  `PeerSession` enforces it at handshake by comparing the peer's advertised frame size
  AND `captureOrientation` against its own `myHello`, and refuses the pair otherwise.
  A peer on a build predating the `captureOrientation` field advertises none, which cannot
  match either mount and is refused with the same message. `peerProtoVersion`
  deliberately stays at 1 so those peers reach this specific, actionable error instead of
  a generic version mismatch.
- The capture side **is** wired to this on the live path:
  `LiveSessionModel.beginPairing()` builds its `PeerSession` with
  `record.camera.orientation`, so each phone advertises the mount it is actually
  capturing in. Two caveats, both real:
  - The mount is read at *pairing* time, when only `startCamera`'s unpinned initial
    guess exists. `RecordModel.applyRecording`'s start path re-resolves and pins the
    mount at record start, and that later value is **not** re-advertised — so a phone
    flipped between CONFIRM and the first START RALLY captures in one mount while
    `Hello` still claims the other. The Play tab is landscape-locked throughout, so
    this is a flip between the two landscapes, not a portrait excursion.
  - `PeerBenchView`'s DEBUG session still takes the `.landscapeRight` default. It
    never records, so nothing is mislabelled; it just means the bench cannot
    exercise the mismatch path.
- Detections flow one way (secondary → primary); events flow back in Phase 3.

## Live path (spec Phase 4)

`LiveSessionModel` owns the whole live lifecycle — the calibration gate, the
role choice, the `PeerSession`, the rally, and the paired upload — and
publishes DESIGN.md §16's `p-pair` table as flat view state
(`linkStatus`/`primaryTitle`/`primaryEnabled`/`showsStop`), so the table is
assertable without a view. `PairingView` (`p-pair`) and `LiveStageView`
(`p-live`) are thin renderers over it; `PairingModel` still does the pure
`PeerSession.Phase` → step mapping underneath. `PlayRootView` owns
`RecordModel` and presents `p-live` for exactly as long as a rally is running
— driven by the rally, not by which screen started it. Recording never depends
on the link: a dropped link shows "Link lost — still recording" and keeps
writing locally.

### Bring-up without hardware

There is a DEBUG stereo demo — the cube button on `p-record` (Play → **Record a
clip**). The peer bench is no longer next to it: it moved to the Play root's
toolbar when `PlayRootView` took over the tab. The demo builds a `StereoEngine`
from the validated golden camera pair and drives it with a court-feet
trajectory projected through both models, so the full engine →
`CallPresentation` → flash/banner path runs on one device with no peer and no
ball model. It should resolve to `IN · high confidence`.

That demo exercises everything except the radio and the camera. What it cannot
tell you is whether the link, the clock sync, or a real detector work.

### Two-phone bring-up (needs the hardware)

1. Same build on both phones, both in the **same** landscape mount — both
   lenses at the same end. Capture is landscape-only now and the Play tab is
   locked to it (`OrientationLock`), so the only way to get this wrong is to
   mount one phone landscape-left and the other landscape-right. A mismatched
   pair fails the handshake outright (see "Known limits"), and the reason lands
   on the status row verbatim. Set the mount **before** tapping PAIR: the mount
   that goes on the wire is the one resolved at pairing time.
2. A 4K **landscape** calibration per phone, or an older one the adoption path
   can scale — note that adoption refuses an aspect-ratio change rather than
   distorting the model, so a portrait (9:16) profile will not load at all now
   that capture targets 3840×2160. That is a change from before: portrait
   profiles used to be the ones that worked.
3. Both phones: Play tab → **Live match**. Entering `p-pair` fetches *this*
   phone's own solved camera model and validates that it can be adopted for
   capture. While that is in flight the row reads "Checking this phone's court
   calibration…" and PAIR is disabled; if it fails, the reason shows verbatim
   and PAIR stays disabled. Deliberately at the gate — an unusable calibration
   must fail before two people have walked to opposite corners.
4. **Pick each phone's job, before PAIR.** One phone taps `THIS PHONE CALLS`
   (the primary: it holds the `StereoEngine` and makes the calls), the other
   `THIS PHONE ASSISTS` (the secondary: it streams detections, and its own
   calibration, to the primary). There is no default — the row reads "Pick this
   phone's job to start." and PAIR stays disabled until each phone has chosen.
   Check both phones before tapping: two set to "calls" would both open a
   central and never find each other. The role segment is visible in the idle
   state only, and the choice is fixed for the life of the session.
5. PAIR on both, compare the 4-digit code, CONFIRM on both. "Codes don't
   match" ends the session and returns to idle — use it, that is the
   stranger's-phone case.
6. Wait for `Paired · sync ±x ms`. If x does not settle under 5 ms the session
   will not leave syncing; that is the gate doing its job, not a hang.
7. **START RALLY lights on the calling phone only, and only once the assisting
   phone's calibration has arrived.** The assisting phone has no primary action
   at all — its row reads "Paired · the other phone starts the rally". The
   calling phone reads "Paired · waiting for the other phone's calibration"
   with START RALLY disabled until that message lands and builds its
   `StereoEngine`; a rally started without it would record perfectly and be
   callable by nothing. The secondary sends that message on reaching `.ready`
   and retries every pump turn until it goes out, so if the row never clears,
   the link is the problem — end the session and pair again.
8. START RALLY. Both phones show `p-live` — the assisting one with no tap on
   it at all, since its rally starts from a control message. While the rally
   runs there is no back button: STOP is the only exit. The calling phone
   always has STOP; the assisting phone gains its own only while the link is
   degraded ("Link lost — still recording"), so a dropped link cannot leave it
   recording 4K60 with nothing able to end it.
9. STOP. Each phone uploads its own clip independently under the same
   **per-rally** `session_id` (`camera_role` "a" on the calling phone, "b" on
   the assisting one); the server starts a `stereo-<session_id>` fuse run once
   both runs complete. `p-live` pops back to `p-pair` with the pairing intact —
   same clock sync, no second code — and START RALLY re-arms for the next rally
   once that clip reports. A session runs as many rallies as the match needs;
   only "Codes don't match" or ending the session returns to idle.

### Still needs hardware before any of this means anything

- A real `BallDetector.mlpackage` in `ios/Model/` — the directory ships empty.
- The Phase 1 transport selection bench (the table above is still blank,
  `hardware/bench/` is empty) and the thermal go/no-go at 4K60.
- The Phase 2 ≤ 2 ms sync assertion against a clap anchor.
- Rolling-shutter correction is **not implemented**: `DetectionTuple.y` carries
  the row for it, but nothing applies `t_effective = PTS + (row/H)·readout`.
  Until it is, the sync budget is optimistic for fast cross-frame motion.
- The live path above has only ever run over `LoopbackTransport` in
  `LiveSessionModelTests`. The calibration exchange, the per-rally session
  identity, and the server-side fuse it feeds have never crossed a real radio.
- The mount guard has never been tried on two real phones. It is a genuine gate
  now — the live path advertises `record.camera.orientation` and the guard
  compares mounts explicitly — but the only coverage is `CaptureOrientationTests`
  and `LiveSessionModelTests` over `LoopbackTransport`. Deliberately mis-mount
  one phone during bring-up and confirm both refuse.
- Nothing re-advertises the mount after pairing (see "Known limits"), so the
  flip-after-CONFIRM case is untested and, today, undetected.
- The Swift suite itself needs a Mac: `xcodebuild`/`xcodegen` do not exist on
  the Windows dev box, so nothing in `ios/` is compiled there. See
  `docs/superpowers/plans/2026-07-25-ios-live-entry-point-MAC-CHECKLIST.md`
  for what a Mac still owes this change.
