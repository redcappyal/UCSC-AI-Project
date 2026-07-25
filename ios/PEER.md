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
- Capture is 4K60: portrait 2160×3840 handheld, landscape 3840×2160 for a
  mounted session (`CaptureSettings.CaptureOrientation`). **Both phones must
  run the same orientation** — mixed orientations give transposed pixel
  spaces that triangulate to plausible-looking nonsense.
- ⚠️ **The frame-space guard does not currently catch that.** `PeerSession`
  compares `theirs.frameW/frameH` against the *static* portrait constants
  `CaptureSettings.frameWidth/frameHeight`, and builds its own hello from the
  same statics — so the advertised value does not depend on the session's
  orientation and the comparison can never fail on an orientation mismatch.
  `RecordModel.peerFrameW/peerFrameH` are hardcoded to the same statics, so
  detection tuples are labelled portrait even in a landscape session. Until
  both read `CaptureSettings.frameSize(for:)`, treat "both phones in the same
  orientation" as an operator responsibility, not something the app enforces.
- Detections flow one way (secondary → primary); events flow back in Phase 3.

## Live path (spec Phase 4)

`p-pair` (`PairingView` + `PairingModel`) maps `PeerSession.Phase` onto the
DESIGN.md §16 state table; `p-live` is the record stage plus the §8.17 call
flash and the §8.18 call banner. Recording never depends on the link: a
dropped link shows "Link lost — still recording" and keeps writing locally.

### Bring-up without hardware

There is a DEBUG stereo demo — the cube button on the record stage, under the
peer-bench button. It builds a `StereoEngine` from the validated golden camera
pair and drives it with a court-feet trajectory projected through both models,
so the full engine → `CallPresentation` → flash/banner path runs on one device
with no peer and no ball model. It should resolve to `IN · high confidence`.

That demo exercises everything except the radio and the camera. What it cannot
tell you is whether the link, the clock sync, or a real detector work.

### Two-phone bring-up (needs the hardware)

1. Same build on both phones, both mounted in the **same** orientation.
2. A 4K calibration per phone, or an older one the adoption path can scale —
   note that adoption refuses an aspect-ratio change rather than distorting
   the model, so a portrait profile will not load for a landscape session.
3. Pair: both phones to `p-pair`, PAIR on both, compare the 4-digit code,
   CONFIRM on both. "Codes don't match" ends the session — use it, that is the
   stranger's-phone case.
4. Wait for `Paired · sync ±x ms`. If x does not settle under 5 ms the session
   will not leave syncing; that is the gate doing its job, not a hang.
5. START RALLY.

### Still needs hardware before any of this means anything

- A real `BallDetector.mlpackage` in `ios/Model/` — the directory ships empty.
- The Phase 1 transport selection bench (the table above is still blank) and
  the thermal go/no-go at 4K60.
- The Phase 2 ≤ 2 ms sync assertion against a clap anchor.
- Rolling-shutter correction is **not implemented**: `DetectionTuple.y` carries
  the row for it, but nothing applies `t_effective = PTS + (row/H)·readout`.
  Until it is, the sync budget is optimistic for fast cross-frame motion.
