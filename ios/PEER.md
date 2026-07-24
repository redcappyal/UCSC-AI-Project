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
- Portrait 1080×1920 capture (landscape lands with spec Phase 4).
- Detections flow one way (secondary → primary); events flow back in Phase 3.
