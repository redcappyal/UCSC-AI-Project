# In-app recording + on-site calibration — design

2026-07-22. Adds a camera capture flow to the single-file app: record rally clips with
the phone that will later run live-match calling, keep them on the device in the app's
own folder, and run the existing calibration wizard against the live camera view — the
first real exercise of the calibration workflow the live-match mode will need.

## Goals

- A **Record a clip** entry on the Play screen, directly under **Judge a clip**.
- Recordings saved **on-device** in a dedicated app folder, listed in the app for easy
  future reference (re-judge, export, delete).
- **On-site calibration**: run the existing lines → wall corners → floor map wizard on a
  frame captured from the live camera, before recording. The stored calibration rides
  along with each recording and can be re-applied when judging it.
- Zero backend changes. Everything lives in `index.html`.

## Non-goals (deferred)

- Live tracking/calling during recording (that's the Live match roadmap item — this flow
  exists partly to de-risk it).
- Server-side copies of recordings; torch/exposure camera controls; multi-camera pick UI
  beyond preferring the rear camera; background-recording resilience.

## Approaches considered

1. **On-device storage (chosen)** — OPFS `recordings/` directory (real per-origin folder;
   blob + JSON sidecar per clip) with an IndexedDB fallback for browsers without
   `createWritable` (iOS Safari < 18.4). Matches "saved locally on the device, own
   folder", works offline on court.
2. Server-side `recordings/` folder via upload — simpler storage, but requires the Mac on
   court and contradicts "locally on the device".
3. Share-sheet-only (no library) — no persistent in-app reference; rejected.

## UX

### Entry (Play screen, §8.15)

New surface-variant hero card **Record a clip** ("Film a rally with this phone") between
the accent "Judge a clip" card and the "Live match · SOON" card. Surface variant keeps
the one-accent rule; a working surface card is distinguished from a future one by the
absence of the SOON tag.

### `p-record` phase (sub-page of Play; back chevron, no nav pill)

- **Stage** shows the live rear-camera preview (`<video id="camVid">`, object-fit
  contain, black well). `getUserMedia`: environment camera, ideal 1920×1080@60, plus
  audio (the pipeline's audio wall-hit rescue wants it). Falls back to video-only if the
  mic is denied; status line reports actual resolution/fps, or the error.
- **Readout row** (reserved height): record dot + `m:ss.t` clock, tabular. Idle: dim.
  Recording: accent-yellow pulsing dot (red is reserved for OUT verdicts) + `--text`
  clock. Pulse is opacity-only behind `prefers-reduced-motion`.
- **Primary (proxied): Record ↔ Stop.** MediaRecorder, mime preference
  mp4/avc1 → webm/h264 → webm/vp9 → webm, 12 Mbps video, 1 s timeslice.
- **Secondary: Calibrate court** (disabled while recording) + calibration status line
  ("Not calibrated" / "Calibrated · lines + corners + floor map").
- **Recordings card** (§8.9): one row per saved clip — date/time, duration, size, a
  `CAL` chip when a calibration is attached — with chip actions **Judge · Save ·
  Delete** (delete arms on first tap: "Again?"). Empty state: dim "No recordings yet."

### On-site calibration flow

Calibrate court captures the current preview frame to the canvas (`S.base`, `S.W/H`
from the stream) and enters the existing wizard at `tap_out` with `S.rec.calibrating`
set. All four wizard exits that normally land on `clip`
(`wallSkipBtn`/`wallDoneBtn` when no court model, `floorSkipAllBtn`/`floorDoneBtn`)
route back to `record` instead, stashing `buildJson()` output as the session
calibration. Back from `tap_out` returns to `record`. The `/api/camera-check` health
strip runs unchanged during wall/floor taps — that is the live-match rehearsal.

The stashed calibration is saved into each subsequent recording's metadata sidecar.

### Judging a recording

**Judge** loads the stored blob through the same path as a picked file (shared
`loadVideoFile()` extracted from the `#fileIn` handler). If the recording carries a
calibration and its dimensions match, the frame phase's profile slot offers "Use this
frame with the on-site calibration" → applies via the existing `applyProfile()`
machinery → `review` phase for visual confirmation. Gotcha handled: Chrome's
MediaRecorder webm blobs report `duration = Infinity`; `loadedmetadata` now applies the
standard seek-to-huge-time fix before routing.

### Export

**Save** uses `navigator.share({files})` (share sheet → Save to Files/Photos) when
available, else an `<a download>` fallback.

## Storage: `RecStore`

- **OPFS** (preferred): `recordings/` directory under `navigator.storage.getDirectory()`;
  per clip `rec-<stamp>.<ext>` + `rec-<stamp>.json` sidecar
  `{id, created_utc, duration_s, mime, size, width, height, calibration}`.
- **IndexedDB fallback**: db `slc-recordings`, store `recs` keyed by id, records
  `{id, meta, blob}` — used when OPFS/`createWritable` is missing.
- API: `mode()`, `list()`, `save(blob, meta)`, `load(id)`, `remove(id)`.
  `navigator.storage.persist()` requested on first save.

## State

`S.rec = {stream, recorder, chunks, startedAt, timer, calibration, calibrating, mime}`;
`S.pendingRecCal` carries a recording's calibration into the judge flow. Camera stream
survives the calibration sub-flow (instant return); it is fully stopped on any other
exit from `record` (back to Play, judging a recording).

## Error handling

- Camera permission denied / no camera → status line explains; Record stays disabled.
- MediaRecorder unsupported → status line; Record disabled (calibration still works).
- Storage failures surface in the error banner; a failed save keeps the blob in memory
  and offers the share-sheet path via the banner message.

## Testing

No backend changes → no new pytest. UI verified per the `/verify` skill with Playwright
(`--use-fake-device-for-media-stream --use-fake-ui-for-media-stream`): record → stop →
row appears → judge routes to frame phase; calibration round-trip back to `record`
with status updated; both themes at 390×844.

## DESIGN.md deltas (same change)

§3.3 phase list + `p-record`; §8.15 documents the second working hero card; new §8.16
(record readout, preview element, recordings rows); §16 blueprint rows for `p-load`
and `p-record`.
