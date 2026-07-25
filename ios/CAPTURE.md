# Court capture settings

Everything here is pinned in `Sources/Record/CaptureSettings.swift`. The code
carries the per-setting rationale; this file covers what an operator has to do
and what breaks if the settings change.

| Setting | Value | Enforced by |
|---|---|---|
| Resolution | 3840x2160 landscape, both mounts | `applyCaptureFormat` via `bestFormatIndex` |
| Frame rate | 60 fps, min and max pinned | `activeVideoMin/MaxFrameDuration` |
| Codec | HEVC, ~53 Mbps | `startRecording` |
| Shutter | 1/1000, falling back to 1/500 | `solveExposure` |
| ISO | Locked, capped at 2500 or the device max | `solveExposure` |
| White balance | Locked, metered per court | `lockForCourt` |
| Focus | Locked | `lockForCourt` |
| Stabilisation | OFF | `configureSession` |
| Lens | Ultrawide, main camera as fallback | `configureSession` |

Capture is always landscape. The phone mounts on its side, and
`CaptureSettings.rotationAngle(for:)` applies 0 or 180 so the recorded frame is upright
whichever end the lens is at. Portrait is not a capture mode.

**Migration: re-solve portrait calibrations.** `CameraModel.scaled` refuses to scale
across an aspect change rather than distorting the geometry, and this change flips the
capture aspect from portrait (9:16) to landscape (16:9). Any calibration solved from
portrait footage will now throw `aspectMismatch` in `adoptedForCapture()` instead of
loading. This is expected, not a bug — those calibrations need to be re-solved from
landscape footage before they will load again.

## Why these are locked rather than automatic

The recording serves two masters: it feeds the live tracker, and it becomes
training footage. Both want the same thing — no free variables. Auto-exposure
hunting as a player in dark kit crosses frame is a nuisance dimension the
detector would otherwise have to spend labelled examples absorbing.

The two that carry the most weight:

- **1/1000 shutter.** A ~28 m/s ball travels 230 mm in 1/120 s, which at the
  front wall is a ~34 px streak of a ~6 px ball. At 1/1000 it travels 28 mm and
  the streak is about the ball's own size. Blur, not resolution, is what hides
  far-wall contacts — and `eval_set/BASELINE-2026-07-23.md` puts 65 of 71
  missed bounces on wall hits.
- **4K.** The detector downsamples to 960 either way
  (`inference_engine.resize_frame_for_inference`), so this costs today's
  pipeline nothing and buys the archive. Pixels not sampled now can never be
  recovered by a future model. Apple encodes 4K60 at ~400 MB/min against
  ~350 MB/min for 1080p120 — ~14% more bytes for 4x the pixels.

## Per-court setup

1. Mount the phone, point it at the court, open Record.
2. `startCamera()` meters once and freezes exposure/WB/focus. The capsule under
   the preview reads back the resolved values, e.g. `Locked 1/1000 · ISO 1450`.
3. If it reads **`Dim court — ...`**, the venue could not hold exposure inside
   the ISO cap and the footage will come out under. Add light, or accept it.
4. Re-open Record to re-meter after any lighting or mount change.

## Consequences to know about

- **Every court needs re-calibrating.** Calibration profiles are keyed on
  frame dimensions (`index.html`, `loadProfiles`), so 1080x1920 profiles will
  not match 4K footage and will not be offered. This fails closed rather than
  applying a wrong-space calibration, which is the correct behaviour — but it
  does mean pre-4K profiles are dead. `clearPre4kProfiles` in `index.html`
  drops them once per browser, stamped `slc-cal-profiles-reset`; bump
  `PROFILE_RESET_VERSION` if a future capture change needs the same treatment.
  Note this is the browser store only — the per-run
  `ui_runs/*/calibration.json` records are deliberately kept, because
  `build_eval_set.py` embeds them into every eval case and deleting them
  would silently zero the judge axes on the next rebuild.
- **Both phones must run the same build.** `PeerSession`'s Hello advertises
  `CaptureSettings.frameWidth/Height`, and detections cross the wire in those
  pixel units. A 4K phone paired with a 1080p phone skews stereo silently.
- **Stabilisation must stay off.** OIS and EIS warp the frame per-frame, which
  would invalidate the fixed floor homography and the 15-correspondence camera
  solve. A calibrated camera has to stay geometrically rigid.
- **The ultrawide is the slower lens** (typically f/2.4 against the main
  camera's f/1.6), costing roughly 1.3 stops. That is the main reason the
  exposure solve needs a fallback shutter at all.

## Not yet verified on hardware

Both need a real device on a real court; neither can be checked in the
simulator.

- **Thermals.** 4K60 HEVC for a full match, phone clamped in still air, is the
  live risk — not storage. Watch for a resolution drop or a halted recording.
  Segment-per-rally recording is the mitigation, and also cuts a match to
  roughly 6 GB per phone.
- **Flicker.** 1/1000 under mains-driven lighting can band or pulse. A few
  minutes of test footage at the venue settles it; if it bands, 1/500 is the
  out.

## Landscape capture — verification status

**Nothing on the landscape-capture branch has been compiled or run.** It was
written on a Windows machine where `xcodebuild`, `xcodegen` and `swift` are all
absent, and the repo has no macOS CI (`.github/workflows/tests.yml` is pytest
only). Every Swift change is verified by inspection and adversarial review, not
by a build. Four review rounds found real defects — including one probable
compile error and one bug that would have recorded upside-down footage — so
treat "reviewed" as meaningfully different from "working".

Run these on a Mac, in order, before trusting any of it:

1. **It builds and the suite passes.**
   `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'`
   Two things are most likely to fail first: `RecordModel.init` assigning
   `captureOrientation` before `tracker` is initialized (a review predicted
   this as an error and the fix — removing the property's declaration default —
   was applied unverified), and `OrientationLockTests`/`LiveWiringTests`
   touching the `@MainActor` `OrientationPolicy` from XCTest.
2. **Play tab locks to landscape.** Rotate the simulator through all four
   orientations on Play: it must never show portrait. Landscape-left and
   landscape-right are both allowed *before* the camera configures.
3. **The pin holds.** Once the exposure note appears, the other landscape must
   be refused. Then switch to Matches and back to Play — the pin must survive
   the round trip, which is what `OrientationPolicy.capturePin` exists for.
4. **The web tabs still rotate.** Matches and Coach must reach portrait, or the
   `UISupportedInterfaceOrientations` superset is wrong.
5. **180° normalization — needs a physical iPhone, no simulator camera.**
   Mount or hold the phone **landscape-left**, record, play back. The video
   must be upright. If it is upside down, `rotationAngle(for:)`'s 0/180 mapping
   is inverted; swap the cases and update `CaptureOrientationTests`, which
   asserts the mapping and must change with it.

Two known limits, both deliberate and neither fixed by this branch:

- **The peer mount guard is correct but unreachable.** `PeerSession` refuses a
  mismatched pair, but nothing in the app constructs a `PeerSession` carrying a
  real mount — `PeerBenchView` (DEBUG) takes the default and `attachPeer` has
  no production caller. Two phones in opposite mounts still pair. Wiring the
  pairing path is separate work.
- **Portrait-solved calibrations no longer load.** `CameraModel.scaled` refuses
  the aspect change rather than distorting geometry, so any calibration solved
  from portrait footage must be re-solved from landscape footage.

When these are run, replace this section with what actually passed, the device
model, and the date. A partially-run checklist recorded as complete is the
failure mode this branch exists to remove.
