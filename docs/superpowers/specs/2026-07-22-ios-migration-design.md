# iOS Migration — Native Capture + On-Device YOLO, Cloud Pipeline, TestFlight Demo

**Date:** 2026-07-22
**Status:** Approved
**Deadline:** Camp demo in ~10 days (~2026-08-01)

## Goal

Ship a single TestFlight app that demos the whole product: a native record screen
with a live on-device ball overlay running on the Apple Neural Engine, native
IN/OUT results, and the existing web review/library/coach UI rendered in-app.
The Python pipeline stays in the cloud, unchanged.

Longer-term (out of scope for v1, but the architecture must not block it):
real-time in/out calls and real-time scoring computed on-device.

## Decisions made

| Question | Decision |
|---|---|
| What does on-device detection do in v1? | Live ball overlay during recording only. Cloud still runs its own detection on the uploaded video, so line-call accuracy is unchanged. |
| Build environment | Mac + paid Apple Developer account available. |
| Detection model | Train **YOLO11n** on the existing Roboflow dataset (`ai-squash-line-tracker`), export to Core ML. Do **not** try to convert the RF-DETR checkpoint — its ONNX export is already documented to fragment under the CoreML execution provider (`inference_engine.py`). |
| App shape | Hybrid, option 3: native record screen + native results card; Matches/Coach/full-review screens are the existing `index.html` in a WKWebView. |
| Webview source | Remote — the deployed Flask app serves `index.html`. No bundled copy, no UI duplication. |
| Cloud detection model | RF-DETR stays as-is in the cloud for v1. |

## Architecture

```
iPhone (Swift, native)                      Cloud (existing Python, deployed)
┌──────────────────────────────┐            ┌────────────────────────────────┐
│ Record screen                │            │ Flask app.py (unchanged API)   │
│  AVCaptureSession            │  video     │  /api/upload → job_runner      │
│  ├─► Core ML YOLO11n (ANE)   │ ─────────► │  RF-DETR detection (for now)   │
│  │    live ball overlay      │   HTTPS    │  bounce → classify → judge     │
│  └─► AVAssetWriter (.mp4)    │            │  coach report                  │
│ Results card (native)        │ ◄───────── │  run JSON                      │
│ Review/Library/Coach         │   poll     │  serves index.html to webview  │
│  (WKWebView → cloud URL)     │            └────────────────────────────────┘
└──────────────────────────────┘
```

### BallTracker: the runway to real-time

The native detection layer is a `BallTracker` module that emits a stream of
`(timestamp, x, y, confidence)` to subscribers. v1 has two consumers: the
overlay renderer and a ring buffer. Future phases add consumers without
rewriting v1:

1. Swift port of `event_engine.py` bounce detection consuming the ring buffer
2. Homography judging (calibration fetched from the cloud) → real-time in/out
3. Real-time scoring

## Model workstream

- Export the Roboflow dataset in YOLO format; train YOLO11n (nano) with
  Ultralytics at `imgsz=960` (ball is small in frame; pipeline already
  standardizes on 960px). Nano at 960 runs >30fps on the ANE.
- Export: `yolo export format=coreml nms=True half=True` → `.mlpackage`
  bundled in the app.
- **Verify ANE residency** with Xcode's Core ML performance report (per-layer
  compute-unit placement). This is exactly where the RF-DETR export failed.
- **Score before shipping**: run the YOLO checkpoint through the existing eval
  harness (`benchmark_tracking.py`, `eval_set/`) against the Bay Club labels
  and compare recall to RF-DETR. Acceptance bar for v1: the overlay looks
  locked-on; frame-for-frame parity with RF-DETR is not required.
- **License flag**: Ultralytics YOLO is AGPL-3.0. Fine for a camp demo;
  commercial distribution needs an Enterprise license or a different
  architecture. Known, accepted for now.

## iOS app

New `ios/` directory in the repo. SwiftUI, iOS 17+, three tabs mirroring the
web app's Play/Matches/Coach IA.

- **Record (native):** `AVCaptureSession` video data output feeds both
  `BallTracker` (Core ML) and an `AVAssetWriter` writing the mp4 — same sample
  buffers, so overlay and recording cannot drift. Overlay drawn in a
  Metal-backed layer over the preview. Stop → upload to `/api/upload` → poll →
  native results card (per-hit IN/OUT chips, confidence, "Open full review").
- **Matches / Coach / full review (WKWebView):** loads the deployed
  Flask-served `index.html`. The results card deep-links into a specific run
  via a small hash-route hook in `index.html` (e.g. `#run=<id>`).
- Native screens follow **DESIGN.md** tokens so native and web read as one app.

## Cloud deployment

Deploy the Flask app once; it serves the upload endpoint, pipeline compute,
and the webview UI. Single VM (CPU-heavy is tolerable at demo volumes; upgrade
to GPU if rehearsal turnaround is slow) behind **Caddy for automatic HTTPS**
(App Transport Security requires HTTPS). Auth for the demo: unguessable
URL/token header only.

## Ten-day schedule

| Days | Deliverable |
|---|---|
| 1–2 | Dataset export, YOLO11n training, offline eval vs labels · Xcode scaffold + TestFlight "hello world" upload (surface signing issues early) |
| 3–5 | Native record screen: camera, Core ML overlay, ANE verification, AVAssetWriter |
| 5–7 | Upload → poll → results card · Flask deployed with HTTPS · webview tabs + run deep-link |
| 8–9 | DESIGN.md polish, app icon, real TestFlight build to camp testers |
| 10 | Buffer + on-court dry run |

**Scope cuts (explicit):** calibration stays in the webview; no on-device
in/out (phase 2); no offline mode; cloud keeps RF-DETR.

## Testing

- Model quality: existing Python eval harness; no new infra.
- Swift: unit tests for the API client (upload/poll/JSON parsing) and
  `BallTracker` output plumbing.
- Camera/overlay path: verified on-device; day-10 on-court dry run is
  budgeted rehearsal, not slack.

## Risks

| Risk | Mitigation |
|---|---|
| YOLO11n recall on a small, fast ball | Train at 960; tune confidence threshold; measure with the eval harness before integrating |
| TestFlight/signing friction | "Hello world" upload on day 2, not day 9 |
| Cloud turnaround too slow for a live demo | Rehearse early; GPU VM as fallback |
| Webview/native seam feels janky | Results card is native; webview only for deep review, styled by the same DESIGN.md tokens |
