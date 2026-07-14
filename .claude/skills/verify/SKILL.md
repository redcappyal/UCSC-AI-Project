---
name: verify
description: Build/launch/drive recipe for verifying UI and pipeline changes in this squash line-calling app.
---

# Verifying changes in this app

## Launch
- Server: `PORT=5177 .venv/bin/python app.py` (repo venv has flask + cv2; system python3 does not). Health check: `curl http://127.0.0.1:5177/api/health`.
- The whole UI is `index.html` (inline HTML/CSS/JS), served by Flask at `/`.

## Drive the browser
- No playwright in any venv, but Playwright browsers are cached at `~/Library/Caches/ms-playwright/`. `npm install playwright-core` in a scratch dir and launch with
  `executablePath: ~/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing` (headless decodes H.264 fine).
- Top-level `const S` (app state) and all functions are reachable from `page.evaluate` — poll `S.phase`, `S.clip`, etc.
- Load a video with `page.setInputFiles('#fileIn', path)`, wait for `S.phase === 'frame'`.
- To reach the clip phase without doing line calibration, `page.evaluate(() => setPhase('clip'))`. Calibration is only needed by the Track button's `buildJson()`.
- If you mutate `S.clip.start/end` directly in evaluate, also call `recenterView(); updateClipTimeline(); scheduleDetailRender(0);` — the UI event handlers do this for you, raw state pokes don't.

## Test videos
- `SquashAnalytics.mp4` in the repo root: real footage, 311.9s @ 60fps 1920x1080 — good "long video" case.
- Generate a short synthetic one with the venv's cv2 (`VideoWriter` fourcc `avc1`) for the short-video/edge cases; burn the timestamp into frames to verify frame-exact seeking.

## Cheap end-to-end track
- `/api/track` accepts any parseable JSON for `calibration_json` at job-creation time; a 0.6s window with `frame_stride=10`, `inference_width=640` runs ~4 inferences and completes in seconds on CPU. Poll `/api/track/status/<run_id>`.

## Gotchas
- Page requests `/favicon.ico` → 404 console error; pre-existing noise, not a failure.
- Filmstrip thumbnail renders seek the single shared `<video>` element; renders are debounced and token-guarded — wait on `S.clip.detailKey` changing rather than fixed timeouts.
