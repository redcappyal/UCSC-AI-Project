# iOS Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One TestFlight app: native SwiftUI record screen with a live on-device YOLO11n ball overlay (Apple Neural Engine), upload to the existing cloud Flask pipeline for IN/OUT calls, native results card, and the existing web UI in a WKWebView for review/matches/coach.

**Architecture:** Hybrid app per `docs/superpowers/specs/2026-07-22-ios-migration-design.md`. The phone's YOLO detections power the live overlay only; the cloud still runs RF-DETR on the uploaded video, so line-call accuracy is unchanged. `BallTracker` is a subscriber-based module (overlay + ring buffer today; bounce detection/judging later). Server changes are two small seams: a latest-calibration endpoint and hash-based deep links in `index.html`.

**Tech Stack:** Python/Flask (existing), Ultralytics YOLO11n → Core ML, SwiftUI + AVFoundation + Vision (iOS 17+), XcodeGen, Caddy, TestFlight.

## Global Constraints

- iOS deployment target **17.0**, SwiftUI, portrait-only, `TARGETED_DEVICE_FAMILY: 1` (iPhone).
- Bundle id: `com.redcappyal.squashlinecalling`.
- YOLO11n at `imgsz=960`; detection confidence threshold **0.25** (mirrors `tracking_common.CONFIDENCE_THRESHOLD`).
- The Python test suite must keep passing with **`requirements-test.txt` only** — no module-level `ultralytics`/`torch`/`roboflow` imports in any file a test imports. Follow the repo's lazy-import pattern (`local_model_eval.py`).
- Native colors come from DESIGN.md dark-theme tokens: bg `#000`, surface `#1C1C1E`, line `#26262A`, dim `#98989F`, text `#FFF`, accent `#FFD60A` (always with black text), IN verdict `#2ECC5E`, OUT verdict `#E03A2F`, unknown `#C7C7CC`. IN/OUT colors are for verdicts ONLY.
- Webview loads the **deployed** Flask origin (remote `index.html`), never a bundled copy.
- Python steps run in this (Windows) workspace and are fully TDD. Steps tagged **[Mac]** (Xcode build/test, Core ML export, TestFlight, deploy) run on the Mac; Swift code and tests are still written here, and each Swift task ends with the exact `[Mac]` command to verify. Batch `[Mac]` verification is acceptable (e.g., run once after Tasks 5–11), but must happen before Task 12.
- Swift test runner: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`.

## Server API contracts (verbatim from app.py — do not change)

- `POST /api/upload` multipart field `video_file` → `{ok, video_id, fps?, frame_count?, duration?}`.
- `POST /api/track` form fields `video_id`, `calibration_json`, `start_time`, `end_time`, `frame_stride` (1–10), `inference_width` (0|640|960|1280), `event_engine` (""), `fusion_3d` ("") → `public_job` JSON.
- `GET /api/track/status/<run_id>` → `public_job`: `{ok, status: queued|running|complete|failed, run_id, stage?, progress, processed_frames, total_frames, message, hits?, error?, rows?}`.
- Hit entries: `{frame, timestamp_seconds, call: IN|OUT|UNKNOWN|AUDIO, margin_px?, event_type?, target_zone?, wall_diagram?, ...}`. Front-wall filter (mirror `app.py:front_wall_hits_from_payload`): `target_zone != null AND wall_diagram != null AND event_type in (null, "wall", "unknown")`.

---

### Task 1: `GET /api/calibration/latest` endpoint

The native app cannot run the calibration wizard; it reuses the most recent run's calibration (`run_dir/calibration.json`, written by every `/api/track`). Demo flow: calibrate once from the web app on-site, then every native recording reuses it.

**Files:**
- Modify: `app.py` (after `get_court_model`, ~line 467)
- Test: `tests/test_calibration_latest.py`

**Interfaces:**
- Produces: `GET /api/calibration/latest` → 200 `{ok: true, run_id: str, saved_at: ISO8601, calibration: object}` | 404 `{ok: false, error: str}`. Task 6's Swift client consumes this.

- [ ] **Step 1: Write the failing test**

```python
"""GET /api/calibration/latest: newest run calibration for native clients."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_run_with_calibration(app_module, run_id, calibration, age_seconds):
    run_dir = app_module.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return run_dir


def test_latest_calibration_returns_newest_run():
    import app as app_module

    client = app_module.app.test_client()
    older = make_run_with_calibration(
        app_module, "cal-latest-older", {"lines": [{"name": "out"}]}, age_seconds=120)
    newer = make_run_with_calibration(
        app_module, "cal-latest-newer", {"lines": [{"name": "tin"}]}, age_seconds=5)
    try:
        response = client.get("/api/calibration/latest")
        assert response.status_code == 200
        body = response.get_json()
        assert body["ok"] is True
        assert body["run_id"] == "cal-latest-newer"
        assert body["calibration"] == {"lines": [{"name": "tin"}]}
        assert body["saved_at"].endswith("Z")
    finally:
        shutil.rmtree(older, ignore_errors=True)
        shutil.rmtree(newer, ignore_errors=True)


def test_latest_calibration_404_when_none_exist():
    import app as app_module

    client = app_module.app.test_client()
    # Only guaranteed-empty when no other test left runs behind; use a marker
    # dir without calibration.json to prove non-calibrated runs are skipped.
    marker = app_module.RUNS_DIR / "cal-latest-empty-run"
    marker.mkdir(parents=True, exist_ok=True)
    had_calibrations = any(app_module.RUNS_DIR.glob("*/calibration.json"))
    try:
        response = client.get("/api/calibration/latest")
        if had_calibrations:
            assert response.status_code == 200  # other fixtures present; endpoint still works
        else:
            assert response.status_code == 404
            assert response.get_json()["ok"] is False
    finally:
        shutil.rmtree(marker, ignore_errors=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_calibration_latest.py -v` (Windows dev box: `venv\Scripts\python -m pytest tests\test_calibration_latest.py -v`)
Expected: FAIL — 404/`ok is False` on the first test (route does not exist).

- [ ] **Step 3: Implement the endpoint in `app.py`**

Insert after `get_court_model` (keep `import time`, `json`, `jsonify`, `error_response` — all already imported):

```python
@app.get("/api/calibration/latest")
def latest_calibration():
    """Most recent run's calibration so a native client can reuse the court
    setup without redoing the wizard. Recency = calibration.json mtime, which
    tracks the last time /api/track accepted that calibration."""
    best = None
    if RUNS_DIR.exists():
        for path in RUNS_DIR.glob("*/calibration.json"):
            try:
                key = (path.stat().st_mtime_ns, path.parent.name)
            except OSError:
                continue
            if best is None or key > best[0]:
                best = (key, path)

    if best is None:
        return error_response(
            "No saved calibration found. Run a calibrated analysis first.", status=404)

    path = best[1]
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return error_response("Latest calibration could not be read.", status=500)

    saved_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    return jsonify({
        "ok": True,
        "run_id": path.parent.name,
        "saved_at": saved_at,
        "calibration": calibration,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_calibration_latest.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass (117 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_calibration_latest.py
git commit -m "feat: /api/calibration/latest for native clients"
```

---

### Task 2: Deep links in `index.html` (`#run=`, `#tab=`)

The native results card opens the full review at a specific run; the Matches/Coach tabs open web sections. A `#run` link seeds the existing restore stash (`RESTORE_KEY`, `index.html:5563`) so `tryRestoreSession()` does all the rehydration work unchanged.

**Files:**
- Modify: `index.html:5668-5671` (the boot block: `updatePlayButtons(); setPhase('load'); tryRestoreSession();`)

**Interfaces:**
- Consumes: `RESTORE_KEY`, `tryRestoreSession()`, `setPhase()` — all existing.
- Produces: URL contract for Task 11: `<origin>/#run=<run_id>&frame=<n>` opens the review of that run; `<origin>/#tab=matches` / `#tab=coach` opens that section.

- [ ] **Step 1: Replace the boot block**

Replace:

```js
updatePlayButtons();
setPhase('load');
tryRestoreSession();
```

with:

```js
/* ---------- deep links (#run=<id>&frame=<n>, #tab=matches|coach) ----------
   The native iOS shell opens the webview at a specific run (results card →
   full review) or a section tab. A #run link seeds the restore stash so
   tryRestoreSession does the rehydration exactly like a returning session. */
function applyDeepLink(){
  const params = new URLSearchParams(location.hash.replace(/^#/, ''));
  const runId = (params.get('run') || '').trim();
  if(runId){
    try{
      localStorage.setItem(RESTORE_KEY, JSON.stringify({
        run_id: runId,
        frame: Number(params.get('frame')) || 0,
        videoName: null,
      }));
    }catch(_){ /* storage unavailable — boot normally */ }
    return 'run';
  }
  const tab = params.get('tab');
  return (tab === 'matches' || tab === 'coach') ? tab : null;
}

updatePlayButtons();
setPhase('load');
const bootLink = applyDeepLink();
if(bootLink === 'matches' || bootLink === 'coach') setPhase(bootLink);
else tryRestoreSession();
```

- [ ] **Step 2: Verify in the browser (use the `/verify` skill recipe)**

Serve the app, then check:
- `http://127.0.0.1:5188/#tab=coach` lands on the Coach section; `#tab=matches` on Matches.
- `http://127.0.0.1:5188/#run=doesnotexist` shows the existing "Previous session could not be restored" status (proves the stash path fired).
- If a completed run exists on disk: `#run=<that id>` restores straight into review.
- Plain `http://127.0.0.1:5188/` boots exactly as before. No visual change; both themes unaffected.

- [ ] **Step 3: Run the Python suite (guards nothing broke server-side)**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add index.html
git commit -m "feat: #run/#tab deep links for the native shell webview"
```

---

### Task 3: `yolo_model_eval.py` — score a YOLO checkpoint with the existing eval stack

Runs a `.pt` checkpoint over a video and writes the **same CSV schema** the pipeline uses (`tracking_common.CSV_FIELDNAMES`), so every existing eval/label tool works on YOLO output unchanged. Mirrors `local_model_eval.py`.

**Files:**
- Create: `yolo_model_eval.py`
- Test: `tests/test_yolo_eval.py`

**Interfaces:**
- Consumes: `tracking_common.select_ball_prediction`, `ball_csv_row`, `CSV_FIELDNAMES`, `draw_predictions`, `CONFIDENCE_THRESHOLD`.
- Produces: `yolo_boxes_to_predictions(rows, names) -> list[dict]` (pure, no ultralytics import); CLI `python yolo_model_eval.py --weights best.pt --video clip.mp4 --output-csv out.csv [--annotated out.mp4] [--stride 1] [--imgsz 960]`.

- [ ] **Step 1: Write the failing test**

```python
"""yolo_model_eval: ultralytics boxes -> the pipeline's prediction dicts.

Module import and the adapter must work without ultralytics installed —
the test env is requirements-test.txt only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking_common import select_ball_prediction
from yolo_model_eval import yolo_boxes_to_predictions


def test_boxes_convert_to_center_size_dicts():
    rows = [(100.0, 200.0, 110.0, 212.0, 0.9, 0)]
    names = {0: "ball"}
    predictions = yolo_boxes_to_predictions(rows, names)
    assert predictions == [{
        "x": 105.0, "y": 206.0, "width": 10.0, "height": 12.0,
        "confidence": 0.9, "class": "ball",
    }]


def test_ball_class_wins_selection_over_other_classes():
    rows = [
        (0.0, 0.0, 50.0, 50.0, 0.95, 1),      # e.g. "player"
        (100.0, 200.0, 110.0, 212.0, 0.6, 0), # "ball"
    ]
    names = {0: "ball", 1: "player"}
    selected = select_ball_prediction(yolo_boxes_to_predictions(rows, names))
    assert selected["class"] == "ball"
    assert selected["confidence"] == 0.6


def test_unknown_class_index_stringifies():
    predictions = yolo_boxes_to_predictions([(0, 0, 2, 2, 0.5, 7)], {})
    assert predictions[0]["class"] == "7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_yolo_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yolo_model_eval'`.

- [ ] **Step 3: Implement `yolo_model_eval.py`**

```python
"""CLI: run a YOLO checkpoint over a video and write ball coordinates.

Writes the same CSV schema as the production tracker (tracking_common), so the
existing eval/label tooling scores YOLO checkpoints with zero changes. This is
the acceptance gate for the on-device model: compare its detection rate and
eval numbers against the RF-DETR CSVs before shipping the Core ML export.

ultralytics is imported lazily inside main(); the adapter stays importable in
the requirements-test.txt environment.
"""

import argparse
import csv
from pathlib import Path

import cv2

from tracking_common import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    ball_csv_row,
    draw_predictions,
    select_ball_prediction,
)


def yolo_boxes_to_predictions(rows, names):
    """ultralytics Boxes.data rows -> prediction dicts for tracking_common.

    rows: iterable of (x1, y1, x2, y2, confidence, class_index); tensors or
    plain sequences both work. names: {class_index: class_name}.
    """
    predictions = []
    for x1, y1, x2, y2, confidence, class_index in rows:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        predictions.append({
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": float(confidence),
            "class": names.get(int(class_index), str(int(class_index))),
        })
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="YOLO .pt checkpoint")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--annotated", default=None,
                        help="Optional annotated .mp4 output path")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO  # lazy: heavy, and absent in the test env

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.annotated:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.annotated), cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, args.stride), (width, height))

    detected = 0
    written = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        frame_index = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.stride:
                continue
            if args.max_frames is not None and written >= args.max_frames:
                break

            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            rows = result.boxes.data.tolist() if result.boxes is not None else []
            predictions = yolo_boxes_to_predictions(rows, result.names)
            ball = select_ball_prediction(predictions)
            csv_writer.writerow(ball_csv_row(frame_index, fps, ball))
            written += 1
            if ball is not None:
                detected += 1
            if writer is not None:
                annotated = frame.copy()
                draw_predictions(annotated, [ball] if ball else [])
                writer.write(annotated)

    capture.release()
    if writer is not None:
        writer.release()
    rate = detected / written * 100 if written else 0.0
    print(f"frames={written} detected={detected} rate={rate:.1f}%")
    print(f"csv={args.output_csv}")


if __name__ == "__main__":
    main()
```

Note: check `draw_predictions`' signature in `tracking_common.py` before wiring the `--annotated` branch; if it expects raw prediction dicts, pass `[ball]` as shown, otherwise adapt the call — do not change `tracking_common`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_yolo_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the whole suite, then commit**

Run: `venv/bin/python -m pytest tests/ -q` — all pass.

```bash
git add yolo_model_eval.py tests/test_yolo_eval.py
git commit -m "feat: eval a YOLO checkpoint through the existing CSV/eval stack"
```

---

### Task 4: `train_yolo_ball.py` — Roboflow dataset → YOLO11n

**Files:**
- Create: `train_yolo_ball.py`
- Test: `tests/test_train_yolo_ball.py`

**Interfaces:**
- Produces: `build_train_kwargs(data_yaml, imgsz, epochs, batch, name, device) -> dict` (pure); CLI `python train_yolo_ball.py --workspace <slug> --dataset-version <n> [--epochs 100] [--imgsz 960]`. Reads `ROBOFLOW_API_KEY` (and optional `ROBOFLOW_WORKSPACE`) from `.env` like the rest of the repo. Prints the best-weights path Task 12 exports.

- [ ] **Step 1: Write the failing test**

```python
"""train_yolo_ball: training-kwargs builder (no ultralytics/roboflow needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_yolo_ball import build_train_kwargs


def test_defaults_pin_imgsz_960_and_name():
    kwargs = build_train_kwargs("data/data.yaml")
    assert kwargs == {
        "data": "data/data.yaml", "imgsz": 960, "epochs": 100,
        "batch": -1, "name": "ball-yolo11n", "cache": True,
    }


def test_device_only_included_when_set():
    assert "device" not in build_train_kwargs("d.yaml", device=None)
    assert build_train_kwargs("d.yaml", device="0")["device"] == "0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_train_yolo_ball.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `train_yolo_ball.py`**

```python
"""CLI: download the Roboflow dataset and train YOLO11n for on-device Core ML.

The production cloud model stays RF-DETR; this trains the *phone* model on the
same labels. Train anywhere with a GPU; export to Core ML on the Mac
(ios/MODEL.md). ultralytics/roboflow import lazily so the test env stays light.
"""

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))


def build_train_kwargs(data_yaml, imgsz=960, epochs=100, batch=-1,
                       name="ball-yolo11n", device=None):
    """Ultralytics train() kwargs. imgsz=960 matches the pipeline's inference
    width — the ball is small in frame and 640 measurably hurts recall."""
    kwargs = {
        "data": str(data_yaml), "imgsz": imgsz, "epochs": epochs,
        "batch": batch, "name": name, "cache": True,
    }
    if device is not None:
        kwargs["device"] = device
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",
                        default=os.environ.get("ROBOFLOW_WORKSPACE"),
                        help="Roboflow workspace slug")
    parser.add_argument("--project", default="ai-squash-line-tracker")
    parser.add_argument("--dataset-version", type=int, required=True,
                        help="Roboflow DATASET version (not the model version)")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (same key the app uses).")
    if not args.workspace:
        raise SystemExit("Pass --workspace or set ROBOFLOW_WORKSPACE in .env.")

    from roboflow import Roboflow  # lazy
    from ultralytics import YOLO   # lazy

    dataset = (
        Roboflow(api_key=api_key)
        .workspace(args.workspace)
        .project(args.project)
        .version(args.dataset_version)
        .download("yolov11")
    )
    data_yaml = Path(dataset.location) / "data.yaml"

    model = YOLO(args.model)
    results = model.train(**build_train_kwargs(
        data_yaml, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device))
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    print("next: score it with yolo_model_eval.py, then export via ios/MODEL.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the suite, then commit**

Run: `venv/bin/python -m pytest tests/test_train_yolo_ball.py tests/ -q`
Expected: all pass.

```bash
git add train_yolo_ball.py tests/test_train_yolo_ball.py
git commit -m "feat: YOLO11n training script for the on-device ball model"
```

---

### Task 5: iOS scaffold — XcodeGen project, theme, tabs, TestFlight hello world

**Files:**
- Create: `ios/project.yml`, `ios/.gitignore`, `ios/README.md`, `ios/TESTFLIGHT.md`
- Create: `ios/Sources/SquashApp.swift`, `ios/Sources/RootTabView.swift`, `ios/Sources/Theme.swift`, `ios/Sources/Config.swift`
- Create: `ios/Tests/SmokeTests.swift`
- Create: `ios/Model/.gitkeep` (the `.mlpackage` arrives in Task 12)

**Interfaces:**
- Produces: `Theme` (colors below), `Config.baseURL: URL` — every later Swift task uses these. Tab placeholders replaced in Tasks 9/11.

- [ ] **Step 1: Write `ios/project.yml`**

```yaml
name: SquashLineCalling
options:
  bundleIdPrefix: com.redcappyal
  deploymentTarget:
    iOS: "17.0"
settings:
  base:
    SWIFT_VERSION: "5.9"
    TARGETED_DEVICE_FAMILY: "1"
    CODE_SIGN_STYLE: Automatic
targets:
  SquashLineCalling:
    type: application
    platform: iOS
    sources:
      - path: Sources
      - path: Model
        optional: true
        type: folder
        buildPhase: resources
    info:
      path: Generated/Info.plist
      properties:
        UILaunchScreen: {}
        NSCameraUsageDescription: Records rallies and runs the live ball tracker.
        NSMicrophoneUsageDescription: Impact sounds help the analyzer find wall hits.
        UISupportedInterfaceOrientations: [UIInterfaceOrientationPortrait]
        ITSAppUsesNonExemptEncryption: false
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.redcappyal.squashlinecalling
  SquashLineCallingTests:
    type: bundle.unit-test
    platform: iOS
    sources: [Tests]
    dependencies:
      - target: SquashLineCalling
```

Note for the implementer: a raw `.mlpackage` referenced as a folder resource is NOT compiled. Task 12 adds the model by placing `BallDetector.mlpackage` under `ios/Model/`; if Xcode does not compile it to `BallDetector.mlmodelc` automatically (check the build log), change the Model entry to a normal source path (`- path: Model`) so Xcode's Core ML compiler picks it up. The app must build and run with the Model dir empty either way.

- [ ] **Step 2: Write `ios/.gitignore`**

```
SquashLineCalling.xcodeproj/
Generated/
DerivedData/
```

- [ ] **Step 3: Write the app skeleton**

`ios/Sources/Theme.swift` — DESIGN.md dark tokens (the native shell ships dark; the webview handles its own theming):

```swift
import SwiftUI

/// DESIGN.md dark-theme tokens. IN/OUT colors are for verdicts ONLY;
/// the accent is always paired with black text.
enum Theme {
    static let bg = Color(hex: 0x000000)
    static let surface = Color(hex: 0x1C1C1E)
    static let line = Color(hex: 0x26262A)
    static let dim = Color(hex: 0x98989F)
    static let text = Color.white
    static let accentBg = Color(hex: 0xFFD60A)
    static let accentText = Color.black
    static let inCall = Color(hex: 0x2ECC5E)
    static let outCall = Color(hex: 0xE03A2F)
    static let unknown = Color(hex: 0xC7C7CC)
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255)
    }
}
```

`ios/Sources/Config.swift`:

```swift
import Foundation

enum Config {
    /// Deployed Flask origin (deploy/DEPLOY.md). Update before archiving.
    /// Any build can override at runtime via UserDefaults key "serverBase"
    /// (e.g. from Xcode scheme arguments) for LAN testing.
    static let defaultBase = "https://squash.example.com"

    static var baseURL: URL {
        if let raw = UserDefaults.standard.string(forKey: "serverBase"),
           let url = URL(string: raw) {
            return url
        }
        return URL(string: defaultBase)!
    }
}
```

`ios/Sources/SquashApp.swift`:

```swift
import SwiftUI

@main
struct SquashApp: App {
    var body: some Scene {
        WindowGroup {
            RootTabView()
                .preferredColorScheme(.dark)
        }
    }
}
```

`ios/Sources/RootTabView.swift` (placeholders; Tasks 9/11 replace the tab contents):

```swift
import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            Text("Record").tabItem { Label("Play", systemImage: "record.circle") }
            Text("Matches").tabItem { Label("Matches", systemImage: "square.stack") }
            Text("Coach").tabItem { Label("Coach", systemImage: "figure.tennis") }
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
    }
}
```

`ios/Tests/SmokeTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

final class SmokeTests: XCTestCase {
    func testConfigDefaultBaseIsHTTPS() {
        XCTAssertEqual(Config.baseURL.scheme, "https")
    }
}
```

- [ ] **Step 4: Write `ios/README.md`**

```markdown
# iOS app

Requires macOS + Xcode 15+.

    brew install xcodegen
    cd ios
    xcodegen generate
    open SquashLineCalling.xcodeproj

Tests: `xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`

- `Sources/` — SwiftUI app (Play = native record; Matches/Coach = webview).
- `Model/` — drop `BallDetector.mlpackage` here (see `MODEL.md`, written in a later task).
- Server origin: `Sources/Config.swift`.
```

- [ ] **Step 5: Write `ios/TESTFLIGHT.md`**

```markdown
# TestFlight

## One-time setup (do this on day 2 — it flushes out signing problems early)

1. Xcode → Settings → Accounts → add the Apple ID on the paid developer team.
2. `cd ios && xcodegen generate && open SquashLineCalling.xcodeproj`.
3. Target SquashLineCalling → Signing & Capabilities → select the Team.
   (Automatic signing; the bundle id com.redcappyal.squashlinecalling registers itself.)
4. App Store Connect → Apps → "+" → New App → iOS, name "Squash Line Calling",
   bundle id from step 3, SKU anything.
5. Select "Any iOS Device (arm64)" → Product → Archive → Distribute App →
   TestFlight & App Store → Upload.
6. App Store Connect → TestFlight tab → wait for processing (~10 min) →
   Internal Testing → "+" group "Camp" → add testers by Apple ID email.
   Internal testers need no Beta App Review.
7. Testers install via the TestFlight app invitation email.

## Every subsequent build

1. Bump build number (target → General → Build, or agvtool).
2. Product → Archive → Distribute → Upload. The Camp group auto-updates.

## Before the real demo build

- Set the deployed server origin in `Sources/Config.swift`.
- Confirm `ios/Model/BallDetector.mlpackage` is present (MODEL.md) so the
  live overlay works.
```

- [ ] **Step 6 [Mac]: Verify scaffold builds and hello-world upload**

Run: `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`
Expected: BUILD SUCCEEDED, 1 test passed. Then follow TESTFLIGHT.md one-time setup through step 7 — a placeholder build lands on a real phone via TestFlight.

- [ ] **Step 7: Commit**

```bash
git add ios/
git commit -m "feat: iOS scaffold — XcodeGen project, theme tokens, tab shell, TestFlight doc"
```

---

### Task 6: API models + client

**Files:**
- Create: `ios/Sources/API/Models.swift`, `ios/Sources/API/Multipart.swift`, `ios/Sources/API/APIClient.swift`
- Test: `ios/Tests/ModelsTests.swift`, `ios/Tests/MultipartTests.swift`

**Interfaces:**
- Consumes: `Config.baseURL` (Task 5); server contracts from the header.
- Produces (Tasks 10–11 rely on these exact names):
  - `struct UploadResponse { ok, videoID, fps?, frameCount?, duration? }`
  - `struct JobStatus { ok, status, runID?, stage?, progress?, processedFrames?, totalFrames?, message?, error?, hits? }`
  - `struct Hit: Identifiable { frame, timestampSeconds, call, marginPx?, eventType?, hasTargetZone, hasWallDiagram }` and `Array<Hit>.frontWall`
  - `struct LatestCalibration { runID: String, calibrationJSON: String }`
  - `protocol APIClientProtocol { latestCalibration(); upload(videoURL:); startTrack(videoID:calibrationJSON:duration:); trackStatus(runID:) }` + `struct APIClient: APIClientProtocol`

- [ ] **Step 1: Write the failing tests**

`ios/Tests/ModelsTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

final class ModelsTests: XCTestCase {
    func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    func testUploadResponseDecodes() throws {
        let response = try decode(UploadResponse.self, #"""
        {"ok": true, "video_id": "abc123", "fps": 30.0,
         "frame_count": 900, "duration": 30.0}
        """#)
        XCTAssertEqual(response.videoID, "abc123")
        XCTAssertEqual(response.duration, 30.0)
    }

    func testJobStatusRunningDecodesWithoutHits() throws {
        let status = try decode(JobStatus.self, #"""
        {"ok": true, "status": "running", "run_id": "1753", "stage": "tracking",
         "progress": 41.5, "processed_frames": 100, "total_frames": 241,
         "message": "Tracking frames..."}
        """#)
        XCTAssertEqual(status.status, "running")
        XCTAssertNil(status.hits)
        XCTAssertEqual(status.progress, 41.5)
    }

    func testHitPresenceFlagsAndFrontWallFilter() throws {
        let status = try decode(JobStatus.self, #"""
        {"ok": true, "status": "complete", "run_id": "1753", "hits": [
          {"frame": 120, "timestamp_seconds": 4.0, "call": "IN",
           "margin_px": 3.5, "event_type": "wall",
           "target_zone": {"zone": 3}, "wall_diagram": {"x": 1.0, "y": 2.0}},
          {"frame": 300, "timestamp_seconds": 10.0, "call": "OUT",
           "margin_px": -2.0,
           "target_zone": {"zone": 1}, "wall_diagram": {"x": 3.0, "y": 4.0}},
          {"frame": 400, "timestamp_seconds": 13.3, "call": "UNKNOWN",
           "event_type": "floor"},
          {"frame": 500, "timestamp_seconds": 16.6, "call": "AUDIO",
           "event_type": "wall"}
        ]}
        """#)
        let hits = try XCTUnwrap(status.hits)
        XCTAssertEqual(hits.count, 4)
        XCTAssertTrue(hits[0].hasTargetZone && hits[0].hasWallDiagram)
        XCTAssertFalse(hits[2].hasTargetZone)
        // Mirrors app.py front_wall_hits_from_payload: needs zone + diagram,
        // event_type in (null, wall, unknown). The floor hit and the
        // diagram-less AUDIO hit drop out.
        XCTAssertEqual(hits.frontWall.map(\.frame), [120, 300])
    }

    func testLatestCalibrationKeepsRawJSON() throws {
        let data = Data(#"{"ok": true, "run_id": "99", "calibration": {"lines": [1, 2]}}"#.utf8)
        let cal = try LatestCalibration(responseData: data)
        XCTAssertEqual(cal.runID, "99")
        // Round-trips as JSON (key order may differ) — it is re-posted verbatim
        // to /api/track, never interpreted by the app.
        let parsed = try JSONSerialization.jsonObject(with: Data(cal.calibrationJSON.utf8)) as? [String: Any]
        XCTAssertEqual(parsed?["lines"] as? [Int], [1, 2])
    }
}
```

`ios/Tests/MultipartTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

final class MultipartTests: XCTestCase {
    func testBodyStructure() {
        let body = Multipart.body(
            boundary: "BOUND", fields: [("video_id", "abc")],
            fileField: "video_file", filename: "clip.mp4",
            contentType: "video/mp4", fileData: Data("FILEBYTES".utf8))
        let text = String(decoding: body, as: UTF8.self)
        XCTAssertTrue(text.contains("--BOUND\r\n"))
        XCTAssertTrue(text.contains("name=\"video_id\"\r\n\r\nabc\r\n"))
        XCTAssertTrue(text.contains(
            "name=\"video_file\"; filename=\"clip.mp4\"\r\nContent-Type: video/mp4\r\n\r\nFILEBYTES\r\n"))
        XCTAssertTrue(text.hasSuffix("--BOUND--\r\n"))
    }

    func testFormURLEncoding() {
        let encoded = Multipart.formURLEncoded([
            ("calibration_json", #"{"a": 1}"#), ("start_time", "0")])
        XCTAssertEqual(encoded, "calibration_json=%7B%22a%22%3A%201%7D&start_time=0")
    }
}
```

- [ ] **Step 2 [Mac]: Run tests to verify they fail**

Run: the standard Swift test command.
Expected: compile failure (types missing). If batching Mac runs, proceed and verify after Task 11.

- [ ] **Step 3: Implement `ios/Sources/API/Models.swift`**

```swift
import Foundation

struct UploadResponse: Decodable, Equatable {
    let ok: Bool
    let videoID: String
    let fps: Double?
    let frameCount: Int?
    let duration: Double?

    enum CodingKeys: String, CodingKey {
        case ok, fps, duration
        case videoID = "video_id"
        case frameCount = "frame_count"
    }
}

struct Hit: Decodable, Equatable, Identifiable {
    var id: Int { frame }
    let frame: Int
    let timestampSeconds: Double
    let call: String            // IN | OUT | UNKNOWN | AUDIO
    let marginPx: Double?
    let eventType: String?
    let hasTargetZone: Bool
    let hasWallDiagram: Bool

    enum CodingKeys: String, CodingKey {
        case frame, call
        case timestampSeconds = "timestamp_seconds"
        case marginPx = "margin_px"
        case eventType = "event_type"
        case targetZone = "target_zone"
        case wallDiagram = "wall_diagram"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        frame = try container.decode(Int.self, forKey: .frame)
        timestampSeconds = try container.decode(Double.self, forKey: .timestampSeconds)
        call = try container.decode(String.self, forKey: .call)
        marginPx = try container.decodeIfPresent(Double.self, forKey: .marginPx)
        eventType = try container.decodeIfPresent(String.self, forKey: .eventType)
        // Presence-only: the shapes are the server's business (opaque here).
        hasTargetZone = container.contains(.targetZone)
            && !((try? container.decodeNil(forKey: .targetZone)) ?? true)
        hasWallDiagram = container.contains(.wallDiagram)
            && !((try? container.decodeNil(forKey: .wallDiagram)) ?? true)
    }
}

extension Array where Element == Hit {
    /// Mirror of app.py front_wall_hits_from_payload.
    var frontWall: [Hit] {
        filter {
            $0.hasTargetZone && $0.hasWallDiagram
                && ($0.eventType == nil || $0.eventType == "wall" || $0.eventType == "unknown")
        }
    }
}

struct JobStatus: Decodable, Equatable {
    let ok: Bool
    let status: String          // queued | running | complete | failed
    let runID: String?
    let stage: String?
    let progress: Double?
    let processedFrames: Int?
    let totalFrames: Int?
    let message: String?
    let error: String?
    let hits: [Hit]?

    enum CodingKeys: String, CodingKey {
        case ok, status, stage, progress, message, error, hits
        case runID = "run_id"
        case processedFrames = "processed_frames"
        case totalFrames = "total_frames"
    }
}

struct LatestCalibration: Equatable {
    let runID: String
    /// Raw JSON re-posted verbatim as /api/track's calibration_json field.
    let calibrationJSON: String

    init(responseData: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let runID = object["run_id"] as? String,
              let calibration = object["calibration"],
              JSONSerialization.isValidJSONObject(calibration) else {
            throw APIError.badResponse
        }
        self.runID = runID
        let data = try JSONSerialization.data(withJSONObject: calibration)
        self.calibrationJSON = String(decoding: data, as: UTF8.self)
    }
}

enum APIError: LocalizedError, Equatable {
    case badResponse
    case http(Int, String?)
    case noCalibration

    var errorDescription: String? {
        switch self {
        case .badResponse: return "The server sent an unexpected response."
        case .http(let code, let message): return message ?? "Server error (\(code))."
        case .noCalibration:
            return "No court calibration found. Calibrate one run from the web app first."
        }
    }
}
```

- [ ] **Step 4: Implement `ios/Sources/API/Multipart.swift`**

```swift
import Foundation

enum Multipart {
    static func body(boundary: String, fields: [(String, String)],
                     fileField: String, filename: String,
                     contentType: String, fileData: Data) -> Data {
        var data = Data()
        func append(_ string: String) { data.append(Data(string.utf8)) }
        for (name, value) in fields {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(contentType)\r\n\r\n")
        data.append(fileData)
        append("\r\n--\(boundary)--\r\n")
        return data
    }

    static func formURLEncoded(_ fields: [(String, String)]) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return fields.map { name, value in
            let encoded = value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
            return "\(name)=\(encoded)"
        }.joined(separator: "&")
    }
}
```

- [ ] **Step 5: Implement `ios/Sources/API/APIClient.swift`**

```swift
import Foundation

protocol APIClientProtocol: Sendable {
    func latestCalibration() async throws -> LatestCalibration
    func upload(videoURL: URL) async throws -> UploadResponse
    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus
    func trackStatus(runID: String) async throws -> JobStatus
}

struct APIClient: APIClientProtocol {
    var baseURL: URL = Config.baseURL
    var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 15 * 60   // big rally uploads
        return URLSession(configuration: config)
    }()

    func latestCalibration() async throws -> LatestCalibration {
        let url = baseURL.appending(path: "api/calibration/latest")
        let (data, response) = try await session.data(from: url)
        if (response as? HTTPURLResponse)?.statusCode == 404 { throw APIError.noCalibration }
        try Self.checkHTTP(response, data: data)
        return try LatestCalibration(responseData: data)
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        let boundary = "slc-\(UUID().uuidString)"
        var request = URLRequest(url: baseURL.appending(path: "api/upload"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        let fileData = try Data(contentsOf: videoURL)   // demo rallies: tens of MB
        let body = Multipart.body(
            boundary: boundary, fields: [],
            fileField: "video_file", filename: videoURL.lastPathComponent,
            contentType: "video/mp4", fileData: fileData)
        let (data, response) = try await session.upload(for: request, from: body)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(UploadResponse.self, from: data)
    }

    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        var request = URLRequest(url: baseURL.appending(path: "api/track"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded",
                         forHTTPHeaderField: "Content-Type")
        let form = Multipart.formURLEncoded([
            ("video_id", videoID),
            ("calibration_json", calibrationJSON),
            ("start_time", "0"),
            ("end_time", String(duration)),
            ("frame_stride", "4"),
            ("inference_width", "960"),
            ("event_engine", ""),
            ("fusion_3d", ""),
        ])
        request.httpBody = Data(form.utf8)
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        let url = baseURL.appending(path: "api/track/status/\(runID)")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    private static func checkHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(
                [String: String].self, from: data))?["error"]
            throw APIError.http(http.statusCode, message)
        }
    }
}
```

- [ ] **Step 6 [Mac]: Run tests to verify they pass**

Run: the standard Swift test command.
Expected: ModelsTests (4) + MultipartTests (2) + Smoke (1) pass.

- [ ] **Step 7: Commit**

```bash
git add ios/Sources/API ios/Tests/ModelsTests.swift ios/Tests/MultipartTests.swift
git commit -m "feat: iOS API models and client for upload/track/status/calibration"
```

---

### Task 7: RingBuffer + BallTracker + CoreMLBallDetector

**Files:**
- Create: `ios/Sources/Tracking/RingBuffer.swift`, `ios/Sources/Tracking/BallTracker.swift`, `ios/Sources/Tracking/CoreMLBallDetector.swift`
- Test: `ios/Tests/RingBufferTests.swift`, `ios/Tests/BallTrackerTests.swift`

**Interfaces:**
- Produces (Task 9 consumes):
  - `struct BallObservation: Equatable { timestamp: TimeInterval; rect: CGRect /* Vision-normalized, origin bottom-left */; confidence: Float }`
  - `protocol BallDetecting { func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? }`
  - `final class BallTracker { init(detector: BallDetecting?); var isEnabled: Bool; func subscribe(_:); func process(_:timestamp:); var recent: [BallObservation] }`
  - `final class CoreMLBallDetector: BallDetecting` (`init?()` — nil when the model bundle is missing)
  - `struct RingBuffer<Element> { init(capacity:); mutating func append(_:); var elements: [Element]; var count: Int }`

- [ ] **Step 1: Write the failing tests**

`ios/Tests/RingBufferTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

final class RingBufferTests: XCTestCase {
    func testAppendsInOrderBelowCapacity() {
        var buffer = RingBuffer<Int>(capacity: 3)
        buffer.append(1); buffer.append(2)
        XCTAssertEqual(buffer.elements, [1, 2])
        XCTAssertEqual(buffer.count, 2)
    }

    func testWrapsKeepingNewestOldestFirst() {
        var buffer = RingBuffer<Int>(capacity: 3)
        for value in 1...5 { buffer.append(value) }
        XCTAssertEqual(buffer.elements, [3, 4, 5])
        XCTAssertEqual(buffer.count, 3)
    }
}
```

`ios/Tests/BallTrackerTests.swift`:

```swift
import XCTest
import CoreVideo
@testable import SquashLineCalling

private final class ScriptedDetector: BallDetecting {
    var results: [BallObservation?]
    init(results: [BallObservation?]) { self.results = results }
    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        results.isEmpty ? nil : results.removeFirst()
    }
}

final class BallTrackerTests: XCTestCase {
    private func pixelBuffer() -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &buffer)
        return buffer!
    }

    private func observation(_ t: TimeInterval) -> BallObservation {
        BallObservation(timestamp: t,
                        rect: CGRect(x: 0.4, y: 0.5, width: 0.02, height: 0.02),
                        confidence: 0.9)
    }

    func testHitNotifiesSubscribersAndBuffers() {
        let tracker = BallTracker(detector: ScriptedDetector(
            results: [observation(1.0), nil, observation(2.0)]))
        var received: [BallObservation] = []
        let expectation = expectation(description: "two notifications")
        expectation.expectedFulfillmentCount = 2
        tracker.subscribe { received.append($0); expectation.fulfill() }

        let buffer = pixelBuffer()
        tracker.process(buffer, timestamp: 1.0)   // hit
        tracker.process(buffer, timestamp: 1.5)   // miss: no notify, no buffer
        tracker.process(buffer, timestamp: 2.0)   // hit

        wait(for: [expectation], timeout: 1.0)
        XCTAssertEqual(received.map(\.timestamp), [1.0, 2.0])
        XCTAssertEqual(tracker.recent.map(\.timestamp), [1.0, 2.0])
    }

    func testNilDetectorDisablesTracking() {
        let tracker = BallTracker(detector: nil)
        XCTAssertFalse(tracker.isEnabled)
        tracker.process(pixelBuffer(), timestamp: 1.0)
        XCTAssertTrue(tracker.recent.isEmpty)
    }
}
```

- [ ] **Step 2: Implement `ios/Sources/Tracking/RingBuffer.swift`**

```swift
/// Fixed-capacity FIFO over an array. Oldest-first snapshot via `elements`.
struct RingBuffer<Element> {
    private var storage: [Element] = []
    private var next = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = max(1, capacity)
        storage.reserveCapacity(self.capacity)
    }

    var count: Int { storage.count }

    mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
        } else {
            storage[next] = element
        }
        next = (next + 1) % capacity
    }

    var elements: [Element] {
        guard storage.count == capacity else { return storage }
        return Array(storage[next...]) + Array(storage[..<next])
    }
}
```

- [ ] **Step 3: Implement `ios/Sources/Tracking/BallTracker.swift`**

```swift
import CoreVideo
import Foundation

struct BallObservation: Equatable {
    let timestamp: TimeInterval
    /// Vision-normalized bounding box: [0,1] with origin at BOTTOM-left.
    let rect: CGRect
    let confidence: Float
}

protocol BallDetecting {
    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation?
}

/// One producer (the capture queue), many consumers. v1 consumers: the live
/// overlay and a ring buffer. The real-time in/out phase adds a bounce
/// detector as a third subscriber — that is the whole migration path, so
/// keep this class free of UI or network concerns.
final class BallTracker {
    static let bufferCapacity = 900   // ~30 s at 30 fps

    private let detector: BallDetecting?
    private let lock = NSLock()
    private var buffer = RingBuffer<BallObservation>(capacity: BallTracker.bufferCapacity)
    private var subscribers: [(BallObservation) -> Void] = []

    var isEnabled: Bool { detector != nil }

    init(detector: BallDetecting?) {
        self.detector = detector
    }

    func subscribe(_ subscriber: @escaping (BallObservation) -> Void) {
        lock.lock(); defer { lock.unlock() }
        subscribers.append(subscriber)
    }

    var recent: [BallObservation] {
        lock.lock(); defer { lock.unlock() }
        return buffer.elements
    }

    /// Called on the capture queue for every frame.
    func process(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        guard let observation = detector?.detect(pixelBuffer, timestamp: timestamp) else { return }
        lock.lock()
        buffer.append(observation)
        let currentSubscribers = subscribers
        lock.unlock()
        DispatchQueue.main.async {
            for subscriber in currentSubscribers { subscriber(observation) }
        }
    }
}
```

- [ ] **Step 4: Implement `ios/Sources/Tracking/CoreMLBallDetector.swift`**

```swift
import CoreVideo
import Foundation
import Vision

/// Runs the bundled YOLO Core ML model (ios/Model/BallDetector.mlpackage,
/// exported with nms=True so Vision yields VNRecognizedObjectObservation).
/// init fails soft when the model is absent so the app still builds/runs
/// before the training workstream lands — RecordView shows a badge instead.
final class CoreMLBallDetector: BallDetecting {
    static let modelName = "BallDetector"
    static let confidenceThreshold: Float = 0.25   // tracking_common parity

    private let model: VNCoreMLModel

    init?() {
        guard let url = Bundle.main.url(forResource: Self.modelName,
                                        withExtension: "mlmodelc") else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // let Core ML place it on the ANE
        guard let coreml = try? MLModel(contentsOf: url, configuration: configuration),
              let vnModel = try? VNCoreMLModel(for: coreml) else { return nil }
        self.model = vnModel
    }

    func detect(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> BallObservation? {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill   // matches YOLO letterbox-free export
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
        let best = (request.results as? [VNRecognizedObjectObservation])?
            .filter { $0.confidence >= Self.confidenceThreshold }
            .max(by: { $0.confidence < $1.confidence })
        guard let best else { return nil }
        return BallObservation(timestamp: timestamp,
                               rect: best.boundingBox,
                               confidence: best.confidence)
    }
}
```

(`import CoreML` comes via Vision; add `import CoreML` explicitly if the compiler asks.)

- [ ] **Step 5 [Mac]: Run tests**

Expected: RingBufferTests (2) + BallTrackerTests (2) pass along with earlier suites.

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Tracking ios/Tests/RingBufferTests.swift ios/Tests/BallTrackerTests.swift
git commit -m "feat: BallTracker pipeline — ring buffer, subscribers, Core ML detector"
```

---

### Task 8: CameraController — capture + synchronized recording

**Files:**
- Create: `ios/Sources/Record/CameraController.swift`

**Interfaces:**
- Produces (Task 9 consumes): `final class CameraController { let session: AVCaptureSession; var onVideoSample: ((CVPixelBuffer, TimeInterval) -> Void)?; func configure() async throws; func start(); func stop(); func startRecording() throws; func stopRecording() async throws -> URL }`. The overlay and the recorded file are fed by the SAME sample buffers, so they cannot drift.
- No unit tests: every path needs camera hardware. Verified on-device in Task 9's step and the day-10 rehearsal.

- [ ] **Step 1: Implement `ios/Sources/Record/CameraController.swift`**

```swift
import AVFoundation
import Foundation

final class CameraController: NSObject {
    enum CameraError: LocalizedError {
        case permissionDenied, configurationFailed, notRecording

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Camera or microphone access was denied. Enable both in Settings."
            case .configurationFailed: return "The camera could not be configured."
            case .notRecording: return "No recording is in progress."
            }
        }
    }

    let session = AVCaptureSession()
    /// Every video frame, on the output queue. RecordView wires this to
    /// BallTracker.process.
    var onVideoSample: ((CVPixelBuffer, TimeInterval) -> Void)?

    private let sessionQueue = DispatchQueue(label: "slc.camera.session")
    // One queue for BOTH outputs: writer state below is queue-confined to it.
    private let outputQueue = DispatchQueue(label: "slc.camera.output")

    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()

    private var writer: AVAssetWriter?
    private var writerVideo: AVAssetWriterInput?
    private var writerAudio: AVAssetWriterInput?
    private var writerSessionStarted = false
    private var outputURL: URL?

    func configure() async throws {
        let camera = await AVCaptureDevice.requestAccess(for: .video)
        let microphone = await AVCaptureDevice.requestAccess(for: .audio)
        guard camera && microphone else { throw CameraError.permissionDenied }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do { try self.configureSession(); continuation.resume() }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let cameraInput = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(cameraInput) else {
            throw CameraError.configurationFailed
        }
        session.addInput(cameraInput)

        if let microphone = AVCaptureDevice.default(for: .audio),
           let microphoneInput = try? AVCaptureDeviceInput(device: microphone),
           session.canAddInput(microphoneInput) {
            session.addInput(microphoneInput)   // audio rescue needs the track
        }

        videoOutput.videoSettings =
            [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        guard session.canAddOutput(videoOutput) else { throw CameraError.configurationFailed }
        session.addOutput(videoOutput)

        if session.canAddOutput(audioOutput) {
            audioOutput.setSampleBufferDelegate(self, queue: outputQueue)
            session.addOutput(audioOutput)
        }

        // Portrait upright to match the locked UI orientation.
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
    }

    func start() {
        sessionQueue.async {
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    func startRecording() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rally-\(Int(Date().timeIntervalSince1970)).mp4")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 1080,     // portrait: rotated 1920x1080
            AVVideoHeightKey: 1920,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 12_000_000],
        ])
        video.expectsMediaDataInRealTime = true

        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 96_000,
        ])
        audio.expectsMediaDataInRealTime = true

        guard writer.canAdd(video), writer.canAdd(audio) else {
            throw CameraError.configurationFailed
        }
        writer.add(video)
        writer.add(audio)
        guard writer.startWriting() else {
            throw writer.error ?? CameraError.configurationFailed
        }

        outputQueue.sync {
            self.writer = writer
            self.writerVideo = video
            self.writerAudio = audio
            self.writerSessionStarted = false
            self.outputURL = url
        }
    }

    func stopRecording() async throws -> URL {
        let (writer, video, audio, url) = outputQueue.sync {
            let state = (self.writer, self.writerVideo, self.writerAudio, self.outputURL)
            self.writer = nil
            self.writerVideo = nil
            self.writerAudio = nil
            self.outputURL = nil
            return state
        }
        guard let writer, let url else { throw CameraError.notRecording }
        video?.markAsFinished()
        audio?.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw writer.error ?? CameraError.configurationFailed
        }
        return url
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate,
                            AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if output === videoOutput,
           let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            onVideoSample?(pixelBuffer, CMTimeGetSeconds(timestamp))
        }

        guard let writer else { return }
        if output === videoOutput {
            if !writerSessionStarted {
                writer.startSession(atSourceTime: timestamp)
                writerSessionStarted = true
            }
            if let input = writerVideo, input.isReadyForMoreMediaData {
                input.append(sampleBuffer)
            }
        } else if writerSessionStarted {
            if let input = writerAudio, input.isReadyForMoreMediaData {
                input.append(sampleBuffer)
            }
        }
    }
}
```

- [ ] **Step 2 [Mac]: Compile check**

Run: the standard Swift test command (build implies compile of app sources).
Expected: BUILD SUCCEEDED; existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add ios/Sources/Record/CameraController.swift
git commit -m "feat: camera capture with synchronized AVAssetWriter recording"
```

---

### Task 9: Record screen — preview, overlay, controls

**Files:**
- Create: `ios/Sources/Record/Geometry.swift`, `ios/Sources/Record/CameraPreviewView.swift`, `ios/Sources/Record/OverlayView.swift`, `ios/Sources/Record/RecordModel.swift`, `ios/Sources/Record/RecordView.swift`
- Modify: `ios/Sources/RootTabView.swift` (Play tab → `RecordView()`)
- Test: `ios/Tests/GeometryTests.swift`

**Interfaces:**
- Consumes: `CameraController` (Task 8), `BallTracker`/`CoreMLBallDetector`/`BallObservation` (Task 7), `Theme` (Task 5).
- Produces: `struct FinishedClip: Identifiable { id: UUID; url: URL; duration: Double }` — Task 10's ResultsView receives it. `Geometry.aspectFitRect(content:container:)` and `Geometry.overlayRect(visionRect:videoRect:)`.

- [ ] **Step 1: Write the failing test**

`ios/Tests/GeometryTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

final class GeometryTests: XCTestCase {
    func testAspectFitLetterboxesTallContainer() {
        // 9:16 video in a 100x300 container: full width, centered vertically.
        let rect = Geometry.aspectFitRect(
            content: CGSize(width: 1080, height: 1920),
            container: CGSize(width: 100, height: 300))
        XCTAssertEqual(rect.origin.x, 0, accuracy: 0.01)
        XCTAssertEqual(rect.width, 100, accuracy: 0.01)
        XCTAssertEqual(rect.height, 100 * 1920 / 1080, accuracy: 0.01)
        XCTAssertEqual(rect.midY, 150, accuracy: 0.01)
    }

    func testOverlayRectFlipsVisionY() {
        // Vision origin is bottom-left; screen origin is top-left.
        let videoRect = CGRect(x: 0, y: 0, width: 100, height: 200)
        let vision = CGRect(x: 0.5, y: 0.0, width: 0.1, height: 0.1) // bottom of frame
        let mapped = Geometry.overlayRect(visionRect: vision, videoRect: videoRect)
        XCTAssertEqual(mapped.origin.x, 50, accuracy: 0.01)
        XCTAssertEqual(mapped.origin.y, 180, accuracy: 0.01)   // near the bottom on screen
        XCTAssertEqual(mapped.size, CGSize(width: 10, height: 20))
    }
}
```

- [ ] **Step 2: Implement `ios/Sources/Record/Geometry.swift`**

```swift
import CoreGraphics

enum Geometry {
    /// Where aspect-fit content lands inside a container (mirrors
    /// AVCaptureVideoPreviewLayer's .resizeAspect).
    static func aspectFitRect(content: CGSize, container: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0 else { return .zero }
        let scale = min(container.width / content.width,
                        container.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(x: (container.width - size.width) / 2,
                      y: (container.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    /// Vision-normalized rect (origin bottom-left) -> screen rect inside the
    /// aspect-fit video area (origin top-left).
    static func overlayRect(visionRect: CGRect, videoRect: CGRect) -> CGRect {
        CGRect(
            x: videoRect.minX + visionRect.minX * videoRect.width,
            y: videoRect.minY + (1 - visionRect.minY - visionRect.height) * videoRect.height,
            width: visionRect.width * videoRect.width,
            height: visionRect.height * videoRect.height)
    }
}
```

- [ ] **Step 3: Implement preview + overlay + model + view**

`ios/Sources/Record/CameraPreviewView.swift`:

```swift
import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}
```

`ios/Sources/Record/OverlayView.swift`:

```swift
import SwiftUI

/// Live ball marker + short fading trail over the camera preview.
/// Portrait capture is 1080x1920; the preview letterboxes with resizeAspect,
/// so map through the same aspect-fit rect.
struct OverlayView: View {
    let trail: [BallObservation]   // oldest first, newest last
    static let contentSize = CGSize(width: 1080, height: 1920)

    var body: some View {
        GeometryReader { proxy in
            let videoRect = Geometry.aspectFitRect(
                content: Self.contentSize, container: proxy.size)
            Canvas { context, _ in
                for (index, observation) in trail.enumerated() {
                    let rect = Geometry.overlayRect(
                        visionRect: observation.rect, videoRect: videoRect)
                    let radius = max(6, rect.width / 2)
                    let circle = CGRect(
                        x: rect.midX - radius, y: rect.midY - radius,
                        width: radius * 2, height: radius * 2)
                    let age = Double(index + 1) / Double(trail.count)   // newest -> 1
                    if index == trail.count - 1 {
                        context.stroke(Path(ellipseIn: circle),
                                       with: .color(Theme.accentBg), lineWidth: 3)
                    } else {
                        context.fill(Path(ellipseIn: circle.insetBy(
                            dx: radius * 0.6, dy: radius * 0.6)),
                            with: .color(Theme.accentBg.opacity(0.15 + 0.5 * age)))
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }
}
```

`ios/Sources/Record/RecordModel.swift`:

```swift
import Foundation

struct FinishedClip: Identifiable {
    let id = UUID()
    let url: URL
    let duration: Double
}

@MainActor
final class RecordModel: ObservableObject {
    let camera = CameraController()
    let tracker: BallTracker

    @Published var trail: [BallObservation] = []
    @Published var isRecording = false
    @Published var recordingStartedAt: Date?
    @Published var errorText: String?
    @Published var finishedClip: FinishedClip?   // non-nil presents ResultsView

    private static let trailLength = 15

    var detectorMissing: Bool { !tracker.isEnabled }

    init(detector: BallDetecting? = CoreMLBallDetector()) {
        tracker = BallTracker(detector: detector)
        tracker.subscribe { [weak self] observation in
            guard let self else { return }
            trail.append(observation)
            if trail.count > Self.trailLength { trail.removeFirst() }
        }
        camera.onVideoSample = { [tracker] pixelBuffer, timestamp in
            tracker.process(pixelBuffer, timestamp: timestamp)
        }
    }

    func startCamera() async {
        do {
            try await camera.configure()
            camera.start()
        } catch {
            errorText = error.localizedDescription
        }
    }

    func toggleRecording() async {
        if isRecording {
            do {
                let url = try await camera.stopRecording()
                let duration = recordingStartedAt.map {
                    Date().timeIntervalSince($0)
                } ?? 0
                isRecording = false
                recordingStartedAt = nil
                finishedClip = FinishedClip(url: url, duration: duration)
            } catch {
                isRecording = false
                recordingStartedAt = nil
                errorText = error.localizedDescription
            }
        } else {
            do {
                try camera.startRecording()
                isRecording = true
                recordingStartedAt = Date()
                errorText = nil
            } catch {
                errorText = error.localizedDescription
            }
        }
    }
}
```

`ios/Sources/Record/RecordView.swift`:

```swift
import SwiftUI

struct RecordView: View {
    @StateObject private var model = RecordModel()

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            CameraPreviewView(session: model.camera.session).ignoresSafeArea()
            OverlayView(trail: model.trail).ignoresSafeArea()

            VStack {
                if model.detectorMissing {
                    Text("Ball model missing — overlay disabled")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.dim)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
                if let errorText = model.errorText {
                    Text(errorText)
                        .font(.footnote)
                        .foregroundStyle(Theme.text)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Theme.surface, in: Capsule())
                        .padding(.top, 8)
                }
                Spacer()
                recordControls
            }
        }
        .task { await model.startCamera() }
        .sheet(item: $model.finishedClip) { clip in
            ResultsView(clip: clip)
        }
    }

    private var recordControls: some View {
        VStack(spacing: 12) {
            if model.isRecording, let start = model.recordingStartedAt {
                Text(start, style: .timer)
                    .font(.system(.title3, design: .monospaced).weight(.semibold))
                    .foregroundStyle(Theme.text)
            }
            Button {
                Task { await model.toggleRecording() }
            } label: {
                ZStack {
                    Circle().stroke(Theme.text, lineWidth: 4).frame(width: 76, height: 76)
                    if model.isRecording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Theme.outCall).frame(width: 32, height: 32)
                    } else {
                        Circle().fill(Theme.accentBg).frame(width: 62, height: 62)
                    }
                }
            }
            .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
        }
        .padding(.bottom, 24)
    }
}
```

Until Task 10 exists, add a temporary `ResultsView` stub so the target compiles — Task 10 replaces it:

```swift
// ios/Sources/Results/ResultsView.swift (stub, replaced in the next task)
import SwiftUI

struct ResultsView: View {
    let clip: FinishedClip
    var body: some View {
        Text("Recorded \(Int(clip.duration))s clip")
            .foregroundStyle(Theme.text)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
    }
}
```

- [ ] **Step 4: Point the Play tab at RecordView**

In `ios/Sources/RootTabView.swift`, replace `Text("Record")` with `RecordView()`.

- [ ] **Step 5 [Mac]: Run tests + on-device smoke**

Run: the standard Swift test command — GeometryTests (2) pass, build green.
On a physical iPhone: camera preview appears, permission prompts fire, record button toggles, stopping presents the stub sheet. (Overlay stays dark until the model lands in Task 12 — the "model missing" badge must show.)

- [ ] **Step 6: Commit**

```bash
git add ios/Sources/Record ios/Sources/Results ios/Sources/RootTabView.swift ios/Tests/GeometryTests.swift
git commit -m "feat: native record screen with live overlay plumbing"
```

---

### Task 10: RunSubmission state machine + ResultsView

**Files:**
- Create: `ios/Sources/Results/RunSubmission.swift`
- Replace: `ios/Sources/Results/ResultsView.swift` (the Task 9 stub)
- Test: `ios/Tests/RunSubmissionTests.swift`

**Interfaces:**
- Consumes: `APIClientProtocol`, `JobStatus`, `Hit`, `.frontWall`, `APIError.noCalibration` (Task 6); `FinishedClip` (Task 9).
- Produces: `RunSubmission.Phase` enum (Task 11's full-review link uses `completedRunID`).

- [ ] **Step 1: Write the failing test**

`ios/Tests/RunSubmissionTests.swift`:

```swift
import XCTest
@testable import SquashLineCalling

private struct FakeAPI: APIClientProtocol {
    var calibration: Result<LatestCalibration, Error>
    var statuses: [JobStatus]   // consumed by successive trackStatus calls

    final class Cursor: @unchecked Sendable { var index = 0 }
    let cursor = Cursor()

    func latestCalibration() async throws -> LatestCalibration {
        try calibration.get()
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        UploadResponse(ok: true, videoID: "vid-1", fps: 30, frameCount: 900, duration: 30)
    }

    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        statuses[0]
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        cursor.index = min(cursor.index + 1, statuses.count - 1)
        return statuses[cursor.index]
    }
}

@MainActor
final class RunSubmissionTests: XCTestCase {
    private func calibration() throws -> LatestCalibration {
        try LatestCalibration(responseData: Data(
            #"{"ok": true, "run_id": "7", "calibration": {"lines": []}}"#.utf8))
    }

    private func status(_ status: String, hits: [Hit]? = nil) -> JobStatus {
        JobStatus(ok: true, status: status, runID: "run-9", stage: nil,
                  progress: 50, processedFrames: 1, totalFrames: 2,
                  message: "msg", error: nil, hits: hits)
    }

    func testHappyPathReachesComplete() async throws {
        let api = FakeAPI(
            calibration: .success(try calibration()),
            statuses: [status("queued"), status("running"), status("complete")])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .complete(let job) = submission.phase else {
            return XCTFail("expected complete, got \(submission.phase)")
        }
        XCTAssertEqual(job.runID, "run-9")
        XCTAssertEqual(submission.completedRunID, "run-9")
    }

    func testMissingCalibrationFailsWithActionableMessage() async {
        let api = FakeAPI(calibration: .failure(APIError.noCalibration), statuses: [])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected failed")
        }
        XCTAssertTrue(message.contains("Calibrate"))
    }

    func testServerFailureSurfacesError() async throws {
        let failed = JobStatus(ok: true, status: "failed", runID: "run-9", stage: nil,
                               progress: nil, processedFrames: nil, totalFrames: nil,
                               message: nil, error: "Tracking failed hard.", hits: nil)
        let api = FakeAPI(calibration: .success(try calibration()),
                          statuses: [status("queued"), failed])
        let submission = RunSubmission(api: api, pollInterval: .zero)
        await submission.submit(videoURL: URL(fileURLWithPath: "/tmp/x.mp4"), duration: 30)
        guard case .failed(let message) = submission.phase else {
            return XCTFail("expected failed")
        }
        XCTAssertEqual(message, "Tracking failed hard.")
    }
}
```

Note: `JobStatus` needs a memberwise initializer for these tests — since it declares CodingKeys only, Swift still synthesizes the memberwise init internally; `@testable import` reaches it.

- [ ] **Step 2: Implement `ios/Sources/Results/RunSubmission.swift`**

```swift
import Foundation

/// Record-to-results pipeline: latest calibration -> upload -> start track ->
/// poll until complete/failed. Mirrors the web app's runTrackBtn flow.
@MainActor
final class RunSubmission: ObservableObject {
    enum Phase: Equatable {
        case idle
        case fetchingCalibration
        case uploading
        case tracking(progress: Double, message: String)
        case complete(JobStatus)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    private let api: APIClientProtocol
    private let pollInterval: Duration

    init(api: APIClientProtocol = APIClient(), pollInterval: Duration = .seconds(1)) {
        self.api = api
        self.pollInterval = pollInterval
    }

    var completedRunID: String? {
        if case .complete(let job) = phase { return job.runID }
        return nil
    }

    func submit(videoURL: URL, duration: Double) async {
        do {
            phase = .fetchingCalibration
            let calibration = try await api.latestCalibration()

            phase = .uploading
            let upload = try await api.upload(videoURL: videoURL)
            let clipDuration = upload.duration ?? duration

            var job = try await api.startTrack(
                videoID: upload.videoID,
                calibrationJSON: calibration.calibrationJSON,
                duration: clipDuration)

            while job.status == "queued" || job.status == "running" {
                phase = .tracking(progress: job.progress ?? 0,
                                  message: job.message ?? "Analyzing…")
                try await Task.sleep(for: pollInterval)
                guard let runID = job.runID else { throw APIError.badResponse }
                job = try await api.trackStatus(runID: runID)
            }

            if job.status == "complete" {
                phase = .complete(job)
            } else {
                phase = .failed(job.error ?? "Tracking failed.")
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }
}
```

- [ ] **Step 3: Replace the ResultsView stub**

```swift
import SwiftUI

struct ResultsView: View {
    let clip: FinishedClip
    @StateObject private var submission = RunSubmission()
    @State private var showFullReview = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.bg.ignoresSafeArea()
                content
            }
            .navigationTitle("Rally")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }.tint(Theme.accentBg)
                }
            }
        }
        .task { await submission.submit(videoURL: clip.url, duration: clip.duration) }
        .fullScreenCover(isPresented: $showFullReview) {
            if let runID = submission.completedRunID {
                WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#run=\(runID)")!,
                          showsClose: true)
            }
        }
    }

    @ViewBuilder private var content: some View {
        switch submission.phase {
        case .idle, .fetchingCalibration:
            progress("Fetching court calibration…")
        case .uploading:
            progress("Uploading rally…")
        case .tracking(let percent, let message):
            VStack(spacing: 12) {
                ProgressView(value: percent, total: 100).tint(Theme.accentBg)
                Text(message).font(.footnote).foregroundStyle(Theme.dim)
            }
            .padding(24)
        case .failed(let message):
            VStack(spacing: 16) {
                Text(message)
                    .foregroundStyle(Theme.text)
                    .multilineTextAlignment(.center)
                Button("Try again") {
                    Task { await submission.submit(videoURL: clip.url,
                                                   duration: clip.duration) }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentBg).foregroundStyle(Theme.accentText)
            }
            .padding(24)
        case .complete(let job):
            completeList(job)
        }
    }

    private func progress(_ label: String) -> some View {
        VStack(spacing: 12) {
            ProgressView().tint(Theme.accentBg)
            Text(label).font(.footnote).foregroundStyle(Theme.dim)
        }
    }

    private func completeList(_ job: JobStatus) -> some View {
        let hits = (job.hits ?? []).frontWall
        return ScrollView {
            VStack(spacing: 10) {
                if hits.isEmpty {
                    Text("No front-wall hits detected in this rally.")
                        .foregroundStyle(Theme.dim)
                        .padding(.top, 40)
                }
                ForEach(Array(hits.enumerated()), id: \.element.id) { index, hit in
                    HStack {
                        Text("Hit \(index + 1)")
                            .foregroundStyle(Theme.text)
                            .font(.body.weight(.semibold))
                        Text(String(format: "%.1fs", hit.timestampSeconds))
                            .foregroundStyle(Theme.dim).font(.footnote)
                        Spacer()
                        callChip(hit)
                    }
                    .padding(14)
                    .background(Theme.surface,
                                in: RoundedRectangle(cornerRadius: 12))
                }
                Button("Open full review") { showFullReview = true }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accentBg).foregroundStyle(Theme.accentText)
                    .padding(.top, 8)
            }
            .padding(16)
        }
    }

    private func callChip(_ hit: Hit) -> some View {
        let color: Color = hit.call == "IN" ? Theme.inCall
            : hit.call == "OUT" ? Theme.outCall : Theme.unknown
        let margin = hit.marginPx.map {
            String(format: " %+.1f px", $0)
        } ?? ""
        return Text(hit.call + margin)
            .font(.footnote.weight(.bold))
            .foregroundStyle(.black)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(color, in: Capsule())
    }
}
```

- [ ] **Step 4 [Mac]: Run tests**

Expected: RunSubmissionTests (3) pass; compile of ResultsView requires `WebScreen` — if executing tasks strictly in order, keep Task 11's `WebScreen` as the next immediate task or add it in the same Mac verification batch.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Results ios/Tests/RunSubmissionTests.swift
git commit -m "feat: upload/track/poll state machine and native results card"
```

---

### Task 11: WebScreen + Matches/Coach tabs

**Files:**
- Create: `ios/Sources/Web/WebScreen.swift`
- Modify: `ios/Sources/RootTabView.swift`

**Interfaces:**
- Consumes: deep-link URL contract from Task 2; `Config.baseURL`.
- Produces: `struct WebScreen: View { init(url: URL, showsClose: Bool = false) }`.

- [ ] **Step 1: Implement `ios/Sources/Web/WebScreen.swift`**

```swift
import SwiftUI
import WebKit

/// The existing web product inside the native shell. Remote by design:
/// web fixes ship without an app update (spec decision).
struct WebScreen: View {
    let url: URL
    var showsClose = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack(alignment: .topTrailing) {
            WebViewRepresentable(url: url).ignoresSafeArea(edges: .bottom)
            if showsClose {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundStyle(Theme.dim)
                        .padding(12)
                }
                .accessibilityLabel("Close review")
            }
        }
        .background(Theme.bg)
    }
}

private struct WebViewRepresentable: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.isOpaque = false
        webView.backgroundColor = .black
        #if DEBUG
        webView.isInspectable = true
        #endif
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Loaded once in makeUIView; SwiftUI re-renders must not reload the page.
    }
}
```

- [ ] **Step 2: Wire the tabs**

`ios/Sources/RootTabView.swift`:

```swift
import SwiftUI

struct RootTabView: View {
    var body: some View {
        TabView {
            RecordView()
                .tabItem { Label("Play", systemImage: "record.circle") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=matches")!)
                .tabItem { Label("Matches", systemImage: "square.stack") }
            WebScreen(url: URL(string: Config.baseURL.absoluteString + "/#tab=coach")!)
                .tabItem { Label("Coach", systemImage: "figure.tennis") }
        }
        .tint(Theme.accentBg)
        .background(Theme.bg)
    }
}
```

- [ ] **Step 3 [Mac]: Full test run + simulator smoke**

Run: the standard Swift test command — all suites pass (Smoke 1, Models 4, Multipart 2, RingBuffer 2, BallTracker 2, Geometry 2, RunSubmission 3 = 16).
Simulator with `serverBase` pointed at a LAN Flask instance (`HOST=0.0.0.0 PORT=5188 venv/bin/python app.py`, set UserDefaults via scheme argument `-serverBase http://<lan-ip>:5188`; for plain-HTTP LAN testing add a DEBUG-only ATS exception or use the deployed HTTPS origin): Matches/Coach tabs render the web app; a completed run's `#run=` URL restores review.

- [ ] **Step 4: Commit**

```bash
git add ios/Sources/Web ios/Sources/RootTabView.swift
git commit -m "feat: webview tabs and full-review deep link"
```

---

### Task 12: Ops — model export doc, cloud deploy, TestFlight release

No code; three sets of runnable artifacts + checklists. `[Mac]`/ops throughout.

**Files:**
- Create: `ios/MODEL.md`, `deploy/Caddyfile`, `deploy/squash-line-calling.service`, `deploy/DEPLOY.md`

**Interfaces:**
- Consumes: `train_yolo_ball.py` (Task 4), `yolo_model_eval.py` (Task 3), scaffold + TESTFLIGHT.md (Task 5), `Config.defaultBase` (Task 5).

- [ ] **Step 1: Write `ios/MODEL.md`**

```markdown
# Ball model: train → score → export → verify ANE

## 1. Train (any GPU box)

    pip install ultralytics roboflow
    python train_yolo_ball.py --workspace <slug> --dataset-version <n>

Prints `best weights: .../best.pt`.

## 2. Score before shipping (acceptance gate)

    python yolo_model_eval.py --weights best.pt --video <bayclub clip>.mp4 \
        --output-csv yolo_eval.csv --annotated yolo_eval.mp4

Accept when BOTH hold:
- Detection rate within 10 points of the RF-DETR run on the same clip
  (produce the baseline with local_model_eval.py if not already on disk).
- The annotated video looks locked-on through rally speed (spec bar:
  "looks locked-on", not frame parity).

## 3. Export to Core ML (Mac only — coremltools)

    pip install ultralytics coremltools
    yolo export model=best.pt format=coreml nms=True half=True imgsz=960

Rename the exported `best.mlpackage` to `BallDetector.mlpackage` and move it
to `ios/Model/`. Regenerate + rebuild; confirm the build log compiles it to
`BallDetector.mlmodelc` (see the note in ios/project.yml if it does not).

## 4. Verify Neural Engine residency

1. Open BallDetector.mlpackage in Xcode → Performance tab → run a
   performance report on a CONNECTED IPHONE (not simulator).
2. Accept when the majority of compute units show Neural Engine and median
   prediction is under 15 ms. This is exactly the check the RF-DETR ONNX
   export failed — do not skip it.
3. In-app sanity: record screen overlay tracks a thrown ball smoothly at
   30 fps with no thermal warnings within a 3-minute rally.
```

- [ ] **Step 2: Write `deploy/Caddyfile`, `deploy/squash-line-calling.service`, `deploy/DEPLOY.md`**

`deploy/Caddyfile`:

```
# Replace with the real domain; Caddy fetches/renews the certificate itself.
squash.example.com {
    reverse_proxy 127.0.0.1:5188
    request_body {
        max_size 2GB
    }
}
```

`deploy/squash-line-calling.service`:

```ini
[Unit]
Description=Squash Line Calling (Flask pipeline)
After=network.target

[Service]
User=squash
WorkingDirectory=/opt/squash-line-calling
Environment=HOST=127.0.0.1
Environment=PORT=5188
ExecStart=/opt/squash-line-calling/venv/bin/python app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

`deploy/DEPLOY.md`:

```markdown
# Cloud deploy (demo-grade)

One VM serves everything: /api/upload, the tracking pipeline (RF-DETR on
CPU), and index.html for the app's webview tabs.

## VM

- 8 vCPU / 16 GB (CPU inference ~275 ms/frame at 960 px; a 30 s rally at
  stride 4 ≈ 225 frames ≈ ~1 min. If rehearsal feels slow, move to a GPU
  box and set TRACKING_BACKEND=torch).
- Ubuntu 22.04+, a DNS A record for the chosen domain.

## Steps

1. `sudo useradd -r -m squash && sudo git clone <repo> /opt/squash-line-calling`
2. `cd /opt/squash-line-calling && python3 -m venv venv && venv/bin/pip install -r requirements.txt`
3. `.env`: ROBOFLOW_API_KEY=... (plus OPENAI_API_KEY for coach text, optional)
4. `sudo cp deploy/squash-line-calling.service /etc/systemd/system/ && sudo systemctl enable --now squash-line-calling`
5. Install Caddy (apt, official repo), edit deploy/Caddyfile with the real
   domain, `sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy`.
6. Check: `curl https://<domain>/api/health` → `{"ok": true, ...}`
7. Point the app at it: `ios/Sources/Config.swift` defaultBase → `https://<domain>`,
   rebuild, upload to TestFlight.

## Access control

Demo-grade: unguessable subdomain, no auth (spec decision). Do not reuse
this setup beyond the demo without adding auth.

## Demo-day flow

1. Mount phone, open the app's Matches tab (webview) → calibrate one run
   from the web flow on-site.
2. Native record → every rally reuses that calibration via
   /api/calibration/latest.
```

- [ ] **Step 3 [Mac/ops]: Execute the checklists**

1. Train + score (MODEL.md 1–2) — record the detection-rate numbers in the PR description.
2. Export + ANE verify (MODEL.md 3–4); commit `ios/Model/BallDetector.mlpackage` (a few MB — fine to commit).
3. Deploy (DEPLOY.md); set `Config.defaultBase`; commit.
4. TESTFLIGHT.md "every subsequent build" → camp group gets the real app.
5. On-court dry run: record a rally → overlay tracks → results card shows calls → full review opens.

- [ ] **Step 4: Commit**

```bash
git add ios/MODEL.md deploy/
git commit -m "docs: model export gate, cloud deploy artifacts, demo runbook"
```

---

## Execution order and the 10-day map

Tasks 1–4 (Python, verifiable here) → Task 5 (scaffold + day-2 TestFlight hello world) → Tasks 6–11 (Swift; batch [Mac] verification allowed) → Task 12 (train/export/deploy/release; steps 1–2 of it can start as soon as Task 4 lands, in parallel with Swift work) → day-10 rehearsal.
