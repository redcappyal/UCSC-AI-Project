# CrossCourt

**Squash training, measured.** Mount a phone on the court, play, and get statistics and
coaching back — then compare them against your last session, so practice becomes a
feedback loop instead of a directionless hour on court.

## Where this is going

Direction set 2026-07-27. Two stages:

**1. Capture.** A recorder built for one purpose: collecting input the analysis stage can
actually use. The phone mounts on the fin beside the back glass door, and the app
recognizes the court lines and builds the court homography while recording. *Not built
yet* — calibration today is a manual tap wizard.

**2. Analysis.** The clip is saved on-device and uploaded by choice. The cloud run returns
a stats dashboard with squash-specific LLM coaching, and keeps past runs so the same
measurables can be tracked over time. *Partly built* — the pipeline and the coaching
report exist; cross-session comparison does not.

Line calling — IN/OUT on every front-wall hit against calibrated out, tin, and service
lines — is where this project started and still works. It is now **one statistic among
many** rather than the product itself.

---

## How it works today

A single-file mobile web app (`index.html`) and a native iOS app over a Flask + OpenCV
pipeline that detects the ball frame by frame, finds the moments it strikes a wall, and
judges each front-wall impact against calibrated out/tin lines. The first contact in each
rally is also checked against a calibrated service line.

```
video ──► ball detection ──► bounce detection ──► classification ──► judging
          (per-frame          (velocity change,    (wall / side wall   (impact point vs
           detection model)    two-stage fit,       / floor / racket)   calibrated wall lines)
                               GB model, audio)
```

| Stage | Where | What it does |
|---|---|---|
| Ball detection | `ball_model.py`, `ball_detector.py`, `job_runner.py` | Per-frame ball positions. Default (`BALL_DETECTOR=local`) is the committed WASB temporal model: native-resolution 416px tiles, 3 consecutive frames per detection, strided coarse pass with true `t±1` neighbours, then a dense refine pass around hit candidates. `BALL_DETECTOR=rfdetr` keeps the hosted Roboflow RF-DETR (`inference_engine.py`) for A/B eval; that is the only path needing `ROBOFLOW_API_KEY`. Every run records which backend produced it (`ball_backend` in job.json and report-v1). **The WASB backend is wired but not yet measured; the line-call eval against `eval_set/BASELINE-2026-07-23.md` is the gate for any recall claim.** |
| Bounce detection | `detect_wall_hits.py`, `bounce_gb_model_detector.py` | Finds impact frames from trajectory kinks. `detect_wall_hits` locates candidates for the refine pass; `bounce_gb_model_detector` labels the events. One path, not a choice — the selectable engines were removed 2026-07-27. |
| Audio rescue | `audio_events.py` | Impact sounds recover bounces the trajectory missed. |
| Classification | `classify_events.py` | Labels each hit wall / side wall / floor / racket. |
| Judging | `judge_call.py`, `court_model.py` | Wall-line calibration → IN/OUT calls plus a service-line call for each rally’s first front-wall contact. Works in raw pixels: a projection preserves which side of a coplanar line a point falls on, so no homography is needed for the call itself. |
| Court mapping | `court_model.py` | The floor homography — image pixels to court feet. Needed for anything *positional* (zones, distances, coverage), which the analysis stage is built on. |
| Coaching | `app.py` | Target-zone analytics over the rally, optionally narrated by an LLM. |
| ~~Stereo fusion~~ | `archive/stereo/` | **Archived 2026-07-27 — do not extend or import.** Two phones fusing into a court-frame 3D track: built and tested, never evaluated. Triangulation needs the ball in *both* views, which multiplies the recall bottleneck instead of fixing it. Restore point: tag `archive/stereo-v1`. Revisit gate: single-view wall-hit recall ~85%. See [archive/stereo/README.md](archive/stereo/README.md). |
| ~~Fusion engine + 3D contacts~~ | `archive/fusion-engine/` | **Archived 2026-07-27 — do not extend or import.** A second selectable bounce engine (audio × derivatives × arcs + a squash sequence grammar) and the `fusion_3d` flag that fitted gravity-constrained 3D arcs in court feet. It was evaluated and failed its gate: the corpus could not measure the 3D delta, and where 3D engaged it placed front-wall contacts 26 ft from the wall. Restore point: tag `archive/fusion-engine-v1`. See [archive/fusion-engine/README.md](archive/fusion-engine/README.md). |

The UI is deliberately one file. `DESIGN.md` is the binding rulebook for anything visual —
read it before touching HTML/CSS/JS, and update it in the same change if you must deviate.

---

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # add your ROBOFLOW_API_KEY
.venv/bin/python app.py
```

Open http://127.0.0.1:5188. For phone access on the same network:

```bash
HOST=0.0.0.0 .venv/bin/python app.py
```

Useful environment variables:

| Variable | Default | Effect |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address. `0.0.0.0` exposes the server to the LAN for phone access. |
| `PORT` | `5188` | Listen port. Avoid 5000 — macOS AirPlay Receiver holds it. |
| `BALL_DETECTOR` | `local` | Ball detector for analysis jobs: `local` (committed WASB temporal model) or `rfdetr` (hosted Roboflow). No silent fallback. |
| `BALL_DEVICE` | `auto` | Device for the local ball detector: CUDA when available, else CPU. MPS is opt-in (`mps`), never auto. |
| `BALL_MAX_BATCH_TILES` | `8` | Tiles per inference batch, capping the manifest's `max_batch_tiles` (32). A bigger batch buys nothing on any measured device — CUDA is flat from batch 1 to 64 (A10: 16.9→18.2 ms/tile) because the model saturates the card on one tile — and above 8 MPS falls off a ~40x cliff (M2 Air: 81.8 ms/tile at batch 8, 3082 at batch 10) and at 4K tile counts returns **wrong numbers**: the Metal command buffer OOMs, torch does not raise, and the heatmap comes back incomplete. Raise it only up to the manifest ceiling, and only with a measurement. |
| `BALL_MODEL_DIR` | `models/crosscourt-wasb-416-v1` | Override the local ball model directory. |
| `ROBOFLOW_API_KEY` | — | Required only when `BALL_DETECTOR=rfdetr`. |
| `ROBOFLOW_MODEL_ID` | `ai-squash-line-tracker/4` | Which hosted detection model to load (rfdetr backend). |
| `TRACKING_BACKEND` | `auto` | rfdetr backend only: `torch` (GPU/MPS) or `onnx` (CPU). |
| `COACH_LLM_PROVIDER` | OpenAI when an API key is present; otherwise local templates | `ollama`, `openai`, or `local`. |
| `OLLAMA_COACH_MODEL` | `qwen3:8b` | Local model used for structured coaching reports. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server used by the Flask app. |
| `OLLAMA_COACH_MAX_TOKENS` | `1200` | Output ceiling for the compact structured coaching report. |
| `OPENAI_API_KEY` | — | Optional. Enables player-specific LLM feedback and drills; falls back to local coaching without it. |
| `OPENAI_COACH_MODEL` | `gpt-5-mini` | Model used for structured coaching reports. |

### Free local coaching with Ollama

On an Apple Silicon Mac, install Ollama and download the default coaching model:

```bash
brew install ollama
brew services start ollama
ollama pull qwen3:8b
```

Then configure `.env`:

```dotenv
COACH_LLM_PROVIDER=ollama
OLLAMA_COACH_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

The coaching endpoint sends only the derived match analytics to the local Ollama
server. If Ollama is stopped or the model is unavailable, the UI falls back to
the built-in local coaching template and displays the reason.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

283 tests, about 9 seconds. They run without the model runtime — `requirements-test.txt` is
the light dependency set CI installs, and it deliberately excludes `inference`/torch.
`ball_track_offline.py`'s one real-model test is marked `requires_model` and deselected by
default; a green run reads "283 passed, 1 deselected". `archive/` is excluded from
collection — its tests are preserved verbatim and import modules that have moved.

---

## Ground truth and evaluation

A line-calling model is only as trustworthy as the labels you measure it against. Two
label streams feed one evaluation set.

**1. In-app corrections** — while reviewing a run, tap a hit to correct its type, ball
position, or bounce timing. Written to `ui_runs/<run_id>/corrections.json`.

**2. Offline labeling** — scrub a video frame by frame and mark every wall hit:

```bash
.venv/bin/python label_hits.py --video path/to/clip.mp4 --labels myclip_wall_hits.csv
```

`h` marks a hit · `a`/`d` step frames · `[`/`]` ±10 · `<`/`>` ±100 · `n`/`N` jump between
labels · `g` go to frame · `s` save · `q` save and quit.

This writes the CSV plus a `.meta.json` sidecar recording which video (by sha256) those
frame numbers index into. Without the sidecar the labels are anonymous integers and cannot
be evaluated against anything.

**Distill and replay:**

```bash
.venv/bin/python build_eval_set.py     # labels -> eval_set/cases.jsonl
.venv/bin/python eval_line_calls.py    # replay all axes
```

`eval_set/` is committed; `ui_runs/` is not. The build therefore **merges** into the
existing set by default, preserving cases whose source runs live on a teammate's machine.
Use `--replace` only when you intend a clean rebuild.

Evaluation axes: IN/OUT accuracy vs human calls, drift since labeling, hit-type confusion
and false-positive rate, ball-position error in pixels, bounce-timing offset, and
missed-bounce rate per type.

A labeled video with no tracking run is excluded from the missed-bounce axis rather than
scored — otherwise "we never ran the model" would look identical to "the model missed
everything."

### Where the system actually stands

Honest numbers, not flattering ones. These track `eval_set/BASELINE-2026-07-23.md` — the
current reference, 113 cases; re-read it rather than this paragraph when they disagree.

- **Recall is the bottleneck.** Of labeled bounces that had a tracking run to compare
  against, the detector missed **71 of 109** — and 65 of those misses are wall hits. The
  precision metrics all read near-perfect, which is not a contradiction: the system is
  accurate about the few events it catches and blind to most of the rally.
- **Recall is now measured at rally scale; precision is not.** The 95 Bay Club label-CSV
  events give the missed-bounce axis real weight, but every precision axis still resolves
  to `n=2`–`n=4`, so none of them can be steered on.
- **There is not a single OUT case in the eval set.** Both human calls on record are IN, so
  the OUT branch of the judge is entirely unmeasured. Line calling is no longer the whole
  product, but an unmeasured branch is still an unmeasured branch — labeling OUT balls is
  cheap and remains high-value.
- **The pivot raises the bar rather than lowering it.** A missed bounce used to mean a
  missing call, which is visible. In a statistics product it means a silently wrong number:
  a rally-length or coverage statistic computed from 35% of the events looks perfectly
  plausible and is simply false. Nothing on the analysis dashboard should ship before the
  recall axis moves.

### Training

```bash
.venv/bin/python train_bounce_classifier.py --labels wall_hits.csv --ball-csv ball_coordinates.csv
```

The train/test split is **chronological, not random**. Feature rows are per-frame with
overlapping context windows, so a random split scatters near-duplicate neighbouring frames
across both sides and reports a score the model never earned. An embargo band drops the
rows whose context windows straddle the cut.

---

## Repository map

| Path | Purpose |
|---|---|
| `index.html` | The entire UI. See `DESIGN.md`. |
| `app.py` | Flask routes: upload, track, judge, corrections, coaching. |
| `job_runner.py` | Tracking pipeline; a run is fully described by its `job.json`. |
| `inference_engine.py` | Model loading and backend selection (torch/MPS or ONNX). |
| `detect_wall_hits.py`, `bounce_gb_model_detector.py` | Bounce detection. |
| `judge_call.py` | The IN/OUT decision, in pixel space against calibrated lines. |
| `court_model.py` | Calibration geometry: the floor homography and the camera solve. |
| `ios/` | Native SwiftUI app. Play records natively; Matches and Coach are webviews. |
| `ball_track_offline.py` | Offline runner for the local ball detector (WASB temporal by default; a YOLOX artifact via `BALL_MODEL_DIR`). |
| `archive/stereo/` | The archived two-phone path. Inert, uncollected, restorable. |
| `label_hits.py` | Offline frame-by-frame labeler. |
| `build_eval_set.py`, `eval_line_calls.py` | Label distillation and evaluation replay. |
| `eval_set/` | The committed, versioned evaluation corpus. |
| `DESIGN.md` | Binding design system for all UI work. |
| `NOTES-overnight.md` | Data-flywheel work log and schema notes. |

Every tracking run records `model_id`, `tracking_backend`, `device`, and `app_version` in
its `job.json`, so a result can always be attributed to the model version that produced it.
