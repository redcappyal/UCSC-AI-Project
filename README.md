# Squash Line Calling

Point a phone at a squash court, record a rally, and get an automated IN/OUT call on
every front-wall hit — plus a coaching report on where the ball actually went.

A single-file mobile web app (`index.html`) over a Flask + OpenCV pipeline that detects
the ball frame by frame, finds the moments it strikes a wall, and judges each front-wall
impact against hand-calibrated out/tin lines. The first contact in each rally is also
checked against a calibrated service line.

---

## How it works

```
video ──► ball detection ──► bounce detection ──► classification ──► judging
          (per-frame          (velocity change,    (wall / side wall   (impact point vs
           detection model)    two-stage fit,       / floor / racket)   calibrated wall lines)
                               GB model, audio)
```

| Stage | Where | What it does |
|---|---|---|
| Ball detection | `inference_engine.py`, `job_runner.py` | Per-frame ball boxes. Coarse strided pass, then a dense refine pass only around hit candidates. |
| Bounce detection | `detect_wall_hits.py`, `bounce_gb_model_detector.py`, `event_engine.py` | Finds impact frames from trajectory kinks. Swappable engines: `votes`, `gb_model`, `fusion`. |
| Audio rescue | `audio_events.py` | Impact sounds recover bounces the trajectory missed. |
| Classification | `classify_events.py` | Labels each hit wall / side wall / floor / racket. |
| Judging | `judge_call.py`, `court_model.py` | Wall-line calibration → normal IN/OUT calls plus a service-line call for each rally’s first front-wall contact. |
| Coaching | `app.py` | Target-zone analytics over the rally, optionally narrated by an LLM. |
| Stereo fusion | `stereo_engine.py`, `stereo_sync.py`, `job_runner.py` | Optional. Two phones recording one rally fuse into a court-frame 3D track and 3D line calls. See [docs/stereo-fusion.md](docs/stereo-fusion.md) — the fused path is built and tested, but **unevaluated** against the monocular baseline. |

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
HOST=0.0.0.0 PORT=5188 .venv/bin/python app.py
```

Useful environment variables:

| Variable | Default | Effect |
|---|---|---|
| `ROBOFLOW_API_KEY` | — | Required for model inference. |
| `ROBOFLOW_MODEL_ID` | `ai-squash-line-tracker/4` | Which detection model to load. |
| `TRACKING_BACKEND` | `auto` | `torch` (GPU/MPS) or `onnx` (CPU). |
| `COACH_LLM_PROVIDER` | OpenAI when an API key is present; otherwise local templates | `ollama`, `openai`, or `local`. |
| `OLLAMA_COACH_MODEL` | `qwen3:8b` | Local model used for structured coaching reports. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server used by the Flask app. |
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

346 tests, about a minute. They run without the model runtime — `requirements-test.txt` is
the light dependency set CI installs, and it deliberately excludes `inference`/torch.

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
- **There is not a single OUT case in the eval set.** Calling a ball OUT is the entire
  point of the app, and it is currently unmeasured — the two human calls on record are
  both IN. Labeling OUT balls is the highest-value thing anyone can do for this project.

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
| `event_engine.py`, `detect_wall_hits.py` | Bounce detection engines. |
| `judge_call.py`, `court_model.py` | Calibration geometry and the IN/OUT decision. |
| `label_hits.py` | Offline frame-by-frame labeler. |
| `build_eval_set.py`, `eval_line_calls.py` | Label distillation and evaluation replay. |
| `eval_set/` | The committed, versioned evaluation corpus. |
| `DESIGN.md` | Binding design system for all UI work. |
| `NOTES-overnight.md` | Data-flywheel work log and schema notes. |

Every tracking run records `model_id`, `tracking_backend`, `device`, and `app_version` in
its `job.json`, so a result can always be attributed to the model version that produced it.
