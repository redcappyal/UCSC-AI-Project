# CLAUDE.md

CrossCourt — a squash **training feedback loop**. Record a session with a phone on the
court, get statistics and coaching back, and compare them against past sessions so a
player can tell whether they are actually improving. Today that ships as a single-file
mobile web app (`index.html`) plus a native iOS app over a Flask pipeline (`app.py`,
`job_runner.py`, `inference_engine.py`) that tracks the ball, finds wall and floor
contacts, and judges front-wall hits IN or OUT.

## Direction (decided 2026-07-27)

The product is the feedback loop, not the referee. **IN/OUT is one statistic among many,
not the goal** — do not justify work on the grounds that it improves line calling alone.
Two stages:

1. **Capture** — a recorder built for one purpose: collecting input the analysis stage can
   use. Phone on the fin beside the back glass door; the app recognizes the court lines and
   builds the homography *during recording*. **Not built** — calibration today is the
   manual tap wizard, and every existing calibration and eval artifact came from it.
2. **Analysis** — clip saved on-device, uploaded by choice, analyzed into a stats dashboard
   with squash-specific LLM coaching, with past runs retained so the same measurables track
   over time. **Partly built** — pipeline and coaching report exist; cross-session
   comparison does not.

What this changes about work in this repo:

- **The floor homography is load-bearing now.** The IN/OUT call needs no homography — a
  projection preserves which side of a coplanar line a point falls on, which is why
  `judge_call.py` works in raw pixels — but positional statistics need real court feet, so
  `court_model.FloorMap` is infrastructure, not a coaching side-path.
- **Detection recall still gates everything downstream** (~35% at rally scale). Statistics
  built on missed bounces are wrong *quietly*, which is worse in a coaching product than in
  a refereeing one, where a miss at least shows up as a missing call.
- **There is one bounce-detection path, not a choice of engines.** The selectable
  `event_engine` (`votes` / `gb_model` / `fusion`) and the `fusion_3d` flag were removed
  2026-07-27: `detect_wall_hits` finds candidates for the refine pass, then
  `bounce_gb_model_detector` + `classify_events` label them. Don't reintroduce a fork
  here — the pipeline needs one well-measured path.
- **Three features are archived, not deleted.** Two-phone stereo in `archive/stereo/`
  (tag `archive/stereo-v1`, revisit gate: single-view wall-hit recall ~85%), the
  fusion engine + 3D contact detection in `archive/fusion-engine/` (tag
  `archive/fusion-engine-v1`; it was evaluated and failed its gate — see
  `eval_set/RESULTS-3d-contact.md`), and the Challenge/corrections UI in
  `archive/challenge-ui/` (2026-07-29). All are inert and uncollected. Do not extend
  them or import from them.
- **Corrections now have one producer, not two.** Archiving the Challenge pane removed
  the in-app way to write `corrections.json`, one of the **two** label streams
  `build_eval_set.py` distills into `eval_set/cases.jsonl`. Existing cases and
  `ui_runs/` files are untouched and still score; labeling mode (`p-label` →
  `ground_truth.json`) is still live and is now the only way to add labels. The
  corrections endpoint and schema stay in `app.py`. See that archive's README before
  planning any work that assumes new correction labels are arriving.

## Design

**For all UI and front-end work, strictly follow the rules in [DESIGN.md](DESIGN.md).**

DESIGN.md is the single source of truth for design tokens (colors, type, spacing, radii,
motion), the component library, per-screen blueprints, and the hard "never do" list. Read
it before writing or changing any HTML/CSS/JS that renders UI. If a change requires
deviating from it, update DESIGN.md deliberately in the same change — never drift
silently. Verify UI changes in both themes at a phone viewport (use the `/verify` skill).

## Environment

The venv is `.venv`; everything runs through it. System `python3` has no flask or cv2.

```bash
.venv/bin/python app.py                  # http://127.0.0.1:5188
.venv/bin/python -m pytest tests/ -q     # 283 tests, ~9s
```

`pytest.ini` deselects `requires_model` by default — one test that needs the exported
TorchScript ball model (see the `ball_track_offline.py` note below). A green run reads
"283 passed, 1 deselected"; that deselection is expected, not a skip to chase.

`pytest.ini` also excludes `archive/` from collection entirely. The archived stereo and
fusion-engine tests are preserved verbatim and still import modules that moved — they must
never be collected. A bare `pytest` from the repo root errors during collection for
unrelated pre-existing reasons; `pytest tests/` is the invocation that works.

Editing a `*.py` that has a paired `tests/test_*.py` auto-runs that file (PostToolUse
hook). Failures come back as a *blocked edit*, not a warning.

On the Windows CUDA training box there is no `.venv` here; that environment lives at
`C:\Users\alann\Code\ball-detector-train\.venv` (cv2 + torch, no pytest or flask), and
the PostToolUse hook above is not configured there.

**Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite`.** On Windows both
reach the CRT's ANSI file API: reads return `None`, and writes return `True` while landing
under a mojibake filename — that is how `ball-crops-2026-07-24` lost 961 of 2,936 train
crops. Use `_imread_unicode`/`_imwrite_unicode` in `prepare_ball_dataset.py`. Crop
filenames are ASCII by construction via `ascii_slug()`, with the readable clip name kept
in the COCO per-image `clip` field. `cv2.VideoCapture`/`VideoWriter` are *not* affected —
FFmpeg does its own UTF-8 conversion — so leave those call sites alone.

## Skills carry the workflow

- `/verify` — launching, driving the browser, test videos, UI gotchas.
- `/eval` — required before claiming any pipeline change improved anything. Judge,
  calibration, and detector changes are scored against the newest `eval_set/BASELINE-*.md`.

## Two clients, one pipeline

`index.html` is the web UI. `ios/` is a native SwiftUI app (Play records natively;
Matches and Coach are webviews). Both talk to the same Flask pipeline.

- `ios/SquashLineCalling.xcodeproj` is generated by `xcodegen generate` and gitignored —
  edit `ios/project.yml`, never the project file.
- `ball_track_offline.py` (formerly `stereo_offline.py`, renamed when the stereo half moved
  to `archive/stereo/python/stereo_offline_fuse.py`) is the offline runner for the locally
  trained YOLOX ball detector (`ball_model.py` + `ball_detector.py`). It needs
  `models/crosscourt-ball-416-v1/` (gitignored; export it with `export_ball_model.py` — see
  `ios/MODEL.md` §2b) and `torch`. YOLOX itself stays training-only and never becomes a
  runtime dependency.
