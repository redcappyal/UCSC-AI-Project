# Architecture

Seven diagrams in [`architecture.puml`](architecture.puml), for someone who just joined and
needs a map. Read them in order — each one assumes the last.

If you read nothing else here, read these three sentences:

> A phone films a rally. The user hand-taps the out line, the tin and the wall corners, so
> the app knows where the court is in the image. Everything after that is: find the ball in
> every frame, work out which frames it hit something, and compare those impact points to
> the tapped lines.

---

## How to view the diagrams

The `.puml` file is the committed source; rendered images are not committed. There is no
renderer in this repo and none installed on the dev machine, so pick one:

**VS Code (easiest).** Install the `jebbs.plantuml` extension, open `architecture.puml`,
press `Alt+D`. It previews all seven blocks. Requires Java.

**Online.** Paste one `@startuml`…`@enduml` block into <https://www.plantuml.com/plantuml>.
Note this sends the diagram source to a third-party server — it contains no secrets, but
it is your call.

**Locally, fully offline.**

```bash
winget install Microsoft.OpenJDK.21
```

then drop `plantuml.jar` next to the file and run:

```bash
java -jar plantuml.jar docs/architecture.puml
```

That writes one PNG per block (`system_context.png`, `python_modules.png`, …) into `docs/`.
Those PNGs are **not** tracked — `.gitignore` allowlists `docs/**/*.puml` and `docs/**/*.md`
only, deliberately, so nobody has to keep a rendered image in sync with the source.

---

## The seven diagrams

**1. `system_context`** — Who talks to what. Two client surfaces (the web app and the
native iOS app) over one Flask backend. The thing to take away: **there is no database.**
A run is a directory of JSON and CSV under `ui_runs/`. Start here.

**2. `python_modules`** — The layering that the file tree hides. All ~35 Python modules sit
flat at the repo root with no package structure, so nothing about the import direction is
visible until you draw it. Also shows the data flywheel (labels → eval corpus → CI gate),
which is a real subsystem here, not a side project.

**3. `tracking_pipeline`** — Everything inside `run_tracking_job()`. Coarse strided pass →
audio → find candidates → dense refine pass only around those candidates → audio rescue →
detect → classify → judge. The two-pass structure is the single most important idea in the
backend: run cheap over the whole clip to find out *where* to look, then run expensive only
there.

**4. `judge_a_clip`** — The request flow, end to end. This is the one to have open the first
time you run the app. Note the 500 ms polling loop — there are no websockets and no SSE.

**5. `web_ui_phases`** — `index.html` has no router and no framework. `setPhase(p)` hides all
twenty `<section>` elements and un-hides one. That function *is* the navigation model; the
diagram is its state machine.

**6. `ios_live_path`** — The native app: on-device Core ML detection across two paired
phones, calling the ball during the rally instead of after it. Components marked
*not yet wired* have unit-tested logic but no production call site — the view is never
instantiated in `ios/Sources/`. Don't go looking for the code path that shows them.

**7. `data_shapes`** — The records that flow between stages. They are plain dicts and CSV
rows; nothing in the code declares them, so this is the map you would otherwise build by
scattering `print()` calls.

---

## Things that will waste your time if nobody tells you

These are real inconsistencies in the repo as it stands. They are recorded here, not fixed —
each is someone's call to make.

1. **The model ID in the README is stale.** `README.md` documents `ROBOFLOW_MODEL_ID`
   defaulting to `ai-squash-line-tracker/4`. `inference_engine.py` sets
   `DEFAULT_MODEL_ID = "squashai/1"`. The code wins; the diagrams follow the code.

2. **`DESIGN.md §16` documents two screens the web app does not have.** It gives full
   blueprints for `p-pair` and `p-live` as current screens. `index.html` has no `p-pair` at
   all, and its `#p-live` is a "Coming soon" placeholder. Those blueprints describe the
   **native** implementation in `ios/`. `DESIGN.md` is a shared spec for both surfaces —
   `ios/Sources/Theme.swift` mirrors its tokens, and the Swift files cite its section
   numbers in their doc comments.

3. **The test count in the README is stale.** It says "117 tests". `tests/` currently holds
   25 modules. Run them rather than trusting a number:
   ```bash
   venv/bin/python -m pytest tests/ -q
   ```

4. **The train/test split is documented two ways.** `README.md` says the bounce-classifier
   split is "chronological, not random"; `tests/test_training_split.py` asserts "the
   restored random stratified train/test split". One of them is out of date. If you touch
   `train_bounce_classifier.py`, resolve this first — it changes what the reported accuracy
   means.

5. **`.claude/worktrees/` holds full duplicate checkouts.** Grep hits from there are not the
   real codebase. Exclude that path from every search.

6. **`bounce_gb_model_detector.py` imports `train_bounce_classifier.py` at request time**,
   for `build_features_for_frame()` and the model path. A training module is on the serving
   path. It looks like a mistake. It is load-bearing.

---

## Before you trust any accuracy number

Read the "Where the system actually stands" section of [`../README.md`](../README.md). The
short version: recall is the bottleneck, the eval corpus is tiny, and **there is not a
single OUT case in it** — which is to say the app's entire purpose is currently unmeasured.
Labeling OUT balls is the highest-value contribution available.

Related reading, in the order it becomes useful:

| Document | When you need it |
|---|---|
| [`../README.md`](../README.md) | First. Setup, how to run it, honest status. |
| [`../DESIGN.md`](../DESIGN.md) | Before touching *any* UI. Binding, not advisory. |
| [`annotation-guide.md`](annotation-guide.md) | Before labeling footage for the detector. |
| [`../ios/MODEL.md`](../ios/MODEL.md) | The on-device model pipeline, and why YOLOX and not Ultralytics (AGPL). |
| [`../ios/CAPTURE.md`](../ios/CAPTURE.md) | Why the camera settings are locked. |
| [`../ios/PEER.md`](../ios/PEER.md) | Two-phone bench and ops runbook. |
| [`superpowers/specs/`](superpowers/specs/) | The approved design docs behind each phase. |
| [`superpowers/plans/`](superpowers/plans/) | The implementation plans, checkbox by checkbox. |

---

## Keeping these diagrams honest

They name **files and functions, never line numbers** — line numbers rot within days and a
diagram that cites stale ones is worse than no diagram.

If you move a module, rename a route, add a phase to `setPhase()`, or wire up one of the
*not yet wired* iOS components, update `architecture.puml` in the same change. A wrong map
costs more than a missing one.
