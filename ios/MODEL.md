# Ball model: data → train → score → export → verify ANE → run

The on-device detector has one job: find a ball **7–24 px wide** in a 4K frame,
at 60 fps, inside a ~16 ms budget, without cooking the phone. Every decision
below follows from that sentence.

**Licensing constraint first, because it rules out the obvious path.**
Ultralytics YOLO (v8/v11) is AGPL-3.0. That is incompatible with shipping
CrossCourt on the App Store, and — via section 13, which treats network
interaction as distribution — with serving inference from the Flask pipeline
too. "Keep YOLO server-side" is not a workaround. We use **YOLOX (Apache-2.0)**.
Never initialise from `yolo11n.pt`.

`train_yolo_ball.py` and `yolo_model_eval.py` are the legacy Ultralytics path,
kept for reference. Do not ship anything they produce.

---

## 1. Data

Built by `prepare_ball_dataset.py` from a Roboflow **COCO** export:

    python3 prepare_ball_dataset.py \
        --source "~/Desktop/Annotated Data/SquashAI.coco" \
        --out ball_crops \
        --val-clips "ModelTrainTest3" \
        --test-clips "Bay Club Clip Compilation 1"

It emits 416 px windows cut at **native source resolution** around each
annotation, plus hard negatives. It never emits resized frames, and that is the
whole point — see §6.

**Exporting from Roboflow:** auto-orient only, **no resize**, **no
augmentation**, versions-per-image 1, unannotated images excluded, format COCO
JSON. `docs/annotation-guide.pdf` covers the reasoning; the short version is that
the deployed v4 model was trained with `resize 512x512 "Stretch to"` plus
`motion-blur: 100 pixels` on a canvas where the median ball is 8.7 px. The blur
erased the ball in roughly one of every three training copies while leaving the
label attached, which teaches the model not to fire.

**What the current set contains, and its limits:**

| | |
| --- | --- |
| Source frames kept | 1,029 of 1,367 (338 dropped below 960 px wide) |
| Annotations | 1,103 |
| **Independent moments** | **66** — 80–95% of labeled frames are consecutive |
| Ball width, source px | p10 7.0, p50 10.7, p90 24 |
| Usable source clips | 4 |

**Neither val nor test measures generalisation.** ModelTrainTest3 (val) shares a
rig and probably a session with ModelTrainTest2 (train); Bay Club (test) is
off-domain. Use val to catch overfitting, never to claim accuracy. A real eval
set has to be captured off the printed mount — until it exists, no number here
predicts production behaviour.

Annotations carry a non-standard `streak` field beside `bbox`: the major axis of
the source polygon. For a motion-streaked ball its endpoints are the position at
the start and end of the exposure. Ignored by training; used by the tracker and
`judge_call.py`.

## 2. Train (CUDA box)

**RTX 50-series (Blackwell) needs a CUDA 12.8+ PyTorch.** Those GPUs are compute
capability sm_120 and the cu118/cu121/cu124 wheels ship no sm_120 kernels —
training dies with "no kernel image is available for execution on the device."

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Verify before walking away:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"

Then YOLOX from source (the PyPI package lags), plus the Apache-2.0 COCO
checkpoint `yolox_tiny.pth` from its releases page:

    git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && pip install -v -e .

    BALL_CROPS=/path/to/ball_crops python -m yolox.tools.train \
        -f yolox_ball_exp.py -d 1 -b 32 --fp16 -c yolox_tiny.pth

`-c` fine-tunes from COCO rather than starting cold; with 66 independent moments
that is not optional. Drop to `-b 16` if 8 GB of VRAM runs out. Expect an
overnight run for 300 epochs at 416.

`yolox_ball_exp.py` documents every non-default setting inline — most
importantly mosaic on / mixup off, no rotation or shear (the mount does not
rotate), and no motion-blur augmentation.

Apple Silicon is not a viable training path at this dataset size — measured, a
local MPS run had not finished epoch 1 after ~19 minutes. Train on CUDA, then
bring the checkpoint back to a Mac for §4, which is macOS-only.

## 3. Score before shipping (acceptance gate)

Accept when BOTH hold:

- Detection rate within 10 points of the RF-DETR run on the same clip (produce
  the baseline with `local_model_eval.py` if it is not already on disk).
- The annotated video looks locked-on through rally speed — "looks locked-on",
  not frame parity.

`yolo_model_eval.py` currently assumes an Ultralytics `.pt` and needs a YOLOX
loader before it can score this.

## 4. Export to Core ML (Mac only)

Do **not** use `yolo export` — that is the AGPL path. coremltools has also
removed its ONNX frontend, so the old ONNX detour no longer works. Trace the
PyTorch model directly:

    import torch, coremltools as ct

    model = exp.get_model().eval()        # load best_ckpt.pth into it first
    traced = torch.jit.trace(model, torch.randn(1, 3, 416, 416))
    ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 416, 416))],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    ).save("BallDetector.mlpackage")

Three things decide whether this lands on the ANE:

- **Static shape.** Flexible or dynamic shapes push the model to CPU/GPU.
- **fp16.** What the ANE runs natively.
- **Raw head outputs only.** Decode and NMS stay in Swift — trivial for one
  class and a handful of candidates, and keeping them out of the graph is what
  keeps the graph resident.

Move `BallDetector.mlpackage` to `ios/Model/`, regenerate, rebuild, and confirm
the build log compiles it to `BallDetector.mlmodelc` (see the note in
`ios/project.yml` if it does not).

## 5. Verify Neural Engine residency

Do not skip this — it is exactly the check the RF-DETR ONNX export failed.

1. Open `BallDetector.mlpackage` in Xcode → Performance tab → run a performance
   report on a **connected iPhone**, not the simulator.
2. Accept when the majority of compute units show Neural Engine and median
   prediction is comfortably inside the frame budget.
3. If SiLU shows up as a fallback, *that* is when to set `act = "relu"` in
   `yolox_ball_exp.py` and retrain — not before. ReLU costs pretrained-feature
   fidelity, which is expensive at this dataset size, so only pay it once the
   profile proves it is needed.
4. In-app sanity: the overlay tracks a thrown ball smoothly with no thermal
   warnings across a 3-minute rally.

## 6. The inference loop

The model is half the system. This is the other half, and it is what makes 4K60
feasible at all.

**Never downscale the whole frame.** `inference_engine.py` today resizes to
960 px wide (`DEFAULT_INFERENCE_WIDTH`), which turns a ~25 px ball into ~6 px
before the model sees it. Instead:

- **Every frame — fine pass.** Crop 416×416 at native resolution, centred on the
  **velocity-extrapolated** position from the Kalman state in
  `tracking_common.py` / `ballistic.py`. The input tensor ends up *smaller* than
  a downscaled frame while the ball keeps every pixel it has.
- **Widen under uncertainty.** At 60 fps a 55 m/s ball travels ~550 px between
  frames at 4K, so a window centred on the last known position will miss. Centre
  on the extrapolation and grow toward 640 as velocity variance rises.
- **Coarse re-acquire, on track loss only.** Tiled sweep at reduced resolution.
  Expensive, and rare enough not to matter.

Training mirrors this: crops are cut the same way, and positives are jittered so
the ball is never reliably centred — otherwise the head learns to predict the
centre rather than to localise.

**Capture settings** (`ios/Sources/Record/CaptureSettings.swift`): 4K60, HEVC,
shutter 1/1000 s, ISO and white balance locked, stabilisation off, ultrawide.
Keep the shutter at 1/1000: contrast-to-noise is flat from 1/1000 to 1/2000, so
a faster shutter buys no detectability and costs ISO, while the streak endpoints
recover the sub-frame localisation it would have bought.

**Open question the rig must settle.** Ultrawide from the back wall may put the
ball near ~7 px at the front wall, which is exactly where line calls happen.
Measure it on real footage from the mount before assuming the crop size and
`--target-ball-px` are right — that is a lens question, not a model one.

## 7. The loop that improves it

1. Capture a frozen eval set off the printed mount — at least 3 courts and 6
   sessions, sparse frames, never trained on. See `docs/annotation-guide.pdf`.
2. Score against it. This is the first number in the project that will mean
   anything.
3. Label where the model fails, not randomly.
4. Regenerate crops, retrain, repeat.
