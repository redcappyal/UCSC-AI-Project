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

**Check the crops resolve before training on them.** Crop filenames inherit the
source clip name, and one clip is a YouTube video whose title contains U+FF5C
(`｜`), so 961 of 2936 train crops carry non-ASCII names. Zipping that on macOS
and extracting on Windows without the UTF-8 flag honored decodes each UTF-8 byte
as CP437 — `｜` (`EF BD 9C`) becomes `∩╜£` — leaving the COCO JSON correct and a
third of the files unreachable. Recoverable with
`name.encode('cp437').decode('utf-8')`, but cheaper to catch up front:

    python -c "import json,os,sys; d=sys.argv[1]; s=sys.argv[2]; \
      j=json.load(open(f'{d}/annotations/instances_{s}.json',encoding='utf-8')); \
      m=[i['file_name'] for i in j['images'] if not os.path.exists(f'{d}/{s}/'+i['file_name'])]; \
      print(len(m),'of',len(j['images']),'missing')" /path/to/ball_crops train

See also the Windows `cv2.imread` limitation in §2, which bites the *correctly*
named files for the same underlying reason.

## 2. Train (CUDA box)

**RTX 50-series (Blackwell) needs a CUDA 12.8+ PyTorch.** Those GPUs are compute
capability sm_120 and the cu118/cu121/cu124 wheels ship no sm_120 kernels —
training dies with "no kernel image is available for execution on the device."

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Verify before walking away. `is_available()` is not enough — it only confirms the
driver handshake, while the failure above happens at *kernel launch*, hours in:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"

`sm_120` must appear in the arch list. Then force one real fp16 conv on the
device; that is the check that actually proves kernels exist.

Then YOLOX from source (the PyPI package lags), plus the Apache-2.0 COCO
checkpoint `yolox_tiny.pth` from its releases page:

    git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX
    pip install -v -e . --no-deps
    pip install opencv-python loguru tqdm thop tabulate psutil tensorboard pycocotools ninja

`--no-deps` is deliberate. YOLOX's `requirements.txt` pins
`onnx-simplifier==0.4.10`, a 2022 C++ extension with no modern wheel, and we do
not need it: §4 exports by tracing the PyTorch model directly, and coremltools
has dropped its ONNX frontend anyway.

    BALL_CROPS=/path/to/ball_crops python -m yolox.tools.train \
        -f yolox_ball_exp.py -d 1 -b 32 --fp16 -c yolox_tiny.pth

`-c` fine-tunes from COCO rather than starting cold; with 66 independent moments
that is not optional. Measured, 456 of 462 tensors transfer — only the six
`cls_preds` layers mismatch (80 classes → 1) and `load_ckpt` skips them.

`yolox_ball_exp.py` documents every non-default setting inline — most
importantly mosaic on / mixup off, no rotation or shear (the mount does not
rotate), and no motion-blur augmentation. It also overrides
`get_dataset`/`get_eval_dataset`, which is **not optional**: stock YOLOX
hardcodes the COCO 2017 layout (`get_dataset` passes no `name` so `COCODataset`
defaults to `train2017`, and `get_eval_dataset` hardcodes `val2017`), and reads no
exp attribute for it. Without the overrides the run dies at startup looking for
folders this dataset does not have.

### Measured on an RTX 5060, 8 GB (2026-07-24)

Python 3.12, torch 2.11.0+cu128, YOLOX 0.3.0.

- **`-b 32` used 2118 MB of 8151.** The old "drop to `-b 16` if 8 GB runs out"
  advice is unnecessary at 416 — and `-b` is not free to change, because
  `basic_lr_per_img` multiplies by batch size, so batch *is* the LR schedule.
- **~28 s/epoch** (92 iters at 0.298 s with mosaic, 0.258 s without), so
  **~2.4 h for 300 epochs — not overnight.**
- No `--cache`: `data_time` was 0.002 s against 0.298 s `iter_time`, so this is
  compute-bound and caching only costs RAM.
- **Val converged at epoch 100 and the remaining 200 epochs bought nothing** —
  best AP[.5:.95] 0.4034 / AP@0.5 0.885 / AP@0.75 0.304 at epoch 100, versus
  0.383 / 0.834 / 0.264 at epoch 300. Epochs 150–280 sat in 0.383–0.395.
  `max_epoch` near 120 should reach the same place in ~1 h; confirm before
  trusting it.
- **Closing mosaic bought nothing measurable.** All 16 mosaic-free tail evals
  (epochs 285–300) landed at 0.382–0.385 with best AP@0.75 0.284 — *below* epoch
  100. Do not assume `no_aug_epochs` recovers localisation here.
- Val AP is very noisy while LR is high — 0.836 at epoch 20, 0.123 at epoch 40,
  0.740 at 50 — then rock-steady once LR anneals. 97 val images means each one is
  ~1% of the metric. Training loss fell monotonically (4.9 → 2.6) the whole time,
  so **treat early swings as noise, not divergence**; the signal to act on would
  be val declining across several consecutive evals while train loss falls.

### If the CUDA box is Windows

Two things break, both because YOLOX never targeted Windows. Patch the clone;
note that a re-clone silently reverts both.

1. **`cv2.imread` cannot open non-ANSI paths** and returns `None`, so any crop
   whose filename carries a character outside the active code page is unreadable
   — this silently hit 961 of 2936 train crops, a third of the set. In
   `yolox/data/datasets/coco.py` `load_image`, use
   `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)`.
2. **The C++ `fast_cocoeval` op is never built** (`setup.py` skips `ext_modules`
   on win32) and killed the run *after* training finished, during eval. The
   subtlety: the *import* succeeds — `COCOeval_opt` JIT-builds lazily in
   `__init__` — so it fails at construction with `RuntimeError("Ninja is
   required...")`, which the stock `except ImportError` in
   `yolox/evaluators/coco_evaluator.py` cannot catch, and widening it to
   `except Exception` does not help either. Import `pycocotools.cocoeval.COCOeval`
   directly. It is a drop-in, and the C++ op only matters at COCO's 5k-image
   scale; val here is 97 images.

Also expect wheel-only installs: with no MSVC compiler, every dependency must
have a prebuilt wheel, which is why 3.12 rather than the newest Python.

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
