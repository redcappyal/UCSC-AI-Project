# Ball model: train → score → export → verify ANE

## 1. Train (any GPU box)

    pip install ultralytics roboflow
    python train_yolo_ball.py --workspace <slug> --dataset-version <n>

Prints `best weights: .../best.pt`.

Concretely, for the current Roboflow dataset on a CUDA box:

    python train_yolo_ball.py --workspace reds-workspace-oc87h \
        --project squashai --dataset-version 3 --device 0

`--batch -1` (the default) auto-sizes to available VRAM; drop to `--batch 8`
if it OOMs. Leave `--cache` off unless the box has ~30 GB of RAM to spare
(see build_train_kwargs) — the v3 dataset is 11.5k images.

**RTX 50-series (Blackwell) needs a CUDA 12.8+ PyTorch.** Those GPUs are
compute capability sm_120, and the cu118/cu121/cu124 wheels ship no sm_120
kernels — training dies with "no kernel image is available for execution on
the device." Install before ultralytics:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Verify the GPU is actually being used before walking away:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Apple Silicon (`--device mps`) is not a viable training path for a dataset
this size — measured: a local run had not finished epoch 1 after ~19 minutes.
Train on CUDA, then bring `best.pt` back to the Mac for step 3 (Core ML
export needs coremltools and is macOS-only).

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
