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
