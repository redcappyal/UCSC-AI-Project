"""Compatibility CLI: run the configured Roboflow detector locally over a video.

This file used to call an old Roboflow serverless workflow. It now uses the
same local cached model path as the app, training scripts, benchmark, and
local_model_eval.py:

    ROBOFLOW_MODEL_ID=squashai/1

Weights are downloaded/cached by inference.get_model(..., countinference=False)
through inference_engine.py.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))

from inference_engine import (  # noqa: E402
    DEFAULT_INFERENCE_WIDTH,
    DEFAULT_MODEL_ID,
    TRACKING_BACKEND,
    configured_providers,
    infer_frame_predictions,
    load_model,
)
from tracking_common import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    ball_csv_row,
    prediction_class_name,
)


VIDEO_INPUT_PATH = Path(__file__).with_name("SquashTestVid.mp4")
VIDEO_OUTPUT_PATH = Path(__file__).with_name("annotated_output.mp4")
CSV_OUTPUT_PATH = Path(__file__).with_name("ball_coordinates.csv")

FRAME_STRIDE = 1
START_FRAME = None
END_FRAME = None
MAX_FRAMES = None
START_SECONDS = None
CONFIDENCE = CONFIDENCE_THRESHOLD
INFERENCE_WIDTH = DEFAULT_INFERENCE_WIDTH


def positive_int_or_none(value):
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured Roboflow detector locally with cached weights "
            "and write an annotated video plus ball-coordinate CSV."
        )
    )
    parser.add_argument("--model-id", default=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--video", type=Path, default=VIDEO_INPUT_PATH)
    parser.add_argument("--output-video", type=Path, default=VIDEO_OUTPUT_PATH)
    parser.add_argument("--csv", type=Path, default=CSV_OUTPUT_PATH)
    parser.add_argument("--start-frame", type=positive_int_or_none, default=START_FRAME)
    parser.add_argument("--end-frame", type=positive_int_or_none, default=END_FRAME)
    parser.add_argument("--start-seconds", type=float, default=START_SECONDS)
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    parser.add_argument("--inference-width", type=int, default=INFERENCE_WIDTH)
    parser.add_argument("--max-frames", type=positive_int_or_none, default=MAX_FRAMES)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def select_raw_highest_confidence_prediction(predictions):
    if not predictions:
        return None
    return max(predictions, key=lambda prediction: float(prediction.get("confidence", 1.0)))


def draw_raw_predictions(frame, predictions):
    output_frame = frame.copy()
    frame_height, frame_width = output_frame.shape[:2]

    for prediction in predictions:
        try:
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        confidence = float(prediction.get("confidence", 1.0))
        label = f"{prediction_class_name(prediction)} {confidence:.2f}"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output_frame


def frame_range(args, source_fps, source_frame_count):
    if args.start_frame is not None:
        start_frame = args.start_frame
    else:
        start_frame = int((args.start_seconds or 0) * source_fps)

    end_frame = args.end_frame if args.end_frame is not None else source_frame_count - 1
    end_frame = min(end_frame, source_frame_count - 1)

    if start_frame >= source_frame_count:
        raise RuntimeError(
            f"START_FRAME={start_frame} starts after the video ends "
            f"({source_frame_count / source_fps:.2f} seconds)."
        )
    if end_frame < start_frame:
        raise RuntimeError(f"END_FRAME={end_frame} is before START_FRAME={start_frame}.")

    return start_frame, end_frame


def api_key_fingerprint(api_key):
    api_key = api_key.strip()
    if len(api_key) <= 8:
        return "(set)" if api_key else "(missing)"
    return f"{api_key[:4]}...{api_key[-4:]}"


def model_config_summary(args):
    return {
        "model_id": args.model_id,
        "env_model_id": os.getenv("ROBOFLOW_MODEL_ID", ""),
        "workspace": os.getenv("ROBOFLOW_WORKSPACE", ""),
        "project_id": os.getenv("ROBOFLOW_PROJECT_ID", ""),
        "package_name": os.getenv("ROBOFLOW_PACKAGE_NAME", ""),
        "api_key_fingerprint": api_key_fingerprint(args.api_key),
        "tracking_backend": TRACKING_BACKEND,
        "default_device": os.getenv("DEFAULT_DEVICE", ""),
        "onnx_providers": configured_providers(),
    }


def main():
    args = parse_args()
    metadata_path = args.metadata_json or args.csv.with_suffix(".metadata.json")

    if args.frame_stride < 1:
        raise RuntimeError("--frame-stride must be 1 or greater.")
    if args.inference_width < 0:
        raise RuntimeError("--inference-width must be 0 or greater.")
    if not args.api_key.strip():
        raise RuntimeError("No Roboflow API key found. Set ROBOFLOW_API_KEY in .env.")

    config = model_config_summary(args)
    print(f"Loading local Roboflow model: {args.model_id}")
    print(
        "Roboflow config: "
        f"workspace={config['workspace'] or '(unset)'}, "
        f"project={config['project_id'] or '(unset)'}, "
        f"package={config['package_name'] or '(unset)'}, "
        f"api_key={config['api_key_fingerprint']}"
    )
    print("Model loading key: inference.get_model uses --model-id / ROBOFLOW_MODEL_ID.")
    print(
        "Inference config: "
        f"backend={config['tracking_backend']}, "
        f"default_device={config['default_device'] or '(unset)'}, "
        f"onnx_providers={config['onnx_providers']}"
    )
    print("This downloads/caches weights locally with countinference=False.")
    model = load_model(args.model_id, args.api_key)
    print("Model loaded.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = source_fps / args.frame_stride
    start_frame, end_frame = frame_range(args, source_fps, source_frame_count)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = None
    read_count = start_frame
    processed_count = 0
    detected_count = 0
    raw_prediction_count = 0

    print(f"Source video: {source_frame_count} frames at {source_fps:.2f} FPS")
    print(f"Processing source frames {start_frame} through {end_frame}")
    print(f"Processing every {args.frame_stride} frame(s)")
    print(
        "Inference width: "
        + ("original" if args.inference_width == 0 else f"{args.inference_width}px")
    )
    if not args.no_video:
        print(f"Output video: {args.output_video}")
    print(f"Output CSV: {args.csv}")

    try:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_video:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)

        with args.csv.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()

            while read_count <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                if (read_count - start_frame) % args.frame_stride != 0:
                    read_count += 1
                    continue

                predictions = infer_frame_predictions(
                    model,
                    frame,
                    args.confidence,
                    args.inference_width,
                )
                raw_prediction_count += len(predictions)
                selected_prediction = select_raw_highest_confidence_prediction(predictions)
                if selected_prediction is not None:
                    detected_count += 1
                csv_writer.writerow(ball_csv_row(read_count, source_fps, selected_prediction))

                if not args.no_video:
                    output_frame = draw_raw_predictions(frame, predictions)
                    if writer is None:
                        height, width = output_frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height),
                        )
                    writer.write(output_frame)

                processed_count += 1
                print(
                    f"Processed source frame {read_count} -> output frame {processed_count} "
                    f"({len(predictions)} raw prediction(s))"
                )

                read_count += 1
                if args.smoke_test:
                    break
                if args.max_frames is not None and processed_count >= args.max_frames:
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    outputs = [str(args.csv)]
    if not args.no_video:
        outputs.insert(0, str(args.output_video))
    metadata = {
        **config,
        "video": str(args.video),
        "output_video": None if args.no_video else str(args.output_video),
        "csv": str(args.csv),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_stride": args.frame_stride,
        "inference_width": args.inference_width,
        "confidence": args.confidence,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "processed_frames": processed_count,
        "frames_with_raw_prediction": detected_count,
        "raw_prediction_count": raw_prediction_count,
        "model_object_type": type(model).__name__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    outputs.append(str(metadata_path))
    print(
        f"Done! Processed {processed_count} frame(s); "
        f"{detected_count} frame(s) had at least one raw prediction; "
        f"{raw_prediction_count} total raw prediction(s) -> "
        + ", ".join(outputs)
    )


if __name__ == "__main__":
    main()
