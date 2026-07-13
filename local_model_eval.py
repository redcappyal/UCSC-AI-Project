import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
os.environ.setdefault("MODEL_CACHE_DIR", str(Path(__file__).with_name(".roboflow-cache")))
os.environ.setdefault("METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

import cv2
from PIL import Image

from modelEval import (
    API_KEY,
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    END_FRAME,
    FRAME_STRIDE,
    START_FRAME,
    VIDEO_INPUT_PATH,
    WORKFLOW_ID,
    ball_csv_row,
    draw_predictions,
    select_ball_prediction,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))


def infer_model_id_from_workflow_id(workflow_id):
    if "-v" not in workflow_id:
        return None

    project_id, workflow_suffix = workflow_id.split("-v", 1)
    repeated_project_prefix = f"{project_id}-"
    if not workflow_suffix.startswith(repeated_project_prefix):
        return None

    version_and_rest = workflow_suffix[len(repeated_project_prefix) :]
    version = version_and_rest.split("-", 1)[0]
    if not version.isdigit():
        return None

    return f"{project_id}/{version}"


DEFAULT_MODEL_ID = os.getenv(
    "ROBOFLOW_MODEL_ID",
    infer_model_id_from_workflow_id(WORKFLOW_ID) or "squash-line-calling-model/1",
)
DEFAULT_OUTPUT_VIDEO_PATH = Path(__file__).with_name("annotated_output_local.mp4")
DEFAULT_CSV_OUTPUT_PATH = Path(__file__).with_name("ball_coordinates_local.csv")
DEFAULT_INFERENCE_WIDTH = int(os.getenv("INFERENCE_WIDTH", "960"))


def object_to_dict(value):
    if isinstance(value, dict):
        return {key: object_to_dict(item) for key, item in value.items()}

    if isinstance(value, list):
        return [object_to_dict(item) for item in value]

    if isinstance(value, tuple):
        return [object_to_dict(item) for item in value]

    if hasattr(value, "model_dump"):
        return object_to_dict(value.model_dump())

    if hasattr(value, "dict"):
        return object_to_dict(value.dict())

    return value


def find_predictions(obj):
    predictions = []

    if isinstance(obj, dict):
        if all(key in obj for key in ("x", "y", "width", "height")):
            predictions.append(obj)

        for value in obj.values():
            predictions.extend(find_predictions(value))

    elif isinstance(obj, list):
        for item in obj:
            predictions.extend(find_predictions(item))

    return predictions


def normalize_prediction(prediction):
    normalized = dict(prediction)

    if "class" not in normalized and "class_name" in normalized:
        normalized["class"] = normalized["class_name"]

    if "class_name" not in normalized and "class" in normalized:
        normalized["class_name"] = normalized["class"]

    if "x" in normalized and "y" in normalized and "width" in normalized and "height" in normalized:
        return normalized

    if all(key in normalized for key in ("x_min", "y_min", "x_max", "y_max")):
        x_min = float(normalized["x_min"])
        y_min = float(normalized["y_min"])
        x_max = float(normalized["x_max"])
        y_max = float(normalized["y_max"])
        normalized["x"] = (x_min + x_max) / 2
        normalized["y"] = (y_min + y_max) / 2
        normalized["width"] = x_max - x_min
        normalized["height"] = y_max - y_min
        return normalized

    if all(key in normalized for key in ("xmin", "ymin", "xmax", "ymax")):
        x_min = float(normalized["xmin"])
        y_min = float(normalized["ymin"])
        x_max = float(normalized["xmax"])
        y_max = float(normalized["ymax"])
        normalized["x"] = (x_min + x_max) / 2
        normalized["y"] = (y_min + y_max) / 2
        normalized["width"] = x_max - x_min
        normalized["height"] = y_max - y_min
        return normalized

    return normalized


def normalize_predictions(raw_result):
    raw_dict = object_to_dict(raw_result)
    predictions = find_predictions(raw_dict)
    return [normalize_prediction(prediction) for prediction in predictions]


def resize_frame_for_inference(frame, max_width):
    if not max_width or max_width <= 0:
        return frame, 1.0, 1.0

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0, 1.0

    scale = max_width / width
    inference_size = (int(max_width), max(1, int(round(height * scale))))
    resized = cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA)
    return resized, width / inference_size[0], height / inference_size[1]


def scale_prediction(prediction, x_scale, y_scale):
    if x_scale == 1.0 and y_scale == 1.0:
        return prediction

    scaled = dict(prediction)
    for key in ("x", "width", "x_center", "x_min", "x_max", "xmin", "xmax"):
        if key in scaled and scaled[key] not in (None, ""):
            scaled[key] = float(scaled[key]) * x_scale

    for key in ("y", "height", "y_center", "y_min", "y_max", "ymin", "ymax"):
        if key in scaled and scaled[key] not in (None, ""):
            scaled[key] = float(scaled[key]) * y_scale

    return scaled


def scale_predictions(predictions, x_scale, y_scale):
    return [scale_prediction(prediction, x_scale, y_scale) for prediction in predictions]


def infer_frame(model, frame, confidence):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    try:
        return model.infer(image=image, confidence=confidence)
    except TypeError:
        return model.infer(image)


def infer_frame_predictions(model, frame, confidence, max_width=DEFAULT_INFERENCE_WIDTH):
    inference_frame, x_scale, y_scale = resize_frame_for_inference(frame, max_width)
    result = infer_frame(model, inference_frame, confidence)
    predictions = normalize_predictions(result)
    return scale_predictions(predictions, x_scale, y_scale)


def positive_int_or_none(value):
    if value is None:
        return None

    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Roboflow model locally with inference.get_model and write ball coordinates."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", API_KEY))
    parser.add_argument("--video", type=Path, default=VIDEO_INPUT_PATH)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--start-frame", type=positive_int_or_none, default=START_FRAME)
    parser.add_argument("--end-frame", type=positive_int_or_none, default=END_FRAME)
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    parser.add_argument("--inference-width", type=int, default=DEFAULT_INFERENCE_WIDTH)
    parser.add_argument("--max-frames", type=positive_int_or_none, default=None)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = args.start_frame if args.start_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else source_frame_count - 1
    end_frame = min(end_frame, source_frame_count - 1)

    if start_frame >= source_frame_count:
        raise RuntimeError(f"START_FRAME={start_frame} starts after the video ends.")

    if end_frame < start_frame:
        raise RuntimeError(f"END_FRAME={end_frame} is before START_FRAME={start_frame}.")

    if args.frame_stride < 1:
        raise RuntimeError("--frame-stride must be 1 or greater.")

    if args.inference_width < 0:
        raise RuntimeError("--inference-width must be 0 or greater.")

    if not args.api_key.strip():
        raise RuntimeError(
            "No Roboflow API key found. Set ROBOFLOW_API_KEY in your shell, "
            "or put the key back into API_KEY in modelEval.py for local testing."
        )

    from inference import get_model

    print(f"Loading local Roboflow model: {args.model_id}")
    model = get_model(model_id=args.model_id, api_key=args.api_key, countinference=False)
    print("Model loaded.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = None
    processed_count = 0
    read_count = start_frame
    output_fps = source_fps / args.frame_stride

    print(f"Source video: {source_frame_count} frames at {source_fps:.2f} FPS")
    print(f"Processing source frames {start_frame} through {end_frame}")
    print(f"Processing every {args.frame_stride} frame(s)")
    print(
        "Inference width: "
        + ("original" if args.inference_width == 0 else f"{args.inference_width}px")
    )
    print(f"Output CSV: {args.csv}")
    if not args.no_video:
        print(f"Output video: {args.output_video}")

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
            ball_prediction = select_ball_prediction(predictions)
            csv_writer.writerow(ball_csv_row(read_count, source_fps, ball_prediction))

            if not args.no_video:
                output_frame = draw_predictions(frame, predictions)
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
            detected_text = "detected" if ball_prediction is not None else "not detected"
            print(f"Processed source frame {read_count} -> {detected_text}")

            read_count += 1
            if args.smoke_test:
                break

            if args.max_frames is not None and processed_count >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()

    print(f"Done. Processed {processed_count} frame(s).")


if __name__ == "__main__":
    main()
