"""CLI: run a Roboflow model locally over a video and write annotations.

This mirrors modelEval.py's one-pass video loop, but uses local
inference.get_model(..., countinference=False) through inference_engine.py.
That downloads/caches weights locally and avoids the remote serverless
workflow.
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

# inference_engine sets model-cache/metrics env defaults before inference loads.
# Keep this import after .env loading so local eval uses the same configured
# model/backend as the app process.
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
    select_motion_consistent_ball_predictions,
)


VIDEO_INPUT_PATH = Path(__file__).with_name("ModelTrainTest3.mp4")
VIDEO_OUTPUT_PATH = Path(__file__).with_name("annotated_output_local.mp4")
CSV_OUTPUT_PATH = Path(__file__).with_name("ball_coordinates_local.csv")

FRAME_STRIDE = 1
START_FRAME = None
END_FRAME = None
MAX_FRAMES = None
START_SECONDS = None
CONFIDENCE = CONFIDENCE_THRESHOLD
INFERENCE_WIDTH = DEFAULT_INFERENCE_WIDTH
TRAJECTORY_FILL_MAX_GAP = 4
TRAJECTORY_FILL_EDGE_MARGIN = 24.0


def positive_int_or_none(value):
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a Roboflow model locally with cached weights and write an "
            "annotated video plus ball-coordinate CSV."
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
    parser.add_argument(
        "--trajectory-fill-max-gap",
        type=int,
        default=TRAJECTORY_FILL_MAX_GAP,
        help=(
            "For the annotated video only, interpolate boxes across this many "
            "consecutive no-detection output frames between two real detections. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--trajectory-fill-edge-margin",
        type=float,
        default=TRAJECTORY_FILL_EDGE_MARGIN,
        help=(
            "Do not interpolate when anchor/interpolated boxes are this close "
            "to the frame edge."
        ),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def select_raw_highest_confidence_prediction(predictions):
    """Pick the raw highest-confidence box. No class/motion/dust filtering."""
    if not predictions:
        return None
    return max(predictions, key=lambda prediction: float(prediction.get("confidence", 1.0)))


def draw_raw_predictions(frame, predictions):
    """Draw every returned prediction. No local confidence/class filtering."""
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
        class_name = prediction_class_name(prediction)
        label = f"{class_name} {confidence:.2f}"

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


def prediction_box(prediction):
    try:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (KeyError, TypeError, ValueError):
        return None

    return x, y, width, height


def prediction_inside_frame(prediction, frame_width, frame_height, edge_margin):
    box = prediction_box(prediction)
    if box is None:
        return False

    x, y, width, height = box
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2

    return (
        x1 >= edge_margin
        and y1 >= edge_margin
        and x2 <= frame_width - edge_margin
        and y2 <= frame_height - edge_margin
    )


def interpolate_prediction(previous_prediction, next_prediction, alpha):
    previous_box = prediction_box(previous_prediction)
    next_box = prediction_box(next_prediction)
    if previous_box is None or next_box is None:
        return None

    interpolated = dict(previous_prediction)
    for key, previous_value, next_value in zip(
        ("x", "y", "width", "height"),
        previous_box,
        next_box,
    ):
        interpolated[key] = previous_value + (next_value - previous_value) * alpha

    previous_confidence = float(previous_prediction.get("confidence", 1.0))
    next_confidence = float(next_prediction.get("confidence", 1.0))
    interpolated["confidence"] = min(previous_confidence, next_confidence)
    interpolated["class"] = "trajectory-estimate"
    interpolated["class_name"] = "trajectory-estimate"
    return interpolated


def draw_estimated_prediction(frame, prediction):
    return draw_tracked_prediction(
        frame,
        prediction,
        label_prefix="trajectory estimate",
        color=(0, 165, 255),
    )


def draw_tracked_prediction(frame, prediction, label_prefix="tracked ball", color=(0, 255, 0)):
    output_frame = frame.copy()
    frame_height, frame_width = output_frame.shape[:2]
    box = prediction_box(prediction)
    if box is None:
        return output_frame

    x, y, width, height = box
    x1 = max(0, min(int(x - width / 2), frame_width - 1))
    y1 = max(0, min(int(y - height / 2), frame_height - 1))
    x2 = max(0, min(int(x + width / 2), frame_width - 1))
    y2 = max(0, min(int(y + height / 2), frame_height - 1))

    confidence = float(prediction.get("confidence", 1.0))
    label = f"{label_prefix} {confidence:.2f}"
    center = (int(max(0, min(x, frame_width - 1))), int(max(0, min(y, frame_height - 1))))

    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(output_frame, center, 5, color, -1)
    cv2.putText(
        output_frame,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return output_frame


def can_fill_trajectory_gap(
    previous_anchor,
    next_prediction,
    missing_frames,
    output_index,
    frame_width,
    frame_height,
    max_gap,
    edge_margin,
):
    if max_gap <= 0 or previous_anchor is None or next_prediction is None:
        return False
    if not missing_frames:
        return False

    previous_index, previous_prediction = previous_anchor
    gap_size = output_index - previous_index - 1
    if gap_size != len(missing_frames) or gap_size > max_gap:
        return False

    if not prediction_inside_frame(previous_prediction, frame_width, frame_height, edge_margin):
        return False
    if not prediction_inside_frame(next_prediction, frame_width, frame_height, edge_margin):
        return False

    for missing_index, _, _ in missing_frames:
        alpha = (missing_index - previous_index) / (output_index - previous_index)
        estimated = interpolate_prediction(previous_prediction, next_prediction, alpha)
        if estimated is None:
            return False
        if not prediction_inside_frame(estimated, frame_width, frame_height, edge_margin):
            return False

    return True


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
    if args.trajectory_fill_max_gap < 0:
        raise RuntimeError("--trajectory-fill-max-gap must be 0 or greater.")
    if args.trajectory_fill_edge_margin < 0:
        raise RuntimeError("--trajectory-fill-edge-margin must be 0 or greater.")
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
    trajectory_estimated_count = 0
    trajectory_tracked_detection_count = 0
    previous_anchor = None
    missing_video_frames = []
    predictions_by_source_frame = {}
    processed_source_frames = []

    def write_output_frame(output_frame):
        nonlocal writer
        if writer is None:
            height, width = output_frame.shape[:2]
            writer = cv2.VideoWriter(
                str(args.output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )
        writer.write(output_frame)

    def flush_missing_video_frames(next_prediction=None, next_output_index=None):
        nonlocal trajectory_estimated_count
        if not missing_video_frames:
            return

        first_frame = missing_video_frames[0][2]
        frame_height, frame_width = first_frame.shape[:2]
        can_fill = (
            next_output_index is not None
            and can_fill_trajectory_gap(
                previous_anchor,
                next_prediction,
                missing_video_frames,
                next_output_index,
                frame_width,
                frame_height,
                args.trajectory_fill_max_gap,
                args.trajectory_fill_edge_margin,
            )
        )

        for missing_index, _, missing_frame in missing_video_frames:
            if can_fill:
                previous_index, previous_prediction = previous_anchor
                alpha = (missing_index - previous_index) / (next_output_index - previous_index)
                estimated = interpolate_prediction(previous_prediction, next_prediction, alpha)
                output_frame = draw_estimated_prediction(missing_frame, estimated)
                trajectory_estimated_count += 1
            else:
                output_frame = draw_raw_predictions(missing_frame, [])
            write_output_frame(output_frame)

        missing_video_frames.clear()

    def annotate_processed_video():
        nonlocal previous_anchor, trajectory_tracked_detection_count
        if args.no_video or not processed_source_frames:
            return

        selected_by_frame = select_motion_consistent_ball_predictions(
            predictions_by_source_frame,
            confidence_threshold=args.confidence,
        )

        previous_anchor = None
        missing_video_frames.clear()

        annotation_cap = cv2.VideoCapture(str(args.video))
        if not annotation_cap.isOpened():
            raise RuntimeError(f"Could not reopen video for annotation: {args.video}")

        try:
            for output_index, source_frame in enumerate(processed_source_frames, start=1):
                annotation_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                ok, frame = annotation_cap.read()
                if not ok:
                    continue

                tracked_prediction = selected_by_frame.get(source_frame)
                if tracked_prediction is None:
                    missing_video_frames.append((output_index, source_frame, frame.copy()))
                    if (
                        previous_anchor is None
                        or args.trajectory_fill_max_gap == 0
                        or len(missing_video_frames) > args.trajectory_fill_max_gap
                    ):
                        flush_missing_video_frames()
                    continue

                flush_missing_video_frames(tracked_prediction, output_index)
                write_output_frame(draw_tracked_prediction(frame, tracked_prediction))
                previous_anchor = (output_index, tracked_prediction)
                trajectory_tracked_detection_count += 1

            flush_missing_video_frames()
        finally:
            annotation_cap.release()

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
                predictions_by_source_frame[read_count] = predictions
                processed_source_frames.append(read_count)
                raw_prediction_count += len(predictions)
                selected_prediction = select_raw_highest_confidence_prediction(predictions)
                if selected_prediction is not None:
                    detected_count += 1
                csv_writer.writerow(ball_csv_row(read_count, source_fps, selected_prediction))

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

    if not args.no_video:
        annotate_processed_video()
        print(
            "Annotated video boxes: "
            f"{trajectory_tracked_detection_count} tracked detection(s), "
            f"{trajectory_estimated_count} trajectory estimate(s)"
        )

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
        "trajectory_tracked_video_frames": trajectory_tracked_detection_count,
        "trajectory_estimated_video_frames": trajectory_estimated_count,
        "trajectory_fill_max_gap": args.trajectory_fill_max_gap,
        "trajectory_fill_edge_margin": args.trajectory_fill_edge_margin,
        "model_object_type": type(model).__name__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    outputs.append(str(metadata_path))
    print(
        f"Done! Processed {processed_count} frame(s); "
        f"{detected_count} frame(s) had at least one raw prediction; "
        f"{raw_prediction_count} total raw prediction(s); "
        f"{trajectory_estimated_count} video-only trajectory estimate(s) -> "
        + ", ".join(outputs)
    )


if __name__ == "__main__":
    main()
