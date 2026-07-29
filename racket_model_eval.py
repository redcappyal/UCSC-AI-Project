"""Annotate a video with the hosted Roboflow racket detector."""

import argparse
import csv
import json
import os
import time
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
    infer_frame_predictions,
)
from tracking_common import prediction_class_name  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = ROOT / "MatchplayEp2Clip.mp4"
DEFAULT_OUTPUT_VIDEO_PATH = ROOT / "annotated_output_racket.mp4"
DEFAULT_CSV_PATH = ROOT / "racket_detections.csv"
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_WORKSPACE_NAME = "alvins-workspace-vzjhh"
DEFAULT_WORKFLOW_ID = "racketdetection-2"
DEFAULT_CONFIDENCE = 0.30
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0

CSV_FIELDNAMES = [
    "source_frame",
    "timestamp_seconds",
    "detected",
    "detection_index",
    "class_name",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


class ServerlessRacketModel:
    """Adapt the hosted SDK client to ``infer_frame_predictions``."""

    def __init__(self, client, workspace_name, workflow_id):
        self.client = client
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id

    def infer(self, frame, confidence=None):
        # Workflow confidence is filtered locally after normalization.
        return self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": frame},
            use_cache=True,
        )


def create_serverless_model(api_url, api_key, workspace_name, workflow_id):
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return ServerlessRacketModel(client, workspace_name, workflow_id)


def positive_int_or_none(value):
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Defaults to the CSV path with a .metadata.json suffix.",
    )
    parser.add_argument(
        "--workspace-name",
        default=os.getenv("RACKET_WORKFLOW_WORKSPACE", DEFAULT_WORKSPACE_NAME),
    )
    parser.add_argument(
        "--workflow-id",
        default=os.getenv("RACKET_WORKFLOW_ID", DEFAULT_WORKFLOW_ID),
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("RACKET_ROBOFLOW_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "RACKET_WORKFLOW_API_KEY",
            os.getenv("RACKET_ROBOFLOW_API_KEY", ""),
        ),
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--inference-width",
        type=int,
        default=DEFAULT_INFERENCE_WIDTH,
    )
    parser.add_argument("--start-frame", type=positive_int_or_none, default=0)
    parser.add_argument("--end-frame", type=positive_int_or_none, default=None)
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=positive_int_or_none, default=None)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Seconds between visible progress and ETA updates.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def filter_predictions(predictions, confidence):
    return [
        prediction
        for prediction in predictions
        if float(prediction.get("confidence", 1.0)) >= confidence
    ]


def prediction_box(prediction):
    try:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, width, height


def draw_predictions(frame, predictions):
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]

    for prediction in predictions:
        box = prediction_box(prediction)
        if box is None:
            continue
        x, y, width, height = box
        x1 = max(0, min(int(round(x - width / 2)), frame_width - 1))
        y1 = max(0, min(int(round(y - height / 2)), frame_height - 1))
        x2 = max(0, min(int(round(x + width / 2)), frame_width - 1))
        y2 = max(0, min(int(round(y + height / 2)), frame_height - 1))
        class_name = prediction_class_name(prediction)
        confidence = float(prediction.get("confidence", 1.0))
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (10, 214, 255), 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        label_right = min(frame_width - 1, x1 + text_width + 8)
        cv2.rectangle(
            output,
            (x1, label_top),
            (label_right, y1),
            (10, 214, 255),
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, max(text_height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def csv_rows_for_frame(frame_index, fps, predictions):
    timestamp = frame_index / fps if fps else 0.0
    if not predictions:
        return [
            {
                "source_frame": frame_index,
                "timestamp_seconds": f"{timestamp:.6f}",
                "detected": False,
                "detection_index": "",
                "class_name": "",
                "confidence": "",
                "x_center": "",
                "y_center": "",
                "width": "",
                "height": "",
                "x_min": "",
                "y_min": "",
                "x_max": "",
                "y_max": "",
            }
        ]

    rows = []
    for detection_index, prediction in enumerate(predictions):
        box = prediction_box(prediction)
        if box is None:
            continue
        x, y, width, height = box
        rows.append(
            {
                "source_frame": frame_index,
                "timestamp_seconds": f"{timestamp:.6f}",
                "detected": True,
                "detection_index": detection_index,
                "class_name": prediction_class_name(prediction),
                "confidence": f"{float(prediction.get('confidence', 1.0)):.6f}",
                "x_center": f"{x:.3f}",
                "y_center": f"{y:.3f}",
                "width": f"{width:.3f}",
                "height": f"{height:.3f}",
                "x_min": f"{x - width / 2:.3f}",
                "y_min": f"{y - height / 2:.3f}",
                "x_max": f"{x + width / 2:.3f}",
                "y_max": f"{y + height / 2:.3f}",
            }
        )
    return rows or csv_rows_for_frame(frame_index, fps, [])


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def frame_range(args, fps, frame_count):
    start_frame = int(args.start_frame or 0)
    if args.start_seconds is not None:
        start_frame = max(0, int(round(args.start_seconds * fps)))
    start_frame = min(start_frame, max(0, frame_count - 1))
    end_frame = frame_count - 1 if args.end_frame is None else int(args.end_frame)
    end_frame = min(end_frame, frame_count - 1)
    if end_frame < start_frame:
        raise RuntimeError(f"End frame {end_frame} is before start frame {start_frame}.")
    return start_frame, end_frame


def validate_args(args):
    if not args.api_key.strip() or "*" in args.api_key:
        raise RuntimeError(
            "Set the complete RACKET_ROBOFLOW_API_KEY in .env; "
            "the redacted example key cannot run inference."
        )
    if not args.workspace_name.strip():
        raise RuntimeError("--workspace-name cannot be empty.")
    if not args.workflow_id.strip():
        raise RuntimeError("--workflow-id cannot be empty.")
    if not 0 <= args.confidence <= 1:
        raise RuntimeError("--confidence must be between 0 and 1.")
    if args.inference_width < 0:
        raise RuntimeError("--inference-width must be 0 or greater.")
    if args.frame_stride < 1:
        raise RuntimeError("--frame-stride must be 1 or greater.")
    if args.progress_interval <= 0:
        raise RuntimeError("--progress-interval must be greater than 0.")


def main():
    args = parse_args()
    validate_args(args)
    metadata_path = args.metadata_json or args.csv.with_suffix(".metadata.json")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame, end_frame = frame_range(args, fps, frame_count)
    planned_frames = (end_frame - start_frame) // args.frame_stride + 1
    if args.max_frames is not None:
        planned_frames = min(planned_frames, args.max_frames)
    if args.smoke_test:
        planned_frames = min(planned_frames, 1)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)

    model = create_serverless_model(
        args.api_url,
        args.api_key,
        args.workspace_name,
        args.workflow_id,
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = None
    processed_frames = 0
    detection_count = 0
    started = time.monotonic()
    last_progress = started

    print(
        f"Racket workflow: {args.workspace_name}/{args.workflow_id} via {args.api_url}\n"
        f"Video: {args.video}\n"
        f"Frames: {start_frame}-{end_frame}, stride {args.frame_stride}, "
        f"planned {planned_frames}\n"
        f"Confidence: {args.confidence:.2f}, inference width: "
        f"{'original' if args.inference_width == 0 else args.inference_width}",
        flush=True,
    )

    try:
        with args.csv.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()
            source_frame = start_frame
            while source_frame <= end_frame and processed_frames < planned_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if (source_frame - start_frame) % args.frame_stride:
                    source_frame += 1
                    continue

                try:
                    predictions = infer_frame_predictions(
                        model,
                        frame,
                        args.confidence,
                        args.inference_width,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"Racket inference failed at source frame {source_frame}: {error}"
                    ) from error
                predictions = filter_predictions(predictions, args.confidence)
                csv_writer.writerows(csv_rows_for_frame(source_frame, fps, predictions))
                detection_count += len(predictions)

                if not args.no_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(args.output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps / args.frame_stride,
                            (frame_width, frame_height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not open output video: {args.output_video}"
                            )
                    writer.write(draw_predictions(frame, predictions))

                processed_frames += 1
                source_frame += 1
                now = time.monotonic()
                if (
                    processed_frames == planned_frames
                    or now - last_progress >= args.progress_interval
                ):
                    elapsed = max(now - started, 1e-9)
                    rate = processed_frames / elapsed
                    eta = (planned_frames - processed_frames) / max(rate, 1e-9)
                    print(
                        f"Progress: {processed_frames}/{planned_frames} "
                        f"({processed_frames / planned_frames * 100:.1f}%) | "
                        f"{rate:.2f} frames/s | ETA {format_duration(eta)} | "
                        f"{detection_count} detection(s)",
                        flush=True,
                    )
                    last_progress = now
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    metadata = {
        "workspace_name": args.workspace_name,
        "workflow_id": args.workflow_id,
        "api_url": args.api_url,
        "video": str(args.video),
        "output_video": None if args.no_video else str(args.output_video),
        "csv": str(args.csv),
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_width": frame_width,
        "source_height": frame_height,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_stride": args.frame_stride,
        "inference_width": args.inference_width,
        "confidence": args.confidence,
        "processed_frames": processed_frames,
        "detection_count": detection_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    outputs = [str(args.csv), str(metadata_path)]
    if not args.no_video:
        outputs.insert(0, str(args.output_video))
    print(
        f"Done: {processed_frames} frame(s), {detection_count} detection(s) -> "
        + ", ".join(outputs),
        flush=True,
    )


if __name__ == "__main__":
    main()
