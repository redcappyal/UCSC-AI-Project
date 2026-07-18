"""CLI: run the local Roboflow model over a video and write ball coordinates."""

import argparse
import csv
import os
from pathlib import Path

import cv2

# inference_engine sets model-cache/metrics env defaults (and the CPU-safe
# torch.cuda.stream patch) on import, before the inference package loads.
from inference_engine import (
    DEFAULT_INFERENCE_WIDTH,
    DEFAULT_MODEL_ID,
    infer_frame_predictions,
    load_model,
)
from tracking_common import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
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


DEFAULT_VIDEO_INPUT_PATH = Path(__file__).with_name("ModelTrainTest.mp4")
DEFAULT_OUTPUT_VIDEO_PATH = Path(__file__).with_name("annotated_output_local.mp4")
DEFAULT_CSV_OUTPUT_PATH = Path(__file__).with_name("ball_coordinates_local.csv")


def positive_int_or_none(value):
    if value is None:
        return None

    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Roboflow model locally with inference.get_model and write ball coordinates."
    )
    parser.add_argument("--model-id", default=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_INPUT_PATH)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--start-frame", type=positive_int_or_none, default=None)
    parser.add_argument("--end-frame", type=positive_int_or_none, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
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
    start_frame = 2200
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
            "No Roboflow API key found. Set ROBOFLOW_API_KEY in your shell or .env."
        )

    print(f"Loading local Roboflow model: {args.model_id}")
    model = load_model(args.model_id, args.api_key)
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
