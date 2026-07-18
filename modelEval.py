"""CLI: run the Roboflow serverless workflow over a video (remote inference).

For local in-process inference use local_model_eval.py; the web app uses
job_runner.py. Shared helpers live in tracking_common.py.
"""

import base64
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np

from tracking_common import (
    CSV_FIELDNAMES,
    ball_csv_row,
    draw_predictions,
    find_predictions,
    select_ball_prediction,
)


API_URL = "https://serverless.roboflow.com"

WORKSPACE = "matthews-workspace-vbemk"
WORKFLOW_ID = "ai-squash-line-judge-3-vai-squash-line-judge-3-hifw5-2-yolo26s-t1-logic"
IMAGE_INPUT = "image"

VIDEO_INPUT_PATH = Path(__file__).with_name("SquashTestVid.mp4")
VIDEO_OUTPUT_PATH = Path(__file__).with_name("annotated_output.mp4")
CSV_OUTPUT_PATH = Path(__file__).with_name("ball_coordinates.csv")

# If your Roboflow workflow has a visualization block, this is usually the
# output key. If it is missing, the script falls back to drawing predictions.
ROBOFLOW_ANNOTATED_OUTPUT = "output_image"

FRAME_STRIDE = 1
START_FRAME = None
END_FRAME = None
MAX_FRAMES = None  # Use an integer like 300 for a quick test.
START_SECONDS = None  # Used only when START_FRAME is None.
MAX_RETRIES = 3


def decode_output_image(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, dict):
        value = value.get("value")

    if isinstance(value, str):
        img_bytes = base64.b64decode(value)
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return None


def process_frame(client, frame):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={IMAGE_INPUT: frame},
                use_cache=False,
            )
        except Exception as error:
            last_error = error
            print(f"Workflow request failed on attempt {attempt}/{MAX_RETRIES}: {error}")
            time.sleep(1)

    raise last_error


def result_to_frame(result, source_frame):
    output = result_to_output(result)

    roboflow_frame = decode_output_image(output.get(ROBOFLOW_ANNOTATED_OUTPUT))
    if roboflow_frame is not None:
        return roboflow_frame

    predictions = find_predictions(output)
    return draw_predictions(source_frame, predictions)


def result_to_output(result):
    output = result[0] if isinstance(result, list) and result else result
    return output if isinstance(output, dict) else {}


def main():
    from inference_sdk import InferenceHTTPClient

    api_key = "fSbrLMfk0LmeJbQfbGRp"
    if not api_key.strip():
        raise RuntimeError("Set ROBOFLOW_API_KEY before running the remote workflow.")

    client = InferenceHTTPClient.init(api_url=API_URL, api_key=api_key)

    cap = cv2.VideoCapture(str(VIDEO_INPUT_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_INPUT_PATH}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = source_fps / FRAME_STRIDE
    if START_FRAME is not None:
        start_frame = START_FRAME
    else:
        start_frame = int((START_SECONDS or 0) * source_fps)

    end_frame = END_FRAME if END_FRAME is not None else source_frame_count - 1

    if start_frame >= source_frame_count:
        raise RuntimeError(
            f"START_FRAME={start_frame} starts after the video ends "
            f"({source_frame_count / source_fps:.2f} seconds)."
        )

    if end_frame < start_frame:
        raise RuntimeError(f"END_FRAME={end_frame} is before START_FRAME={start_frame}.")

    end_frame = min(end_frame, source_frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = None
    read_count = start_frame
    processed_count = 0

    print(f"Source video: {source_frame_count} frames at {source_fps:.2f} FPS")
    print(f"Processing source frames {start_frame} through {end_frame}")
    print(f"Processing every {FRAME_STRIDE} frame(s)")
    print(f"Output video: {VIDEO_OUTPUT_PATH}")
    print(f"Output CSV: {CSV_OUTPUT_PATH}")

    try:
        with CSV_OUTPUT_PATH.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()

            while True:
                if read_count > end_frame:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                if read_count % FRAME_STRIDE != 0:
                    read_count += 1
                    continue

                result = process_frame(client, frame)
                output = result_to_output(result)
                predictions = find_predictions(output)
                ball_prediction = select_ball_prediction(predictions)
                csv_writer.writerow(ball_csv_row(read_count, source_fps, ball_prediction))

                output_frame = result_to_frame(result, frame)

                if writer is None:
                    height, width = output_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(VIDEO_OUTPUT_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )

                writer.write(output_frame)
                processed_count += 1

                print(f"Processed source frame {read_count} -> output frame {processed_count}")

                read_count += 1
                if MAX_FRAMES is not None and processed_count >= MAX_FRAMES:
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(
        f"Done! Processed {processed_count} frame(s) -> "
        f"{VIDEO_OUTPUT_PATH}, {CSV_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
