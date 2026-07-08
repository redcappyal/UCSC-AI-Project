import base64
import time
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


API_URL = "https://serverless.roboflow.com"
API_KEY = ""

WORKSPACE = "squash-line-calling-model"
WORKFLOW_ID = "squash-line-calling-vsquash-line-calling-1-rfdetr-medium-t1-logic"
IMAGE_INPUT = "image"

VIDEO_INPUT_PATH = Path("/Users/Alvin/Downloads/SquashAnalytics/SquashAnalytics.mp4")
VIDEO_OUTPUT_PATH = Path(__file__).with_name("annotated_output.mp4")

# If your Roboflow workflow has a visualization block, this is usually the
# output key. If it is missing, the script falls back to drawing predictions.
ROBOFLOW_ANNOTATED_OUTPUT = "output_image"

FRAME_STRIDE = 2
MAX_FRAMES = 3000  # Use an integer like 300 for a quick test.
START_SECONDS = 70  # 1:10 into the source video.
CONFIDENCE_THRESHOLD = 0.25
MAX_RETRIES = 3


client = InferenceHTTPClient.init(
    api_url=API_URL,
    api_key=API_KEY,
)


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


def draw_predictions(frame, predictions):
    output_frame = frame.copy()
    frame_height, frame_width = output_frame.shape[:2]

    for prediction in predictions:
        confidence = prediction.get("confidence", 1.0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        class_name = prediction.get("class", prediction.get("class_name", "object"))
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


def process_frame(frame):
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
    output = result[0] if isinstance(result, list) and result else result
    output = output if isinstance(output, dict) else {}

    roboflow_frame = decode_output_image(output.get(ROBOFLOW_ANNOTATED_OUTPUT))
    if roboflow_frame is not None:
        return roboflow_frame

    predictions = find_predictions(output)
    return draw_predictions(source_frame, predictions)


def main():
    cap = cv2.VideoCapture(str(VIDEO_INPUT_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_INPUT_PATH}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = source_fps / FRAME_STRIDE
    start_frame = int(START_SECONDS * source_fps)

    if start_frame >= source_frame_count:
        raise RuntimeError(
            f"START_SECONDS={START_SECONDS} starts after the video ends "
            f"({source_frame_count / source_fps:.2f} seconds)."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = None
    read_count = start_frame
    processed_count = 0

    print(f"Source video: {source_frame_count} frames at {source_fps:.2f} FPS")
    print(f"Starting at {START_SECONDS:.2f}s / source frame {start_frame}")
    print(f"Processing every {FRAME_STRIDE} frame(s)")
    print(f"Output video: {VIDEO_OUTPUT_PATH}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if read_count % FRAME_STRIDE != 0:
                read_count += 1
                continue

            result = process_frame(frame)
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

    print(f"Done! Processed {processed_count} frame(s) -> {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
