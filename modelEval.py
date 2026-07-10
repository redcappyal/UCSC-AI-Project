import base64
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


API_URL = "https://serverless.roboflow.com"
API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

WORKSPACE = "squash-line-calling-model"
WORKFLOW_ID = "squash-line-calling-vsquash-line-calling-1-rfdetr-medium-t1-logic"
IMAGE_INPUT = "image"

VIDEO_INPUT_PATH = Path(__file__).with_name("SquashAnalytics.mp4")
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
CONFIDENCE_THRESHOLD = 0.25
MAX_RETRIES = 3
BALL_CLASS_NAMES = {"ball", "squash ball", "squash-ball", "squash_ball"}
CSV_FIELDNAMES = [
    "source_frame",
    "timestamp_seconds",
    "detected",
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


def prediction_class_name(prediction):
    return str(
        prediction.get(
            "class",
            prediction.get("class_name", prediction.get("name", "object")),
        )
    )


def is_ball_prediction(prediction):
    class_name = prediction_class_name(prediction).strip().lower()
    return class_name in BALL_CLASS_NAMES


def select_ball_prediction(predictions):
    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("confidence", 1.0) >= CONFIDENCE_THRESHOLD
    ]
    ball_predictions = [
        prediction for prediction in valid_predictions if is_ball_prediction(prediction)
    ]

    if not ball_predictions and len(valid_predictions) == 1:
        ball_predictions = valid_predictions

    if not ball_predictions:
        return None

    return max(ball_predictions, key=lambda prediction: prediction.get("confidence", 1.0))


def ball_csv_row(source_frame, source_fps, prediction):
    row = {
        "source_frame": source_frame,
        "timestamp_seconds": f"{source_frame / source_fps:.6f}",
        "detected": False,
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

    if prediction is None:
        return row

    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    row.update(
        {
            "detected": True,
            "class_name": prediction_class_name(prediction),
            "confidence": f"{prediction.get('confidence', 1.0):.6f}",
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
    return row


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

                result = process_frame(frame)
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
