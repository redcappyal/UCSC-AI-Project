"""Shared ball-tracking constants and pure helpers.

Importing this module has no side effects (no network clients, no model
loads); anything here is safe for the Flask app, the CLIs, and tests.
"""

import cv2


CONFIDENCE_THRESHOLD = 0.25
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
