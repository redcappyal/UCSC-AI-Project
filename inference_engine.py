"""Local Roboflow model loading and frame inference.

This module owns the one copy of the inference environment setup and the
cached in-process model. The `inference` package is imported lazily inside
get_tracking_model so importing this module stays side-effect free apart
from environment defaults, which must be set before `inference` loads.
"""

import os
import threading
from pathlib import Path

import cv2

from tracking_common import find_predictions


ROOT = Path(__file__).resolve().parent

os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / ".roboflow-cache"))
os.environ.setdefault("METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
# The CoreML execution provider compiles this RF-DETR export but fails at
# inference time on macOS, and onnxruntime tries it by default. Pin CPU; a
# GPU server can override this in .env (e.g. CUDAExecutionProvider).
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CPUExecutionProvider]")

DEFAULT_MODEL_ID = "squash-line-calling-model/1"
DEFAULT_INFERENCE_WIDTH = int(os.getenv("INFERENCE_WIDTH", "960"))

_MODEL = None
_MODEL_ID = None
_MODEL_LOAD_LOCK = threading.Lock()


def get_tracking_model():
    global _MODEL, _MODEL_ID

    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key.strip():
        raise RuntimeError("No Roboflow API key found. Set ROBOFLOW_API_KEY in .env.")

    with _MODEL_LOAD_LOCK:
        if _MODEL is not None and _MODEL_ID == model_id:
            return _MODEL

        from inference import get_model

        _MODEL = get_model(
            model_id=model_id,
            api_key=api_key,
            countinference=False,
        )
        _MODEL_ID = model_id
        return _MODEL


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


def normalize_prediction(prediction):
    normalized = dict(prediction)

    if "class" not in normalized and "class_name" in normalized:
        normalized["class"] = normalized["class_name"]

    if "class_name" not in normalized and "class" in normalized:
        normalized["class_name"] = normalized["class"]

    if "x" in normalized and "y" in normalized and "width" in normalized and "height" in normalized:
        return normalized

    for min_x, min_y, max_x, max_y in (("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax")):
        if all(key in normalized for key in (min_x, min_y, max_x, max_y)):
            x_min = float(normalized[min_x])
            y_min = float(normalized[min_y])
            x_max = float(normalized[max_x])
            y_max = float(normalized[max_y])
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
    # Roboflow local models accept BGR numpy arrays directly; converting to
    # PIL would add a full extra copy of every frame.
    try:
        return model.infer(frame, confidence=confidence)
    except TypeError:
        return model.infer(frame)


def infer_frame_predictions(model, frame, confidence, max_width=DEFAULT_INFERENCE_WIDTH):
    inference_frame, x_scale, y_scale = resize_frame_for_inference(frame, max_width)
    result = infer_frame(model, inference_frame, confidence)
    predictions = normalize_predictions(result)
    return scale_predictions(predictions, x_scale, y_scale)
