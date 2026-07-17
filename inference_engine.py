"""Local Roboflow model loading and frame inference.

This module owns the one copy of the inference environment setup and the
cached in-process model. The `inference` package is imported lazily inside
get_tracking_model so importing this module stays side-effect free apart
from environment defaults, which must be set before `inference` loads.
"""

import os
import threading
import warnings
from contextlib import nullcontext
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
# The CoreML execution provider runs this RF-DETR export but only claims
# ~80% of the graph, fragmenting it into 100+ partitions that end up slower
# than pure CPU. Pin CPU for the ONNX fallback path; a GPU server can
# override this in .env (e.g. CUDAExecutionProvider).
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CPUExecutionProvider]")

# Parts of the inference stack call torch.cuda.stream(None) even on CPU-only
# machines, which crashes mid-run; make a None stream a no-op context.
try:
    import torch
except ImportError:
    torch = None

if torch is not None and not getattr(torch.cuda.stream, "_squash_cpu_safe", False):
    _torch_cuda_stream = torch.cuda.stream

    def _cpu_safe_cuda_stream(stream):
        if stream is None:
            return nullcontext()
        return _torch_cuda_stream(stream)

    _cpu_safe_cuda_stream._squash_cpu_safe = True
    torch.cuda.stream = _cpu_safe_cuda_stream


def best_torch_device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# The torch backend runs RF-DETR on the GPU (MPS on Apple Silicon), measured
# ~4x faster than ONNX on CPU (68ms vs 275ms per 960px frame on an M2).
# "auto" picks it whenever a GPU exists; TRACKING_BACKEND=onnx forces the
# ONNX/CPU path.
TRACKING_BACKEND = os.environ.get("TRACKING_BACKEND", "auto").strip().lower()
if TRACKING_BACKEND not in {"torch", "onnx"}:
    TRACKING_BACKEND = "torch" if best_torch_device() != "cpu" else "onnx"

if TRACKING_BACKEND == "torch":
    os.environ.setdefault("DEFAULT_DEVICE", best_torch_device())
    # The auto-loader in inference-models tries backends in its own order;
    # disabling the alternatives is the only way to select the torch package.
    # Must be set before the first `inference` import.
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "onnx,trt,torch-script,ultralytics,hugging-face",
    )
else:
    os.environ.setdefault("DEFAULT_DEVICE", "cpu")

DEFAULT_MODEL_ID = "ai-squash-line-tracker/4"
DEFAULT_INFERENCE_WIDTH = int(os.getenv("INFERENCE_WIDTH", "960"))

_MODEL = None
_MODEL_ID = None
_MODEL_LOAD_LOCK = threading.Lock()


def configured_providers():
    raw = os.environ.get("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CPUExecutionProvider]")
    return [item.strip() for item in raw.strip("[]").split(",") if item.strip()]


def _patch_rfdetr_checkpoint_loading():
    # Roboflow's torch checkpoints for RF-DETR carry a stray _kp_active_mask
    # buffer from keypoint-capable training code that inference-models
    # (<= 0.31.0) rejects during strict state_dict loading.
    try:
        from inference_models.models.rfdetr.rfdetr_base_pytorch import LWDETR
    except ImportError:
        return

    if getattr(LWDETR.load_state_dict, "_squash_kp_mask_safe", False):
        return

    original = LWDETR.load_state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        cleaned = {key: value for key, value in state_dict.items() if key != "_kp_active_mask"}
        return original(self, cleaned, *args, **kwargs)

    load_state_dict._squash_kp_mask_safe = True
    LWDETR.load_state_dict = load_state_dict


def _optimize_torch_model(model, device):
    inner = getattr(model, "_model", None)
    if inner is None or not hasattr(inner, "optimize_for_inference"):
        return

    # fp16 halves inference time on GPU devices; on CPU it is slower.
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    try:
        inner.optimize_for_inference(compile=True, batch_size=1, dtype=dtype)
    except Exception as error:
        warnings.warn(f"optimize_for_inference failed; using the plain torch model: {error}")
        try:
            inner.remove_optimized_model()
        except Exception:
            pass


def _load_onnx_model(model_id, api_key):
    from inference import get_model
    from inference.core import env as inference_env

    # Re-enable the backends disabled above to prefer torch; the adapter
    # re-reads this set on every model construction.
    disabled = getattr(inference_env, "DISABLED_INFERENCE_MODELS_BACKENDS", None)
    if isinstance(disabled, set):
        disabled.clear()

    return get_model(
        model_id=model_id,
        api_key=api_key,
        countinference=False,
        device="cpu",
        onnx_execution_providers=configured_providers(),
    )


def load_model(model_id, api_key):
    if TRACKING_BACKEND == "torch":
        device = os.environ.get("DEFAULT_DEVICE", "cpu")
        try:
            from inference import get_model

            _patch_rfdetr_checkpoint_loading()
            model = get_model(
                model_id=model_id,
                api_key=api_key,
                countinference=False,
                device=device,
            )
            _optimize_torch_model(model, device)
            return model
        except Exception as error:
            warnings.warn(
                f"Torch backend unavailable for {model_id} on {device}; "
                f"falling back to ONNX on CPU: {error}"
            )

    return _load_onnx_model(model_id, api_key)


def get_tracking_model():
    global _MODEL, _MODEL_ID

    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key.strip():
        raise RuntimeError("No Roboflow API key found. Set ROBOFLOW_API_KEY in .env.")

    with _MODEL_LOAD_LOCK:
        if _MODEL is not None and _MODEL_ID == model_id:
            return _MODEL

        _MODEL = load_model(model_id, api_key)
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
