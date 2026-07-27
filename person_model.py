"""Person detection seam for player attribution.

RF-DETR Keypoint (Apache-2.0) is the only real backend. rfdetr is imported
lazily so the test suite and any environment without the package still import
this module. When the backend is unavailable the pipeline falls back to the
assumed-alternation attribution and says so (spec §4.1 — never silently).
"""

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

PERSON_SCHEMA_VERSION = "person-model-v1"
PERSON_CONFIDENCE_THRESHOLD = 0.5
CROP_PAD_RATIO = 0.25  # padding added around a bbox crop, fraction of each side

COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class PersonDetection:
    x: float          # bbox center x, px
    y: float          # bbox center y, px
    width: float
    height: float
    confidence: float
    keypoints: tuple  # 17 x (x_px, y_px, confidence), or () when unavailable

    @property
    def foot_px(self):
        return (self.x, self.y + self.height / 2.0)


def keypoints_result_to_detections(xyxy, det_conf, kp_xy, kp_conf,
                                   threshold=PERSON_CONFIDENCE_THRESHOLD):
    """sv.KeyPoints arrays -> PersonDetection list. Pure numpy; no rfdetr."""
    detections = []
    if xyxy is None or det_conf is None:
        return detections
    xyxy = np.asarray(xyxy, dtype=float)
    det_conf = np.asarray(det_conf, dtype=float)
    for index in range(xyxy.shape[0]):
        confidence = float(det_conf[index])
        if confidence < threshold:
            continue
        x_min, y_min, x_max, y_max = (float(v) for v in xyxy[index])
        keypoints = ()
        if kp_xy is not None and kp_conf is not None:
            kp_xy_arr = np.asarray(kp_xy, dtype=float)
            kp_conf_arr = np.asarray(kp_conf, dtype=float)
            keypoints = tuple(
                (float(kp_xy_arr[index, k, 0]),
                 float(kp_xy_arr[index, k, 1]),
                 float(kp_conf_arr[index, k]))
                for k in range(kp_xy_arr.shape[1])
            )
        detections.append(PersonDetection(
            x=(x_min + x_max) / 2.0,
            y=(y_min + y_max) / 2.0,
            width=x_max - x_min,
            height=y_max - y_min,
            confidence=confidence,
            keypoints=keypoints,
        ))
    return detections


def _import_rfdetr():
    """Lazy rfdetr import. Returns the model class or None."""
    try:
        from rfdetr import RFDETRKeypointPreview  # noqa: PLC0415 — deliberate lazy import
    except Exception:
        return None
    return RFDETRKeypointPreview


class RFDETRPersonDetector:
    backend = "rfdetr"

    def __init__(self, model_class):
        # First construction downloads the COCO checkpoint (docs/PERSON_MODEL.md).
        self._model = model_class()

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._model.predict(frame_rgb, threshold=PERSON_CONFIDENCE_THRESHOLD)
        xyxy = None
        data = getattr(result, "data", None) or {}
        if "xyxy" in data:
            xyxy = data["xyxy"]
        return keypoints_result_to_detections(
            xyxy,
            getattr(result, "detection_confidence", None),
            getattr(result, "xy", None),
            getattr(result, "keypoint_confidence", None),
        )


def available_backend():
    if os.getenv("PERSON_DETECTOR", "").strip().lower() == "none":
        return "none"
    return "rfdetr" if _import_rfdetr() is not None else "none"


def load_person_detector():
    if os.getenv("PERSON_DETECTOR", "").strip().lower() == "none":
        return None
    model_class = _import_rfdetr()
    if model_class is None:
        return None
    return RFDETRPersonDetector(model_class)


def save_person_crop(video_path, frame_idx, detection, out_path,
                     pad_ratio=CROP_PAD_RATIO):
    """Seek one frame and write the padded bbox crop. ASCII out_path only
    (CLAUDE.md: cv2.imwrite must never see a non-ASCII path)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return False

    height, width = frame.shape[:2]
    pad_x = detection.width * pad_ratio
    pad_y = detection.height * pad_ratio
    x_min = max(0, int(detection.x - detection.width / 2 - pad_x))
    x_max = min(width, int(detection.x + detection.width / 2 + pad_x))
    y_min = max(0, int(detection.y - detection.height / 2 - pad_y))
    y_max = min(height, int(detection.y + detection.height / 2 + pad_y))
    if x_max <= x_min or y_max <= y_min:
        return False

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame[y_min:y_max, x_min:x_max]))
