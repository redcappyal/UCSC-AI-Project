"""Model-based wall-hit detector for wall_hit_model.pkl.

The current train_wall_hit_model.py artifact stores metadata, but this loader
also accepts the older raw sklearn Pipeline while local experiments migrate.
"""

import os
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "wall_hit_model.pkl"
WINDOW_RADIUS = 10
DEFAULT_THRESHOLD = float(os.getenv("WALL_HIT_MODEL_THRESHOLD", "0.50"))
DEFAULT_MIN_GAP_FRAMES = 10

_MODEL_CACHE = {}


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_artifact(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    artifact = _MODEL_CACHE.get(model_path)
    if artifact is None:
        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            artifact = {
                "model": artifact,
                "window_radius": WINDOW_RADIUS,
                "hit_threshold": DEFAULT_THRESHOLD,
                "feature_format": "wall_hit_frame_window_v1",
                "model_type": "unknown",
            }
        _MODEL_CACHE[model_path] = artifact
    return artifact


def row_to_position(row):
    detected = parse_bool(row.get("detected"))
    if detected and row.get("x_center") and row.get("y_center"):
        x = float(row["x_center"])
        y = float(row["y_center"])
    else:
        x = 0.0
        y = 0.0
    return {
        "frame": int(row["source_frame"]),
        "timestamp": float(row.get("timestamp_seconds") or 0.0),
        "detected": 1.0 if detected else 0.0,
        "x_center": x,
        "y_center": y,
    }


def rows_by_frame(rows):
    return {int(row["source_frame"]): row_to_position(row) for row in rows}


def build_feature_vector(positions, center_frame, window_radius=WINDOW_RADIUS):
    features = []
    for offset in range(-window_radius, window_radius + 1):
        row = positions.get(center_frame + offset)
        if row is None:
            features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([row["x_center"], row["y_center"], row["detected"]])
    return features


def normalize_x_range(wall_x_range):
    if wall_x_range is None:
        return None
    left, right = map(float, wall_x_range)
    if right < left:
        left, right = right, left
    if right <= left:
        return None
    return left, right


def inside_x_range(x, wall_x_range):
    wall_x_range = normalize_x_range(wall_x_range)
    if wall_x_range is None:
        return True
    return wall_x_range[0] <= float(x) <= wall_x_range[1]


def pick_probability_peaks(candidates, min_gap):
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    picked = []
    for candidate in ranked:
        if any(abs(candidate["hit_frame"] - hit["hit_frame"]) < min_gap for hit in picked):
            continue
        picked.append(candidate)
    return sorted(picked, key=lambda item: item["hit_frame"])


def detect_hits_with_model(
    rows,
    *,
    model_path=DEFAULT_MODEL_PATH,
    threshold=None,
    min_gap=DEFAULT_MIN_GAP_FRAMES,
    wall_x_range=None,
):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    window_radius = int(artifact.get("window_radius", WINDOW_RADIUS))
    threshold = float(artifact.get("hit_threshold", DEFAULT_THRESHOLD) if threshold is None else threshold)
    positions = rows_by_frame(rows)
    if not positions:
        return []

    frames = sorted(positions)
    features = np.array(
        [build_feature_vector(positions, frame, window_radius) for frame in frames],
        dtype=np.float32,
    )
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_class_index]

    candidates = []
    for frame, probability in zip(frames, probabilities):
        if probability < threshold:
            continue
        row = positions[frame]
        if not row["detected"]:
            continue
        candidate_x = float(row["x_center"])
        candidate_y = float(row["y_center"])
        if not inside_x_range(candidate_x, wall_x_range):
            continue
        candidates.append(
            {
                "hit_frame": int(frame),
                "timestamp_seconds": float(row["timestamp"]),
                "score": float(probability),
                "model_probability": float(probability),
                "dv_magnitude": 0.0,
                "speed_before": 0.0,
                "speed_after": 0.0,
                "turn_degrees": 0.0,
                "after_gap": False,
                "candidate_x": candidate_x,
                "candidate_y": candidate_y,
                "detector": "wall_hit_model",
                "model_type": artifact.get("model_type", "unknown"),
            }
        )

    return pick_probability_peaks(candidates, min_gap)


# Backwards-compatible name for local experiments that imported it earlier.
detect_hits_with_legacy_model = detect_hits_with_model
