"""Runtime detector for GradientBoostingClassifier bounce artifacts."""

from pathlib import Path

import joblib
import pandas as pd

from judge_call import load_calibration_lines
from train_bounce_classifier import build_features_for_frame, finite_point, parse_bool


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "bounce_gb_better_features.pkl"
DEFAULT_MIN_GAP_FRAMES = 10
DEFAULT_THRESHOLD = 0.25

_ARTIFACT_CACHE = {}


def load_artifact(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    artifact = _ARTIFACT_CACHE.get(model_path)
    if artifact is None:
        artifact = joblib.load(model_path)
        _ARTIFACT_CACHE[model_path] = artifact
    return artifact


def infer_context(feature_columns):
    offsets = []
    for column in feature_columns:
        if not column.startswith("t"):
            continue
        prefix = column.split("_", 1)[0]
        try:
            offsets.append(abs(int(prefix[1:])))
        except ValueError:
            continue
    return max(offsets, default=3)


def calibration_geometry(calibration):
    if not calibration:
        return None
    top_line, bottom_line = load_calibration_lines(calibration)
    tin_left = min(bottom_line.left.x, bottom_line.right.x)
    tin_right = max(bottom_line.left.x, bottom_line.right.x)
    return {
        "top_line": top_line,
        "bottom_line": bottom_line,
        "tin_left": tin_left,
        "tin_right": tin_right,
    }


def row_to_feature_row(row):
    detected = parse_bool(row.get("detected"))
    return {
        "frame": int(row["source_frame"]),
        "timestamp": float(row.get("timestamp_seconds") or 0.0),
        "detected": detected,
        "confidence": float(row["confidence"]) if row.get("confidence") else 0.0,
        "x": float(row["x_center"]) if detected and row.get("x_center") else float("nan"),
        "y": float(row["y_center"]) if detected and row.get("y_center") else float("nan"),
        "width": float(row["width"]) if detected and row.get("width") else 0.0,
        "height": float(row["height"]) if detected and row.get("height") else 0.0,
    }


def rows_by_frame(rows):
    return {int(row["source_frame"]): row_to_feature_row(row) for row in rows}


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


def detect_hits_with_gb_model(
    rows,
    *,
    model_path=DEFAULT_MODEL_PATH,
    threshold=None,
    min_gap=DEFAULT_MIN_GAP_FRAMES,
    wall_x_range=None,
    calibration=None,
):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact.get("hit_threshold", DEFAULT_THRESHOLD) if threshold is None else threshold)
    context = infer_context(feature_columns)
    include_geometry = any(
        column in feature_columns
        for column in (
            "inside_tin_x_range",
            "nearest_wall_line_distance_px",
            "min_nearest_wall_line_distance_context",
        )
    )
    geometry = calibration_geometry(calibration) if include_geometry else None

    parsed_rows = rows_by_frame(rows)
    if not parsed_rows:
        return []

    frames = sorted(parsed_rows)
    records = []
    full_records = []
    for frame in frames:
        record = build_features_for_frame(
            frame,
            parsed_rows,
            geometry,
            context,
            include_geometry,
            None,
            5,
        )
        full_records.append(record)
        records.append({column: record.get(column, 0.0) for column in feature_columns})

    frame_features = pd.DataFrame(records, columns=feature_columns)
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(frame_features)[:, positive_class_index]

    candidates = []
    for frame, probability, record in zip(frames, probabilities, full_records):
        if probability < threshold:
            continue
        row = parsed_rows[frame]
        if not finite_point(row):
            continue
        candidate_x = float(row["x"])
        candidate_y = float(row["y"])
        if not inside_x_range(candidate_x, wall_x_range):
            continue
        candidates.append(
            {
                "hit_frame": int(frame),
                "timestamp_seconds": float(row["timestamp"]),
                "score": float(probability),
                "model_probability": float(probability),
                "dv_magnitude": float(record.get("velocity_change_px_s", 0.0)),
                "speed_before": float(record.get("speed_before_px_s", 0.0)),
                "speed_after": float(record.get("speed_after_px_s", 0.0)),
                "turn_degrees": float(record.get("turn_degrees", 0.0)),
                "after_gap": False,
                "candidate_x": candidate_x,
                "candidate_y": candidate_y,
                "detector": "gradient_boosting",
            }
        )

    return pick_probability_peaks(candidates, min_gap)
