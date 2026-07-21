"""Runtime detector for GradientBoostingClassifier bounce artifacts."""

import os
from pathlib import Path

import joblib
import pandas as pd

from judge_call import Point, load_calibration_lines
from train_bounce_classifier import (
    DEFAULT_MODEL_PATH as TRAIN_DEFAULT_MODEL_PATH,
    build_features_for_frame,
    finite_point,
    parse_bool,
)


DEFAULT_MODEL_PATH = TRAIN_DEFAULT_MODEL_PATH
DEFAULT_MIN_GAP_FRAMES = 10
DEFAULT_THRESHOLD = 0.20
DEFAULT_WALL_GATE_PAD_PX = 80.0
DEFAULT_WALL_GATE_PAD_FRACTION = 0.85
DEFAULT_SIDEWALL_GATE_PAD_PX = 120.0
DEFAULT_SIDEWALL_GATE_PAD_FRACTION = 0.25
DEFAULT_WALL_VISIT_GAP_FRAMES = 24

_ARTIFACT_CACHE = {}


def load_artifact(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    stat = model_path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _ARTIFACT_CACHE.get(model_path)
    if cached is None or cached["signature"] != signature:
        artifact = joblib.load(model_path)
        _ARTIFACT_CACHE[model_path] = {
            "artifact": artifact,
            "signature": signature,
        }
    else:
        artifact = cached["artifact"]
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


def inside_x_range(x, wall_x_range, pad_px=0.0):
    wall_x_range = normalize_x_range(wall_x_range)
    if wall_x_range is None:
        return True
    pad_px = max(0.0, float(pad_px))
    return wall_x_range[0] - pad_px <= float(x) <= wall_x_range[1] + pad_px


def env_float(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def env_int(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def runtime_threshold(artifact, threshold):
    if threshold is not None:
        return float(threshold)
    artifact_threshold = float(artifact.get("hit_threshold", DEFAULT_THRESHOLD))
    default_threshold = env_float("BOUNCE_GB_RUNTIME_THRESHOLD", DEFAULT_THRESHOLD)
    return min(artifact_threshold, default_threshold)


def point_progress_on_line(point, line):
    length_squared = line.dx * line.dx + line.dy * line.dy
    if length_squared <= 1e-9:
        return 0.0
    return (
        (point.x - line.left.x) * line.dx
        + (point.y - line.left.y) * line.dy
    ) / length_squared


def calibrated_wall_gate(calibration):
    if not calibration:
        return None
    try:
        top_line, bottom_line = load_calibration_lines(calibration)
    except ValueError:
        return None
    return top_line, bottom_line


def inside_calibrated_wall_gate(x, y, wall_gate, wall_x_range):
    point = Point(float(x), float(y))
    pad_px = env_float("BOUNCE_GB_WALL_GATE_PAD_PX", DEFAULT_WALL_GATE_PAD_PX)
    pad_fraction = env_float(
        "BOUNCE_GB_WALL_GATE_PAD_FRACTION",
        DEFAULT_WALL_GATE_PAD_FRACTION,
    )
    if wall_gate is None:
        line_width = 0.0
        normalized = normalize_x_range(wall_x_range)
        if normalized is not None:
            line_width = normalized[1] - normalized[0]
        return inside_x_range(x, wall_x_range, max(pad_px, pad_fraction * line_width))

    top_line, bottom_line = wall_gate
    top_margin = top_line.signed_distance_below(point)
    bottom_margin = -bottom_line.signed_distance_below(point)
    if top_margin < -pad_px or bottom_margin < -pad_px:
        return False

    # Camera tilt means raw x can be misleading. Use progress along the
    # calibrated tin/out-line direction, then allow a generous extension
    # because users often label only the clearly visible middle of the line.
    u = point_progress_on_line(point, bottom_line)
    return -pad_fraction <= u <= 1.0 + pad_fraction


def inside_lenient_sidewall_gate(x, y, wall_gate, wall_x_range):
    pad_px = env_float("BOUNCE_GB_SIDEWALL_GATE_PAD_PX", DEFAULT_SIDEWALL_GATE_PAD_PX)
    pad_fraction = env_float(
        "BOUNCE_GB_SIDEWALL_GATE_PAD_FRACTION",
        DEFAULT_SIDEWALL_GATE_PAD_FRACTION,
    )
    normalized = normalize_x_range(wall_x_range)
    if wall_gate is None:
        line_width = (normalized[1] - normalized[0]) if normalized is not None else 0.0
        return inside_x_range(x, wall_x_range, max(pad_px, pad_fraction * line_width))

    # Only gate horizontally. This removes obvious side-wall detections while
    # staying lenient about height, camera tilt, and imperfect line taps.
    point = Point(float(x), float(y))
    _, bottom_line = wall_gate
    u = point_progress_on_line(point, bottom_line)
    return -pad_fraction <= u <= 1.0 + pad_fraction


def pick_probability_peaks(candidates, min_gap):
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    picked = []
    for candidate in ranked:
        if any(abs(candidate["hit_frame"] - hit["hit_frame"]) < min_gap for hit in picked):
            continue
        picked.append(candidate)
    return sorted(picked, key=lambda item: item["hit_frame"])


def collapse_wall_area_duplicates(candidates, max_gap=DEFAULT_WALL_VISIT_GAP_FRAMES):
    if not candidates:
        return []

    max_gap = max(0, int(max_gap))
    grouped = []
    current = []
    last_frame = None
    for candidate in sorted(candidates, key=lambda item: item["hit_frame"]):
        frame = int(candidate["hit_frame"])
        if current and last_frame is not None and frame - last_frame > max_gap:
            grouped.append(current)
            current = []
        current.append(candidate)
        last_frame = frame
    if current:
        grouped.append(current)

    picked = []
    for group in grouped:
        best = max(group, key=lambda item: item["score"])
        best = dict(best)
        best["wall_visit_candidate_count"] = len(group)
        best["wall_visit_frames"] = [int(item["hit_frame"]) for item in group]
        picked.append(best)
    return sorted(picked, key=lambda item: item["hit_frame"])


def detect_hits_with_gb_model(
    rows,
    *,
    model_path=DEFAULT_MODEL_PATH,
    threshold=None,
    min_gap=DEFAULT_MIN_GAP_FRAMES,
    wall_x_range=None,
    calibration=None,
    apply_spatial_filter=True,
    spatial_filter_mode="wall",
    collapse_wall_area=True,
):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    threshold = runtime_threshold(artifact, threshold)
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
    wall_gate = calibrated_wall_gate(calibration)

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
        if apply_spatial_filter:
            if spatial_filter_mode == "sidewall":
                inside_gate = inside_lenient_sidewall_gate(
                    candidate_x,
                    candidate_y,
                    wall_gate,
                    wall_x_range,
                )
            else:
                inside_gate = inside_calibrated_wall_gate(
                    candidate_x,
                    candidate_y,
                    wall_gate,
                    wall_x_range,
                )
            if not inside_gate:
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
                "event_type": "wall",
                "classification_source": "gradient_boosting_model",
            }
        )

    if collapse_wall_area:
        visit_gap = env_int("BOUNCE_GB_WALL_VISIT_GAP_FRAMES", DEFAULT_WALL_VISIT_GAP_FRAMES)
        candidates = collapse_wall_area_duplicates(candidates, visit_gap)

    return pick_probability_peaks(candidates, min_gap)
