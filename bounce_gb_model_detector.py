"""Runtime detector for GradientBoostingClassifier bounce artifacts."""

import os
from pathlib import Path

import joblib
import pandas as pd

from judge_call import Point, load_calibration_lines, load_wall_corners
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
DEFAULT_FRONT_WALL_CHUNK_PAD_PX = 24.0
DEFAULT_FRONT_WALL_CHUNK_PAD_FRACTION = 0.02
DEFAULT_STATIONARY_WINDOW_FRAMES = 8
DEFAULT_STATIONARY_MIN_DETECTIONS = 4
DEFAULT_STATIONARY_MIN_SPAN_PX = 12.0
DEFAULT_STATIONARY_MIN_PATH_PX = 18.0

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
    wall_corners = load_wall_corners(calibration)
    try:
        top_line, bottom_line = load_calibration_lines(calibration)
    except ValueError:
        if wall_corners is not None:
            return ("wall_corners", wall_corners, None, None)
        return None
    if wall_corners is not None:
        return ("wall_corners", wall_corners, top_line, bottom_line)
    return ("lines", top_line, bottom_line)


def wall_corners_y_bounds(wall_corners):
    return (
        min(wall_corners.top_left.y, wall_corners.top_right.y),
        max(wall_corners.bottom_left.y, wall_corners.bottom_right.y),
    )


def wall_corners_x_bounds_at_y(wall_corners, y):
    top_y, bottom_y = wall_corners_y_bounds(wall_corners)
    clamped_y = min(max(float(y), top_y), bottom_y)
    return wall_corners.x_bounds_at_y(clamped_y)


def inside_wall_corners_gate(x, y, wall_corners, *, horizontal_only, pad_px, pad_fraction):
    x = float(x)
    y = float(y)
    top_y, bottom_y = wall_corners_y_bounds(wall_corners)
    left, right = wall_corners_x_bounds_at_y(wall_corners, y)
    wall_width = max(0.0, right - left)
    pad = max(float(pad_px), float(pad_fraction) * wall_width)

    if x < left - pad or x > right + pad:
        return False

    if horizontal_only:
        return True

    return top_y - pad <= y <= bottom_y + pad


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

    if wall_gate[0] == "wall_corners":
        return inside_wall_corners_gate(
            x,
            y,
            wall_gate[1],
            horizontal_only=False,
            pad_px=pad_px,
            pad_fraction=pad_fraction,
        )

    _, top_line, bottom_line = wall_gate
    top_margin = top_line.signed_distance_below(point)
    bottom_margin = -bottom_line.signed_distance_below(point)
    if top_margin < -pad_px or bottom_margin < -pad_px:
        return False

    # Camera tilt means raw x can be misleading. Use progress along the
    # calibrated tin/out-line direction, then allow a generous extension
    # because users often label only the clearly visible middle of the line.
    u = point_progress_on_line(point, bottom_line)
    return -pad_fraction <= u <= 1.0 + pad_fraction


def inside_front_wall_chunk_gate(x, y, wall_gate, wall_x_range):
    """True while the tracked ball is plausibly on the calibrated front wall.

    This is intentionally stricter than the sidewall gate. The sidewall gate is
    only a horizontal sanity check, while this gate defines the start/end of a
    front-wall visit. Leaving this region breaks the visit into a new chunk.
    """
    pad_px = env_float("BOUNCE_GB_FRONT_WALL_CHUNK_PAD_PX", DEFAULT_FRONT_WALL_CHUNK_PAD_PX)
    pad_fraction = env_float(
        "BOUNCE_GB_FRONT_WALL_CHUNK_PAD_FRACTION",
        DEFAULT_FRONT_WALL_CHUNK_PAD_FRACTION,
    )

    if wall_gate is None:
        normalized = normalize_x_range(wall_x_range)
        line_width = (normalized[1] - normalized[0]) if normalized is not None else 0.0
        return inside_x_range(x, wall_x_range, max(pad_px, pad_fraction * line_width))

    point = Point(float(x), float(y))
    if wall_gate[0] == "wall_corners":
        horizontally_inside_wall = inside_wall_corners_gate(
            x,
            y,
            wall_gate[1],
            horizontal_only=True,
            pad_px=pad_px,
            pad_fraction=pad_fraction,
        )
        if not horizontally_inside_wall:
            return False
        bottom_line = wall_gate[3] if len(wall_gate) > 3 else None
        if bottom_line is not None:
            bottom_margin = -bottom_line.signed_distance_below(point)
            return bottom_margin >= -pad_px

        _, bottom_y = wall_corners_y_bounds(wall_gate[1])
        return float(y) <= bottom_y + pad_px

    _, top_line, bottom_line = wall_gate
    bottom_margin = -bottom_line.signed_distance_below(point)
    if bottom_margin < -pad_px:
        return False

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
    if wall_gate[0] == "wall_corners":
        return inside_wall_corners_gate(
            x,
            y,
            wall_gate[1],
            horizontal_only=True,
            pad_px=pad_px,
            pad_fraction=pad_fraction,
        )

    _, _, bottom_line = wall_gate
    u = point_progress_on_line(point, bottom_line)
    return -pad_fraction <= u <= 1.0 + pad_fraction


def motion_stats_around_frame(parsed_rows, frame, window_frames):
    window_frames = max(0, int(window_frames))
    points = []
    for sample_frame in range(int(frame) - window_frames, int(frame) + window_frames + 1):
        row = parsed_rows.get(sample_frame)
        if finite_point(row):
            points.append((sample_frame, float(row["x"]), float(row["y"])))

    if not points:
        return {"detected": 0, "span_px": 0.0, "path_px": 0.0}

    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    path = 0.0
    for prev, cur in zip(points, points[1:]):
        path += ((cur[1] - prev[1]) ** 2 + (cur[2] - prev[2]) ** 2) ** 0.5
    return {
        "detected": len(points),
        "span_px": (x_span * x_span + y_span * y_span) ** 0.5,
        "path_px": path,
    }


def is_stationary_false_track(parsed_rows, frame):
    min_detections = env_int(
        "BOUNCE_GB_STATIONARY_MIN_DETECTIONS",
        DEFAULT_STATIONARY_MIN_DETECTIONS,
    )
    stats = motion_stats_around_frame(
        parsed_rows,
        frame,
        env_int("BOUNCE_GB_STATIONARY_WINDOW_FRAMES", DEFAULT_STATIONARY_WINDOW_FRAMES),
    )
    if stats["detected"] < min_detections:
        return False, stats

    min_span = env_float("BOUNCE_GB_STATIONARY_MIN_SPAN_PX", DEFAULT_STATIONARY_MIN_SPAN_PX)
    min_path = env_float("BOUNCE_GB_STATIONARY_MIN_PATH_PX", DEFAULT_STATIONARY_MIN_PATH_PX)
    is_stationary = stats["span_px"] < min_span and stats["path_px"] < min_path
    return is_stationary, stats


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


def collapse_front_wall_chunks(
    candidates,
    parsed_rows,
    wall_gate,
    wall_x_range,
    fallback_gap=DEFAULT_WALL_VISIT_GAP_FRAMES,
):
    if not candidates:
        return []

    # Without calibration there is no reliable way to know that the ball has
    # left the front wall, so keep the older fixed-gap behavior as fallback.
    if wall_gate is None:
        return collapse_wall_area_duplicates(candidates, fallback_gap)

    candidates_by_frame = {int(candidate["hit_frame"]): candidate for candidate in candidates}
    chunks = []
    current_candidates = []
    current_frames = []

    for frame in sorted(parsed_rows):
        row = parsed_rows[frame]
        if not finite_point(row):
            continue

        inside_front_wall = inside_front_wall_chunk_gate(
            row["x"],
            row["y"],
            wall_gate,
            wall_x_range,
        )
        if not inside_front_wall:
            if current_frames:
                chunks.append((current_frames, current_candidates))
                current_frames = []
                current_candidates = []
            continue

        current_frames.append(int(frame))
        candidate = candidates_by_frame.get(int(frame))
        if candidate is not None:
            current_candidates.append(candidate)

    if current_frames:
        chunks.append((current_frames, current_candidates))

    picked = []
    for chunk_frames, chunk_candidates in chunks:
        if not chunk_candidates:
            continue
        best = max(chunk_candidates, key=lambda item: item["score"])
        best = dict(best)
        best["wall_visit_candidate_count"] = len(chunk_candidates)
        best["wall_visit_frames"] = [int(item["hit_frame"]) for item in chunk_candidates]
        best["front_wall_chunk_start_frame"] = int(chunk_frames[0])
        best["front_wall_chunk_end_frame"] = int(chunk_frames[-1])
        best["front_wall_chunk_frame_count"] = len(chunk_frames)
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
    apply_stationary_filter=True,
    collapse_wall_area=True,
):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    threshold = runtime_threshold(artifact, threshold)
    min_gap = env_int("BOUNCE_GB_MIN_GAP_FRAMES", min_gap)
    context = infer_context(feature_columns)
    include_geometry = any(
        column in feature_columns
        for column in (
            "inside_tin_x_range",
            "nearest_wall_line_distance_px",
            "min_nearest_wall_line_distance_context",
            "calibration_wall_height_px",
            "calibration_wall_width_px",
            "calibration_roll_degrees",
            "calibration_perspective_shear",
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
        stationary_stats = None
        if apply_stationary_filter:
            stationary, stationary_stats = is_stationary_false_track(parsed_rows, frame)
            if stationary:
                continue
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
                "motion_span_px": (
                    float(stationary_stats["span_px"]) if stationary_stats is not None else None
                ),
                "motion_path_px": (
                    float(stationary_stats["path_px"]) if stationary_stats is not None else None
                ),
                "detector": "gradient_boosting",
                "event_type": "wall",
                "classification_source": "gradient_boosting_model",
            }
        )

    # Group only after stationary and location filters have removed bad
    # candidates. Otherwise a sidewall/static false positive can bridge two
    # real front-wall predictions into one wall visit.
    if collapse_wall_area:
        visit_gap = env_int("BOUNCE_GB_WALL_VISIT_GAP_FRAMES", DEFAULT_WALL_VISIT_GAP_FRAMES)
        candidates = collapse_front_wall_chunks(
            candidates,
            parsed_rows,
            wall_gate,
            wall_x_range,
            fallback_gap=visit_gap,
        )

    return pick_probability_peaks(candidates, min_gap)
