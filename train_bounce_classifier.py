"""Train a GradientBoostingClassifier for squash wall-hit frames.

This script builds a tabular dataset from labeled hit frames and Roboflow ball
tracking output. Each training row is one source frame, with features from a
context window around that frame.

Typical training run after you already have a ball coordinate CSV:

    python train_bounce_classifier.py \
        --labels wall_hits.csv \
        --ball-csv ball_coordinates.csv \
        --model-output bounce_gb_model.pkl

To pre-download/cache the Roboflow weights before any tracking:

    python train_bounce_classifier.py --prepare-model-only

To generate only the minimum missing ball-coordinate rows needed for training:

    python train_bounce_classifier.py --generate-ball-csv
"""

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from audio_events import detect_audio_candidates_from_file
from inference_engine import (
    DEFAULT_INFERENCE_WIDTH,
    get_tracking_model,
    infer_frame_predictions,
)
from judge_call import Point, judge_ball, load_calibration_lines
from tracking_common import CONFIDENCE_THRESHOLD, CSV_FIELDNAMES, ball_csv_row, select_ball_prediction

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = ROOT / "ModelTrainTest.mp4"
DEFAULT_LABELS_PATH = ROOT / "wall_hits.csv"
DEFAULT_CALIBRATION_PATH = ROOT / "calibration.json"
DEFAULT_BALL_CSV_PATH = ROOT / "ball_coordinates_gb.csv"
DEFAULT_EXISTING_BALL_CSV_PATH = ROOT / "ball_coordinates.csv"
DEFAULT_FEATURES_PATH = ROOT / "bounce_training_features.csv"
DEFAULT_MODEL_PATH = ROOT / "bounce_gb_model.pkl"

CONTEXT_FRAMES = 4
NEGATIVE_EXCLUSION_FRAMES = 8
POSITIVE_WINDOW_FRAMES = 1
AUDIO_PEAK_WINDOW_FRAMES = 5
DEFAULT_HIT_THRESHOLD = 0.25
MODEL_FEATURE_COLUMNS = [
    "velocity_change_px_s",
    "speed_after_px_s",
    "x_span_context",
    "vy_after_px_s",
    "area_span_context",
    "local_velocity_change_px_s",
    "min_nearest_wall_line_distance_context",
    "local_accel_mag_px_s2",
    "local_turn_degrees",
    "speed_before_px_s",
]


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_hit_labels(labels_path):
    labels = set()
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with labels_path.open(newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        if reader.fieldnames:
            frame_column = next(
                (
                    column
                    for column in ("hit_frame", "source_frame", "frame")
                    if column in reader.fieldnames
                ),
                None,
            )
            if frame_column is None:
                raise ValueError(f"{labels_path} must contain hit_frame, source_frame, or frame.")

            for row in reader:
                value = row.get(frame_column, "").strip()
                if value:
                    labels.add(int(value))
            return labels

    return labels


def expand_labels(labels, start_frame, end_frame, positive_window):
    expanded = set()
    for frame in labels:
        if frame < start_frame - positive_window or frame > end_frame + positive_window:
            continue
        for candidate in range(frame - positive_window, frame + positive_window + 1):
            if start_frame <= candidate <= end_frame:
                expanded.add(candidate)
    return expanded


def video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def raw_ball_csv_rows(csv_path):
    if not csv_path.exists():
        return {}

    rows = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("source_frame"):
                rows[int(row["source_frame"])] = row
    return rows


def write_raw_ball_csv_rows(csv_path, rows_by_frame):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for frame in sorted(rows_by_frame):
            writer.writerow(rows_by_frame[frame])


def required_context_frames(training_frames, start_frame, end_frame, context):
    frames = set()
    for frame in training_frames:
        for candidate in range(frame - context, frame + context + 1):
            if start_frame <= candidate <= end_frame:
                frames.add(candidate)
    return frames


def tracking_frame_plan(training_frames, start_frame, end_frame, context, track_all_frames):
    if track_all_frames:
        return set(range(start_frame, end_frame + 1))
    return required_context_frames(training_frames, start_frame, end_frame, context)


def track_selected_frames(video_path, frames, inference_width, confidence):
    if not frames:
        return {}

    print("Loading local Roboflow model. This also ensures weights are cached before inference.")
    model = get_tracking_model()
    print("Model loaded.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    metadata = video_metadata(video_path)
    fps = metadata["fps"]
    sorted_frames = sorted(frames)
    tracked_rows = {}
    last_requested_frame = None
    processed = 0
    progress_interval = max(25, min(500, max(1, len(sorted_frames) // 20)))
    print(
        f"Tracking {len(sorted_frames)} selected frame(s), "
        f"from {sorted_frames[0]} to {sorted_frames[-1]}...",
        flush=True,
    )

    try:
        for frame_idx in sorted_frames:
            if last_requested_frame is None or frame_idx != last_requested_frame + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ok, frame = cap.read()
            if not ok:
                print(f"Warning: could not read source frame {frame_idx}")
                last_requested_frame = None
                continue

            predictions = infer_frame_predictions(
                model,
                frame,
                confidence,
                inference_width,
            )
            ball_prediction = select_ball_prediction(predictions)
            tracked_rows[frame_idx] = ball_csv_row(frame_idx, fps, ball_prediction)

            processed += 1
            if processed == 1 or processed % progress_interval == 0 or processed == len(sorted_frames):
                percent = processed / len(sorted_frames) * 100
                print(
                    f"Tracked {processed}/{len(sorted_frames)} frame(s) "
                    f"({percent:.1f}%), source frame {frame_idx}",
                    flush=True,
                )
            last_requested_frame = frame_idx
    finally:
        cap.release()

    return tracked_rows


def ensure_ball_csv(
    video_path,
    csv_path,
    required_frames,
    inference_width,
    confidence,
    generate_ball_csv,
    force,
):
    existing_rows = raw_ball_csv_rows(csv_path)
    existing_frames = set(existing_rows)
    required_frames = set(required_frames)
    missing_frames = sorted(required_frames - existing_frames)
    print(
        f"Coordinate coverage: {len(existing_frames)} row(s) in {csv_path}; "
        f"{len(required_frames)} required; {len(missing_frames)} missing.",
        flush=True,
    )

    if not force and not missing_frames:
        print(f"Using existing ball CSV: {csv_path}")
        return

    if force:
        frames_to_track = sorted(required_frames)
    else:
        frames_to_track = missing_frames

    if not generate_ball_csv and not force:
        preview = ", ".join(str(frame) for frame in missing_frames[:12])
        suffix = "..." if len(missing_frames) > 12 else ""
        raise RuntimeError(
            f"{csv_path} is missing {len(missing_frames)} required coordinate row(s): "
            f"{preview}{suffix}. Reuse an existing full ball CSV, or run with "
            f"--generate-ball-csv to track only the missing training/context frames."
        )

    action = "Regenerating" if force else "Generating"
    print(f"{action} {len(frames_to_track)} ball-coordinate row(s) with local Roboflow inference.")
    print("This uses get_model(..., countinference=False); it avoids the remote serverless workflow.")
    tracked_rows = track_selected_frames(video_path, frames_to_track, inference_width, confidence)
    existing_rows.update(tracked_rows)
    write_raw_ball_csv_rows(csv_path, existing_rows)
    print(f"Saved {len(existing_rows)} total coordinate row(s) to {csv_path}")


def load_ball_rows(csv_path):
    rows = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame = int(row["source_frame"])
            detected = parse_bool(row.get("detected"))
            parsed = {
                "frame": frame,
                "timestamp": float(row.get("timestamp_seconds") or 0.0),
                "detected": detected,
                "confidence": float(row["confidence"]) if row.get("confidence") else 0.0,
                "x": float(row["x_center"]) if detected and row.get("x_center") else np.nan,
                "y": float(row["y_center"]) if detected and row.get("y_center") else np.nan,
                "width": float(row["width"]) if detected and row.get("width") else 0.0,
                "height": float(row["height"]) if detected and row.get("height") else 0.0,
            }
            rows[frame] = parsed
    return rows


def load_geometry(calibration_path):
    if not calibration_path or not calibration_path.exists():
        return None
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    top_line, bottom_line = load_calibration_lines(calibration)
    tin_left = min(bottom_line.left.x, bottom_line.right.x)
    tin_right = max(bottom_line.left.x, bottom_line.right.x)
    return {
        "top_line": top_line,
        "bottom_line": bottom_line,
        "tin_left": tin_left,
        "tin_right": tin_right,
    }


def load_audio_candidates_file(path, start_frame, end_frame):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_candidates = raw if isinstance(raw, list) else raw.get("candidates", [])
    candidates = []

    for item in raw_candidates:
        try:
            frame = int(item["frame"])
            window_start = int(item.get("window_start_frame", frame))
            window_end = int(item.get("window_end_frame", frame))
            score = float(item.get("score", 0.0))
            rms = float(item.get("rms", 0.0))
            time_seconds = float(item.get("time_seconds", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        if window_end < start_frame or window_start > end_frame:
            continue
        candidates.append(
            {
                "frame": min(end_frame, max(start_frame, frame)),
                "window_start_frame": max(start_frame, window_start),
                "window_end_frame": min(end_frame, window_end),
                "time_seconds": time_seconds,
                "score": score,
                "rms": rms,
            }
        )

    candidates.sort(key=lambda item: item["frame"])
    print(f"Loaded {len(candidates)} audio candidate(s) from {path}.", flush=True)
    return candidates


def load_audio_features(args, start_frame, end_frame, fps):
    if args.audio_candidates:
        return load_audio_candidates_file(args.audio_candidates, start_frame, end_frame)
    if args.audio_file:
        candidates = detect_audio_candidates_from_file(
            args.audio_file,
            start_frame,
            end_frame,
            fps,
            args.audio_max_peaks,
            log=lambda message: print(message, flush=True),
        )
        if args.audio_candidates_output:
            output_path = Path(args.audio_candidates_output)
            output_path.write_text(json.dumps({"candidates": candidates}, indent=2), encoding="utf-8")
            print(f"Saved audio candidates to {output_path}.", flush=True)
        return candidates
    return None


def audio_features_for_frame(frame, audio_candidates, audio_window_frames):
    defaults = {
        "audio_peak_in_window": 0.0,
        "audio_peak_score": 0.0,
        "audio_peak_rms": 0.0,
        "audio_peak_distance_frames": float(audio_window_frames + 1),
        "audio_peak_count_near": 0.0,
    }
    if not audio_candidates:
        return defaults

    nearby = [
        candidate
        for candidate in audio_candidates
        if abs(int(candidate["frame"]) - frame) <= audio_window_frames
        or int(candidate["window_start_frame"]) <= frame <= int(candidate["window_end_frame"])
    ]
    if not nearby:
        return defaults

    best = max(nearby, key=lambda item: float(item.get("score", 0.0)))
    distance = abs(int(best["frame"]) - frame)
    return {
        "audio_peak_in_window": 1.0,
        "audio_peak_score": float(best.get("score", 0.0)),
        "audio_peak_rms": float(best.get("rms", 0.0)),
        "audio_peak_distance_frames": float(distance),
        "audio_peak_count_near": float(len(nearby)),
    }


def finite_point(row):
    return row and row["detected"] and math.isfinite(row["x"]) and math.isfinite(row["y"])


def velocity_between(prev_row, next_row):
    if not finite_point(prev_row) or not finite_point(next_row):
        return np.nan, np.nan, np.nan
    dt = next_row["timestamp"] - prev_row["timestamp"]
    if dt <= 0:
        return np.nan, np.nan, np.nan
    vx = (next_row["x"] - prev_row["x"]) / dt
    vy = (next_row["y"] - prev_row["y"]) / dt
    return vx, vy, math.hypot(vx, vy)


def finite_or_zero(value):
    return 0.0 if not math.isfinite(value) else value


def angle_between(v1, v2):
    if any(not math.isfinite(value) for value in (*v1, *v2)):
        return np.nan
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_angle))


def line_distances(row, geometry):
    if geometry is None or not finite_point(row):
        return None
    x = row["x"]
    y = row["y"]
    try:
        out_distance = y - geometry["top_line"].y_at_x(x)
        tin_distance = geometry["bottom_line"].y_at_x(x) - y
    except ValueError:
        return None
    return out_distance, tin_distance


def geometry_features(row, geometry):
    features = {
        "inside_tin_x_range": 0.0,
        "distance_to_left_tin_px": 0.0,
        "distance_to_right_tin_px": 0.0,
        "normalized_wall_x": 0.0,
        "normalized_wall_y": 0.0,
        "distance_to_out_line_px": 0.0,
        "distance_to_tin_line_px": 0.0,
        "nearest_wall_line_distance_px": 0.0,
        "judge_margin_px": 0.0,
    }
    if geometry is None or not finite_point(row):
        return features

    x = row["x"]
    y = row["y"]
    tin_left = geometry["tin_left"]
    tin_right = geometry["tin_right"]
    top_line = geometry["top_line"]
    bottom_line = geometry["bottom_line"]

    features["inside_tin_x_range"] = 1.0 if tin_left <= x <= tin_right else 0.0
    features["distance_to_left_tin_px"] = x - tin_left
    features["distance_to_right_tin_px"] = tin_right - x
    if tin_right > tin_left:
        features["normalized_wall_x"] = (x - tin_left) / (tin_right - tin_left)

    try:
        top_y = top_line.y_at_x(x)
        bottom_y = bottom_line.y_at_x(x)
        features["distance_to_out_line_px"] = y - top_y
        features["distance_to_tin_line_px"] = bottom_y - y
        features["nearest_wall_line_distance_px"] = min(
            abs(features["distance_to_out_line_px"]),
            abs(features["distance_to_tin_line_px"]),
        )
        if bottom_y > top_y:
            features["normalized_wall_y"] = (y - top_y) / (bottom_y - top_y)
        _, _, _, _ = judge_ball(Point(x, y), top_line, bottom_line)
        features["judge_margin_px"] = min(y - top_y, bottom_y - y)
    except ValueError:
        pass

    return features


def build_features_for_frame(
    frame,
    rows,
    geometry,
    context,
    include_geometry,
    audio_candidates,
    audio_window_frames,
):
    center = rows.get(frame)
    features = {"frame": frame}

    detected_count = 0
    confidences = []
    xs = []
    ys = []
    widths = []
    heights = []
    areas = []
    out_distances = []
    tin_distances = []
    nearest_line_distances = []

    for offset in range(-context, context + 1):
        row = rows.get(frame + offset)
        prefix = f"t{offset:+d}"
        detected = 1.0 if finite_point(row) else 0.0
        features[f"{prefix}_detected"] = detected
        features[f"{prefix}_confidence"] = row["confidence"] if row else 0.0
        features[f"{prefix}_x"] = row["x"] if finite_point(row) else 0.0
        features[f"{prefix}_y"] = row["y"] if finite_point(row) else 0.0
        features[f"{prefix}_width"] = row["width"] if row else 0.0
        features[f"{prefix}_height"] = row["height"] if row else 0.0

        width = row["width"] if row else 0.0
        height = row["height"] if row else 0.0
        if width > 0:
            widths.append(width)
        if height > 0:
            heights.append(height)
        if width > 0 and height > 0:
            areas.append(width * height)

        if finite_point(row):
            detected_count += 1
            confidences.append(row["confidence"])
            xs.append(row["x"])
            ys.append(row["y"])
            distances = line_distances(row, geometry) if include_geometry else None
            if distances is not None:
                out_distance, tin_distance = distances
                out_distances.append(out_distance)
                tin_distances.append(tin_distance)
                nearest_line_distances.append(min(abs(out_distance), abs(tin_distance)))

    features["detected_count_context"] = detected_count
    features["missing_count_context"] = (context * 2 + 1) - detected_count
    features["mean_confidence_context"] = float(np.mean(confidences)) if confidences else 0.0
    features["max_confidence_context"] = max(confidences) if confidences else 0.0
    features["x_span_context"] = max(xs) - min(xs) if xs else 0.0
    features["y_span_context"] = max(ys) - min(ys) if ys else 0.0
    features["mean_width_context"] = float(np.mean(widths)) if widths else 0.0
    features["mean_height_context"] = float(np.mean(heights)) if heights else 0.0
    features["mean_area_context"] = float(np.mean(areas)) if areas else 0.0
    features["max_area_context"] = max(areas) if areas else 0.0
    features["area_span_context"] = max(areas) - min(areas) if areas else 0.0

    before = rows.get(frame - 1)
    after = rows.get(frame + 1)
    vx_before, vy_before, speed_before = velocity_between(rows.get(frame - context), center)
    vx_after, vy_after, speed_after = velocity_between(center, rows.get(frame + context))
    vx_local, vy_local, speed_local = velocity_between(before, after)
    vx_1f_before, vy_1f_before, speed_1f_before = velocity_between(before, center)
    vx_1f_after, vy_1f_after, speed_1f_after = velocity_between(center, after)
    features["vx_before_px_s"] = finite_or_zero(vx_before)
    features["vy_before_px_s"] = finite_or_zero(vy_before)
    features["speed_before_px_s"] = finite_or_zero(speed_before)
    features["vx_after_px_s"] = finite_or_zero(vx_after)
    features["vy_after_px_s"] = finite_or_zero(vy_after)
    features["speed_after_px_s"] = finite_or_zero(speed_after)
    features["local_vx_px_s"] = finite_or_zero(vx_local)
    features["local_vy_px_s"] = finite_or_zero(vy_local)
    features["local_speed_px_s"] = finite_or_zero(speed_local)
    features["vx_1f_before_px_s"] = finite_or_zero(vx_1f_before)
    features["vy_1f_before_px_s"] = finite_or_zero(vy_1f_before)
    features["speed_1f_before_px_s"] = finite_or_zero(speed_1f_before)
    features["vx_1f_after_px_s"] = finite_or_zero(vx_1f_after)
    features["vy_1f_after_px_s"] = finite_or_zero(vy_1f_after)
    features["speed_1f_after_px_s"] = finite_or_zero(speed_1f_after)

    dvx = vx_after - vx_before if math.isfinite(vx_after) and math.isfinite(vx_before) else np.nan
    dvy = vy_after - vy_before if math.isfinite(vy_after) and math.isfinite(vy_before) else np.nan
    local_dvx = (
        vx_1f_after - vx_1f_before
        if math.isfinite(vx_1f_after) and math.isfinite(vx_1f_before)
        else np.nan
    )
    local_dvy = (
        vy_1f_after - vy_1f_before
        if math.isfinite(vy_1f_after) and math.isfinite(vy_1f_before)
        else np.nan
    )
    features["velocity_change_px_s"] = finite_or_zero(math.hypot(dvx, dvy) if math.isfinite(dvx) else np.nan)
    features["local_velocity_change_px_s"] = finite_or_zero(
        math.hypot(local_dvx, local_dvy) if math.isfinite(local_dvx) else np.nan
    )
    accel_dt = (
        after["timestamp"] - before["timestamp"]
        if finite_point(before) and finite_point(after)
        else np.nan
    )
    accel_x = local_dvx / accel_dt if math.isfinite(local_dvx) and accel_dt > 0 else np.nan
    accel_y = local_dvy / accel_dt if math.isfinite(local_dvy) and accel_dt > 0 else np.nan
    features["local_accel_x_px_s2"] = finite_or_zero(accel_x)
    features["local_accel_y_px_s2"] = finite_or_zero(accel_y)
    features["local_accel_mag_px_s2"] = finite_or_zero(
        math.hypot(accel_x, accel_y) if math.isfinite(accel_x) else np.nan
    )
    turn = angle_between((vx_before, vy_before), (vx_after, vy_after))
    local_turn = angle_between((vx_1f_before, vy_1f_before), (vx_1f_after, vy_1f_after))
    features["turn_degrees"] = finite_or_zero(turn)
    features["local_turn_degrees"] = finite_or_zero(local_turn)

    if include_geometry:
        features["min_abs_out_line_distance_context"] = (
            min(abs(value) for value in out_distances) if out_distances else 0.0
        )
        features["min_abs_tin_line_distance_context"] = (
            min(abs(value) for value in tin_distances) if tin_distances else 0.0
        )
        features["min_nearest_wall_line_distance_context"] = (
            min(nearest_line_distances) if nearest_line_distances else 0.0
        )
        features["nearest_wall_line_distance_span_context"] = (
            max(nearest_line_distances) - min(nearest_line_distances)
            if nearest_line_distances
            else 0.0
        )
        center_distance = (
            min(abs(value) for value in line_distances(center, geometry))
            if line_distances(center, geometry) is not None
            else 0.0
        )
        prev_distance = (
            min(abs(value) for value in line_distances(before, geometry))
            if line_distances(before, geometry) is not None
            else center_distance
        )
        next_distance = (
            min(abs(value) for value in line_distances(after, geometry))
            if line_distances(after, geometry) is not None
            else center_distance
        )
        features["wall_line_distance_valley_depth_px"] = max(
            0.0,
            min(prev_distance - center_distance, next_distance - center_distance),
        )
        features["wall_line_distance_decreases_then_increases"] = (
            1.0 if prev_distance > center_distance and next_distance > center_distance else 0.0
        )

    if include_geometry:
        if finite_point(center):
            features.update(geometry_features(center, geometry))
        else:
            features.update(geometry_features(None, geometry))

    if audio_candidates is not None:
        features.update(audio_features_for_frame(frame, audio_candidates, audio_window_frames))

    return features


def sample_training_frames(labels, start_frame, end_frame, negative_ratio, exclusion_frames, random_seed):
    positives = sorted(frame for frame in labels if start_frame <= frame <= end_frame)
    positive_set = set(positives)
    excluded = set()
    for frame in positives:
        excluded.update(range(frame - exclusion_frames, frame + exclusion_frames + 1))

    all_negatives = [
        frame
        for frame in range(start_frame, end_frame + 1)
        if frame not in excluded and frame not in positive_set
    ]
    rng = random.Random(random_seed)
    rng.shuffle(all_negatives)
    negative_count = min(len(all_negatives), max(len(positives) * negative_ratio, len(positives)))
    negatives = sorted(all_negatives[:negative_count])
    return positives, negatives


def build_training_table(
    positives,
    negatives,
    rows,
    geometry,
    context,
    include_geometry,
    audio_candidates,
    audio_window_frames,
):
    records = []
    total = len(positives) + len(negatives)
    progress_interval = max(25, min(500, max(1, total // 20)))
    processed = 0
    print(
        f"Building feature rows for {total} frame(s): "
        f"{len(positives)} positive, {len(negatives)} negative...",
        flush=True,
    )

    for label, frames in ((1, positives), (0, negatives)):
        for frame in frames:
            record = build_features_for_frame(
                frame,
                rows,
                geometry,
                context,
                include_geometry,
                audio_candidates,
                audio_window_frames,
            )
            record["is_wall_hit"] = label
            records.append(record)
            processed += 1
            if processed == 1 or processed % progress_interval == 0 or processed == total:
                percent = processed / total * 100
                print(
                    f"Built {processed}/{total} feature row(s) ({percent:.1f}%), "
                    f"current frame {frame}",
                    flush=True,
                )

    if not records:
        raise RuntimeError("No training rows were created.")

    return pd.DataFrame(records).sort_values("frame").reset_index(drop=True)


def metrics_from_confusion(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 4.0
    f2 = (
        (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tn + fp + fn + tp)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "accuracy": float(accuracy),
    }


def train_model(
    features,
    model_output,
    random_seed,
    hit_threshold,
    class_balance,
    gb_params=None,
    save_model=True,
):
    y = features["is_wall_hit"].astype(int)
    missing_features = [column for column in MODEL_FEATURE_COLUMNS if column not in features.columns]
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise RuntimeError(f"Training table is missing model feature column(s): {missing_text}")
    x = features[MODEL_FEATURE_COLUMNS]

    if y.nunique() < 2:
        raise RuntimeError("Need both positive and negative examples to train.")

    print(
        f"Training GradientBoostingClassifier on {len(features)} row(s) "
        f"with {len(x.columns)} feature column(s)...",
        flush=True,
    )
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_seed,
        stratify=stratify,
    )
    print(
        f"Train/test split: {len(x_train)} train row(s), {len(x_test)} test row(s).",
        flush=True,
    )

    gb_params = dict(gb_params or {})
    model = GradientBoostingClassifier(random_state=random_seed, **gb_params)
    if gb_params:
        print(f"GradientBoosting hyperparameters: {gb_params}", flush=True)
    print("Fitting classifier...", flush=True)
    sample_weight = None
    if class_balance:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        negative_weight = float(sample_weight[y_train.to_numpy() == 0][0])
        positive_weight = float(sample_weight[y_train.to_numpy() == 1][0])
        print(
            "Using balanced sample weights: "
            f"negative={negative_weight:.3f}, positive={positive_weight:.3f}",
            flush=True,
        )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    print("Classifier fit complete. Evaluating...", flush=True)

    positive_class_index = list(model.classes_).index(1)
    hit_probabilities = model.predict_proba(x_test)[:, positive_class_index]
    predictions = (hit_probabilities >= hit_threshold).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = metrics_from_confusion(cm)
    print(f"Using hit probability threshold: {hit_threshold:.3f}", flush=True)
    print("Confusion matrix:")
    print(cm)
    print()
    print(classification_report(y_test, predictions, digits=3, zero_division=0))

    artifact = {
        "model": model,
        "feature_columns": list(x.columns),
        "positive_label": "is_wall_hit",
        "hit_threshold": hit_threshold,
        "class_balance": bool(class_balance),
        "gb_params": gb_params,
    }
    if save_model:
        joblib.dump(artifact, model_output)
        print(f"Saved model artifact to {model_output}")

    importances = sorted(
        zip(x.columns, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nTop feature importances:")
    for name, value in importances[:20]:
        print(f"  {name}: {value:.4f}")

    return {
        "artifact": artifact,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_importances": importances,
    }


def parse_int_grid(value):
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one integer")
    return values


def parse_float_grid(value):
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return values


def fbeta_score(precision, recall, beta):
    if precision + recall <= 0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def metric_sort_key(result, min_precision=0.30, beta=2.0):
    metrics = result["result"]["metrics"]
    selection_score = fbeta_score(metrics["precision"], metrics["recall"], beta)
    precision_floor_met = metrics["precision"] >= min_precision
    return (
        1 if precision_floor_met else 0,
        selection_score,
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        metrics["accuracy"],
    )


def print_best_result(best, min_precision=0.30, beta=2.0):
    metrics = best["result"]["metrics"]
    params = best["params"]
    selection_score = fbeta_score(metrics["precision"], metrics["recall"], beta)
    print("\nBest hyperparameters:")
    print(f"  context_frames: {params['context']}")
    print(f"  positive_window_frames: {params['positive_window']}")
    print(f"  negative_exclusion_frames: {params['negative_exclusion']}")
    print(f"  gb_n_estimators: {params['gb_n_estimators']}")
    print(f"  gb_learning_rate: {params['gb_learning_rate']}")
    print(f"  gb_max_depth: {params['gb_max_depth']}")
    print(f"  gb_min_samples_leaf: {params['gb_min_samples_leaf']}")
    print(f"  gb_subsample: {params['gb_subsample']}")
    print("\nBest metrics:")
    print(f"  precision: {metrics['precision']:.3f}")
    print(f"  recall: {metrics['recall']:.3f}")
    print(f"  f1: {metrics['f1']:.3f}")
    print(f"  f2: {metrics['f2']:.3f}")
    print(f"  selection_f_beta(beta={beta:g}): {selection_score:.3f}")
    print(f"  precision_floor_met: {metrics['precision'] >= min_precision}")
    print(f"  accuracy: {metrics['accuracy']:.3f}")
    print(
        "  confusion_matrix: "
        f"[[{metrics['tn']} {metrics['fp']}], [{metrics['fn']} {metrics['tp']}]]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GradientBoostingClassifier from wall-hit labels and ball tracking features."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--ball-csv", type=Path, default=DEFAULT_BALL_CSV_PATH)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--context", type=int, default=CONTEXT_FRAMES)
    parser.add_argument(
        "--hyperparameter-matrix",
        action="store_true",
        help=(
            "Train every combination from the comma-separated grid flags, "
            "then save the best model by precision floor plus recall-weighted F-beta."
        ),
    )
    parser.add_argument("--context-grid", type=parse_int_grid, default=None)
    parser.add_argument("--positive-window-grid", type=parse_int_grid, default=None)
    parser.add_argument("--negative-exclusion-grid", type=parse_int_grid, default=None)
    parser.add_argument("--gb-n-estimators", type=int, default=100)
    parser.add_argument("--gb-n-estimators-grid", type=parse_int_grid, default=None)
    parser.add_argument("--gb-learning-rate", type=float, default=0.1)
    parser.add_argument("--gb-learning-rate-grid", type=parse_float_grid, default=None)
    parser.add_argument("--gb-max-depth", type=int, default=3)
    parser.add_argument("--gb-max-depth-grid", type=parse_int_grid, default=None)
    parser.add_argument("--gb-min-samples-leaf", type=int, default=1)
    parser.add_argument("--gb-min-samples-leaf-grid", type=parse_int_grid, default=None)
    parser.add_argument("--gb-subsample", type=float, default=1.0)
    parser.add_argument("--gb-subsample-grid", type=parse_float_grid, default=None)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument(
        "--negative-exclusion",
        type=int,
        default=NEGATIVE_EXCLUSION_FRAMES,
        help="Do not sample negative examples within +/- this many frames of a positive label.",
    )
    parser.add_argument(
        "--positive-window",
        type=int,
        default=POSITIVE_WINDOW_FRAMES,
        help="Treat frames within +/- this many frames of each labeled hit as positive.",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=DEFAULT_HIT_THRESHOLD,
        help="Classify a frame as a hit when predict_proba(hit) is at least this value.",
    )
    parser.add_argument(
        "--selection-beta",
        type=float,
        default=2.0,
        help=(
            "F-beta beta used to rank hyperparameter-matrix results. "
            "Values above 1 prioritize recall over precision."
        ),
    )
    parser.add_argument(
        "--min-selection-precision",
        type=float,
        default=0.30,
        help=(
            "Prefer matrix results with at least this precision. "
            "If none meet it, fall back to the best F-beta result overall."
        ),
    )
    parser.add_argument(
        "--no-class-balance",
        action="store_true",
        help="Disable balanced sample weights during GradientBoostingClassifier training.",
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--inference-width", type=int, default=DEFAULT_INFERENCE_WIDTH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--include-geometry",
        action="store_true",
        help="Include calibration/tin-line geometry features. Off by default for now.",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Optional audio recording to convert into impact-peak features.",
    )
    parser.add_argument(
        "--audio-candidates",
        type=Path,
        default=None,
        help="Optional JSON of precomputed audio candidates from the app.",
    )
    parser.add_argument(
        "--audio-candidates-output",
        type=Path,
        default=None,
        help="Where to save candidates detected from --audio-file.",
    )
    parser.add_argument(
        "--audio-window-frames",
        type=int,
        default=AUDIO_PEAK_WINDOW_FRAMES,
        help="Frame radius used when matching audio peaks to training rows.",
    )
    parser.add_argument(
        "--audio-max-peaks",
        type=int,
        default=300,
        help="Maximum number of audio peaks to keep from --audio-file.",
    )
    parser.add_argument(
        "--generate-ball-csv",
        action="store_true",
        help=(
            "Run local Roboflow inference only for missing training/context frames. "
            "Without this flag, the script refuses to call Roboflow if coordinates are missing."
        ),
    )
    parser.add_argument(
        "--force-track",
        action="store_true",
        help="Regenerate the required coordinate rows even if they already exist.",
    )
    parser.add_argument(
        "--track-all-frames",
        action="store_true",
        help="Track every frame in the selected range instead of only sampled training/context frames.",
    )
    parser.add_argument(
        "--prepare-model-only",
        action="store_true",
        help="Load/cache the Roboflow model and exit before tracking or training.",
    )
    return parser.parse_args()


def main():
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    args = parse_args()
    print("Starting bounce classifier training pipeline.", flush=True)
    print(f"Video: {args.video}", flush=True)
    print(f"Labels: {args.labels}", flush=True)
    print(f"Ball coordinates: {args.ball_csv}", flush=True)

    if args.prepare_model_only:
        print("Preparing local Roboflow model cache...")
        print("This loads the local model with countinference=False and exits before frame inference.")
        get_tracking_model()
        print("Model is available locally.")
        return

    metadata = video_metadata(args.video)
    frame_count = metadata["frame_count"]
    start_frame = max(0, args.start_frame)
    end_frame = args.end_frame if args.end_frame is not None else frame_count - 1
    end_frame = min(end_frame, frame_count - 1)
    print(
        f"Video metadata: {frame_count} frame(s), {metadata['fps']:.3f} fps. "
        f"Selected range: {start_frame}-{end_frame}.",
        flush=True,
    )

    labels = load_hit_labels(args.labels)
    if not labels:
        raise RuntimeError(f"No hit labels found in {args.labels}")
    labels_in_range = sorted(frame for frame in labels if start_frame <= frame <= end_frame)
    if not labels_in_range:
        raise RuntimeError(
            f"No labels from {args.labels} fall inside frame range {start_frame}-{end_frame}."
        )
    print(
        f"Loaded {len(labels)} total labeled hit frame(s); "
        f"{len(labels_in_range)} inside selected range.",
        flush=True,
    )

    if (
        args.ball_csv == DEFAULT_BALL_CSV_PATH
        and not args.ball_csv.exists()
        and DEFAULT_EXISTING_BALL_CSV_PATH.exists()
    ):
        print(
            "Using existing app coordinate CSV instead of creating a new one: "
            f"{DEFAULT_EXISTING_BALL_CSV_PATH}"
        )
        args.ball_csv = DEFAULT_EXISTING_BALL_CSV_PATH

    if args.hyperparameter_matrix:
        context_values = args.context_grid or [args.context]
        positive_window_values = args.positive_window_grid or [max(0, args.positive_window)]
        negative_exclusion_values = args.negative_exclusion_grid or [max(0, args.negative_exclusion)]
        gb_n_estimators_values = args.gb_n_estimators_grid or [args.gb_n_estimators]
        gb_learning_rate_values = args.gb_learning_rate_grid or [args.gb_learning_rate]
        gb_max_depth_values = args.gb_max_depth_grid or [args.gb_max_depth]
        gb_min_samples_leaf_values = args.gb_min_samples_leaf_grid or [args.gb_min_samples_leaf]
        gb_subsample_values = args.gb_subsample_grid or [args.gb_subsample]
    else:
        context_values = [args.context]
        positive_window_values = [max(0, args.positive_window)]
        negative_exclusion_values = [max(0, args.negative_exclusion)]
        gb_n_estimators_values = [args.gb_n_estimators]
        gb_learning_rate_values = [args.gb_learning_rate]
        gb_max_depth_values = [args.gb_max_depth]
        gb_min_samples_leaf_values = [args.gb_min_samples_leaf]
        gb_subsample_values = [args.gb_subsample]

    config_plans = []
    required_frames = set()
    data_window_plans = {}
    for context, positive_window, negative_exclusion in itertools.product(
        context_values,
        positive_window_values,
        negative_exclusion_values,
    ):
        data_key = (context, positive_window, negative_exclusion)
        positive_labels = expand_labels(labels, start_frame, end_frame, max(0, positive_window))
        positives, negatives = sample_training_frames(
            positive_labels,
            start_frame,
            end_frame,
            args.negative_ratio,
            max(0, negative_exclusion),
            args.random_seed,
        )
        training_frames = sorted(set(positives) | set(negatives))
        plan_required_frames = tracking_frame_plan(
            training_frames,
            start_frame,
            end_frame,
            context,
            args.track_all_frames,
        )
        required_frames.update(plan_required_frames)
        data_window_plans[data_key] = {
            "positive_labels": positive_labels,
            "positives": positives,
            "negatives": negatives,
            "training_frames": training_frames,
            "required_frames": plan_required_frames,
        }

    for (
        context,
        positive_window,
        negative_exclusion,
        gb_n_estimators,
        gb_learning_rate,
        gb_max_depth,
        gb_min_samples_leaf,
        gb_subsample,
    ) in itertools.product(
        context_values,
        positive_window_values,
        negative_exclusion_values,
        gb_n_estimators_values,
        gb_learning_rate_values,
        gb_max_depth_values,
        gb_min_samples_leaf_values,
        gb_subsample_values,
    ):
        data_key = (context, positive_window, negative_exclusion)
        data_plan = data_window_plans[data_key]
        config_plans.append(
            {
                "params": {
                    "context": context,
                    "positive_window": positive_window,
                    "negative_exclusion": negative_exclusion,
                    "gb_n_estimators": gb_n_estimators,
                    "gb_learning_rate": gb_learning_rate,
                    "gb_max_depth": gb_max_depth,
                    "gb_min_samples_leaf": gb_min_samples_leaf,
                    "gb_subsample": gb_subsample,
                },
                "positives": data_plan["positives"],
                "negatives": data_plan["negatives"],
                "positive_labels": data_plan["positive_labels"],
                "training_frames": data_plan["training_frames"],
                "required_frames": data_plan["required_frames"],
            }
        )

    if args.hyperparameter_matrix:
        print(
            "Hyperparameter matrix: "
            f"{len(context_values)} context value(s) x "
            f"{len(positive_window_values)} positive-window value(s) x "
            f"{len(negative_exclusion_values)} negative-exclusion value(s) x "
            f"{len(gb_n_estimators_values)} estimator value(s) x "
            f"{len(gb_learning_rate_values)} learning-rate value(s) x "
            f"{len(gb_max_depth_values)} max-depth value(s) x "
            f"{len(gb_min_samples_leaf_values)} min-leaf value(s) x "
            f"{len(gb_subsample_values)} subsample value(s) = "
            f"{len(config_plans)} run(s).",
            flush=True,
        )

    first_plan = config_plans[0]
    print(
        f"First config positive label window: +/-{first_plan['params']['positive_window']} frame(s), "
        f"creating {len(first_plan['positive_labels'])} positive training frame(s).",
        flush=True,
    )
    print(
        f"First config sampled frames: {len(first_plan['positives'])} positive and "
        f"{len(first_plan['negatives'])} negative.",
        flush=True,
    )
    if args.track_all_frames:
        print(
            f"Training will use {len(first_plan['training_frames'])} labeled/sampled frames "
            f"for the first config; "
            f"tracking plan covers all {len(required_frames)} frame(s)."
        )
    else:
        print(
            f"Training will use {len(first_plan['training_frames'])} labeled/sampled frames "
            f"for the first config; "
            f"tracking plan covers {len(required_frames)} required context frame(s)."
        )

    ensure_ball_csv(
        args.video,
        args.ball_csv,
        required_frames,
        args.inference_width,
        args.confidence,
        args.generate_ball_csv,
        args.force_track,
    )

    rows = load_ball_rows(args.ball_csv)
    detected_rows = sum(1 for row in rows.values() if finite_point(row))
    print(
        f"Loaded {len(rows)} coordinate row(s) from {args.ball_csv}; "
        f"{detected_rows} have detected ball positions.",
        flush=True,
    )
    missing_label_rows = [frame for frame in labels_in_range if frame not in rows]
    if missing_label_rows:
        preview = ", ".join(str(frame) for frame in missing_label_rows[:8])
        print(
            f"Warning: {len(missing_label_rows)} labeled frame(s) are missing from "
            f"{args.ball_csv}: {preview}. Use --force-track to regenerate full coverage."
        )
    geometry = load_geometry(args.calibration) if args.include_geometry and args.calibration else None
    if args.include_geometry:
        print("Including calibration geometry features.")
    else:
        print("Geometry features are disabled; training uses tracking and motion features only.")
    audio_candidates = load_audio_features(args, start_frame, end_frame, metadata["fps"])
    if audio_candidates is None:
        print("Audio features are disabled.", flush=True)
    else:
        print(
            f"Audio features are enabled using {len(audio_candidates)} candidate peak(s); "
            f"matching radius +/-{args.audio_window_frames} frame(s).",
            flush=True,
        )

    best = None
    all_results = []
    feature_cache = {}
    for index, plan in enumerate(config_plans, start=1):
        params = plan["params"]
        if args.hyperparameter_matrix:
            print(
                "\n"
                f"Matrix run {index}/{len(config_plans)}: "
                f"context={params['context']}, "
                f"positive_window={params['positive_window']}, "
                f"negative_exclusion={params['negative_exclusion']}, "
                f"gb_n_estimators={params['gb_n_estimators']}, "
                f"gb_learning_rate={params['gb_learning_rate']}, "
                f"gb_max_depth={params['gb_max_depth']}, "
                f"gb_min_samples_leaf={params['gb_min_samples_leaf']}, "
                f"gb_subsample={params['gb_subsample']}",
                flush=True,
            )
        feature_key = (params["context"], params["positive_window"], params["negative_exclusion"])
        features = feature_cache.get(feature_key)
        if features is None:
            features = build_training_table(
                plan["positives"],
                plan["negatives"],
                rows,
                geometry,
                params["context"],
                args.include_geometry,
                audio_candidates,
                max(0, args.audio_window_frames),
            )
            feature_cache[feature_key] = features
        elif args.hyperparameter_matrix:
            print(
                "Reusing feature table for "
                f"context={params['context']}, "
                f"positive_window={params['positive_window']}, "
                f"negative_exclusion={params['negative_exclusion']}.",
                flush=True,
            )

        result = train_model(
            features,
            args.model_output,
            args.random_seed,
            args.hit_threshold,
            class_balance=not args.no_class_balance,
            gb_params={
                "n_estimators": params["gb_n_estimators"],
                "learning_rate": params["gb_learning_rate"],
                "max_depth": params["gb_max_depth"],
                "min_samples_leaf": params["gb_min_samples_leaf"],
                "subsample": params["gb_subsample"],
            },
            save_model=not args.hyperparameter_matrix,
        )
        entry = {"params": params, "features": features, "result": result}
        all_results.append(entry)
        if best is None or metric_sort_key(
            entry,
            min_precision=args.min_selection_precision,
            beta=args.selection_beta,
        ) > metric_sort_key(
            best,
            min_precision=args.min_selection_precision,
            beta=args.selection_beta,
        ):
            best = entry

    if args.hyperparameter_matrix:
        print(
            "\nHyperparameter matrix summary "
            f"(selection: require precision >= {args.min_selection_precision:.3f} when possible, "
            f"then maximize F-beta with beta={args.selection_beta:g}):"
        )
        for entry in all_results:
            params = entry["params"]
            metrics = entry["result"]["metrics"]
            selection_score = fbeta_score(
                metrics["precision"],
                metrics["recall"],
                args.selection_beta,
            )
            print(
                f"  context={params['context']}, "
                f"positive_window={params['positive_window']}, "
                f"negative_exclusion={params['negative_exclusion']} -> "
                f"gb=(n={params['gb_n_estimators']}, "
                f"lr={params['gb_learning_rate']}, "
                f"depth={params['gb_max_depth']}, "
                f"leaf={params['gb_min_samples_leaf']}, "
                f"subsample={params['gb_subsample']}) -> "
                f"precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, "
                f"f1={metrics['f1']:.3f}, "
                f"f2={metrics['f2']:.3f}, "
                f"selection={selection_score:.3f}, "
                f"precision_ok={metrics['precision'] >= args.min_selection_precision}, "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"fp={metrics['fp']}, fn={metrics['fn']}"
            )
        print_best_result(
            best,
            min_precision=args.min_selection_precision,
            beta=args.selection_beta,
        )
        best["result"]["artifact"]["hyperparameters"] = dict(best["params"])
        best["result"]["artifact"]["selection"] = {
            "metric": "fbeta_with_precision_floor",
            "beta": float(args.selection_beta),
            "min_precision": float(args.min_selection_precision),
            "score": float(
                fbeta_score(
                    best["result"]["metrics"]["precision"],
                    best["result"]["metrics"]["recall"],
                    args.selection_beta,
                )
            ),
            "precision_floor_met": bool(
                best["result"]["metrics"]["precision"] >= args.min_selection_precision
            ),
        }
        joblib.dump(best["result"]["artifact"], args.model_output)
        print(f"\nSaved best model artifact to {args.model_output}")
        features = best["features"]

    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_output, index=False)
    print(
        f"Saved best/last training rows: {len(features)} row(s) "
        f"({int(features['is_wall_hit'].sum())} positive) to {args.features_output}",
        flush=True,
    )
    print("Training pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
