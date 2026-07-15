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
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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

CONTEXT_FRAMES = 10
NEGATIVE_EXCLUSION_FRAMES = 8
POSITIVE_WINDOW_FRAMES = 1
AUDIO_PEAK_WINDOW_FRAMES = 5
DEFAULT_HIT_THRESHOLD = 0.25


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


def audio_to_mono_float(audio_path):
    audio_path = Path(audio_path)
    try:
        import av
    except ImportError:
        av = None

    if av is not None:
        try:
            container = av.open(str(audio_path))
            stream = next(stream for stream in container.streams if stream.type == "audio")
            chunks = []
            sample_rate = int(stream.rate or 0)
            for frame in container.decode(stream):
                frame_samples = frame.to_ndarray()
                if frame_samples.ndim == 2:
                    frame_samples = frame_samples.mean(axis=0)
                chunks.append(frame_samples.astype(np.float32))
                if not sample_rate:
                    sample_rate = int(frame.sample_rate)
            container.close()
            if chunks and sample_rate:
                return sample_rate, np.concatenate(chunks)
        except Exception as exc:
            print(f"PyAV audio decode failed ({exc}); trying WAV/afconvert fallback.", flush=True)

    cleanup_path = None
    source_path = audio_path

    if audio_path.suffix.lower() != ".wav":
        afconvert = shutil.which("afconvert")
        if afconvert is None:
            raise RuntimeError(
                f"{audio_path} is not a WAV file, and afconvert is not available. "
                "Convert the audio to WAV first or pass --audio-candidates."
            )
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cleanup_path = Path(temp_file.name)
        temp_file.close()
        print(f"Converting {audio_path} to temporary WAV for audio features...", flush=True)
        subprocess.run(
            [afconvert, "-f", "WAVE", "-d", "LEI16", str(audio_path), str(cleanup_path)],
            check=True,
        )
        source_path = cleanup_path

    try:
        sample_rate, samples = wavfile.read(source_path)
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    if np.issubdtype(samples.dtype, np.integer):
        scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / float(scale)
    else:
        samples = samples.astype(np.float32)

    return sample_rate, samples


def percentile(sorted_values, p):
    if len(sorted_values) == 0:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(math.floor((len(sorted_values) - 1) * p))))
    return float(sorted_values[index])


def detect_audio_candidates_from_file(audio_path, start_frame, end_frame, fps, max_peaks):
    print(f"Analyzing audio file for impact peaks: {audio_path}", flush=True)
    sample_rate, samples = audio_to_mono_float(audio_path)
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    window_size = max(256, int(round(sample_rate * 0.012)))
    hop = max(128, int(round(sample_rate * 0.005)))
    start_sample = max(0, int(math.floor(max(0.0, start_seconds - 0.5) * sample_rate)))
    end_sample = min(len(samples) - window_size, int(math.ceil((end_seconds + 0.5) * sample_rate)))

    if end_sample <= start_sample or len(samples) < window_size:
        print("Audio is too short for peak detection in the selected frame range.", flush=True)
        return []

    starts = np.arange(start_sample, end_sample + 1, hop, dtype=np.int64)
    squared = samples.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    rms = np.sqrt((cumulative[starts + window_size] - cumulative[starts]) / window_size)
    db = 20 * np.log10(rms + 1e-7)
    times = (starts + window_size / 2) / sample_rate

    selected_range = (times >= start_seconds) & (times <= end_seconds)
    if selected_range.sum() < 3:
        print("Audio has too few analysis windows in the selected frame range.", flush=True)
        return []

    sorted_db = np.sort(db)
    median = percentile(sorted_db, 0.5)
    p90 = percentile(sorted_db, 0.9)
    threshold = max(median + 10.0, p90 + 2.0)
    local_peaks = []
    for index in range(1, len(db) - 1):
        if not selected_range[index]:
            continue
        if db[index] < threshold:
            continue
        if db[index] < db[index - 1] or db[index] < db[index + 1]:
            continue
        local_peaks.append(
            {
                "time_seconds": float(times[index]),
                "score": float(db[index] - median),
                "rms": float(rms[index]),
            }
        )

    local_peaks.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    min_separation_seconds = 0.12
    for peak in local_peaks:
        if any(abs(peak["time_seconds"] - item["time_seconds"]) < min_separation_seconds for item in selected):
            continue
        selected.append(peak)
        if len(selected) >= max_peaks:
            break

    half_window_seconds = 0.08
    candidates = []
    for peak in sorted(selected, key=lambda item: item["time_seconds"]):
        frame = int(round(peak["time_seconds"] * fps))
        candidates.append(
            {
                "frame": frame,
                "time_seconds": peak["time_seconds"],
                "window_start_frame": int(round((peak["time_seconds"] - half_window_seconds) * fps)),
                "window_end_frame": int(round((peak["time_seconds"] + half_window_seconds) * fps)),
                "score": peak["score"],
                "rms": peak["rms"],
            }
        )

    print(
        f"Audio peak detection found {len(candidates)} candidate(s) "
        f"(threshold {threshold:.2f} dB, median {median:.2f} dB).",
        flush=True,
    )
    return candidates


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


def angle_between(v1, v2):
    if any(not math.isfinite(value) for value in (*v1, *v2)):
        return np.nan
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_angle))


def geometry_features(row, geometry):
    features = {
        "inside_tin_x_range": 0.0,
        "distance_to_left_tin_px": 0.0,
        "distance_to_right_tin_px": 0.0,
        "normalized_wall_x": 0.0,
        "normalized_wall_y": 0.0,
        "distance_to_out_line_px": 0.0,
        "distance_to_tin_line_px": 0.0,
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

        if finite_point(row):
            detected_count += 1
            confidences.append(row["confidence"])
            xs.append(row["x"])
            ys.append(row["y"])

    features["detected_count_context"] = detected_count
    features["missing_count_context"] = (context * 2 + 1) - detected_count
    features["mean_confidence_context"] = float(np.mean(confidences)) if confidences else 0.0
    features["max_confidence_context"] = max(confidences) if confidences else 0.0
    features["x_span_context"] = max(xs) - min(xs) if xs else 0.0
    features["y_span_context"] = max(ys) - min(ys) if ys else 0.0

    before = rows.get(frame - 1)
    after = rows.get(frame + 1)
    vx_before, vy_before, speed_before = velocity_between(rows.get(frame - context), center)
    vx_after, vy_after, speed_after = velocity_between(center, rows.get(frame + context))
    vx_local, vy_local, speed_local = velocity_between(before, after)
    features["vx_before_px_s"] = 0.0 if not math.isfinite(vx_before) else vx_before
    features["vy_before_px_s"] = 0.0 if not math.isfinite(vy_before) else vy_before
    features["speed_before_px_s"] = 0.0 if not math.isfinite(speed_before) else speed_before
    features["vx_after_px_s"] = 0.0 if not math.isfinite(vx_after) else vx_after
    features["vy_after_px_s"] = 0.0 if not math.isfinite(vy_after) else vy_after
    features["speed_after_px_s"] = 0.0 if not math.isfinite(speed_after) else speed_after
    features["local_speed_px_s"] = 0.0 if not math.isfinite(speed_local) else speed_local

    dvx = vx_after - vx_before if math.isfinite(vx_after) and math.isfinite(vx_before) else np.nan
    dvy = vy_after - vy_before if math.isfinite(vy_after) and math.isfinite(vy_before) else np.nan
    features["velocity_change_px_s"] = 0.0 if not math.isfinite(dvx) else math.hypot(dvx, dvy)
    turn = angle_between((vx_before, vy_before), (vx_after, vy_after))
    features["turn_degrees"] = 0.0 if not math.isfinite(turn) else turn

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


def train_model(features, model_output, random_seed, hit_threshold):
    y = features["is_wall_hit"].astype(int)
    x = features.drop(columns=["is_wall_hit", "frame"])

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

    model = GradientBoostingClassifier(random_state=random_seed)
    print("Fitting classifier...", flush=True)
    model.fit(x_train, y_train)
    print("Classifier fit complete. Evaluating...", flush=True)

    positive_class_index = list(model.classes_).index(1)
    hit_probabilities = model.predict_proba(x_test)[:, positive_class_index]
    predictions = (hit_probabilities >= hit_threshold).astype(int)
    print(f"Using hit probability threshold: {hit_threshold:.3f}", flush=True)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print(classification_report(y_test, predictions, digits=3, zero_division=0))

    artifact = {
        "model": model,
        "feature_columns": list(x.columns),
        "positive_label": "is_wall_hit",
        "hit_threshold": hit_threshold,
    }
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
    parser.add_argument("--negative-ratio", type=int, default=6)
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
    positive_labels = expand_labels(labels, start_frame, end_frame, max(0, args.positive_window))
    print(
        f"Loaded {len(labels)} total labeled hit frame(s); "
        f"{len(labels_in_range)} inside selected range.",
        flush=True,
    )
    print(
        f"Positive label window: +/-{max(0, args.positive_window)} frame(s), "
        f"creating {len(positive_labels)} positive training frame(s).",
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

    positives, negatives = sample_training_frames(
        positive_labels,
        start_frame,
        end_frame,
        args.negative_ratio,
        NEGATIVE_EXCLUSION_FRAMES,
        args.random_seed,
    )
    print(
        f"Sampled training frames: {len(positives)} positive and "
        f"{len(negatives)} negative.",
        flush=True,
    )
    training_frames = sorted(set(positives) | set(negatives))
    required_frames = tracking_frame_plan(
        training_frames,
        start_frame,
        end_frame,
        args.context,
        args.track_all_frames,
    )
    if args.track_all_frames:
        print(
            f"Training will use {len(training_frames)} labeled/sampled frames; "
            f"tracking plan covers all {len(required_frames)} frame(s)."
        )
    else:
        print(
            f"Training will use {len(training_frames)} labeled/sampled frames; "
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
    features = build_training_table(
        positives,
        negatives,
        rows,
        geometry,
        args.context,
        args.include_geometry,
        audio_candidates,
        max(0, args.audio_window_frames),
    )
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_output, index=False)
    print(
        f"Saved {len(features)} training row(s) "
        f"({int(features['is_wall_hit'].sum())} positive) to {args.features_output}",
        flush=True,
    )

    train_model(features, args.model_output, args.random_seed, args.hit_threshold)
    print("Training pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
