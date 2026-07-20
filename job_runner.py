"""Tracking job pipeline: coarse pass -> hit-candidate refine pass -> judging.

A job is fully described by ui_runs/<run_id>/job.json, so the same code can
run on a thread inside Flask today or as a standalone worker process later:

    python job_runner.py ui_runs/<run_id>
"""

import csv
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import cv2

import court_model
from audio_events import extract_audio_candidates, extract_repeating_audio_windows
from bounce_gb_model_detector import DEFAULT_MODEL_PATH as BOUNCE_GB_MODEL_PATH
from bounce_gb_model_detector import detect_hits_with_gb_model
from event_engine import detect_events_fused
from classify_events import classify_events
from detect_wall_hits import MAX_GAP_FRAMES, detect_hits_from_rows
from inference_engine import get_tracking_model, infer_frame_predictions
from judge_call import Point, judge_ball, judge_margin_px, load_calibration_lines
from judge_call import wall_diagram_coordinates
from tracking_common import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    ball_csv_row,
    select_ball_prediction,
)


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "ui_runs"
UPLOADS_DIR = RUNS_DIR / "uploads"

JOBS = {}
JOBS_LOCK = threading.Lock()
# One tracking job at a time; the model serializes its own session.run, so a
# per-frame lock would only block preprocessing for no benefit.
TRACKING_JOB_SEMAPHORE = threading.Semaphore(1)

DECODE_QUEUE_SIZE = 8
PROGRESS_UPDATE_FRAMES = 10
PROGRESS_UPDATE_SECONDS = 0.5
COARSE_INFERENCE_WIDTH = 640
REFINE_WINDOW_MIN_RADIUS = 12
# Which event engine labels detected events. "gb_model" = trained
# GradientBoostingClassifier; "fusion" = audio repetition x derivative peaks x
# parabolic arcs + squash sequence grammar (event_engine); "votes" = the prior
# detect + classify_events pipeline, kept for comparison.
EVENT_ENGINE = "gb_model"
AUDIO_WINDOW_PAD_FRAMES = 4
# Audio windows with no ball detections get one re-track at this width so
# the event has a position to judge (the normal refine pass skips stride 1).
AUDIO_RESCUE_INFERENCE_WIDTH = 1600
AUDIO_RESCUE_PAD_FRAMES = 8
DISPLAY_FRAME_SEARCH_RADIUS = 4
TIN_WIDTH_FEET = 21.0
FEET_PER_SECOND_TO_MPH = 0.6818181818
TARGET_CENTER_LEFT = 0.21
TARGET_CENTER_RIGHT = 0.79
TARGET_SERVICE_Y = (4.57 - 1.78) / (4.57 - 0.48)
TARGET_TIN_Y = 1.0
TARGET_SIDE_ZONE_BOUNDS = (0.0, 0.324, TARGET_SERVICE_Y, TARGET_TIN_Y)
TARGET_CENTER_ZONE_BOUNDS = (0.0, TARGET_SERVICE_Y, TARGET_TIN_Y)
TARGET_ZONE_IDS = (1, 2, 3, 4, 5)


def persist_job(job):
    run_dir = job.get("run_dir")
    if not run_dir:
        return

    job_path = Path(run_dir) / "job.json"
    tmp_path = job_path.with_name("job.json.tmp")
    try:
        tmp_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(tmp_path, job_path)
    except OSError:
        pass


def update_job(run_id, /, **updates):
    with JOBS_LOCK:
        job = JOBS.setdefault(run_id, {})
        job.update(updates)
        snapshot = dict(job)

    persist_job(snapshot)
    return snapshot


def get_job(run_id):
    with JOBS_LOCK:
        job = JOBS.get(run_id)
        if job is not None:
            return dict(job)

    job_path = RUNS_DIR / run_id / "job.json"
    if not job_path.exists():
        return None

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # A queued/running job that is absent from memory died with the server.
    if job.get("status") in {"queued", "running"}:
        job["status"] = "failed"
        job["error"] = "The server restarted while this job was running."
        job["message"] = "Tracking failed."

    return job


def create_job(run_id, run_dir, **fields):
    return update_job(run_id, run_id=run_id, run_dir=str(run_dir), **fields)


def start_tracking_job(run_id):
    thread = threading.Thread(target=run_tracking_job, args=(run_id,), daemon=True)
    thread.start()
    return thread


def decode_segments_to_queue(video_path, segments, frame_queue, stop_event, decode_errors):
    """Producer thread: decode (start, end, stride) segments into frame_queue."""

    def enqueue(item):
        while not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        for seg_start, seg_end, stride in segments:
            if stop_event.is_set():
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            read_count = seg_start

            while read_count <= seg_end and not stop_event.is_set():
                if (read_count - seg_start) % stride != 0:
                    # grab() skips the decode-to-BGR step for strided-out frames.
                    if not cap.grab():
                        break
                    read_count += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                enqueue((read_count, frame))
                read_count += 1
    except Exception as error:
        decode_errors.append(error)
    finally:
        cap.release()
        enqueue(None)


def track_segments(model, video_path, segments, inference_width, source_fps, results, on_frame):
    """Consumer loop: infer frames from the decode queue into `results`.

    Decode runs on its own thread so it overlaps inference, which dominates.
    """
    frame_queue = queue.Queue(maxsize=DECODE_QUEUE_SIZE)
    stop_event = threading.Event()
    decode_errors = []
    decoder = threading.Thread(
        target=decode_segments_to_queue,
        args=(video_path, segments, frame_queue, stop_event, decode_errors),
        daemon=True,
    )
    decoder.start()

    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break

            frame_idx, frame = item
            predictions = infer_frame_predictions(
                model,
                frame,
                CONFIDENCE_THRESHOLD,
                inference_width,
            )
            ball_prediction = select_ball_prediction(predictions)
            results[frame_idx] = ball_csv_row(frame_idx, source_fps, ball_prediction)
            on_frame(frame_idx)
    finally:
        stop_event.set()
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        decoder.join(timeout=5)

    if decode_errors:
        raise decode_errors[0]


def write_results_csv(csv_path, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        for frame_idx in sorted(results):
            csv_writer.writerow(results[frame_idx])


def merge_frame_windows(windows):
    merged = []
    for low, high in sorted(windows):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))

    return [(low, high, 1) for low, high in merged]


def refine_segments_for_hits(hits, start_frame, end_frame, stride):
    radius = max(REFINE_WINDOW_MIN_RADIUS, 4 * stride)
    return merge_frame_windows(
        (
            max(start_frame, int(hit["hit_frame"]) - radius),
            min(end_frame, int(hit["hit_frame"]) + radius),
        )
        for hit in hits
    )


def refine_segments_for_audio_candidates(candidates, start_frame, end_frame):
    return merge_frame_windows(
        (
            max(start_frame, int(candidate["window_start_frame"]) - AUDIO_WINDOW_PAD_FRAMES),
            min(end_frame, int(candidate["window_end_frame"]) + AUDIO_WINDOW_PAD_FRAMES),
        )
        for candidate in candidates
    )


def audio_hits_from_candidates(candidates, fps):
    return [
        {
            "frame": int(candidate["frame"]),
            "source": "audio",
            "call": "AUDIO",
            "score": candidate.get("score"),
            "timestamp_seconds": int(candidate["frame"]) / fps,
            "window_start_seconds": int(candidate["window_start_frame"]) / fps,
            "window_end_seconds": int(candidate["window_end_frame"]) / fps,
        }
        for candidate in candidates
    ]


def ball_point_from_row(row):
    if row is None:
        return None

    detected = str(row.get("detected", "")).strip().lower()
    if detected not in {"true", "1", "yes"} or not row.get("x_center"):
        return None

    return Point(float(row["x_center"]), float(row["y_center"]))


def row_has_ball_detection(row):
    return ball_point_from_row(row) is not None


def nearest_detected_frame(results, target_frame, max_distance=DISPLAY_FRAME_SEARCH_RADIUS):
    low = int(target_frame - max_distance)
    high = int(target_frame + max_distance)
    best_frame = None
    best_distance = None

    for frame in range(low, high + 1):
        if not row_has_ball_detection(results.get(frame)):
            continue

        distance = abs(frame - target_frame)
        if best_distance is None or distance < best_distance:
            best_frame = frame
            best_distance = distance

    return best_frame


def display_frame_for_hit(results, hit):
    if "impact_frame" in hit:
        return nearest_detected_frame(results, float(hit["impact_frame"]))

    frame = int(hit["hit_frame"])
    return frame if row_has_ball_detection(results.get(frame)) else None


def line_length_px(line):
    return math.hypot(line.right.x - line.left.x, line.right.y - line.left.y)


def velocity_scale_from_tin_line(bottom_line):
    pixels_per_foot = line_length_px(bottom_line) / TIN_WIDTH_FEET
    if pixels_per_foot <= 0:
        return None
    return pixels_per_foot


def tin_horizontal_range_from_run(run_dir):
    calibration_path = Path(run_dir) / "calibration.json"
    if not calibration_path.exists():
        return None

    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        _, bottom_line = load_calibration_lines(calibration)
    except (ValueError, json.JSONDecodeError, OSError):
        return None

    return (
        min(bottom_line.left.x, bottom_line.right.x),
        max(bottom_line.left.x, bottom_line.right.x),
    )


def calibrated_velocity(hit, pixels_per_foot):
    if not pixels_per_foot:
        return None

    def convert(px_per_second):
        feet_per_second = float(px_per_second) / pixels_per_foot
        return {
            "px_per_second": float(px_per_second),
            "feet_per_second": feet_per_second,
            "mph": feet_per_second * FEET_PER_SECOND_TO_MPH,
        }

    return {
        "scale_source": "tin_top_edge_21ft",
        "pixels_per_foot": pixels_per_foot,
        "speed_before": convert(hit["speed_before"]),
        "speed_after": convert(hit["speed_after"]),
        "velocity_change": convert(hit["dv_magnitude"]),
    }


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def is_front_wall_hit(hit):
    event_type = hit.get("event_type")
    return event_type in (None, "wall", "unknown")


def target_zone_for_diagram(diagram):
    x = clamp(diagram["x"])
    y = clamp(diagram["y"])

    if TARGET_CENTER_LEFT <= x <= TARGET_CENTER_RIGHT:
        zone = 4 if y < TARGET_CENTER_ZONE_BOUNDS[1] else 5
        side = "center"
    else:
        zone = 3
        for index in range(3):
            if TARGET_SIDE_ZONE_BOUNDS[index] <= y < TARGET_SIDE_ZONE_BOUNDS[index + 1]:
                zone = index + 1
                break
        side = "left" if x < TARGET_CENTER_LEFT else "right"

    return {
        "zone": zone,
        "side": side,
        "x": float(diagram["x"]),
        "y": float(diagram["y"]),
    }


def build_target_zone_summary(hits):
    zones = [
        {
            "zone": zone,
            "count": 0,
            "percentage": 0.0,
        }
        for zone in TARGET_ZONE_IDS
    ]
    by_zone = {zone["zone"]: zone for zone in zones}
    target_hits = [
        hit
        for hit in hits
        if is_front_wall_hit(hit) and hit.get("target_zone") is not None
    ]
    for hit in target_hits:
        zone = by_zone.get(int(hit["target_zone"]["zone"]))
        if zone is not None:
            zone["count"] += 1

    total = len(target_hits)
    if total:
        for zone in zones:
            zone["percentage"] = zone["count"] / total * 100.0

    common = [
        dict(zone)
        for zone in sorted(zones, key=lambda item: (-item["count"], item["zone"]))
        if zone["count"] > 0
    ][:3]
    missing = [dict(zone) for zone in zones if zone["count"] == 0]
    return {
        "layout": "front_wall_5_target",
        "rows": 3,
        "columns": 3,
        "total_wall_hits": total,
        "zones": zones,
        "common_zones": common,
        "missing_zones": missing,
    }


def judge_hits(run_dir, results, detected, audio_available=None):
    top_line = bottom_line = None
    pixels_per_foot = None
    floor_map = None
    calibration_path = Path(run_dir) / "calibration.json"
    if calibration_path.exists():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            floor_map = court_model.load_floor_calibration(calibration)
            top_line, bottom_line = load_calibration_lines(calibration)
            pixels_per_foot = velocity_scale_from_tin_line(bottom_line)
        except (ValueError, json.JSONDecodeError):
            pass

    hits = []
    for hit in detected:
        candidate_frame = int(hit["hit_frame"])
        frame = display_frame_for_hit(results, hit)
        if frame is None:
            if not hit.get("audio_assisted"):
                continue
            # Audio-rescued events may have no ball detection near the peak;
            # recall-first means they still surface, just without a line call.
            frame = nearest_detected_frame(results, candidate_frame)
            if frame is None:
                frame = candidate_frame
        display_row = results.get(frame)

        call, reason = "UNKNOWN", "judging_unavailable"
        judge_source = None
        margin_px = None

        event_type = hit.get("event_type")
        if event_type in ("racket", "floor", "side_wall"):
            # Verdicts (IN/OUT line calls) apply to front-wall hits only;
            # other events carry their classification but no call.
            call, reason = None, f"classified_as_{event_type}"
        elif top_line is not None:
            # Prefer the fitted impact point (where the ball met the wall)
            # over the detected center at the nearest sampled frame.
            if "impact_x" in hit:
                point = Point(hit["impact_x"], hit["impact_y"])
                judge_source = "impact_estimate"
            else:
                point = ball_point_from_row(display_row)
                judge_source = "detected_center" if point is not None else None

            if point is None:
                reason = f"No ball detection recorded for frame {frame}."
            else:
                try:
                    call, reason, _, _ = judge_ball(point, top_line, bottom_line)
                    # Positive: IN by this many pixels; negative: OUT by |margin|.
                    margin_px = judge_margin_px(point, top_line, bottom_line)
                except ValueError as error:
                    call, reason, judge_source = "UNKNOWN", str(error), None

        entry = {
            "frame": frame,
            "timestamp_seconds": (
                float(display_row["timestamp_seconds"])
                if display_row is not None
                else float(hit["timestamp_seconds"])
            ),
            "score": hit.get("score"),
            "dv_magnitude": hit.get("dv_magnitude"),
            "after_gap": hit.get("after_gap"),
            "call": call,
            "reason": reason,
            "judge_source": judge_source,
            "margin_px": margin_px,
        }
        if "speed_before" in hit:
            velocity = calibrated_velocity(hit, pixels_per_foot)
            if velocity is not None:
                entry["velocity"] = velocity
        if hit.get("audio_assisted"):
            entry["audio_assisted"] = True
        if "source" in hit:
            entry["source"] = hit["source"]
        if "methods" in hit:
            entry["methods"] = hit["methods"]
        if candidate_frame != frame:
            entry["candidate_frame"] = candidate_frame
        if "event_type" in hit:
            entry["event_type"] = hit["event_type"]
            entry["wall_score"] = hit.get("wall_score")
            entry["signals"] = hit.get("signals")
        if "method" in hit:
            entry["method"] = hit["method"]
        if "diagnostics" in hit:
            entry["diagnostics"] = hit["diagnostics"]
        if "impact_x" in hit:
            entry["impact"] = {
                "x": hit["impact_x"],
                "y": hit["impact_y"],
                "frame": hit.get("impact_frame"),
                "time": hit.get("impact_time"),
                "mismatch_px": hit.get("impact_mismatch_px"),
            }
        if top_line is not None and is_front_wall_hit(entry):
            zone_point = None
            if "impact_x" in hit:
                zone_point = Point(hit["impact_x"], hit["impact_y"])
            elif display_row is not None:
                zone_point = ball_point_from_row(display_row)
            if zone_point is not None:
                diagram = wall_diagram_coordinates(
                    zone_point,
                    top_line,
                    bottom_line,
                )
                entry["wall_diagram"] = {
                    "x": diagram["x"],
                    "y": diagram["y"],
                    "x_span": diagram["x_span"],
                    "y_reference": "0 is the out-line lower edge; 1 is the tin top edge",
                }
                entry["target_zone"] = target_zone_for_diagram(diagram)
        if floor_map is not None and entry.get("event_type") == "floor":
            bounce_point = None
            if "impact_x" in hit:
                bounce_point = (hit["impact_x"], hit["impact_y"])
            elif display_row is not None:
                detected_point = ball_point_from_row(display_row)
                if detected_point is not None:
                    bounce_point = (detected_point.x, detected_point.y)
            if bounce_point is not None:
                try:
                    x_ft, y_ft = floor_map.image_to_court(*bounce_point)
                except ValueError:
                    pass
                else:
                    entry["court_position_ft"] = {
                        "x": round(x_ft, 2),
                        "y": round(y_ft, 2),
                    }
                    entry["floor_zone"] = court_model.floor_zone_for_point(x_ft, y_ft)
        hits.append(entry)

    payload = {"hits": hits, "target_zones": build_target_zone_summary(hits)}
    if floor_map is not None:
        payload["floor_zones"] = court_model.build_floor_zone_summary(hits)
    if audio_available is not None:
        payload["audio_available"] = audio_available
    (Path(run_dir) / "detected_hits.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return hits


def floor_zones_from_run(run_dir):
    """Read the floor-bounce summary judge_hits just wrote (None when the run
    had no floor calibration)."""
    try:
        payload = json.loads(
            (Path(run_dir) / "detected_hits.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("floor_zones")


def sorted_rows(results):
    return [results[frame_idx] for frame_idx in sorted(results)]


def run_tracking_job(run_id):
    job = get_job(run_id)
    run_dir = Path(job["run_dir"])
    video_path = Path(job["video_path"])
    csv_path = run_dir / "ball_coordinates.csv"
    start_frame = int(job["start_frame"])
    end_frame = int(job["end_frame"])
    frame_stride = int(job["frame_stride"])
    inference_width = int(job["inference_width"])
    source_fps = float(job["fps"]) or 30.0
    with TRACKING_JOB_SEMAPHORE:
        processed_frames = 0
        last_update = time.monotonic()

        def make_progress_callback(label):
            def on_frame(frame_idx):
                nonlocal processed_frames, last_update
                processed_frames += 1
                now = time.monotonic()
                if (
                    processed_frames % PROGRESS_UPDATE_FRAMES == 0
                    or now - last_update >= PROGRESS_UPDATE_SECONDS
                ):
                    last_update = now
                    update_job(
                        run_id,
                        processed_frames=processed_frames,
                        message=f"{label}: source frame {frame_idx}",
                    )

            return on_frame

        try:
            update_job(run_id, status="running", stage="coarse", message="Loading local model...")
            model = get_tracking_model()
            wall_x_range = tin_horizontal_range_from_run(run_dir)

            results = {}
            # A pass at stride > 1 only needs to be good enough to locate hit
            # candidates, so it can also run at a reduced inference width.
            coarse_width = COARSE_INFERENCE_WIDTH if frame_stride > 1 else inference_width
            track_segments(
                model,
                video_path,
                [(start_frame, end_frame, frame_stride)],
                coarse_width,
                source_fps,
                results,
                make_progress_callback("Coarse pass"),
            )
            write_results_csv(csv_path, results)

            # The two-stage bounce detector needs the calibrated wall lines.
            calibration = None
            calibration_path = run_dir / "calibration.json"
            if calibration_path.exists():
                try:
                    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    calibration = None

            # Audio impact events drive recall: they add refine windows around
            # events the coarse trajectory pass missed, and later feed the
            # event engine. They corroborate or add detections; they never
            # move trajectory-detected ones.
            engine = job.get("event_engine") or EVENT_ENGINE
            update_job(run_id, stage="coarse", message="Analyzing audio...")
            audio_windows = audio_candidates = None
            if engine == "fusion":
                audio_windows = extract_repeating_audio_windows(
                    video_path, start_frame, end_frame, source_fps
                )
            else:
                audio_candidates = extract_audio_candidates(
                    video_path, start_frame, end_frame, source_fps
                )
            refine_audio = audio_windows if audio_windows else audio_candidates

            # Coarse samples are frame_stride apart; the default max_gap would
            # split every track at stride > 3.
            max_gap = max(MAX_GAP_FRAMES, frame_stride)
            detected = detect_hits_from_rows(
                sorted_rows(results),
                max_gap=max_gap,
                wall_x_range=wall_x_range,
                calibration=calibration,
            )
            segments = refine_segments_for_hits(detected, start_frame, end_frame, frame_stride)
            if refine_audio:
                audio_segments = refine_segments_for_audio_candidates(
                    refine_audio, start_frame, end_frame
                )
                segments = merge_frame_windows(
                    (low, high) for low, high, _ in segments + audio_segments
                )

            if frame_stride > 1 and segments:
                refine_total = sum(high - low + 1 for low, high, _ in segments)
                update_job(
                    run_id,
                    stage="refine",
                    total_frames=processed_frames + refine_total,
                    message=f"Refining {len(segments)} window(s) around hit candidates...",
                )
                track_segments(
                    model,
                    video_path,
                    segments,
                    inference_width,
                    source_fps,
                    results,
                    make_progress_callback("Refine pass"),
                )
                write_results_csv(csv_path, results)

            # Audio events with no ball detections anywhere near them cannot
            # be positioned or judged. Re-track just those windows at a
            # boosted inference width — even at stride 1, where the normal
            # refine pass is skipped because the whole clip was already
            # tracked at the requested width.
            if refine_audio:
                unseen = [
                    window
                    for window in refine_audio
                    if not any(
                        row_has_ball_detection(results.get(f))
                        for f in range(
                            int(window["window_start_frame"]) - AUDIO_RESCUE_PAD_FRAMES,
                            int(window["window_end_frame"]) + AUDIO_RESCUE_PAD_FRAMES + 1,
                        )
                    )
                ]
                if unseen:
                    rescue_segments = merge_frame_windows(
                        (
                            max(start_frame, int(w["window_start_frame"]) - AUDIO_RESCUE_PAD_FRAMES),
                            min(end_frame, int(w["window_end_frame"]) + AUDIO_RESCUE_PAD_FRAMES),
                        )
                        for w in unseen
                    )
                    rescue_total = sum(high - low + 1 for low, high, _ in rescue_segments)
                    update_job(
                        run_id,
                        stage="refine",
                        total_frames=processed_frames + rescue_total,
                        message=(
                            f"Re-tracking {len(rescue_segments)} audio window(s) "
                            "with no ball detections..."
                        ),
                    )
                    track_segments(
                        model,
                        video_path,
                        rescue_segments,
                        max(inference_width, AUDIO_RESCUE_INFERENCE_WIDTH),
                        source_fps,
                        results,
                        make_progress_callback("Audio rescue"),
                    )
                    write_results_csv(csv_path, results)

            hits = []
            hits_error = None
            try:
                if engine == "fusion":
                    update_job(run_id, stage="judging", message="Judging wall hits...")
                    classified = detect_events_fused(
                        sorted_rows(results),
                        audio_windows=audio_windows,
                        calibration=calibration,
                        wall_x_range=wall_x_range,
                        config=job.get("fusion"),
                        max_gap=max(MAX_GAP_FRAMES, frame_stride),
                    )
                    audio_available = audio_windows is not None
                elif engine == "gb_model":
                    update_job(run_id, stage="judging", message="Analyzing audio...")
                    audio_candidates = extract_audio_candidates(
                        video_path, start_frame, end_frame, source_fps
                    )
                    update_job(
                        run_id,
                        stage="judging",
                        message=f"Judging wall hits with {BOUNCE_GB_MODEL_PATH.name}...",
                    )
                    detected = detect_hits_with_gb_model(
                        sorted_rows(results),
                        wall_x_range=wall_x_range,
                        calibration=calibration,
                        apply_spatial_filter=True,
                        spatial_filter_mode="sidewall",
                    )
                    classified = classify_events(
                        detected,
                        results,
                        audio_candidates,
                        source_fps,
                        config=job.get("classify"),
                    )
                    audio_available = audio_candidates is not None
                else:
                    update_job(run_id, stage="judging", message="Judging wall hits...")
                    detected = detect_hits_from_rows(
                        sorted_rows(results),
                        max_gap=max(MAX_GAP_FRAMES, frame_stride),
                        wall_x_range=wall_x_range,
                        calibration=calibration,
                        audio_candidates=audio_candidates,
                    )
                    classified = classify_events(
                        detected,
                        results,
                        audio_candidates,
                        source_fps,
                        config=job.get("classify"),
                    )
                    audio_available = audio_candidates is not None
                hits = judge_hits(
                    run_dir,
                    results,
                    classified,
                    audio_available=audio_available,
                )
                target_zones = build_target_zone_summary(hits)
                floor_zones = floor_zones_from_run(run_dir)
            except Exception as error:
                hits_error = str(error)
                target_zones = build_target_zone_summary(hits)
                floor_zones = None

            job = get_job(run_id) or {}
            total_frames = int(job.get("total_frames", processed_frames or 1))
            extra_fields = {}
            if floor_zones is not None:
                extra_fields["floor_zones"] = floor_zones
            update_job(
                run_id,
                status="complete",
                stage="complete",
                processed_frames=total_frames,
                rows=len(results),
                hits=hits,
                target_zones=target_zones,
                hits_error=hits_error,
                message="Tracking complete.",
                **extra_fields,
            )
        except Exception as error:
            update_job(
                run_id,
                status="failed",
                error=f"Tracking failed.\n\n{error}",
                message="Tracking failed.",
            )


def main():
    import argparse

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a tracking job from its run directory.")
    parser.add_argument("run_dir", type=Path, help="Directory containing job.json")
    args = parser.parse_args()

    job = json.loads((args.run_dir / "job.json").read_text(encoding="utf-8"))
    run_id = job["run_id"]
    with JOBS_LOCK:
        JOBS[run_id] = job

    update_job(run_id, status="queued", message="Queued tracking job.")
    run_tracking_job(run_id)

    final = get_job(run_id)
    print(f"Job {run_id} finished with status: {final.get('status')}")
    if final.get("error"):
        print(final["error"])


if __name__ == "__main__":
    main()
