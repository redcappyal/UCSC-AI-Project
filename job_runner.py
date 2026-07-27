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
import person_model
import player_attribution
from audio_events import extract_audio_candidates
from bounce_gb_model_detector import DEFAULT_MODEL_PATH as BOUNCE_GB_MODEL_PATH
from bounce_gb_model_detector import detect_hits_with_gb_model
from classify_events import classify_events
from detect_wall_hits import MAX_GAP_FRAMES, detect_hits_from_rows
from inference_engine import get_tracking_model, infer_frame_predictions
from judge_call import (
    Point,
    judge_ball,
    judge_margin_px,
    judge_serve_ball,
    load_calibration_lines,
    load_service_line,
    load_wall_corners,
)
from judge_call import wall_diagram_coordinates
from player_tracker import PersonFramePass
from tracking_common import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    ball_csv_row,
    select_motion_consistent_ball_predictions,
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
TARGET_ZONE_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_RALLY_GAP_SECONDS = 5.0
DEFAULT_FIRST_SERVER_PLAYER = 1
RALLY_GAP_MIN_SPLIT_SECONDS = 4.0
RALLY_GAP_RATIO_SPLIT = 1.75


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


def build_person_pass(source_fps, frame_stride):
    """PersonFramePass when the detector backend is available, else None.
    Detector load failures must never kill a tracking job."""
    try:
        detector = person_model.load_person_detector()
    except Exception:
        return None
    if detector is None:
        return None
    return PersonFramePass(detector, source_fps, frame_stride)


def write_track_samples(run_dir, samples_by_track, ambiguity_times):
    """Best-effort persistence: a write failure (e.g. an unwritable run dir)
    must not fail the tracking job, so this returns True/False instead of
    raising -- same pattern as persist_job's OSError guard."""
    payload = {
        "schema": "player-tracks-v1",
        "ambiguity_times": [float(t) for t in ambiguity_times],
        "tracks": {
            track: [
                {
                    "t_s": s.t_s,
                    "frame_idx": s.frame_idx,
                    "foot_px": [s.foot_px[0], s.foot_px[1]],
                    "bbox": [s.bbox[0], s.bbox[1], s.bbox[2], s.bbox[3]],
                    "confidence": s.confidence,
                    "coasted": s.coasted,
                }
                for s in samples
            ]
            for track, samples in samples_by_track.items()
        },
    }
    try:
        players_dir = Path(run_dir) / "players"
        players_dir.mkdir(parents=True, exist_ok=True)
        (players_dir / "track_samples.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    except OSError:
        return False
    return True


def update_job(run_id, /, **updates):
    with JOBS_LOCK:
        if run_id not in JOBS:
            # This run may exist only on disk (server restarted since it last
            # ran in memory). Rehydrate from job.json before merging, or the
            # merge lands on a fresh empty stub that then SHADOWS the real
            # job.json for every later get_job call, since get_job short-
            # circuits on any in-memory hit -- even an empty one.
            #
            # Read raw: do not apply get_job's queued/running->failed
            # rewrite here. That rewrite is a read-view decision about a live
            # status, not stored state, and this path also rehydrates jobs
            # that are already "complete" (e.g. a post-hoc players rename
            # long after the run finished).
            #
            # The disk read stays inside JOBS_LOCK on purpose: reading
            # outside the lock could race a concurrent update_job for the
            # same run_id that persists first, so it would either duplicate
            # a rehydrate or clobber a fresher in-memory update with a stale
            # disk read. persist_job's own I/O still happens outside the
            # lock, same as before -- only this rehydrate read is kept in.
            job_path = RUNS_DIR / run_id / "job.json"
            try:
                JOBS[run_id] = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                JOBS[run_id] = {}
        job = JOBS[run_id]
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


def track_segments(model, video_path, segments, inference_width, source_fps, results, on_frame,
                   frame_observer=None):
    """Consumer loop: infer frames from the decode queue into `results`.

    Decode runs on its own thread so it overlaps inference, which dominates.
    `frame_observer(frame_idx, frame)`, when given, is called for every
    decoded frame in this call — callers wire it only on the coarse pass so
    the person-detection cadence rides the coarse decode, never refine or
    audio-rescue (spec §4.2).
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

    raw_predictions = {}
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break

            frame_idx, frame = item
            if frame_observer is not None:
                frame_observer(frame_idx, frame)
            predictions = infer_frame_predictions(
                model,
                frame,
                CONFIDENCE_THRESHOLD,
                inference_width,
            )
            raw_predictions[frame_idx] = predictions
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

    selected_predictions = select_motion_consistent_ball_predictions(
        raw_predictions,
        CONFIDENCE_THRESHOLD,
    )
    for frame_idx, ball_prediction in selected_predictions.items():
        results[frame_idx] = ball_csv_row(frame_idx, source_fps, ball_prediction)


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


def is_serve_hit(hit):
    if hit.get("is_serve"):
        return True
    try:
        return int(hit.get("rally_hit_sequence") or 0) == 1
    except (TypeError, ValueError):
        return False


def env_float(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value)
    except ValueError:
        return float(default)


def configured_env_float(name):
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def other_player(player_number):
    return 2 if int(player_number) == 1 else 1


def is_player_assignable_front_wall_hit(hit):
    return (
        is_front_wall_hit(hit)
        and hit.get("target_zone") is not None
        and hit.get("call") in ("IN", "OUT")
    )


def hit_sort_key(hit):
    return (
        float(hit.get("timestamp_seconds", 0.0)),
        int(hit.get("frame", hit.get("hit_frame", 0)) or 0),
    )


def infer_rally_gap_seconds(front_wall_hits):
    ordered = sorted(front_wall_hits, key=hit_sort_key)
    gaps = [
        float(cur.get("timestamp_seconds", 0.0)) - float(prev.get("timestamp_seconds", 0.0))
        for prev, cur in zip(ordered, ordered[1:])
    ]
    positive_gaps = sorted(gap for gap in gaps if gap > 0)
    best = None
    for lower, upper in zip(positive_gaps, positive_gaps[1:]):
        if lower <= 0:
            continue
        ratio = upper / lower
        if upper < RALLY_GAP_MIN_SPLIT_SECONDS or ratio < RALLY_GAP_RATIO_SPLIT:
            continue
        if best is None or ratio > best[0]:
            best = (ratio, (lower + upper) / 2)

    if best is not None:
        return max(RALLY_GAP_MIN_SPLIT_SECONDS, best[1]), "adaptive_gap_ratio"
    return DEFAULT_RALLY_GAP_SECONDS, "default_gap_seconds"


def target_zone_for_diagram(diagram):
    x = clamp(diagram["x"])
    y = clamp(diagram["y"])

    if TARGET_CENTER_LEFT <= x <= TARGET_CENTER_RIGHT:
        zone = 4 if y < TARGET_CENTER_ZONE_BOUNDS[1] else 5
        side = "center"
    else:
        side_offset = 0 if x < TARGET_CENTER_LEFT else 5
        zone = side_offset + 3
        for index in range(3):
            if TARGET_SIDE_ZONE_BOUNDS[index] <= y < TARGET_SIDE_ZONE_BOUNDS[index + 1]:
                zone = side_offset + index + 1
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
        if (
            is_front_wall_hit(hit)
            and not is_serve_hit(hit)
            and hit.get("target_zone") is not None
        )
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
        "layout": "front_wall_8_target",
        "rows": 3,
        "columns": 3,
        "total_wall_hits": total,
        "zones": zones,
        "common_zones": common,
        "missing_zones": missing,
    }


def segment_front_wall_hits_into_rallies(front_wall_hits, rally_gap_seconds=None):
    if rally_gap_seconds is None:
        configured_gap = configured_env_float("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS")
        if configured_gap is None:
            rally_gap_seconds, gap_method = infer_rally_gap_seconds(front_wall_hits)
        else:
            rally_gap_seconds = configured_gap
            gap_method = "env_PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS"
    else:
        rally_gap_seconds = float(rally_gap_seconds)
        gap_method = "explicit"
    rally_gap_seconds = max(0.0, rally_gap_seconds)
    rallies = []
    current = []
    previous_time = None
    for hit in sorted(front_wall_hits, key=hit_sort_key):
        timestamp = float(hit.get("timestamp_seconds", 0.0))
        if (
            current
            and previous_time is not None
            and timestamp - previous_time >= rally_gap_seconds
        ):
            rallies.append(current)
            current = []
        current.append(hit)
        previous_time = timestamp
    if current:
        rallies.append(current)
    return rallies, rally_gap_seconds, gap_method


def assign_front_wall_hit_players(hits, serve_resolver=None):
    for hit in hits:
        for key in (
            "front_wall_sequence",
            "rally_number",
            "rally_hit_sequence",
            "server_player_number",
            "player_number",
        ):
            hit.pop(key, None)

    assignable_hits = [
        hit for hit in hits if is_player_assignable_front_wall_hit(hit)
    ]
    rallies, rally_gap_seconds, rally_gap_method = segment_front_wall_hits_into_rallies(
        assignable_hits
    )
    try:
        first_server = int(
            env_float("PLAYER_ASSIGNMENT_FIRST_SERVER", DEFAULT_FIRST_SERVER_PLAYER)
        )
    except (OverflowError, ValueError):
        first_server = DEFAULT_FIRST_SERVER_PLAYER
    if first_server not in (1, 2):
        first_server = DEFAULT_FIRST_SERVER_PLAYER

    resolved_tracks = [
        serve_resolver(rally_hits) if serve_resolver is not None else None
        for rally_hits in rallies
    ]

    sequence = 0
    server = first_server
    track_player_map = None
    rally_summaries = []
    for rally_index, rally_hits in enumerate(rallies, start=1):
        resolved = resolved_tracks[rally_index - 1]
        if resolved in ("A", "B") and track_player_map is None:
            # Anchor: the first observed rally binds its track to the
            # propagated server at this point in the chain (spec §4.4).
            track_player_map = {
                resolved: server,
                ("B" if resolved == "A" else "A"): other_player(server),
            }
        if resolved in ("A", "B") and track_player_map is not None:
            rally_server = track_player_map[resolved]
            server_source = "observed"
        else:
            rally_server = server
            server_source = "propagated"

        for rally_sequence, hit in enumerate(rally_hits, start=1):
            sequence += 1
            player_number = (
                rally_server
                if rally_sequence % 2 == 1
                else other_player(rally_server)
            )
            hit["front_wall_sequence"] = sequence
            hit["rally_number"] = rally_index
            hit["rally_hit_sequence"] = rally_sequence
            hit["server_player_number"] = rally_server
            hit["player_number"] = player_number

        serve_hit = rally_hits[0]
        last_hit = rally_hits[-1]
        last_player = int(last_hit.get("player_number") or rally_server)
        last_call = last_hit.get("call")
        serve_call = serve_hit.get("call") if serve_hit.get("is_serve") else None
        if serve_call == "OUT":
            winner = other_player(rally_server)
            winner_reason = "serve_out"
        elif last_call == "OUT":
            winner = other_player(last_player)
            winner_reason = "last_front_wall_hit_out"
        elif last_call == "IN":
            winner = last_player
            winner_reason = "last_front_wall_hit_in_winner"
        else:
            winner = None
            winner_reason = "last_front_wall_hit_unjudged"

        rally_start_time = float(rally_hits[0].get("timestamp_seconds", 0.0))
        rally_end_time = float(rally_hits[-1].get("timestamp_seconds", 0.0))
        rally_summaries.append({
            "rally_number": rally_index,
            "start_frame": int(rally_hits[0].get("frame", 0) or 0),
            "end_frame": int(rally_hits[-1].get("frame", 0) or 0),
            "start_time_seconds": rally_start_time,
            "end_time_seconds": rally_end_time,
            "duration_seconds": round(max(0.0, rally_end_time - rally_start_time), 3),
            "front_wall_hit_count": len(rally_hits),
            "server_player_number": rally_server,
            "server_track": resolved if resolved in ("A", "B") else None,
            "server_source": server_source,
            "winner_player_number": winner,
            "winner_reason": winner_reason,
            "winner_source": "est" if winner is not None else None,
            "winner_crosscheck_agrees": None,
            "last_call": last_call,
            "last_player_number": last_player,
            "serve_frame": (
                int(serve_hit.get("frame", 0) or 0)
                if serve_hit.get("is_serve")
                else None
            ),
            "serve_call": serve_call,
        })
        if winner in (1, 2):
            server = winner

    # Winner back-fill: winner of rally N := observed server of rally N+1
    # (squash rule — the winner serves next). The est winner stays as a
    # silent cross-check (spec §4.4). Last rally keeps est.
    for index in range(len(rally_summaries) - 1):
        next_summary = rally_summaries[index + 1]
        if next_summary["server_source"] != "observed":
            continue
        summary = rally_summaries[index]
        est_winner = summary["winner_player_number"]
        observed_winner = next_summary["server_player_number"]
        summary["winner_player_number"] = observed_winner
        summary["winner_source"] = "next_serve"
        summary["winner_reason"] = "next_rally_serve_observed"
        summary["winner_crosscheck_agrees"] = (
            (est_winner == observed_winner) if est_winner is not None else None
        )

    observed_serve_count = sum(
        1 for summary in rally_summaries if summary["server_source"] == "observed"
    )
    return {
        "method": (
            "rally_gap_observed_serves"
            if observed_serve_count
            else "rally_gap_server_alternation"
        ),
        "rally_gap_seconds": rally_gap_seconds,
        "rally_gap_method": rally_gap_method,
        "first_server_player_number": first_server,
        "rally_count": len(rally_summaries),
        "assigned_front_wall_hit_count": len(assignable_hits),
        "observed_serve_count": observed_serve_count,
        "rallies": rally_summaries,
    }


def build_player_target_zone_summaries(hits):
    summaries = {}
    for player_number in (1, 2):
        player_hits = [
            hit
            for hit in hits
            if int(hit.get("player_number") or 0) == player_number
        ]
        summary = build_target_zone_summary(player_hits)
        summary["player_number"] = player_number
        summary["label"] = f"Player {player_number}"
        summaries[str(player_number)] = summary
    return summaries


def judge_hits(run_dir, results, detected, audio_available=None, serve_resolver=None):
    top_line = bottom_line = None
    service_line = None
    wall_corners = None
    pixels_per_foot = None
    floor_map = None
    calibration = None
    calibration_path = Path(run_dir) / "calibration.json"
    if calibration_path.exists():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            floor_map = court_model.load_floor_calibration(calibration)
            top_line, bottom_line = load_calibration_lines(calibration)
            wall_corners = load_wall_corners(calibration)
            pixels_per_foot = velocity_scale_from_tin_line(bottom_line)
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            service_line = load_service_line(calibration)
        except (ValueError, TypeError):
            service_line = None

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
                    call, reason, _, _ = judge_ball(point, top_line, bottom_line, wall_corners)
                    # Positive: IN by this many pixels; negative: OUT by |margin|.
                    if call in ("IN", "OUT"):
                        margin_px = judge_margin_px(point, top_line, bottom_line, wall_corners)
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
        if top_line is not None and is_front_wall_hit(entry) and entry.get("call") in ("IN", "OUT"):
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
                    wall_corners=wall_corners,
                )
                entry["wall_diagram"] = {
                    "x": diagram["x"],
                    "y": diagram["y"],
                    "x_span": diagram["x_span"],
                    "x_reference": diagram.get("x_reference"),
                    "y_reference": "0 is the out-line lower edge; 1 is the tin top edge",
                }
                entry["target_zone"] = target_zone_for_diagram(diagram)
        if entry.get("event_type") == "floor" and floor_map is not None:
            # Floor bounces are the one event on the floor plane, so the
            # homography is exact for them.
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

    player_assignment = assign_front_wall_hit_players(hits, serve_resolver=serve_resolver)
    if top_line is not None and service_line is not None:
        for entry in hits:
            if int(entry.get("rally_hit_sequence") or 0) != 1:
                continue
            point = None
            if entry.get("impact"):
                point = Point(float(entry["impact"]["x"]), float(entry["impact"]["y"]))
            else:
                display_row = results.get(int(entry.get("frame", -1)))
                if display_row is not None:
                    point = ball_point_from_row(display_row)
            if point is None:
                continue

            entry["is_serve"] = True
            entry["standard_call"] = entry.get("call")
            try:
                call, reason, _, service_y = judge_serve_ball(
                    point, top_line, service_line, wall_corners
                )
                entry["call"] = call
                entry["reason"] = reason
                entry["margin_px"] = (
                    judge_margin_px(point, top_line, service_line, wall_corners)
                    if call in ("IN", "OUT")
                    else None
                )
                entry["serve_call"] = call
                entry["serve_reason"] = reason
                entry["service_line_y"] = service_y
            except ValueError as error:
                entry["call"] = "UNKNOWN"
                entry["reason"] = str(error)
                entry["margin_px"] = None
                entry["serve_call"] = "UNKNOWN"
                entry["serve_reason"] = str(error)

        # Serve OUT can change the rally winner and therefore the next rally's
        # inferred server. Re-run the deterministic assignment after applying
        # the first-contact serve rule.
        player_assignment = assign_front_wall_hit_players(hits, serve_resolver=serve_resolver)
    payload = {
        "hits": hits,
        "target_zones": build_target_zone_summary(hits),
        "target_zones_by_player": build_player_target_zone_summaries(hits),
        "player_assignment": player_assignment,
        "rallies": player_assignment["rallies"],
    }
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

            person_pass = build_person_pass(source_fps, frame_stride)
            if person_pass is not None:
                update_job(run_id, message="Person detector: rfdetr")

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
                frame_observer=person_pass.observe if person_pass is not None else None,
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
            # bounce detector. They corroborate or add detections; they never
            # move trajectory-detected ones.
            update_job(run_id, stage="coarse", message="Analyzing audio...")
            audio_candidates = extract_audio_candidates(
                video_path, start_frame, end_frame, source_fps
            )

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
            if audio_candidates:
                audio_segments = refine_segments_for_audio_candidates(
                    audio_candidates, start_frame, end_frame
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
            if audio_candidates:
                unseen = [
                    window
                    for window in audio_candidates
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

            serve_resolver = None
            samples_by_track = None
            ambiguity_times = []
            if person_pass is not None:
                samples_by_track = person_pass.tracker.samples()
                ambiguity_times = person_pass.tracker.ambiguity_times()
                serve_resolver = player_attribution.build_serve_resolver(
                    samples_by_track, sorted_rows(results)
                )

            hits = []
            hits_error = None
            try:
                update_job(
                    run_id,
                    stage="judging",
                    message=f"Judging wall hits with {BOUNCE_GB_MODEL_PATH.name}...",
                )
                detected = detect_hits_with_gb_model(
                    sorted_rows(results),
                    wall_x_range=wall_x_range,
                    calibration=calibration,
                    # Location filtering runs inside the detector before
                    # wall-visit grouping, so sidewall/outside-wall
                    # candidates cannot merge otherwise valid wall hits.
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
                hits = judge_hits(
                    run_dir,
                    results,
                    classified,
                    audio_available=audio_candidates is not None,
                    serve_resolver=serve_resolver,
                )
                player_assignment = assign_front_wall_hit_players(hits, serve_resolver=serve_resolver)
                target_zones = build_target_zone_summary(hits)
                target_zones_by_player = build_player_target_zone_summaries(hits)
                floor_zones = floor_zones_from_run(run_dir)
            except Exception as error:
                hits_error = str(error)
                player_assignment = assign_front_wall_hit_players(hits, serve_resolver=serve_resolver)
                target_zones = build_target_zone_summary(hits)
                target_zones_by_player = build_player_target_zone_summaries(hits)
                floor_zones = None

            players_v1 = None
            if person_pass is not None:
                confidences = player_attribution.rally_identity_confidences(
                    ambiguity_times, player_assignment["rallies"]
                )
                for rally in player_assignment["rallies"]:
                    rally["identity_confidence"] = confidences.get(rally["rally_number"])
                write_track_samples(run_dir, samples_by_track, ambiguity_times)
                serve_crop_relpath = None
                crop_target = player_attribution.serve_crop_target(
                    player_assignment, samples_by_track
                )
                if crop_target is not None:
                    crop_frame_idx, crop_sample = crop_target
                    crop_detection = person_model.PersonDetection(
                        x=crop_sample.bbox[0], y=crop_sample.bbox[1],
                        width=crop_sample.bbox[2], height=crop_sample.bbox[3],
                        confidence=crop_sample.confidence, keypoints=(),
                    )
                    if person_model.save_person_crop(
                        video_path, crop_frame_idx, crop_detection,
                        run_dir / "players" / "serve_rally1.jpg",
                    ):
                        serve_crop_relpath = "players/serve_rally1.jpg"
                players_v1 = player_attribution.build_players_v1(
                    player_assignment,
                    person_pass.stats(),
                    detector_backend=person_pass.detector.backend,
                    serve_crop_relpath=serve_crop_relpath,
                )
                if person_pass.detect_failures:
                    players_v1["detector_error"] = person_pass.detect_error
            else:
                players_v1 = player_attribution.build_players_v1(
                    player_assignment, None, detector_backend="none"
                )

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
                target_zones_by_player=target_zones_by_player,
                player_assignment=player_assignment,
                hits_error=hits_error,
                players_v1=players_v1,
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
