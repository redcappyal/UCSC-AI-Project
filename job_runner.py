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
import stereo_engine
import stereo_sync
from audio_events import extract_audio_candidates, extract_repeating_audio_windows
from bounce_gb_model_detector import DEFAULT_MODEL_PATH as BOUNCE_GB_MODEL_PATH
from bounce_gb_model_detector import detect_hits_with_gb_model
from event_engine import detect_events_fused
from classify_events import classify_events
from detect_wall_hits import MAX_GAP_FRAMES, detect_hits_from_rows
from inference_engine import get_tracking_model, infer_frame_predictions
from judge_call import Point, judge_ball, judge_margin_px, load_calibration_lines, load_wall_corners
from judge_call import wall_diagram_coordinates
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


# --- Phase 5: paired sessions -----------------------------------------------
#
# A fuse run lives in its own directory named `stereo-<session_id>`. That name
# is load-bearing twice over: it makes the claim atomic (whoever wins the mkdir
# race owns the job), and it lets discovery skip fuse runs so one can never be
# mistaken for a camera run.

STEREO_FUSE_PREFIX = "stereo-"


def stereo_fuse_run_id(session_id):
    return f"{STEREO_FUSE_PREFIX}{session_id}"


def session_runs(session_id):
    """`{camera_role: job}` for the completed camera runs of one session.

    Globs `ui_runs/*/job.json` the way /api/calibration/latest does — there is
    no session index, and building one would be a second source of truth for
    something the run dirs already record. A retried camera leaves two runs
    under the same role; the newest wins.
    """
    found = {}
    if not session_id:
        return found
    try:
        candidates = sorted(RUNS_DIR.glob("*/job.json"))
    except OSError:
        return found

    for job_path in candidates:
        if job_path.parent.name.startswith(STEREO_FUSE_PREFIX):
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(job, dict):
            continue
        if job.get("session_id") != session_id or job.get("status") != "complete":
            continue
        role = job.get("camera_role")
        if role not in {"a", "b"}:
            continue
        previous = found.get(role)
        # run_id is a millisecond wall-clock stamp, so string order is time
        # order for same-width ids and a stable tiebreak otherwise.
        if previous is None or str(job.get("run_id", "")) >= str(previous.get("run_id", "")):
            found[role] = job
    return found


def maybe_start_stereo_fuse(session_id):
    """Start the fuse job iff both roles are complete and nobody else claimed it.

    Returns the fuse run_id when this caller started it, else None. Both
    cameras finishing at once is the normal case, so the claim is the
    directory create itself: `exist_ok=False` makes exactly one caller win.
    """
    runs = session_runs(session_id)
    if set(runs) != {"a", "b"}:
        return None

    fuse_id = stereo_fuse_run_id(session_id)
    fuse_dir = RUNS_DIR / fuse_id
    try:
        fuse_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None            # someone else got there first
    except OSError:
        return None

    run_a, run_b = runs["a"]["run_id"], runs["b"]["run_id"]
    create_job(
        fuse_id,
        fuse_dir,
        status="queued",
        job_type="stereo_fuse",
        message="Queued stereo fusion.",
        session_id=session_id,
        run_a=run_a,
        run_b=run_b,
        processed_frames=0,
        total_frames=1,
        # Deliberately NO video_path: build_eval_set keeps one run per
        # video_sha, so a fuse run carrying role a's video could displace the
        # mono run it was derived from and silently rebaseline the eval set.
    )
    for run_id in (run_a, run_b):
        # Hydrate before updating. A completed run is usually gone from JOBS
        # (the server may have restarted, or this may be a different process),
        # and update_job would then build a one-key dict with no run_dir —
        # which persist_job silently declines to write. Round-tripping the
        # whole job keeps the fuse id and everything else.
        existing = get_job(run_id)
        if existing is not None:
            update_job(run_id, **{**existing, "stereo_fuse_run_id": fuse_id})
    start_stereo_fuse_job(fuse_id)
    return fuse_id


STEREO_TIMELINE_HZ = 120.0


def _stereo_camera(run_dir, side):
    """Solve one side's camera, in the pixel space its CSV is written in."""
    calibration_path = Path(run_dir) / "calibration.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"side {side!r} calibration unreadable: {error}")
    model, info = court_model.solve_camera_model(calibration)
    if model is None:
        raise ValueError(
            f"side {side!r} calibration failed to solve "
            f"(status={info.get('status')!r})")
    return model, calibration


def _stereo_hit_entry(impact, fps_a):
    """One `stereo_engine.Impact` in the existing detected_hits vocabulary.

    Speaking judge_hits' vocabulary rather than stereo_offline's is deliberate:
    build_eval_set filters hits with `isinstance(frame, int)`, so a float frame
    would match nothing and read as total detector failure. Stereo's own values
    ride along losslessly in the additive `stereo` block.
    """
    x_ft, _y_ft, z_ft = (float(v) for v in impact.point_ft)

    event_type = {
        "front_wall": "wall",
        "left_wall": "side_wall",
        "right_wall": "side_wall",
        "back_wall": "side_wall",
        "floor": "floor",
    }.get(impact.surface, "unknown")

    # "down" is a tin hit — it ends the rally exactly as an out ball does, and
    # the mono vocabulary has no third word for it.
    call = {"in": "IN", "out": "OUT", "down": "OUT"}.get(impact.call)

    # confidence is the engine's own signal for how the point was obtained, so
    # method/view_count derive from it and stereo_engine needs no change (and
    # the goldens do not move).
    method, view_count = {
        "high": ("plane_snap", 2),
        "one_view": ("plane_snap", 1),
        "no_call": ("triangulated", 2),
    }.get(impact.confidence, ("triangulated", 2))

    entry = {
        # role a's clock and fps, so the frame joins role a's CSV and labels.
        "frame": int(round(float(impact.t_s) * float(fps_a))),
        "timestamp_seconds": float(impact.t_s),
        "event_type": event_type,
        "call": call,
        "reason": f"stereo_{impact.surface}",
        "judge_source": "stereo_plane_snap",
        # A stereo call has no pixel margin. None is honest, and the mono
        # schema already permits it.
        "margin_px": None,
        "view_count": view_count,
        "method": method,
        "stereo": {
            "surface": impact.surface,
            "point_ft": [float(v) for v in impact.point_ft],
            "margin_ft": float(impact.margin_ft),
            "confidence": impact.confidence,
            "snap_disagreement_ft": (
                None if impact.snap_disagreement_ft is None
                else float(impact.snap_disagreement_ft)),
        },
    }

    # Front-wall contacts carry the wall diagram and target zone the coaching
    # analytics key off, computed straight from the 3D point: x across the
    # 21 ft wall, y from the out line (0) down to the tin top (1), matching
    # judge_call.wall_diagram_coordinates' convention.
    if impact.surface == "front_wall":
        span = court_model.OUT_LINE_HEIGHT_FT - court_model.TIN_TOP_HEIGHT_FT
        diagram = {
            "x": x_ft / court_model.COURT_WIDTH_FT,
            "y": (court_model.OUT_LINE_HEIGHT_FT - z_ft) / span,
            "x_reference": "stereo_court_frame",
        }
        entry["wall_diagram"] = diagram
        entry["target_zone"] = target_zone_for_diagram(diagram)
    return entry


def run_stereo_fuse_job(run_id):
    """Fuse the two camera runs of a paired session into one 3D result.

    Reads both runs' ball_coordinates.csv rather than re-decoding video: the
    detections are the expensive part and both runs already paid for them.
    """
    job = get_job(run_id) or {}
    run_dir = Path(job.get("run_dir", RUNS_DIR / run_id))
    try:
        update_job(run_id, status="running", stage="fusing",
                   message="Solving both cameras.")

        run_a_id, run_b_id = job.get("run_a"), job.get("run_b")
        job_a, job_b = get_job(run_a_id) or {}, get_job(run_b_id) or {}
        dir_a = Path(job_a.get("run_dir", RUNS_DIR / str(run_a_id)))
        dir_b = Path(job_b.get("run_dir", RUNS_DIR / str(run_b_id)))
        fps_a = float(job_a.get("fps") or 60.0)
        fps_b = float(job_b.get("fps") or 60.0)

        model_a, _cal_a = _stereo_camera(dir_a, "a")
        model_b, _cal_b = _stereo_camera(dir_b, "b")

        update_job(run_id, message="Refining the clock offset.")
        samples_a = stereo_sync.track_samples_from_csv(
            dir_a / "ball_coordinates.csv", fps=fps_a)
        samples_b = stereo_sync.track_samples_from_csv(
            dir_b / "ball_coordinates.csv", fps=fps_b)

        manifest = None
        manifest_path = job_a.get("sync_manifest_path") or job_b.get("sync_manifest_path")
        if manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
        seed_s, seed_source = stereo_sync.seed_offset_from_manifest(manifest)

        report = stereo_sync.refine_offset(
            model_a, samples_a, model_b, samples_b, seed_s=seed_s)
        report["seed"]["source"] = seed_source
        report["schema"] = "stereo-sync-v1"
        report["session_id"] = job.get("session_id")
        report["runs"] = {"a": run_a_id, "b": run_b_id}
        offset_s = float(report["refined"]["offset_s"])

        update_job(run_id, message="Fusing tracks.")
        # Samples b, restamped onto a's clock — build the track once and hand
        # the same object to detect_impacts, so the file written to disk is
        # provably the track the impacts came from.
        aligned_b = [
            stereo_engine.TrackSample(t_s=s.t_s + offset_s, px=s.px) for s in samples_b]
        track = stereo_engine.build_track3d(
            model_a, samples_a, model_b, aligned_b, hz=STEREO_TIMELINE_HZ)
        impacts = stereo_engine.detect_impacts(
            model_a, samples_a, model_b, aligned_b, track=track)

        with (run_dir / "stereo_track.jsonl").open("w", encoding="utf-8") as handle:
            for point in track:
                handle.write(json.dumps({
                    "t_s": float(point.t_s),
                    "point_ft": [float(v) for v in point.point_ft],
                    "gap_ft": float(point.gap_ft),
                    "view_count": 2,
                }, sort_keys=True) + "\n")

        try:
            image_px_a, court_xyz_a, _ = court_model._camera_correspondences(_cal_a)
            image_px_b, court_xyz_b, _ = court_model._camera_correspondences(_cal_b)
            report["pair_agreement"] = stereo_engine.pair_agreement(
                model_a, list(zip(court_xyz_a, image_px_a)),
                model_b, list(zip(court_xyz_b, image_px_b)))
        except Exception as error:                   # noqa: BLE001 - reported, not raised
            report["pair_agreement"] = {"error": str(error)}

        (run_dir / "sync_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

        hits = [_stereo_hit_entry(impact, fps_a) for impact in impacts]
        player_assignment = assign_front_wall_hit_players(hits)
        payload = {
            "hits": hits,
            "target_zones": build_target_zone_summary(hits),
            "target_zones_by_player": build_player_target_zone_summaries(hits),
            "player_assignment": player_assignment,
            "rallies": player_assignment.get("rallies", []),
            "stereo": {
                "session_id": job.get("session_id"),
                "run_a": run_a_id,
                "run_b": run_b_id,
                "offset_s": offset_s,
                "offset_accepted": bool(report["refined"]["accepted"]),
                "track_points": len(track),
            },
        }
        (run_dir / "detected_hits.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

        update_job(
            run_id,
            status="complete",
            stage="complete",
            processed_frames=1,
            hits=hits,
            target_zones=payload["target_zones"],
            target_zones_by_player=payload["target_zones_by_player"],
            player_assignment=player_assignment,
            message=(f"Fused {len(track)} track points into {len(hits)} calls "
                     f"at offset {offset_s * 1000:.1f} ms."),
        )
    except Exception as error:                       # noqa: BLE001 - surfaced as job state
        update_job(run_id, status="failed",
                   error=f"Stereo fusion failed.\n\n{error}",
                   message="Stereo fusion failed.")


def try_start_stereo_fuse(session_id):
    """`maybe_start_stereo_fuse`, but it can never damage its caller.

    This runs at the tail of a finished tracking job. That run is already
    complete and its artifacts are already on disk; nothing about fusing a
    session may retroactively fail it.
    """
    if not session_id:
        return None
    try:
        return maybe_start_stereo_fuse(session_id)
    except Exception:
        return None


def create_job(run_id, run_dir, **fields):
    return update_job(run_id, run_id=run_id, run_dir=str(run_dir), **fields)


def start_tracking_job(run_id):
    thread = threading.Thread(target=run_tracking_job, args=(run_id,), daemon=True)
    thread.start()
    return thread


def start_stereo_fuse_job(run_id):
    thread = threading.Thread(target=run_stereo_fuse_job, args=(run_id,), daemon=True)
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

    raw_predictions = {}
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


def assign_front_wall_hit_players(hits):
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

    sequence = 0
    server = first_server
    rally_summaries = []
    for rally_index, rally_hits in enumerate(rallies, start=1):
        rally_server = server
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

        last_hit = rally_hits[-1]
        last_player = int(last_hit.get("player_number") or rally_server)
        last_call = last_hit.get("call")
        if last_call == "OUT":
            winner = other_player(last_player)
            winner_reason = "last_front_wall_hit_out"
        elif last_call == "IN":
            winner = last_player
            winner_reason = "last_front_wall_hit_in_winner"
        else:
            winner = None
            winner_reason = "last_front_wall_hit_unjudged"

        rally_summaries.append({
            "rally_number": rally_index,
            "start_frame": int(rally_hits[0].get("frame", 0) or 0),
            "end_frame": int(rally_hits[-1].get("frame", 0) or 0),
            "start_time_seconds": float(rally_hits[0].get("timestamp_seconds", 0.0)),
            "end_time_seconds": float(rally_hits[-1].get("timestamp_seconds", 0.0)),
            "front_wall_hit_count": len(rally_hits),
            "server_player_number": rally_server,
            "winner_player_number": winner,
            "winner_reason": winner_reason,
            "last_call": last_call,
            "last_player_number": last_player,
        })
        if winner in (1, 2):
            server = winner

    return {
        "method": "rally_gap_server_alternation",
        "rally_gap_seconds": rally_gap_seconds,
        "rally_gap_method": rally_gap_method,
        "first_server_player_number": first_server,
        "rally_count": len(rally_summaries),
        "assigned_front_wall_hit_count": len(assignable_hits),
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


def judge_hits(run_dir, results, detected, audio_available=None, camera=None, camera_info=None):
    top_line = bottom_line = None
    wall_corners = None
    pixels_per_foot = None
    floor_map = None
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
        if entry.get("event_type") == "floor":
            if hit.get("court_position_ft"):
                entry["court_position_ft"] = hit["court_position_ft"]
                x_ft = hit["court_position_ft"]["x"]
                y_ft = hit["court_position_ft"]["y"]
                entry["floor_zone"] = court_model.floor_zone_for_point(x_ft, y_ft)
            elif floor_map is not None:
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

    player_assignment = assign_front_wall_hit_players(hits)
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
    if camera is not None:
        payload["camera_model"] = camera.to_dict()
    elif camera_info is not None:
        payload["camera_warning"] = camera_info
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
                camera = camera_info = None
                if engine == "fusion":
                    update_job(run_id, stage="judging", message="Judging wall hits...")
                    if job.get("fusion_3d") and calibration:
                        camera, camera_info = court_model.solve_camera_model(calibration)
                    classified = detect_events_fused(
                        sorted_rows(results),
                        audio_windows=audio_windows,
                        calibration=calibration,
                        wall_x_range=wall_x_range,
                        config=job.get("fusion"),
                        max_gap=max(MAX_GAP_FRAMES, frame_stride),
                        camera=camera,
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
                    camera=camera,
                    camera_info=camera_info,
                )
                player_assignment = assign_front_wall_hit_players(hits)
                target_zones = build_target_zone_summary(hits)
                target_zones_by_player = build_player_target_zone_summaries(hits)
                floor_zones = floor_zones_from_run(run_dir)
            except Exception as error:
                hits_error = str(error)
                player_assignment = assign_front_wall_hit_players(hits)
                target_zones = build_target_zone_summary(hits)
                target_zones_by_player = build_player_target_zone_summaries(hits)
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
                target_zones_by_player=target_zones_by_player,
                player_assignment=player_assignment,
                hits_error=hits_error,
                message="Tracking complete.",
                **extra_fields,
            )
            # Phase 5: this run may be half of a paired session. The trigger is
            # deliberately the last thing that happens and deliberately cannot
            # raise — this run is already complete and its artifacts are
            # already on disk.
            try_start_stereo_fuse((get_job(run_id) or {}).get("session_id"))
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
