import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import court_model
from judge_call import (
    Point,
    judge_ball,
    judge_margin_px,
    load_ball_positions,
    load_calibration_lines,
    load_wall_corners,
    wall_diagram_coordinates,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
APP_VERSION = "wall-corner-calibration-2026-07-20-1"

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

# inference_engine sets the model-cache/metrics env defaults on import;
# import it (via job_runner) before anything touches the inference package.
from job_runner import (
    RUNS_DIR,
    UPLOADS_DIR,
    create_job,
    get_job,
    start_tracking_job,
)
from inference_engine import TRACKING_BACKEND

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

BY_HASH_DIR = UPLOADS_DIR / "by-hash"

BALL_POSITIONS_CACHE = {}
BALL_POSITIONS_LOCK = threading.Lock()
RUN_HITS_CACHE = {}
JUDGE_HIT_FRAME_TOLERANCE = 2
FRONT_WALL_OUT_HEIGHT_FT = 4.57 * 3.280839895
FRONT_WALL_TIN_HEIGHT_FT = 0.48 * 3.280839895
FRONT_WALL_SERVICE_HEIGHT_FT = 1.78 * 3.280839895
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def error_response(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps else 0,
    }


def public_job(job):
    total_frames = max(1, int(job.get("total_frames", 1)))
    processed_frames = int(job.get("processed_frames", 0))
    progress = min(100.0, processed_frames / total_frames * 100)
    response = {
        "ok": True,
        "status": job.get("status", "queued"),
        "run_id": job.get("run_id"),
        "stage": job.get("stage"),
        "start_frame": job.get("start_frame"),
        "end_frame": job.get("end_frame"),
        "fps": job.get("fps"),
        "frame_stride": job.get("frame_stride", 1),
        "inference_width": job.get("inference_width", 0),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "progress": progress,
        "message": job.get("message", ""),
    }

    for key in (
        "rows",
        "hits",
        "target_zones",
        "floor_zones",
        "calibration_warning",
        "hits_error",
        "annotated_video_url",
        "csv_url",
        "error",
    ):
        if key in job:
            response[key] = job[key]

    return response


def get_ball_positions(run_id, csv_path):
    mtime = csv_path.stat().st_mtime_ns
    with BALL_POSITIONS_LOCK:
        cached = BALL_POSITIONS_CACHE.get(run_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    positions = load_ball_positions(csv_path)
    with BALL_POSITIONS_LOCK:
        BALL_POSITIONS_CACHE[run_id] = (mtime, positions)
    return positions


def get_run_hits(run_id, run_dir):
    hits_path = run_dir / "detected_hits.json"
    if not hits_path.exists():
        return []

    mtime = hits_path.stat().st_mtime_ns
    with BALL_POSITIONS_LOCK:
        cached = RUN_HITS_CACHE.get(run_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        hits = json.loads(hits_path.read_text(encoding="utf-8")).get("hits", [])
    except (OSError, json.JSONDecodeError):
        hits = []

    with BALL_POSITIONS_LOCK:
        RUN_HITS_CACHE[run_id] = (mtime, hits)
    return hits


def find_hit_impact_near_frame(run_id, run_dir, frame):
    best = None
    for hit in get_run_hits(run_id, run_dir):
        impact = hit.get("impact")
        if impact is None:
            continue
        distance = abs(int(hit.get("frame", -10**9)) - frame)
        if distance <= JUDGE_HIT_FRAME_TOLERANCE and (best is None or distance < best[0]):
            best = (distance, impact)
    return best[1] if best else None


def front_wall_hits_from_payload(payload):
    hits = payload.get("hits", [])
    return [
        hit for hit in hits
        if hit.get("target_zone") is not None
        and hit.get("wall_diagram") is not None
        and hit.get("event_type") in (None, "wall", "unknown")
    ]


def average(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def rounded(value, digits=1):
    return None if value is None else round(float(value), digits)


def wall_height_from_diagram_y(y):
    y = max(0.0, min(1.0, float(y)))
    return FRONT_WALL_OUT_HEIGHT_FT - y * (FRONT_WALL_OUT_HEIGHT_FT - FRONT_WALL_TIN_HEIGHT_FT)


def zone_lookup(zones):
    return {int(zone["zone"]): zone for zone in zones or [] if "zone" in zone}


def build_coaching_analytics(payload):
    target_summary = payload.get("target_zones") or {}
    floor_summary = payload.get("floor_zones") or {}
    front_wall_hits = front_wall_hits_from_payload(payload)
    target_zones = zone_lookup(target_summary.get("zones"))

    heights_ft = [
        wall_height_from_diagram_y(hit["wall_diagram"]["y"])
        for hit in front_wall_hits
        if hit.get("wall_diagram") and hit["wall_diagram"].get("y") is not None
    ]
    speed_before_mph = [
        hit.get("velocity", {}).get("speed_before", {}).get("mph")
        for hit in front_wall_hits
        if hit.get("velocity")
    ]
    speed_after_mph = [
        hit.get("velocity", {}).get("speed_after", {}).get("mph")
        for hit in front_wall_hits
        if hit.get("velocity")
    ]
    speed_change_mph = [
        hit.get("velocity", {}).get("velocity_change", {}).get("mph")
        for hit in front_wall_hits
        if hit.get("velocity")
    ]

    total_wall_hits = int(target_summary.get("total_wall_hits") or len(front_wall_hits))
    center_hits = sum(int(target_zones.get(zone, {}).get("count", 0)) for zone in (4, 5))
    side_hits = sum(int(target_zones.get(zone, {}).get("count", 0)) for zone in (1, 2, 3))
    low_hits = sum(int(target_zones.get(zone, {}).get("count", 0)) for zone in (3, 5))
    high_hits = sum(int(target_zones.get(zone, {}).get("count", 0)) for zone in (1, 4))
    calls = [hit.get("call") for hit in front_wall_hits]

    return {
        "total_wall_hits": total_wall_hits,
        "total_floor_bounces": int(floor_summary.get("total_floor_bounces") or 0),
        "common_target_zones": target_summary.get("common_zones") or [],
        "missing_target_zones": target_summary.get("missing_zones") or [],
        "common_floor_zones": floor_summary.get("common_zones") or [],
        "missing_floor_zones": floor_summary.get("missing_zones") or [],
        "target_zone_percentages": [
            {
                "zone": int(zone.get("zone")),
                "count": int(zone.get("count", 0)),
                "percentage": rounded(zone.get("percentage", 0.0), 1),
            }
            for zone in (target_summary.get("zones") or [])
        ],
        "average_wall_height_ft": rounded(average(heights_ft), 1),
        "average_wall_height_reference": {
            "tin_ft": round(FRONT_WALL_TIN_HEIGHT_FT, 1),
            "service_line_ft": round(FRONT_WALL_SERVICE_HEIGHT_FT, 1),
            "out_line_ft": round(FRONT_WALL_OUT_HEIGHT_FT, 1),
        },
        "average_incoming_speed_mph": rounded(average(speed_before_mph), 1),
        "average_exit_speed_mph": rounded(average(speed_after_mph), 1),
        "average_velocity_change_mph": rounded(average(speed_change_mph), 1),
        "max_incoming_speed_mph": rounded(max(speed_before_mph), 1) if speed_before_mph else None,
        "center_target_rate": rounded(center_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "side_target_rate": rounded(side_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "low_target_rate": rounded(low_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "high_target_rate": rounded(high_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "in_count": sum(1 for call in calls if call == "IN"),
        "out_count": sum(1 for call in calls if call == "OUT"),
    }


def local_coaching_feedback(analytics):
    if not analytics.get("total_wall_hits"):
        return (
            "No reliable front-wall target pattern was detected yet. Start by reviewing the "
            "bounce labels and making sure the wall-hit detector is finding the main rally shots."
        )

    notes = []
    center_rate = analytics.get("center_target_rate")
    side_rate = analytics.get("side_target_rate")
    low_rate = analytics.get("low_target_rate")
    avg_height = analytics.get("average_wall_height_ft")
    avg_speed = analytics.get("average_incoming_speed_mph")
    common = analytics.get("common_target_zones") or []
    missing = analytics.get("missing_target_zones") or []

    if common:
        notes.append(
            f"Your most common front-wall target was zone {common[0]['zone']} "
            f"({common[0]['percentage']:.0f}% of detected wall hits)."
        )
    if center_rate is not None and center_rate >= 45:
        notes.append(
            f"{center_rate:.0f}% of shots went through the center targets. Mix in more width to "
            "move the opponent off the T instead of feeding the middle."
        )
    elif side_rate is not None and side_rate >= 55:
        notes.append(
            f"{side_rate:.0f}% of shots used side targets, which is a useful pattern for creating width."
        )
    if low_rate is not None and low_rate >= 45:
        notes.append(
            f"{low_rate:.0f}% of shots were in lower target zones. That can be attacking, but "
            "watch error risk if those are not intentional drops or drives."
        )
    if avg_height is not None:
        if avg_height < FRONT_WALL_SERVICE_HEIGHT_FT:
            notes.append(
                f"Average wall height was {avg_height:.1f} ft, below the service line. "
                "That suggests flatter, more attacking contact."
            )
        else:
            notes.append(
                f"Average wall height was {avg_height:.1f} ft. Look for chances to vary height "
                "between safer length and lower attacking drives."
            )
    if avg_speed is not None:
        notes.append(f"Average incoming ball speed was about {avg_speed:.1f} mph.")
    if missing:
        shown = ", ".join(str(zone["zone"]) for zone in missing[:3])
        notes.append(f"Unused or rarely used target zones include {shown}; those are good practice targets.")

    return " ".join(notes[:5])


def extract_openai_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def llm_coaching_feedback(analytics):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "missing_api_key"

    model = os.getenv("OPENAI_COACH_MODEL", "gpt-5-mini")
    prompt = (
        "You are a squash coach. Give concise, practical feedback from these "
        "automated match analytics. Mention limitations if sample size is small. "
        "Return 3 short bullets and one practice focus.\n\n"
        + json.dumps(analytics, indent=2)
    )
    body = json.dumps(
        {
            "model": model,
            "input": prompt,
            "text": {"verbosity": "low"},
            "max_output_tokens": 450,
        }
    ).encode("utf-8")
    request_obj = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return None, "request_failed"

    text = extract_openai_text(data)
    return (text or None), ("ok" if text else "empty_response")


@app.get("/")
def index():
    response = send_file(ROOT / "index.html", max_age=0)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "version": APP_VERSION,
            "root": str(ROOT),
            "tracking_backend": TRACKING_BACKEND,
            "default_device": os.environ.get("DEFAULT_DEVICE"),
            "onnx_providers": os.environ.get("ONNXRUNTIME_EXECUTION_PROVIDERS"),
        }
    )


def video_path_for_id(video_id):
    video_id = secure_filename(str(video_id))
    if not video_id:
        return None

    matches = sorted(BY_HASH_DIR.glob(f"{video_id}.*"))
    return matches[0] if matches else None


@app.post("/api/upload")
def upload_video():
    video_file = request.files.get("video_file")
    if video_file is None or not video_file.filename:
        return error_response("No video file provided.")

    suffix = Path(secure_filename(video_file.filename)).suffix or ".mp4"
    BY_HASH_DIR.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    tmp_path = BY_HASH_DIR / f"upload-{int(time.time() * 1000)}-{threading.get_ident()}.tmp"
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = video_file.stream.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)

        video_id = hasher.hexdigest()
        final_path = BY_HASH_DIR / f"{video_id}{suffix}"
        if final_path.exists():
            tmp_path.unlink()
        else:
            os.replace(tmp_path, final_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    payload = {"ok": True, "video_id": video_id}
    try:
        info = video_info(final_path)
        payload.update(
            fps=info["fps"],
            frame_count=info["frame_count"],
            duration=info["duration"],
        )
    except ValueError:
        pass

    return jsonify(payload)


@app.get("/api/court-model")
def get_court_model():
    """Court dimensions, landmarks, and wireframe for the calibration wizard."""
    return jsonify({"ok": True, **court_model.court_model_public()})


@app.post("/api/camera-check")
def camera_check():
    """Run the full camera solve on a candidate calibration and report health.

    Read-only wizard feedback: nothing is stored. Always 200 with a status
    field, mirroring solve_camera_model's never-raise contract.
    """
    payload = request.get_json(silent=True) or {}
    calibration = payload.get("calibration")
    if calibration is None:
        try:
            calibration = json.loads(payload.get("calibration_json") or "")
        except (json.JSONDecodeError, TypeError):
            return jsonify({"ok": True, "status": "invalid_json"})
    _, info = court_model.solve_camera_model(calibration)
    return jsonify({"ok": True, **info})


def validate_floor_calibration(calibration):
    """Return a warning string (and strip the floor plane) if it cannot be used.

    Floor mapping is additive: a bad floor plane must never fail the run or
    regress front-wall judging, so we drop it and surface a warning instead.
    """
    planes = calibration.get("planes")
    if not isinstance(planes, dict) or "floor" not in planes:
        return None

    try:
        floor_map = court_model.load_floor_calibration(calibration)
    except Exception:
        floor_map = None
    if floor_map is not None:
        return None

    planes.pop("floor", None)
    return "Floor calibration was invalid and has been ignored for this run."


@app.post("/api/track")
def track_clip():
    video_file = request.files.get("video_file")
    video_id = request.form.get("video_id", "").strip()
    calibration_text = request.form.get("calibration_json", "")

    if not video_id and (video_file is None or not video_file.filename):
        return error_response("Upload the source video before tracking.")

    try:
        calibration = json.loads(calibration_text)
    except json.JSONDecodeError:
        return error_response("Calibration JSON was invalid.")

    try:
        start_time = float(request.form.get("start_time", "0"))
        end_time = float(request.form.get("end_time", "0"))
        frame_stride = int(request.form.get("frame_stride", "4"))
        inference_width = int(request.form.get("inference_width", "960"))
    except ValueError:
        return error_response(
            "Clip start/end times, frame stride, and inference width must be numbers."
        )

    if end_time <= start_time:
        return error_response("Clip end must be after clip start.")

    if frame_stride < 1 or frame_stride > 10:
        return error_response("Frame stride must be between 1 and 10.")

    if inference_width not in {0, 640, 960, 1280}:
        return error_response("Inference width must be 0, 640, 960, or 1280.")

    # Debug-only override of the wall-hit event engine; empty falls back to the
    # job_runner default.
    event_engine = request.form.get("event_engine", "").strip()
    if event_engine not in {"", "votes", "gb_model", "fusion"}:
        return error_response("Event engine must be votes, gb_model, or fusion.")

    # Experimental 3D contact detection (fusion engine only; needs a solvable
    # camera from the calibration — degrades to 2D per-run otherwise).
    fusion_3d = request.form.get("fusion_3d", "").strip()
    if fusion_3d not in {"", "1"}:
        return error_response("fusion_3d must be empty or 1.")

    run_id = str(int(time.time() * 1000))
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if video_id:
        video_path = video_path_for_id(video_id)
        if video_path is None:
            return error_response("Uploaded video was not found. Upload it again.", status=404)
    else:
        upload_dir = UPLOADS_DIR / run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_name = secure_filename(video_file.filename) or "source_video.mp4"
        video_path = upload_dir / video_name
        video_file.save(video_path)

    try:
        info = video_info(video_path)
    except ValueError as error:
        return error_response(str(error))

    start_frame = max(0, int(round(start_time * info["fps"])))
    end_frame = min(info["frame_count"] - 1, int(round(end_time * info["fps"])))
    if end_frame < start_frame:
        return error_response("Selected clip is outside the video duration.")

    calibration_warning = validate_floor_calibration(calibration)
    if calibration_warning:
        app.logger.warning("run %s: %s", run_id, calibration_warning)

    calibration_path = run_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    selected_frames = end_frame - start_frame + 1
    total_frames = (selected_frames + frame_stride - 1) // frame_stride
    extra_job_fields = {}
    if calibration_warning:
        extra_job_fields["calibration_warning"] = calibration_warning
    create_job(
        run_id,
        run_dir,
        status="queued",
        message="Queued tracking job.",
        video_path=str(video_path),
        start_frame=start_frame,
        end_frame=end_frame,
        fps=info["fps"],
        frame_stride=frame_stride,
        inference_width=inference_width,
        event_engine=event_engine or None,
        fusion_3d=fusion_3d == "1",
        processed_frames=0,
        total_frames=total_frames,
        csv_url=f"/api/runs/{run_id}/ball_coordinates.csv",
        **extra_job_fields,
    )
    start_tracking_job(run_id)

    return jsonify(public_job(get_job(run_id)))


@app.get("/api/track/status/<run_id>")
def track_status(run_id):
    job = get_job(secure_filename(run_id))
    if job is None:
        return error_response("Tracking job was not found.", status=404)

    return jsonify(public_job(job))


@app.post("/api/judge")
def judge_frame():
    data = request.get_json(silent=True) or {}
    run_id = str(data.get("run_id", "")).strip()
    if not run_id:
        return error_response("Missing run_id.")

    try:
        frame = int(data.get("frame"))
    except (TypeError, ValueError):
        return error_response("Frame must be an integer.")

    run_dir = RUNS_DIR / secure_filename(run_id)
    csv_path = run_dir / "ball_coordinates.csv"
    calibration_path = run_dir / "calibration.json"

    if not csv_path.exists() or not calibration_path.exists():
        return error_response("Tracking result was not found.", status=404)

    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        top_line, bottom_line = load_calibration_lines(calibration)
        wall_corners = load_wall_corners(calibration)

        impact = find_hit_impact_near_frame(run_id, run_dir, frame)
        if impact is not None:
            ball = Point(float(impact["x"]), float(impact["y"]))
            source = "impact_estimate"
        else:
            ball = get_ball_positions(run_id, csv_path).get(frame)
            if ball is None:
                raise ValueError(f"No ball detection recorded for frame {frame}.")
            source = "detected_center"

        call, reason, top_y, bottom_y = judge_ball(ball, top_line, bottom_line, wall_corners)
        margin_px = judge_margin_px(ball, top_line, bottom_line, wall_corners)
        diagram = wall_diagram_coordinates(
            ball,
            top_line,
            bottom_line,
            calibration.get("frame_width", 1),
            wall_corners,
        )
    except Exception as error:
        return error_response(str(error))

    return jsonify(
        {
            "ok": True,
            "frame": frame,
            "call": call,
            "reason": reason,
            "source": source,
            "margin_px": margin_px,
            "ball": {"x": ball.x, "y": ball.y},
            "top_y": top_y,
            "bottom_y": bottom_y,
            "wall_diagram": {
                "x": diagram["x"],
                "y": diagram["y"],
                "x_span": diagram["x_span"],
                "x_reference": diagram.get("x_reference"),
                "y_reference": "0 is the out-line lower edge; 1 is the tin top edge",
            },
            "outside_line_span": (
                not wall_corners.contains_point(ball)
                if wall_corners is not None
                else (not top_line.contains_x(ball.x) or not bottom_line.contains_x(ball.x))
            ),
        }
    )


@app.get("/api/runs/<run_id>/coach")
def coach_run(run_id):
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    hits_path = run_dir / "detected_hits.json"
    if not hits_path.exists():
        return error_response("Run analytics were not found.", status=404)

    try:
        payload = json.loads(hits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return error_response("Run analytics could not be read.", status=500)

    analytics = build_coaching_analytics(payload)
    feedback, llm_status = llm_coaching_feedback(analytics)
    source = "llm" if feedback else "local"
    if not feedback:
        feedback = local_coaching_feedback(analytics)

    return jsonify(
        {
            "ok": True,
            "analytics": analytics,
            "feedback": feedback,
            "feedback_source": source,
            "llm_status": llm_status,
        }
    )


@app.get("/api/runs/<run_id>/<path:filename>")
def run_file(run_id, filename):
    run_dir = RUNS_DIR / secure_filename(run_id)
    return send_from_directory(run_dir, filename, as_attachment=False)


@app.get("/api/runs/<run_id>/source_video")
def run_source_video(run_id):
    """Stream a run's source clip so the UI can rehydrate after a page refresh.

    The clip lives outside the run dir (uploads/by-hash or uploads/<run_id>),
    so resolve it through the persisted job rather than run_file above.
    send_file(conditional=True) honors Range requests, which <video> seeking
    needs.
    """
    job = get_job(secure_filename(run_id))
    if job is None:
        return error_response("Run was not found.", status=404)

    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        return error_response("Source clip is no longer on the server.", status=404)

    return send_file(Path(video_path), conditional=True)


@app.get("/api/dev/index-mtime")
def index_mtime():
    """Dev live-reload probe: the UI polls this and reloads the tab on change."""
    try:
        mtime = (ROOT / "index.html").stat().st_mtime_ns
    except OSError:
        mtime = 0
    return jsonify({"ok": True, "mtime": mtime, "debug": bool(app.debug)})


GROUND_TRUTH_TYPES = {"wall", "racket", "floor", "side_wall"}


@app.post("/api/runs/<run_id>/ground_truth")
def save_ground_truth(run_id):
    """Persist user-labeled bounce events for a run (the eval ground truth)."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True) or {}
    events = data.get("events")
    if not isinstance(events, list):
        return error_response("Body must include an events list.")

    cleaned = []
    for event in events:
        try:
            frame = int(event["frame"])
            kind = str(event["type"])
        except (KeyError, TypeError, ValueError):
            return error_response("Each event needs an integer frame and a type.")
        if kind not in GROUND_TRUTH_TYPES:
            return error_response(f"Unknown event type: {kind}")
        cleaned.append({"frame": frame, "type": kind})
    cleaned.sort(key=lambda event: event["frame"])

    payload = {"tolerance_frames": 1, "events": cleaned}
    (run_dir / "ground_truth.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "count": len(cleaned)})


CORRECTION_SCHEMA_VERSION = "corrections-v2"
CORRECTION_TYPES = GROUND_TRUTH_TYPES | {"none"}
CORRECTION_CALLS = {"IN", "OUT"}


def parse_ball_point(value):
    """-> {x, y} floats, or raise ValueError. None passes through."""
    if value is None:
        return None
    try:
        return {"x": float(value["x"]), "y": float(value["y"])}
    except (KeyError, TypeError, ValueError):
        raise ValueError("ball must be {x, y} numbers")


def parse_corrected(data):
    """Validate the human half of a correction -> dict, or raise ValueError.

    type drives which other fields are legal: call is wall-only, and a
    'none' (detector false positive) carries no position or timing —
    there is no bounce to locate."""
    hit_type = str(data.get("type", "")).lower()
    if hit_type not in CORRECTION_TYPES:
        raise ValueError(
            "corrected.type must be one of " + ", ".join(sorted(CORRECTION_TYPES))
        )

    call = data.get("call")
    if hit_type == "wall":
        call = None if call is None else str(call).upper()
        if call not in CORRECTION_CALLS:
            raise ValueError("corrected.call must be IN or OUT for wall hits.")
    elif call is not None:
        raise ValueError("corrected.call applies to front-wall corrections only.")

    if hit_type == "none":
        if data.get("ball") is not None:
            raise ValueError("a 'none' correction cannot carry a ball position.")
        return {"type": hit_type, "call": None, "ball": None,
                "frame_is_bounce": None, "frame": None}

    ball = parse_ball_point(data.get("ball"))
    if ball is None:
        raise ValueError("corrected.ball {x, y} is required unless type is none.")
    frame_is_bounce = data.get("frame_is_bounce")
    if not isinstance(frame_is_bounce, bool):
        raise ValueError("corrected.frame_is_bounce must be true or false.")
    corrected_frame = data.get("frame")
    if frame_is_bounce:
        if corrected_frame is not None:
            raise ValueError("corrected.frame is only valid when "
                             "frame_is_bounce is false.")
    else:
        try:
            corrected_frame = int(corrected_frame)
        except (TypeError, ValueError):
            raise ValueError("corrected.frame (int) is required when "
                             "frame_is_bounce is false.")
    return {"type": hit_type, "call": call, "ball": ball,
            "frame_is_bounce": frame_is_bounce, "frame": corrected_frame}


def parse_predicted(data):
    """Normalize the label-time model snapshot -> dict, or raise ValueError.
    Every field is nullable; this input is programmatic, so malformed
    values are client bugs and rejected rather than coerced to null."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("predicted must be an object.")
    hit_type = data.get("type")
    if hit_type is not None:
        hit_type = str(hit_type).lower()
        if hit_type not in CORRECTION_TYPES - {"none"}:
            raise ValueError("predicted.type must be a detector hit type.")
    call = data.get("call")
    if call is not None:
        call = str(call).upper()
        if call not in CORRECTION_CALLS | {"UNKNOWN"}:
            raise ValueError("predicted.call must be IN, OUT, or UNKNOWN.")
    margin = data.get("margin_px")
    if margin is not None:
        try:
            margin = float(margin)
        except (TypeError, ValueError):
            raise ValueError("predicted.margin_px must be a number.")
    source = data.get("source")
    return {"type": hit_type, "call": call,
            "source": None if source is None else str(source),
            "margin_px": margin, "ball": parse_ball_point(data.get("ball"))}


def correction_agreement(corrected, predicted):
    """Server-derived agreement flags; null where comparison is undefined."""
    return {
        "type": (None if predicted["type"] is None
                 else predicted["type"] == corrected["type"]),
        "call": (predicted["call"] == corrected["call"]
                 if corrected["type"] == "wall" and corrected["call"]
                 and predicted["call"] in CORRECTION_CALLS else None),
        "frame": corrected["frame_is_bounce"],
    }


@app.get("/api/runs/<run_id>/corrections")
def get_corrections(run_id):
    """The UI's load path: an empty list (not a 404) when nothing is recorded
    yet, so every fresh run doesn't log a console error."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)
    try:
        data = json.loads((run_dir / "corrections.json").read_text(encoding="utf-8"))
        corrections = data.get("corrections", [])
        schema_version = data.get("schema_version")
    except (OSError, json.JSONDecodeError):
        corrections = []
        schema_version = None
    return jsonify({"ok": True, "schema_version": schema_version,
                    "corrections": corrections})


@app.post("/api/runs/<run_id>/corrections")
def save_correction(run_id):
    """Record a human bounce correction: hit type, ball position, and bounce
    timing (plus IN/OUT for wall hits). One correction per frame, latest
    wins; corrected null removes the frame's entry (undo). corrections.json
    is the raw feed for the eval set, so each entry keeps the model's
    label-time snapshot alongside the human values."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True) or {}
    try:
        frame = int(data["frame"])
    except (KeyError, TypeError, ValueError):
        return error_response("Correction needs an integer frame.")

    corrected_data = data.get("corrected")
    entry = None
    if corrected_data is not None:
        if not isinstance(corrected_data, dict):
            return error_response("corrected must be an object or null to undo.")
        try:
            corrected = parse_corrected(corrected_data)
            predicted = parse_predicted(data.get("predicted"))
        except ValueError as error:
            return error_response(str(error))
        entry = {
            "frame": frame,
            "corrected": corrected,
            "predicted": predicted,
            "agrees": correction_agreement(corrected, predicted),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "app_version": APP_VERSION,
        }

    corrections_path = run_dir / "corrections.json"
    try:
        existing = json.loads(corrections_path.read_text(encoding="utf-8"))
        corrections = existing.get("corrections", [])
    except (OSError, json.JSONDecodeError):
        corrections = []
    corrections = [c for c in corrections if c.get("frame") != frame]
    if entry is not None:
        corrections.append(entry)

    corrections.sort(key=lambda c: c.get("frame", 0))
    corrections_path.write_text(
        json.dumps({"schema_version": CORRECTION_SCHEMA_VERSION,
                    "corrections": corrections}, indent=2),
        encoding="utf-8",
    )
    return jsonify({"ok": True, "count": len(corrections), "correction": entry})


@app.post("/api/label_runs")
def create_label_run():
    """A run directory for labeling only: video reference and frame range,
    no tracking. The id is deterministic per video hash so every labeling
    session for the same clip lands in one place, and a later tracking or
    training pass can locate the source video from label_run.json."""
    data = request.get_json(silent=True) or {}
    video_id = secure_filename(str(data.get("video_id", "")).strip())
    if not video_id:
        return error_response("Missing video_id.")
    matches = sorted(BY_HASH_DIR.glob(f"{video_id}.*"))
    if not matches:
        return error_response("Uploaded video was not found. Upload it again.", status=404)
    try:
        info = video_info(matches[0])
    except ValueError as error:
        return error_response(str(error))

    run_id = f"label-{video_id[:12]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "video_path": str(matches[0]),
        "fps": info["fps"],
        "start_frame": 0,
        "end_frame": max(0, int(info["frame_count"]) - 1),
        "label_only": True,
    }
    (run_dir / "label_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return jsonify(meta)


@app.get("/fonts/<path:filename>")
def font_file(filename):
    return send_from_directory(ROOT / "fonts", filename, max_age=86400)


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SquashAnalytics {APP_VERSION} from {ROOT}")
    print(f"TRACKING_BACKEND={TRACKING_BACKEND} DEFAULT_DEVICE={os.environ.get('DEFAULT_DEVICE')}")
    print(f"ONNXRUNTIME_EXECUTION_PROVIDERS={os.environ.get('ONNXRUNTIME_EXECUTION_PROVIDERS')}")
    print(f"Open http://127.0.0.1:{port}/ on this Mac.")
    if host == "127.0.0.1":
        print(f"For phone access, restart with: HOST=0.0.0.0 PORT={port} python app.py")
    app.run(host=host, port=port, debug=False)
