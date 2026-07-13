import hashlib
import json
import os
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from judge_call import (
    Point,
    calibration_wall_x_bounds,
    judge_ball,
    load_ball_positions,
    load_calibration_lines,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

BY_HASH_DIR = UPLOADS_DIR / "by-hash"

BALL_POSITIONS_CACHE = {}
BALL_POSITIONS_LOCK = threading.Lock()
RUN_HITS_CACHE = {}
JUDGE_HIT_FRAME_TOLERANCE = 2


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


@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


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

    return jsonify({"ok": True, "video_id": video_id})


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
        frame_stride = int(request.form.get("frame_stride", "1"))
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

    calibration_path = run_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    selected_frames = end_frame - start_frame + 1
    total_frames = (selected_frames + frame_stride - 1) // frame_stride
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
        processed_frames=0,
        total_frames=total_frames,
        csv_url=f"/api/runs/{run_id}/ball_coordinates.csv",
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

        impact = find_hit_impact_near_frame(run_id, run_dir, frame)
        if impact is not None:
            ball = Point(float(impact["x"]), float(impact["y"]))
            source = "impact_estimate"
        else:
            ball = get_ball_positions(run_id, csv_path).get(frame)
            if ball is None:
                raise ValueError(f"No ball detection recorded for frame {frame}.")
            source = "detected_center"

        call, reason, top_y, bottom_y = judge_ball(ball, top_line, bottom_line)
        margin_px = min(ball.y - top_y, bottom_y - ball.y)
        wall_left, wall_right = calibration_wall_x_bounds(
            top_line,
            bottom_line,
            calibration.get("frame_width", 1),
        )
        wall_x = (ball.x - wall_left) / (wall_right - wall_left)
        wall_y = (ball.y - top_y) / (bottom_y - top_y)
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
                "x": wall_x,
                "y": wall_y,
                "x_span": [wall_left, wall_right],
                "y_reference": "0 is the out-line lower edge; 1 is the tin top edge",
            },
            "outside_line_span": (
                not top_line.contains_x(ball.x) or not bottom_line.contains_x(ball.x)
            ),
        }
    )


@app.get("/api/runs/<run_id>/<path:filename>")
def run_file(run_id, filename):
    run_dir = RUNS_DIR / secure_filename(run_id)
    return send_from_directory(run_dir, filename, as_attachment=False)


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
