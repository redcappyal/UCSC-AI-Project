import csv
import json
import os
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from judge_call import Line, Point, judge_ball, load_ball_position
from train_wall_hit_model import WINDOW_RADIUS, build_feature_vector, load_ball_positions

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "ui_runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
WALL_HIT_MODEL_PATH = ROOT / "wall_hit_model.pkl"
ROBOFLOW_CACHE_DIR = ROOT / ".roboflow-cache"

os.environ.setdefault("MODEL_CACHE_DIR", str(ROBOFLOW_CACHE_DIR))
os.environ.setdefault("METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

from local_model_eval import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    DEFAULT_MODEL_ID,
    ball_csv_row,
    infer_frame,
    normalize_predictions,
    select_ball_prediction,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
JOBS = {}
JOBS_LOCK = threading.Lock()
TRACKING_MODEL = None
TRACKING_MODEL_ID = None
TRACKING_MODEL_LOCK = threading.Lock()
TRACKING_MODEL_LOAD_LOCK = threading.Lock()


def error_response(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def load_calibration_lines(calibration):
    lines = {line.get("name"): line for line in calibration.get("lines", [])}
    top = lines.get("out_line_lower_edge")
    bottom = lines.get("tin_top_edge")

    if top is None or bottom is None:
        raise ValueError("Calibration must include out_line_lower_edge and tin_top_edge.")

    def line_from_calibration(line):
        endpoints = line["endpoints"]
        return Line(
            Point(float(endpoints[0][0]), float(endpoints[0][1])),
            Point(float(endpoints[1][0]), float(endpoints[1][1])),
        )

    return line_from_calibration(top), line_from_calibration(bottom)


def line_x_bounds(line):
    return (
        min(line.left.x, line.right.x),
        max(line.left.x, line.right.x),
    )


def calibration_wall_x_bounds(top_line, bottom_line, frame_width):
    top_min, top_max = line_x_bounds(top_line)
    bottom_min, bottom_max = line_x_bounds(bottom_line)
    left = max(top_min, bottom_min)
    right = min(top_max, bottom_max)

    if right <= left:
        return 0.0, max(1.0, float(frame_width or 1))

    return left, right


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


def count_csv_rows(path):
    with path.open(newline="") as csv_file:
        return max(0, sum(1 for _ in csv_file) - 1)


def update_job(job_id, **updates):
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def public_job(job):
    total_frames = max(1, int(job.get("total_frames", 1)))
    processed_frames = int(job.get("processed_frames", 0))
    progress = min(100.0, processed_frames / total_frames * 100)
    response = {
        "ok": True,
        "status": job.get("status", "queued"),
        "run_id": job.get("run_id"),
        "start_frame": job.get("start_frame"),
        "end_frame": job.get("end_frame"),
        "fps": job.get("fps"),
        "frame_stride": job.get("frame_stride", 1),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "progress": progress,
        "message": job.get("message", ""),
    }

    for key in (
        "rows",
        "hit_prediction",
        "annotated_video_url",
        "csv_url",
        "error",
    ):
        if key in job:
            response[key] = job[key]

    return response


def predict_wall_hit_frame(csv_path):
    if not WALL_HIT_MODEL_PATH.exists():
        return None

    positions = load_ball_positions(csv_path)
    frames = np.array(sorted(positions), dtype=np.int64)
    if len(frames) == 0:
        return None

    X = np.array(
        [build_feature_vector(positions, int(frame), WINDOW_RADIUS) for frame in frames],
        dtype=np.float32,
    )

    with WALL_HIT_MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        scores = 1 / (1 + np.exp(-raw_scores))
    else:
        scores = model.predict(X).astype(float)

    best_index = int(np.argmax(scores))
    return {
        "frame": int(frames[best_index]),
        "score": float(scores[best_index]),
    }


def get_tracking_model():
    global TRACKING_MODEL, TRACKING_MODEL_ID

    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key.strip():
        raise RuntimeError("No Roboflow API key found. Set ROBOFLOW_API_KEY in .env.")

    with TRACKING_MODEL_LOAD_LOCK:
        if TRACKING_MODEL is not None and TRACKING_MODEL_ID == model_id:
            return TRACKING_MODEL

        from inference import get_model

        TRACKING_MODEL = get_model(
            model_id=model_id,
            api_key=api_key,
            countinference=False,
        )
        TRACKING_MODEL_ID = model_id
        return TRACKING_MODEL


def run_tracking_job(run_id, video_path, start_frame, end_frame, frame_stride, csv_path):
    processed_frames = 0

    try:
        update_job(run_id, status="running", message="Loading local model...")
        model = get_tracking_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        read_count = start_frame

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()

            while read_count <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                if (read_count - start_frame) % frame_stride != 0:
                    read_count += 1
                    continue

                with TRACKING_MODEL_LOCK:
                    result = infer_frame(model, frame, CONFIDENCE_THRESHOLD)
                predictions = normalize_predictions(result)
                ball_prediction = select_ball_prediction(predictions)
                csv_writer.writerow(ball_csv_row(read_count, source_fps, ball_prediction))

                processed_frames += 1
                detected_text = "detected" if ball_prediction is not None else "not detected"
                message = f"Processed source frame {read_count} -> {detected_text}"
                update_job(
                    run_id,
                    processed_frames=processed_frames,
                    message=message,
                )
                read_count += 1
    except Exception as error:
        update_job(
            run_id,
            status="failed",
            error=f"Tracking failed.\n\n{error}",
            message="Tracking failed.",
        )
        return
    finally:
        if "cap" in locals():
            cap.release()

    hit_prediction = None
    try:
        hit_prediction = predict_wall_hit_frame(csv_path)
    except Exception as error:
        hit_prediction = {"error": str(error)}

    job = get_job(run_id) or {}
    total_frames = int(job.get("total_frames", processed_frames or 1))
    update_job(
        run_id,
        status="complete",
        processed_frames=total_frames,
        rows=count_csv_rows(csv_path),
        hit_prediction=hit_prediction,
        message="Tracking complete.",
    )


@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/track")
def track_clip():
    video_file = request.files.get("video_file")
    calibration_text = request.form.get("calibration_json", "")

    if video_file is None or not video_file.filename:
        return error_response("Upload the source video before tracking.")

    try:
        calibration = json.loads(calibration_text)
    except json.JSONDecodeError:
        return error_response("Calibration JSON was invalid.")

    try:
        start_time = float(request.form.get("start_time", "0"))
        end_time = float(request.form.get("end_time", "0"))
        frame_stride = int(request.form.get("frame_stride", "1"))
    except ValueError:
        return error_response("Clip start/end times and frame stride must be numbers.")

    if end_time <= start_time:
        return error_response("Clip end must be after clip start.")

    if frame_stride < 1 or frame_stride > 10:
        return error_response("Frame stride must be between 1 and 10.")

    run_id = str(int(time.time() * 1000))
    run_dir = RUNS_DIR / run_id
    upload_dir = UPLOADS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
    csv_path = run_dir / "ball_coordinates.csv"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    selected_frames = end_frame - start_frame + 1
    total_frames = (selected_frames + frame_stride - 1) // frame_stride
    update_job(
        run_id,
        run_id=run_id,
        status="queued",
        message="Queued tracking job.",
        start_frame=start_frame,
        end_frame=end_frame,
        fps=info["fps"],
        frame_stride=frame_stride,
        processed_frames=0,
        total_frames=total_frames,
        csv_url=f"/api/runs/{run_id}/ball_coordinates.csv",
    )

    thread = threading.Thread(
        target=run_tracking_job,
        args=(run_id, video_path, start_frame, end_frame, frame_stride, csv_path),
        daemon=True,
    )
    thread.start()

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
        ball = load_ball_position(csv_path, frame)
        call, reason, top_y, bottom_y = judge_ball(ball, top_line, bottom_line)
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
    UPLOADS_DIR.mkdir(exist_ok=True)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
