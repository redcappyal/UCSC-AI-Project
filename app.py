import csv
import json
import os
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from detect_wall_hits import detect_hits
from judge_call import Line, Point, judge_ball, load_ball_position

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "ui_runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
ROBOFLOW_CACHE_DIR = ROOT / ".roboflow-cache"
APP_VERSION = "iphone-cpu-2026-07-13-3"

os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ["DEFAULT_DEVICE"] = "cpu"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CPUExecutionProvider]"
os.environ.setdefault("MODEL_CACHE_DIR", str(ROBOFLOW_CACHE_DIR))
os.environ.setdefault("METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

from local_model_eval import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    DEFAULT_MODEL_ID,
    ball_csv_row,
    infer_frame_predictions,
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


def detect_and_judge_hits(run_dir, csv_path):
    detected = detect_hits(csv_path)

    top_line = bottom_line = None
    calibration_path = run_dir / "calibration.json"
    if calibration_path.exists():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            top_line, bottom_line = load_calibration_lines(calibration)
        except (ValueError, json.JSONDecodeError):
            pass

    hits = []
    for hit in detected:
        frame = int(hit["hit_frame"])
        call, reason = "UNKNOWN", "judging_unavailable"
        if top_line is not None:
            try:
                ball = load_ball_position(csv_path, frame)
                call, reason, _, _ = judge_ball(ball, top_line, bottom_line)
            except ValueError as error:
                call, reason = "UNKNOWN", str(error)

        hits.append(
            {
                "frame": frame,
                "timestamp_seconds": hit["timestamp_seconds"],
                "dv_magnitude": hit["dv_magnitude"],
                "after_gap": hit["after_gap"],
                "call": call,
                "reason": reason,
            }
        )

    (run_dir / "detected_hits.json").write_text(
        json.dumps({"hits": hits}, indent=2), encoding="utf-8"
    )
    return hits


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
            device="cpu",
            onnx_execution_providers=["CPUExecutionProvider"],
        )
        TRACKING_MODEL_ID = model_id
        return TRACKING_MODEL


def run_tracking_job(
    run_id,
    video_path,
    start_frame,
    end_frame,
    frame_stride,
    inference_width,
    run_dir,
    csv_path,
):
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
                    predictions = infer_frame_predictions(
                        model,
                        frame,
                        CONFIDENCE_THRESHOLD,
                        inference_width,
                    )
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

    hits = []
    hits_error = None
    try:
        hits = detect_and_judge_hits(run_dir, csv_path)
    except Exception as error:
        hits_error = str(error)

    job = get_job(run_id) or {}
    total_frames = int(job.get("total_frames", processed_frames or 1))
    update_job(
        run_id,
        status="complete",
        processed_frames=total_frames,
        rows=count_csv_rows(csv_path),
        hits=hits,
        hits_error=hits_error,
        message="Tracking complete.",
    )


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
            "default_device": os.environ.get("DEFAULT_DEVICE"),
            "onnx_providers": os.environ.get("ONNXRUNTIME_EXECUTION_PROVIDERS"),
        }
    )


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
        inference_width=inference_width,
        processed_frames=0,
        total_frames=total_frames,
        csv_url=f"/api/runs/{run_id}/ball_coordinates.csv",
    )

    thread = threading.Thread(
        target=run_tracking_job,
        args=(
            run_id,
            video_path,
            start_frame,
            end_frame,
            frame_stride,
            inference_width,
            run_dir,
            csv_path,
        ),
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
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SquashAnalytics {APP_VERSION} from {ROOT}")
    print(f"DEFAULT_DEVICE={os.environ.get('DEFAULT_DEVICE')}")
    print(f"ONNXRUNTIME_EXECUTION_PROVIDERS={os.environ.get('ONNXRUNTIME_EXECUTION_PROVIDERS')}")
    print(f"Open http://127.0.0.1:{port}/ on this Mac.")
    if host == "127.0.0.1":
        print(f"For phone access, restart with: HOST=0.0.0.0 PORT={port} python app.py")
    app.run(host=host, port=port, debug=False)
