"""The single-camera calibration endpoints: /api/camera-model and
/api/calibration/latest.

Was `tests/test_stereo_endpoints.py` -- a misleading name: 4 of its 5 tests
were always single-camera. The one that was not
(`test_camera_pair_check_endpoint`) moved to archive/stereo/tests/ on
2026-07-27 with the endpoint it covered.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from court_model import CameraModel
from synthetic3d import make_camera
from test_calibration_latest import make_run_with_calibration
from test_camera_model import _synthetic_calibration


def _client():
    import app as app_module
    return app_module.app.test_client()


def test_camera_model_endpoint_returns_solved_model():
    client = _client()
    camera = make_camera()
    body = client.post("/api/camera-model",
                       json={"calibration": _synthetic_calibration(camera)}).get_json()
    assert body["ok"] is True and body["status"] == "ok"
    model = CameraModel.from_dict(body["camera_model"])
    # The solved model must project a known court point close to the
    # synthetic camera's own projection.
    point = np.array([10.5, 16.0, 3.0])
    assert np.linalg.norm(
        np.asarray(model.project(point)) - np.asarray(camera.project(point))) < 2.0


def test_camera_model_endpoint_failure_has_no_model_key():
    client = _client()
    body = client.post("/api/camera-model", json={}).get_json()
    assert body["ok"] is True
    assert body["status"] in ("no_frame_size", "invalid_json")
    assert "camera_model" not in body


def test_calibration_latest_camera_id_filter(runs_dir):
    import app as app_module

    client = app_module.app.test_client()
    make_run_with_calibration(
        app_module, "cal-camera-id-left",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-left-fin"},
        age_seconds=30)
    make_run_with_calibration(
        app_module, "cal-camera-id-right",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-right-fin"},
        age_seconds=15)
    make_run_with_calibration(
        app_module, "cal-camera-id-legacy",
        {"schema": "squash-calibration-v2"},  # no camera_id (legacy)
        age_seconds=1)

    # Unfiltered: newest overall (run_legacy), exactly today's behavior. Only
    # meaningful because runs_dir holds nothing but these three.
    body = client.get("/api/calibration/latest").get_json()
    assert body["ok"] is True and body["run_id"] == "cal-camera-id-legacy"

    # Filtered: newest with the matching id, skipping the newer no-id run.
    body = client.get(
        "/api/calibration/latest?camera_id=ucsc-left-fin").get_json()
    assert body["ok"] is True and body["run_id"] == "cal-camera-id-left"
    assert body["calibration"]["camera_id"] == "ucsc-left-fin"

    # Filtered with no match: 404.
    response = client.get("/api/calibration/latest?camera_id=nope")
    assert response.status_code == 404


def test_calibration_latest_500_on_unreadable_newest(runs_dir):
    import app as app_module

    client = app_module.app.test_client()
    run_dir = make_run_with_calibration(
        app_module, "cal-camera-id-corrupt",
        {"schema": "squash-calibration-v2"},
        age_seconds=0)
    # Corrupt the newest run's calibration.json after creation so it remains
    # the freshest candidate on disk but fails to parse.
    (run_dir / "calibration.json").write_text(
        "{not valid json", encoding="utf-8")

    response = client.get("/api/calibration/latest")
    assert response.status_code == 500
    assert response.get_json()["ok"] is False


import cv2

import court_model
from judge_call import load_calibration_lines
from synthetic_court import court_camera, render_court


def _jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buffer.tobytes()


def test_detect_court_endpoint_returns_usable_calibration_structures():
    import io

    client = _client()
    # focal_px=700 keeps the short line inside the frame (the default focal
    # pushes it below frame) without swinging so wide it fabricates a
    # phantom line that outranks the real one (focal_px=500 does that) --
    # see synthetic_court.court_camera and tests/synthetic3d.make_camera.
    image, _ = render_court(court_camera(focal_px=700.0), noise_sigma=2.0)
    payload = {"frames": [(io.BytesIO(_jpeg(image)), f"frame{index}.jpg")
                          for index in range(3)]}

    body = client.post("/api/detect-court", data=payload,
                       content_type="multipart/form-data").get_json()

    assert body["ok"] is True and body["status"] == "ok"
    calibration = {
        "schema": "squash-calibration-v2",
        "frame_width": body["frame_width"],
        "frame_height": body["frame_height"],
        "lines": body["lines"],
        "planes": body["planes"],
        "distortion": None,
    }
    assert load_calibration_lines(calibration) is not None
    assert court_model.load_floor_calibration(calibration) is not None


def test_detect_court_endpoint_is_200_with_a_status_when_given_nothing():
    client = _client()
    body = client.post("/api/detect-court", data={},
                       content_type="multipart/form-data").get_json()
    assert body["ok"] is True and body["status"] == "no_frames"
