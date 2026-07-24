import shutil
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


def test_calibration_latest_camera_id_filter():
    import app as app_module

    client = app_module.app.test_client()
    run_left = make_run_with_calibration(
        app_module, "cal-camera-id-left",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-left-fin"},
        age_seconds=30)
    run_right = make_run_with_calibration(
        app_module, "cal-camera-id-right",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-right-fin"},
        age_seconds=15)
    run_legacy = make_run_with_calibration(
        app_module, "cal-camera-id-legacy",
        {"schema": "squash-calibration-v2"},  # no camera_id (legacy)
        age_seconds=1)
    try:
        # Unfiltered: newest overall (run_legacy), exactly today's behavior.
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
    finally:
        shutil.rmtree(run_left, ignore_errors=True)
        shutil.rmtree(run_right, ignore_errors=True)
        shutil.rmtree(run_legacy, ignore_errors=True)


def test_calibration_latest_500_on_unreadable_newest():
    import app as app_module

    client = app_module.app.test_client()
    run_dir = make_run_with_calibration(
        app_module, "cal-camera-id-corrupt",
        {"schema": "squash-calibration-v2"},
        age_seconds=0)
    try:
        # Corrupt the newest run's calibration.json after creation so it
        # remains the freshest candidate on disk but fails to parse.
        (run_dir / "calibration.json").write_text(
            "{not valid json", encoding="utf-8")

        response = client.get("/api/calibration/latest")
        assert response.status_code == 500
        assert response.get_json()["ok"] is False
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
