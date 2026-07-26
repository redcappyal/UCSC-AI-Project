import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from court_model import CameraModel
from synthetic3d import make_camera
# runs_dir is a fixture; importing it registers it for this module's tests.
from test_calibration_latest import make_run_with_calibration, runs_dir  # noqa: F401
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
        runs_dir, "cal-camera-id-left",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-left-fin"},
        age_seconds=30)
    make_run_with_calibration(
        runs_dir, "cal-camera-id-right",
        {"schema": "squash-calibration-v2", "camera_id": "ucsc-right-fin"},
        age_seconds=15)
    make_run_with_calibration(
        runs_dir, "cal-camera-id-legacy",
        {"schema": "squash-calibration-v2"},  # no camera_id (legacy)
        age_seconds=1)

    # Unfiltered: newest overall (the legacy run), exactly today's behavior.
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
        runs_dir, "cal-camera-id-corrupt",
        {"schema": "squash-calibration-v2"},
        age_seconds=0)
    # Corrupt the newest run's calibration.json after creation so it remains
    # the freshest candidate on disk but fails to parse.
    (run_dir / "calibration.json").write_text("{not valid json", encoding="utf-8")

    response = client.get("/api/calibration/latest")
    assert response.status_code == 500
    assert response.get_json()["ok"] is False


def test_camera_pair_check_endpoint():
    client = _client()
    cam_a = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    cam_b = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    body = client.post("/api/camera-pair-check", json={
        "calibration_a": _synthetic_calibration(cam_a),
        "calibration_b": _synthetic_calibration(cam_b),
    }).get_json()
    assert body["ok"] is True and body["status"] == "ok"
    assert body["ok_pair"] is True
    assert body["median_err_ft"] < 0.05
    assert 2.0 < body["baseline_ft"] < 4.0

    bad = client.post("/api/camera-pair-check", json={
        "calibration_a": _synthetic_calibration(cam_a),
        "calibration_b": {},
    }).get_json()
    assert bad["ok"] is True and bad["status"] == "solve_failed"
    assert bad["status_a"] == "ok" and bad["status_b"] == "no_frame_size"
