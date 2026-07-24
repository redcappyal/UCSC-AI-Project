import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from court_model import CameraModel
from synthetic3d import make_camera
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
