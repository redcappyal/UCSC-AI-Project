"""ARCHIVED 2026-07-27 -- see archive/stereo/README.md. Not collected by pytest.

Covered `POST /api/camera-pair-check`, the cross-camera agreement gate. Split
out of the former `tests/test_stereo_endpoints.py`, whose other four tests were
single-camera and stayed as `tests/test_camera_endpoints.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from synthetic3d import make_camera
from test_camera_model import _synthetic_calibration


def _client():
    import app as app_module
    return app_module.app.test_client()


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
