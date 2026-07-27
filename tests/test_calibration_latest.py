"""GET /api/calibration/latest: newest run calibration for native clients."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_run_with_calibration(app_module, run_id, calibration, age_seconds):
    """A run carrying a calibration of a given age.

    Reads RUNS_DIR off the module at call time, so it follows the `runs_dir`
    fixture's redirect. test_camera_endpoints.py imports this too.
    """
    run_dir = app_module.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return run_dir


def test_latest_calibration_returns_newest_run(runs_dir):
    import app as app_module

    client = app_module.app.test_client()
    make_run_with_calibration(
        app_module, "cal-latest-older", {"lines": [{"name": "out"}]}, age_seconds=120)
    make_run_with_calibration(
        app_module, "cal-latest-newer", {"lines": [{"name": "tin"}]}, age_seconds=5)

    response = client.get("/api/calibration/latest")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["run_id"] == "cal-latest-newer"
    assert body["calibration"] == {"lines": [{"name": "tin"}]}
    assert body["saved_at"].endswith("Z")


def test_latest_calibration_404_when_none_exist(runs_dir):
    import app as app_module

    client = app_module.app.test_client()
    # A run without a calibration.json must not count — the directory is there,
    # the calibration is not. Isolation is what lets this assert the 404
    # outright rather than tolerating whatever another test left behind.
    (runs_dir / "cal-latest-empty-run").mkdir(parents=True, exist_ok=True)

    response = client.get("/api/calibration/latest")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False
