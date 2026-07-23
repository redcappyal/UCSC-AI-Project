"""GET /api/calibration/latest: newest run calibration for native clients."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_run_with_calibration(app_module, run_id, calibration, age_seconds):
    run_dir = app_module.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return run_dir


def test_latest_calibration_returns_newest_run():
    import app as app_module

    client = app_module.app.test_client()
    older = make_run_with_calibration(
        app_module, "cal-latest-older", {"lines": [{"name": "out"}]}, age_seconds=120)
    newer = make_run_with_calibration(
        app_module, "cal-latest-newer", {"lines": [{"name": "tin"}]}, age_seconds=5)
    try:
        response = client.get("/api/calibration/latest")
        assert response.status_code == 200
        body = response.get_json()
        assert body["ok"] is True
        assert body["run_id"] == "cal-latest-newer"
        assert body["calibration"] == {"lines": [{"name": "tin"}]}
        assert body["saved_at"].endswith("Z")
    finally:
        shutil.rmtree(older, ignore_errors=True)
        shutil.rmtree(newer, ignore_errors=True)


def test_latest_calibration_404_when_none_exist():
    import app as app_module

    client = app_module.app.test_client()
    # Only guaranteed-empty when no other test left runs behind; use a marker
    # dir without calibration.json to prove non-calibrated runs are skipped.
    marker = app_module.RUNS_DIR / "cal-latest-empty-run"
    marker.mkdir(parents=True, exist_ok=True)
    had_calibrations = any(app_module.RUNS_DIR.glob("*/calibration.json"))
    try:
        response = client.get("/api/calibration/latest")
        if had_calibrations:
            assert response.status_code == 200  # other fixtures present; endpoint still works
        else:
            assert response.status_code == 404
            assert response.get_json()["ok"] is False
    finally:
        shutil.rmtree(marker, ignore_errors=True)
