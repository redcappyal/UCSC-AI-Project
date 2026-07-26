"""GET /api/calibration/latest: newest run calibration for native clients."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point run storage at tmp_path so these tests never touch ui_runs/.

    The endpoint scans every run for a calibration.json, so a real RUNS_DIR
    makes the result depend on whatever the checkout (or an earlier test) left
    lying around. app.py does `from job_runner import RUNS_DIR`, so the route
    reads app's own global — both names need patching.
    """
    import app as app_module
    import job_runner

    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    return tmp_path


def make_run_with_calibration(runs_dir, run_id, calibration, age_seconds):
    run_dir = runs_dir / run_id
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
        runs_dir, "cal-latest-older", {"lines": [{"name": "out"}]}, age_seconds=120)
    make_run_with_calibration(
        runs_dir, "cal-latest-newer", {"lines": [{"name": "tin"}]}, age_seconds=5)

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
    # A marker dir without calibration.json: proves non-calibrated runs are
    # skipped rather than counted. An isolated RUNS_DIR is genuinely empty, so
    # the 404 is asserted outright instead of being conditional on the checkout.
    (runs_dir / "cal-latest-empty-run").mkdir(parents=True, exist_ok=True)

    response = client.get("/api/calibration/latest")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False
