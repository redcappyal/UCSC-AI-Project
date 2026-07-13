"""Pipeline checks that need no model: windows, persistence, dedup, routes."""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_runner import refine_segments_for_hits


def test_refine_windows_merge_and_clamp():
    segments = refine_segments_for_hits(
        [{"hit_frame": 100}, {"hit_frame": 110}, {"hit_frame": 300}], 0, 1000, 3
    )
    assert segments == [(88, 122, 1), (288, 312, 1)]


def test_refine_windows_clamped_to_clip():
    segments = refine_segments_for_hits([{"hit_frame": 5}], 0, 1000, 1)
    assert segments == [(0, 17, 1)]


def test_job_restart_recovery(tmp_path, monkeypatch):
    import job_runner

    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    run_id = "restart-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id, "run_dir": str(run_dir), "status": "running",
    }))
    job = job_runner.get_job(run_id)
    assert job["status"] == "failed"
    assert "restarted" in job["error"]

    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id, "run_dir": str(run_dir), "status": "complete", "rows": 5,
    }))
    job = job_runner.get_job(run_id)
    assert job["status"] == "complete" and job["rows"] == 5


def test_update_job_persists_atomically(tmp_path):
    import job_runner

    run_id = "persist-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    try:
        job_runner.update_job(run_id, run_dir=str(run_dir), status="queued", note="x")
        on_disk = json.loads((run_dir / "job.json").read_text())
        assert on_disk["status"] == "queued" and on_disk["note"] == "x"
    finally:
        with job_runner.JOBS_LOCK:
            job_runner.JOBS.pop(run_id, None)


def test_upload_dedup_and_track_validation():
    import app as app_module

    client = app_module.app.test_client()

    health = client.get("/api/health").get_json()
    assert health["ok"] is True and "version" in health

    response = client.post("/api/track", data={})
    assert response.status_code == 400

    response = client.post("/api/track", data={
        "video_id": "deadbeef", "calibration_json": "{}",
        "start_time": "0", "end_time": "5",
    })
    assert response.status_code == 404

    payload = b"fake video bytes for hashing"
    ids = []
    for _ in range(2):
        response = client.post(
            "/api/upload",
            data={"video_file": (io.BytesIO(payload), "clip.mp4")},
            content_type="multipart/form-data",
        )
        ids.append(response.get_json()["video_id"])

    assert ids[0] == ids[1]
    matches = list(app_module.BY_HASH_DIR.glob(f"{ids[0]}.*"))
    try:
        assert len(matches) == 1
        assert app_module.video_path_for_id(ids[0]) == matches[0]
    finally:
        for match in matches:
            match.unlink()

    assert client.get("/api/track/status/does-not-exist").status_code == 404
