# ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
# Not imported by any runtime module, not collected by pytest, not linted.
# Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
"""Phase 5: the optional paired-session inputs on /api/track.

The load-bearing property is *absence*: a single-camera request must behave
exactly as it did before Phase 5 existed, byte for byte. Every test here either
pins that, or pins that a supplied field lands where the fuse job will look for
it.
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The four form fields Phase 5 adds. Named once so the absence assertions and
# the presence assertions cannot drift apart.
SESSION_FORM_FIELDS = ("session_id", "camera_role", "peer_video_id", "sync_manifest_json")
SESSION_JOB_KEYS = ("session_id", "camera_role", "peer_video_id", "sync_manifest_path")


def _track_env(monkeypatch, tmp_path):
    """A /api/track that needs no real video, no cv2 decode, and no worker.

    Everything Phase 5 touches happens before the tracking thread starts, so
    stubbing the video probe and the job start keeps these tests in the
    no-model tier the rest of the suite lives in.
    """
    import app as app_module
    import job_runner

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(app_module, "RUNS_DIR", runs)
    monkeypatch.setattr(app_module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(job_runner, "RUNS_DIR", runs)
    monkeypatch.setattr(
        app_module, "video_info",
        lambda path: {"fps": 60.0, "frame_count": 600, "duration": 10.0})
    monkeypatch.setattr(app_module, "start_tracking_job", lambda run_id: None)
    return app_module, runs


def _post_track(client, **extra):
    data = {
        "calibration_json": "{}",
        "start_time": "0",
        "end_time": "5",
        "video_file": (io.BytesIO(b"not really a video"), "clip.mp4"),
    }
    data.update(extra)
    return client.post("/api/track", data=data, content_type="multipart/form-data")


def _job_on_disk(runs, run_id):
    return json.loads((runs / run_id / "job.json").read_text(encoding="utf-8"))


def _cleanup(run_id):
    import job_runner
    with job_runner.JOBS_LOCK:
        job_runner.JOBS.pop(run_id, None)


# --- absence: the single-camera flow is untouched ---------------------------

def test_single_camera_track_carries_no_session_state(monkeypatch, tmp_path):
    app_module, runs = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client())
    assert response.status_code == 200
    body = response.get_json()
    run_id = body["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        for key in SESSION_JOB_KEYS:
            assert key not in job, f"{key} leaked into a single-camera job"
            assert key not in body, f"{key} leaked into a single-camera response"
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


def test_empty_session_fields_are_treated_as_absent(monkeypatch, tmp_path):
    """A form that sends the keys with empty values is still single-camera.

    index.html builds FormData unconditionally in places, so "" must not
    become a session.
    """
    app_module, runs = _track_env(monkeypatch, tmp_path)
    blanks = {field: "" for field in SESSION_FORM_FIELDS}
    response = _post_track(app_module.app.test_client(), **blanks)
    assert response.status_code == 200
    run_id = response.get_json()["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        for key in SESSION_JOB_KEYS:
            assert key not in job
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


# --- presence: the fields land where the fuse job will look ------------------

def test_paired_track_records_every_session_field(monkeypatch, tmp_path):
    app_module, runs = _track_env(monkeypatch, tmp_path)
    manifest = {"offset_series": [0.004, 0.005], "clap_anchor_s": 1.25,
                "readout_time_s": {"a": 0.008, "b": 0.008}}
    response = _post_track(
        app_module.app.test_client(),
        session_id="sess-1", camera_role="a", peer_video_id="peersha",
        sync_manifest_json=json.dumps(manifest))
    assert response.status_code == 200
    body = response.get_json()
    run_id = body["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        assert job["session_id"] == "sess-1"
        assert job["camera_role"] == "a"
        assert job["peer_video_id"] == "peersha"

        manifest_path = runs / run_id / "sync_manifest.json"
        assert manifest_path.exists()
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        assert job["sync_manifest_path"] == str(manifest_path)

        # Clients get the session identity, never a server filesystem path.
        assert body["session_id"] == "sess-1"
        assert body["camera_role"] == "a"
        assert body["peer_video_id"] == "peersha"
        assert "sync_manifest_path" not in body
    finally:
        _cleanup(run_id)


def test_session_without_a_manifest_is_allowed(monkeypatch, tmp_path):
    """The manifest only seeds the refiner, which has its own fallback."""
    app_module, runs = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(),
                           session_id="sess-2", camera_role="b")
    assert response.status_code == 200
    run_id = response.get_json()["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        assert job["session_id"] == "sess-2" and job["camera_role"] == "b"
        assert "sync_manifest_path" not in job
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


# --- rejection --------------------------------------------------------------

def test_unknown_camera_role_is_rejected(monkeypatch, tmp_path):
    app_module, _ = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(),
                           session_id="sess-3", camera_role="left")
    assert response.status_code == 400
    assert "camera role" in response.get_json()["error"].lower()


def test_camera_role_without_a_session_is_rejected(monkeypatch, tmp_path):
    """A role with nothing to pair against would never fuse — fail loudly."""
    app_module, _ = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(), camera_role="a")
    assert response.status_code == 400


def test_session_without_a_role_is_rejected(monkeypatch, tmp_path):
    """Two runs that both omit the role cannot be told apart at fuse time."""
    app_module, _ = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(), session_id="sess-4")
    assert response.status_code == 400


def test_malformed_sync_manifest_is_rejected(monkeypatch, tmp_path):
    app_module, _ = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(),
                           session_id="sess-5", camera_role="a",
                           sync_manifest_json="{not json")
    assert response.status_code == 400
    assert "sync manifest" in response.get_json()["error"].lower()


def test_non_object_sync_manifest_is_rejected(monkeypatch, tmp_path):
    """json.loads accepts a bare scalar; the fuse job needs a mapping."""
    app_module, _ = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(),
                           session_id="sess-6", camera_role="a",
                           sync_manifest_json="[1, 2, 3]")
    assert response.status_code == 400
