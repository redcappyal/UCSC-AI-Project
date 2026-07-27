"""/api/track's form contract: what it accepts, and what it must ignore.

The load-bearing property here has always been *absence*: a single-camera
request must behave exactly as it did before the paired-session fields existed,
byte for byte. That was the leading test of the old `tests/test_paired_session.py`
(archived 2026-07-27 with the rest of the stereo feature — see
archive/stereo/README.md), and it survives the archiving because it is the one
assertion that outlives the fields it was written about.

It is now a *stronger* claim than it was. Before, the four fields were parsed,
validated, and stored, and this file pinned that omitting them changed nothing.
Now they are not read at all, so the assertion is that supplying them changes
nothing either: a client still sending `session_id` / `camera_role` /
`peer_video_id` / `sync_manifest_json` — an old build of the iOS app, a stale
FormData in index.html — gets a plain single-camera run and no error, rather
than a 400 from validation that no longer exists.
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The four form fields the archived stereo feature added to /api/track. Named
# here so this file documents exactly what is being ignored.
RETIRED_FORM_FIELDS = ("session_id", "camera_role", "peer_video_id", "sync_manifest_json")
RETIRED_JOB_KEYS = ("session_id", "camera_role", "peer_video_id", "sync_manifest_path",
                    "stereo_fuse_run_id")


def _track_env(monkeypatch, tmp_path):
    """A /api/track that needs no real video, no cv2 decode, and no worker.

    Everything asserted below happens before the tracking thread starts, so
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


def test_single_camera_track_carries_no_session_state(monkeypatch, tmp_path):
    """The original leading test, unchanged in intent: a plain request is plain."""
    app_module, runs = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client())
    assert response.status_code == 200
    body = response.get_json()
    run_id = body["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        for key in RETIRED_JOB_KEYS:
            assert key not in job, f"{key} leaked into a single-camera job"
            assert key not in body, f"{key} leaked into a single-camera response"
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


def test_retired_session_fields_are_ignored_not_rejected(monkeypatch, tmp_path):
    """A client still sending the archived fields gets a normal run, not a 400.

    Values chosen to be exactly what the removed validation used to reject: a
    `camera_role` outside {a, b}, and a `sync_manifest_json` that is not even
    valid JSON. Both used to be 400s. Neither is read now, so both must sail
    through and produce an ordinary single-camera run.
    """
    app_module, runs = _track_env(monkeypatch, tmp_path)
    response = _post_track(
        app_module.app.test_client(),
        session_id="sess-1", camera_role="left", peer_video_id="peersha",
        sync_manifest_json="{not json")
    assert response.status_code == 200
    body = response.get_json()
    run_id = body["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        for key in RETIRED_JOB_KEYS:
            assert key not in job, f"{key} was stored from a form field nothing reads"
            assert key not in body
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


def test_empty_retired_fields_are_ignored(monkeypatch, tmp_path):
    """index.html builds FormData unconditionally in places, so "" must be inert."""
    app_module, runs = _track_env(monkeypatch, tmp_path)
    blanks = {field: "" for field in RETIRED_FORM_FIELDS}
    response = _post_track(app_module.app.test_client(), **blanks)
    assert response.status_code == 200
    run_id = response.get_json()["run_id"]
    try:
        job = _job_on_disk(runs, run_id)
        for key in RETIRED_JOB_KEYS:
            assert key not in job
        assert not (runs / run_id / "sync_manifest.json").exists()
    finally:
        _cleanup(run_id)


def test_the_paired_run_directory_convention_is_gone(monkeypatch, tmp_path):
    """No run may be created under the archived `stereo-<session_id>` name.

    The fuse job claimed its work by `mkdir(RUNS_DIR / "stereo-<id>")`. Nothing
    creates that any more, and a plain run id is a millisecond timestamp, so the
    prefix must never appear.
    """
    app_module, runs = _track_env(monkeypatch, tmp_path)
    response = _post_track(app_module.app.test_client(),
                           session_id="sess-2", camera_role="a")
    run_id = response.get_json()["run_id"]
    try:
        assert not run_id.startswith("stereo-")
        assert [p.name for p in runs.iterdir()] == [run_id]
    finally:
        _cleanup(run_id)
