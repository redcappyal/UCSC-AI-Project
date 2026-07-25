"""Phase 5: session discovery and the stereo_fuse trigger.

The trigger fires from inside a finished tracking job, so the property that
matters most is that it can never damage the run it fires from — a fuse that
explodes must leave a completed camera run completed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import job_runner


def make_run(runs, run_id, *, session_id=None, camera_role=None,
             status="complete", **extra):
    # This helper mints a fresh run, so it owns the id: drop any in-memory job
    # a previous test left behind. Without this, JOBS keeps a stale run_dir
    # pointing at that test's tmp_path and get_job returns it in preference to
    # the file we are about to write.
    with job_runner.JOBS_LOCK:
        job_runner.JOBS.pop(run_id, None)
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = {"run_id": run_id, "run_dir": str(run_dir), "status": status,
           "fps": 60.0, **extra}
    if session_id:
        job["session_id"] = session_id
    if camera_role:
        job["camera_role"] = camera_role
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return run_dir


def _clear(*run_ids):
    with job_runner.JOBS_LOCK:
        for run_id in run_ids:
            job_runner.JOBS.pop(run_id, None)


# --- discovery --------------------------------------------------------------

def test_session_runs_finds_both_roles(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b")
    make_run(tmp_path, "r-other", session_id="s2", camera_role="a")
    make_run(tmp_path, "r-solo")

    found = job_runner.session_runs("s1")
    assert set(found) == {"a", "b"}
    assert found["a"]["run_id"] == "r-a"
    assert found["b"]["run_id"] == "r-b"


def test_session_runs_ignores_incomplete_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b", status="running")

    assert set(job_runner.session_runs("s1")) == {"a"}


def test_session_runs_ignores_failed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b", status="failed")

    assert set(job_runner.session_runs("s1")) == {"a"}


def test_session_runs_keeps_the_newest_of_duplicate_roles(tmp_path, monkeypatch):
    """A retried camera posts a second run under the same session and role."""
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "1000", session_id="s1", camera_role="a")
    make_run(tmp_path, "2000", session_id="s1", camera_role="a")

    assert job_runner.session_runs("s1")["a"]["run_id"] == "2000"


def test_session_runs_skips_fuse_dirs(tmp_path, monkeypatch):
    """A fuse run must never be mistaken for a camera run."""
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "stereo-s1", session_id="s1", camera_role="a")

    assert job_runner.session_runs("s1")["a"]["run_id"] == "r-a"


def test_session_runs_tolerates_unreadable_job_files(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    broken = tmp_path / "r-broken"
    broken.mkdir()
    (broken / "job.json").write_text("{ not json", encoding="utf-8")

    assert set(job_runner.session_runs("s1")) == {"a"}


def test_session_runs_is_empty_when_the_dir_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path / "nope")
    assert job_runner.session_runs("s1") == {}


# --- the trigger ------------------------------------------------------------

def test_no_fuse_until_both_roles_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")

    assert job_runner.maybe_start_stereo_fuse("s1") is None
    assert started == []
    assert not (tmp_path / "stereo-s1").exists()


def test_fuse_starts_once_both_roles_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b")

    fuse_id = job_runner.maybe_start_stereo_fuse("s1")
    try:
        assert fuse_id == "stereo-s1"
        assert started == ["stereo-s1"]
        job = json.loads((tmp_path / "stereo-s1" / "job.json").read_text())
        assert job["session_id"] == "s1"
        assert job["run_a"] == "r-a" and job["run_b"] == "r-b"
        # The eval set keeps one run per video_sha; a fuse run with a
        # video_path could displace the mono run it came from.
        assert "video_path" not in job
    finally:
        _clear(fuse_id)


def test_only_one_of_two_concurrent_triggers_wins(tmp_path, monkeypatch):
    """Both cameras finishing at once is the normal case, not an edge case."""
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b")

    first = job_runner.maybe_start_stereo_fuse("s1")
    second = job_runner.maybe_start_stereo_fuse("s1")
    try:
        assert first == "stereo-s1"
        assert second is None, "the second trigger must not start a duplicate"
        assert started == ["stereo-s1"]
    finally:
        _clear("stereo-s1")


def test_no_fuse_when_the_peer_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b", status="failed")

    assert job_runner.maybe_start_stereo_fuse("s1") is None
    assert started == []


def test_trigger_records_the_fuse_id_on_both_camera_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", lambda run_id: None)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b")

    job_runner.maybe_start_stereo_fuse("s1")
    try:
        for run_id in ("r-a", "r-b"):
            job = json.loads((tmp_path / run_id / "job.json").read_text())
            assert job["stereo_fuse_run_id"] == "stereo-s1"
            assert job["status"] == "complete", "the trigger must not disturb status"
    finally:
        _clear("stereo-s1", "r-a", "r-b")


def test_a_failing_trigger_cannot_fail_a_completed_run(tmp_path, monkeypatch):
    """The trigger runs inside a finished tracking job. It must be inert."""
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)

    def explode(_run_id):
        raise RuntimeError("fuse blew up")

    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", explode)
    make_run(tmp_path, "r-a", session_id="s1", camera_role="a")
    make_run(tmp_path, "r-b", session_id="s1", camera_role="b")

    # Must not propagate.
    job_runner.try_start_stereo_fuse("s1")
    try:
        job = json.loads((tmp_path / "r-a" / "job.json").read_text())
        assert job["status"] == "complete"
    finally:
        _clear("stereo-s1", "r-a", "r-b")


def test_try_start_is_a_no_op_without_a_session(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    job_runner.try_start_stereo_fuse(None)
    job_runner.try_start_stereo_fuse("")
    assert started == []
