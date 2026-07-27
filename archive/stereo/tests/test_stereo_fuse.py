# ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
# Not imported by any runtime module, not collected by pytest, not linted.
# Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
"""Phase 5: session discovery and the stereo_fuse trigger.

The trigger fires from inside a finished tracking job, so the property that
matters most is that it can never damage the run it fires from — a fuse that
explodes must leave a completed camera run completed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

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


# --- the fuse job itself ----------------------------------------------------


def _paired_session(runs, session_id, offset_s=0.012, fps=60.0):
    """Two completed camera runs of one rally, written as real artifacts.

    No video and no model: the CSVs are the projection of one synthetic
    trajectory through a synthetic camera pair, which is exactly what the fuse
    job consumes.
    """
    from synthetic3d import make_camera            # noqa: PLC0415
    from test_camera_model import _synthetic_calibration   # noqa: PLC0415
    from test_stereo_sync import write_csv         # noqa: PLC0415
    from test_stereo_track import (                # noqa: PLC0415
        sample_camera, simulate_front_wall_shot)
    from stereo_engine import TrackSample          # noqa: PLC0415

    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=fps)
    samples_b = sample_camera(states, right, fps=fps, phase_s=1.0 / (2 * fps))
    # Camera b's clock runs `offset_s` fast; the refiner has to find it back.
    samples_b = [TrackSample(t_s=s.t_s - offset_s, px=s.px) for s in samples_b]

    made = {}
    for role, model, samples in (("a", left, samples_a), ("b", right, samples_b)):
        run_id = f"{session_id}-{role}"
        run_dir = make_run(runs, run_id, session_id=session_id, camera_role=role,
                           fps=fps)
        write_csv(run_dir / "ball_coordinates.csv", samples, fps=fps)
        # The repo's own fixture, so the calibration provably solves back to
        # the camera it was projected through.
        (run_dir / "calibration.json").write_text(
            json.dumps(_synthetic_calibration(model)), encoding="utf-8")
        made[role] = run_id
    return made


def test_fuse_job_produces_all_three_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    session = "s-fuse"
    made = _paired_session(tmp_path, session, offset_s=0.012)
    fuse_id = job_runner.stereo_fuse_run_id(session)
    fuse_dir = tmp_path / fuse_id
    fuse_dir.mkdir()
    job_runner.create_job(fuse_id, fuse_dir, status="queued", session_id=session,
                          run_a=made["a"], run_b=made["b"])
    try:
        job_runner.run_stereo_fuse_job(fuse_id)
        job = job_runner.get_job(fuse_id)
        assert job["status"] == "complete", job.get("error")

        # sync_report.json: the offset was recovered.
        report = json.loads((fuse_dir / "sync_report.json").read_text())
        assert report["schema"] == "stereo-sync-v1"
        assert report["refined"]["accepted"] is True
        assert abs(report["refined"]["offset_s"] - 0.012) < 0.002
        assert report["runs"] == {"a": made["a"], "b": made["b"]}

        # stereo_track.jsonl: one JSON object per line, court-frame points.
        lines = (fuse_dir / "stereo_track.jsonl").read_text().strip().splitlines()
        assert lines
        first = json.loads(lines[0])
        assert set(first) == {"t_s", "point_ft", "gap_ft", "view_count"}
        assert len(first["point_ft"]) == 3
        for line in lines:
            json.loads(line)          # every line parses on its own

        # detected_hits.json: the front-wall call, in the mono vocabulary.
        payload = json.loads((fuse_dir / "detected_hits.json").read_text())
        wall = [h for h in payload["hits"] if h["event_type"] == "wall"]
        assert wall, payload["hits"]
        hit = wall[0]
        assert isinstance(hit["frame"], int), "a float frame matches nothing in build_eval_set"
        assert hit["call"] in {"IN", "OUT"}
        assert hit["margin_px"] is None
        assert hit["view_count"] in {1, 2}
        assert hit["method"] in {"triangulated", "plane_snap", "estimated"}
        assert hit["stereo"]["surface"] == "front_wall"
        assert hit["target_zone"] is not None and hit["wall_diagram"] is not None
        assert payload["stereo"]["offset_accepted"] is True
    finally:
        _clear(fuse_id, made["a"], made["b"])


def test_fuse_job_fails_by_name_when_a_camera_will_not_solve(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    session = "s-badcal"
    made = _paired_session(tmp_path, session)
    (tmp_path / made["b"] / "calibration.json").write_text("{}", encoding="utf-8")
    fuse_id = job_runner.stereo_fuse_run_id(session)
    fuse_dir = tmp_path / fuse_id
    fuse_dir.mkdir()
    job_runner.create_job(fuse_id, fuse_dir, status="queued", session_id=session,
                          run_a=made["a"], run_b=made["b"])
    try:
        job_runner.run_stereo_fuse_job(fuse_id)
        job = job_runner.get_job(fuse_id)
        assert job["status"] == "failed"
        assert "side 'b'" in job["error"]
        # The camera runs are untouched by a fuse failure.
        for run_id in made.values():
            assert json.loads(
                (tmp_path / run_id / "job.json").read_text())["status"] == "complete"
    finally:
        _clear(fuse_id, made["a"], made["b"])


def test_fused_hits_survive_the_coaching_consumers(tmp_path, monkeypatch):
    """The fused file has to be readable by everything that reads a mono one."""
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    import app as app_module

    session = "s-consume"
    made = _paired_session(tmp_path, session)
    fuse_id = job_runner.stereo_fuse_run_id(session)
    fuse_dir = tmp_path / fuse_id
    fuse_dir.mkdir()
    job_runner.create_job(fuse_id, fuse_dir, status="queued", session_id=session,
                          run_a=made["a"], run_b=made["b"])
    try:
        job_runner.run_stereo_fuse_job(fuse_id)
        payload = json.loads((fuse_dir / "detected_hits.json").read_text())

        # Does not raise, and only keeps hits carrying the analytics blocks.
        front = app_module.front_wall_hits_from_payload(payload)
        for hit in front:
            assert hit["target_zone"] is not None

        analytics = app_module.build_coaching_analytics(payload)
        assert "players" in analytics
    finally:
        _clear(fuse_id, made["a"], made["b"])


def test_try_start_is_a_no_op_without_a_session(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    started = []
    monkeypatch.setattr(job_runner, "start_stereo_fuse_job", started.append)
    job_runner.try_start_stereo_fuse(None)
    job_runner.try_start_stereo_fuse("")
    assert started == []
