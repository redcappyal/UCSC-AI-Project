import json


def _make_run(runs_dir, run_id, status="complete"):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
    }))
    (run_dir / "annotations.json").write_text("{}")
    return run_dir


def test_delete_run_removes_only_the_session(runs_dir):
    import app as app_module

    run_id = "delete-session-test"
    run_dir = _make_run(runs_dir, run_id)
    shared_upload = runs_dir / "uploads" / "by-hash" / "video.mp4"
    shared_upload.parent.mkdir(parents=True)
    shared_upload.write_bytes(b"video")
    app_module.BALL_POSITIONS_CACHE[run_id] = (1, [])
    app_module.RUN_HITS_CACHE[run_id] = (1, [])

    response = app_module.app.test_client().delete(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "run_id": run_id}
    assert not run_dir.exists()
    assert shared_upload.exists()
    assert run_id not in app_module.BALL_POSITIONS_CACHE
    assert run_id not in app_module.RUN_HITS_CACHE


def test_delete_run_rejects_active_and_missing_sessions(runs_dir):
    import app as app_module
    import job_runner

    run_id = "delete-running-test"
    run_dir = _make_run(runs_dir, run_id, status="running")
    job_runner.JOBS[run_id] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "running",
    }
    client = app_module.app.test_client()

    assert client.delete(f"/api/runs/{run_id}").status_code == 409
    assert run_dir.exists()
    assert client.delete("/api/runs/missing-session").status_code == 404
    job_runner.forget_job(run_id)


def test_delete_run_rejects_ids_changed_by_sanitizing(runs_dir):
    import app as app_module

    response = app_module.app.test_client().delete("/api/runs/session%20name")

    assert response.status_code == 400
