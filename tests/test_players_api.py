"""POST /api/runs/<id>/players and players_v1 passthrough."""

import json


def make_client():
    import app as app_module
    return app_module.app.test_client()


def make_run(runs_dir, run_id="run-players"):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def test_players_v1_passes_through_public_job():
    import app as app_module
    payload = app_module.public_job({
        "status": "complete",
        "players_v1": {"attribution_backend": "observed"},
    })
    assert payload["players_v1"] == {"attribution_backend": "observed"}


def test_post_players_names(runs_dir):
    import job_runner
    client = make_client()
    run_dir = make_run(runs_dir)
    job_runner.update_job(
        "run-players",
        run_dir=str(run_dir),
        status="complete",
        players_v1={"attribution_backend": "observed",
                    "player_names": {"A": None, "B": None}},
    )

    response = client.post("/api/runs/run-players/players",
                           json={"A": "  Ian ", "B": "Alvin"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["player_names"] == {"A": "Ian", "B": "Alvin"}

    stored = json.loads((run_dir / "player_names.json").read_text())
    assert stored == {"A": "Ian", "B": "Alvin"}
    job = job_runner.get_job("run-players")
    assert job["players_v1"]["player_names"] == {"A": "Ian", "B": "Alvin"}


def test_post_players_validation(runs_dir):
    client = make_client()
    make_run(runs_dir)
    assert client.post("/api/runs/run-players/players",
                       json={"A": "x" * 41}).status_code == 400
    assert client.post("/api/runs/run-players/players",
                       json={"C": "nope"}).status_code == 400
    assert client.post("/api/runs/missing-run/players",
                       json={"A": "Ian"}).status_code == 404


def test_post_players_rejects_non_object_body(runs_dir):
    client = make_client()
    make_run(runs_dir)
    assert client.post("/api/runs/run-players/players",
                       json=42).status_code == 400


def test_post_players_rejects_non_string_names(runs_dir):
    client = make_client()
    make_run(runs_dir)
    assert client.post("/api/runs/run-players/players",
                       json={"A": 5}).status_code == 400
    assert client.post("/api/runs/run-players/players",
                       json={"A": True}).status_code == 400


def test_post_players_rehydrates_disk_only_job(runs_dir, monkeypatch):
    """Finding 1 repro: a run whose job lives only in job.json (server
    restarted since the job finished) must not be shadowed by an empty
    in-memory stub. update_job has to rehydrate from disk before merging,
    and the read-view queued/running->failed rewrite must NOT apply here —
    this is a stored 'complete' job, not a live status read."""
    import job_runner

    # The runs_dir fixture only redirects app.RUNS_DIR; job_runner's own
    # disk fallback (used by get_job/update_job when a run is absent from
    # JOBS) reads via job_runner.RUNS_DIR, so it needs the same redirect —
    # see test_job_restart_recovery in test_pipeline.py for precedent.
    monkeypatch.setattr(job_runner, "RUNS_DIR", runs_dir)

    run_id = "run-restart"
    run_dir = make_run(runs_dir, run_id)
    job_runner.JOBS.pop(run_id, None)
    try:
        on_disk_job = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "complete",
            "rows": 42,
            "hits": [{"frame": 10, "call": "IN"}],
            "players_v1": {
                "attribution_backend": "observed",
                "player_names": {"A": None, "B": None},
            },
        }
        (run_dir / "job.json").write_text(
            json.dumps(on_disk_job, indent=2), encoding="utf-8"
        )
        job_runner.JOBS.pop(run_id, None)

        client = make_client()
        response = client.post(f"/api/runs/{run_id}/players",
                               json={"A": "Ian", "B": "Alvin"})
        assert response.status_code == 200
        body = response.get_json()
        assert body["player_names"] == {"A": "Ian", "B": "Alvin"}

        assert (run_dir / "player_names.json").exists()
        stored_names = json.loads((run_dir / "player_names.json").read_text())
        assert stored_names == {"A": "Ian", "B": "Alvin"}

        job = job_runner.get_job(run_id)
        assert job["status"] == "complete"
        assert job["rows"] == 42
        assert job["hits"] == [{"frame": 10, "call": "IN"}]
        assert job["players_v1"]["player_names"] == {"A": "Ian", "B": "Alvin"}

        on_disk = json.loads((run_dir / "job.json").read_text())
        assert on_disk["status"] == "complete"
        assert on_disk["rows"] == 42
        assert on_disk["players_v1"]["player_names"] == {"A": "Ian", "B": "Alvin"}
    finally:
        job_runner.JOBS.pop(run_id, None)
