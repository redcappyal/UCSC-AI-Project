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
