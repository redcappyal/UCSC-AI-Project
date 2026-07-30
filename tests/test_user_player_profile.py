import json

import app as app_module
import job_runner


def _hit(frame, player_number, zone):
    return {
        "frame": frame,
        "timestamp_seconds": frame / 60.0,
        "event_type": "wall",
        "surface": "front_wall",
        "call": "IN",
        "player_number": player_number,
        "target_zone": {
            "zone": zone,
            "side": "left" if zone <= 3 else "right",
            "band": "normal",
        },
        "wall_diagram": {"x": 0.2 if zone <= 3 else 0.8, "y": 0.5},
    }


def _write_selected_run(
    runs_dir, run_id, selected_player, player_one_zones, player_two_zones
):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    hits = []
    for index, zone in enumerate(player_one_zones):
        hits.append(_hit(index * 120, 1, zone))
    for index, zone in enumerate(player_two_zones):
        hits.append(_hit(index * 120 + 60, 2, zone))
    (run_dir / "detected_hits.json").write_text(
        json.dumps({"hits": hits, "rallies": []}), encoding="utf-8"
    )
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "complete",
        "user_player_number": selected_player,
        "players_v1": {
            "player_names": {"A": "Alvin", "B": "Opponent"},
        },
    }), encoding="utf-8")
    return run_dir


def test_user_can_select_their_player_for_a_run(runs_dir):
    run_id = "run-user-player"
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    job_runner.update_job(
        run_id,
        run_id=run_id,
        run_dir=str(run_dir),
        status="complete",
    )
    client = app_module.app.test_client()

    response = client.post(
        f"/api/runs/{run_id}/me", json={"player_number": 2}
    )

    assert response.status_code == 200
    assert response.get_json()["track"] == "B"
    saved = json.loads((run_dir / "player_profile.json").read_text())
    assert saved["player_number"] == 2
    assert job_runner.get_job(run_id)["user_player_number"] == 2


def test_user_player_selection_validates_the_run_and_player(runs_dir):
    run_dir = runs_dir / "run-user-player"
    run_dir.mkdir()
    client = app_module.app.test_client()

    assert client.post(
        "/api/runs/run-user-player/me", json={"player_number": 3}
    ).status_code == 400
    assert client.post(
        "/api/runs/run-user-player/me", json=[]
    ).status_code == 400
    assert client.post(
        "/api/runs/missing/me", json={"player_number": 1}
    ).status_code == 404


def test_pooled_me_metrics_follow_each_runs_selected_player(
    runs_dir, monkeypatch
):
    _write_selected_run(
        runs_dir,
        "1785000000001",
        selected_player=1,
        player_one_zones=[1, 2],
        player_two_zones=[8, 9, 8, 9],
    )
    _write_selected_run(
        runs_dir,
        "1785000000002",
        selected_player=2,
        player_one_zones=[1, 1, 1],
        player_two_zones=[7, 8],
    )
    monkeypatch.setenv("COACH_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        app_module,
        "ollama_multi_match_coaching_feedback",
        lambda history, pooled: (None, "ollama_unavailable"),
    )
    client = app_module.app.test_client()

    body = client.get("/api/coach/advice").get_json()

    assert body["me"]["session_count"] == 2
    assert body["me"]["analytics"]["total_wall_hits"] == 4
    assert [session["player_number"] for session in body["me"]["sessions"]] == [
        2, 1,
    ]
    assert "Only matches where you selected your player" in body["me_pooling_note"]


def test_runs_index_exposes_selection_and_player_names(runs_dir):
    _write_selected_run(
        runs_dir,
        "1785000000001",
        selected_player=2,
        player_one_zones=[1],
        player_two_zones=[8],
    )
    client = app_module.app.test_client()

    run = client.get("/api/runs").get_json()["runs"][0]

    assert run["user_player_number"] == 2
    assert run["player_names"] == {"A": "Alvin", "B": "Opponent"}
