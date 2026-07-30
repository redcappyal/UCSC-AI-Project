"""Personal coaching over several identified matches."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    app_module.POOLED_COACH_CACHE.clear()
    with app_module.app.test_client() as test_client:
        yield test_client


def hit(index, player_number, zone=5, band="normal", side="center", y=0.5):
    return {
        "frame": index * 60,
        "hit_frame": index * 60,
        "timestamp_seconds": index * 1.0,
        "event_type": "wall",
        "surface": "front_wall",
        "call": "IN",
        "player_number": player_number,
        "target_zone": {"zone": zone, "side": side, "band": band},
        "wall_diagram": {"x": 0.5, "y": y},
        "velocity": {
            "speed_before": {"mph": 30.0},
            "speed_after": {"mph": 25.0},
            "velocity_change": {"mph": 5.0},
        },
    }


def write_run(runs_dir, run_id, hits_per_player, selected_player=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    hits = []
    for index in range(hits_per_player):
        hits.append(hit(index * 2, 1))
        hits.append(hit(index * 2 + 1, 2))
    payload = {
        "hits": hits,
        "rallies": [{
            "start_frame": 0,
            "end_frame": len(hits) * 60,
            "last_player_number": 2,
            "winner_player_number": 1,
            "last_call": "IN",
        }],
    }
    (run_dir / "detected_hits.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    if selected_player is not None:
        (run_dir / "job.json").write_text(json.dumps({
            "user_player_number": selected_player,
            "players_v1": {
                "player_names": {"A": "Alvin", "B": "Opponent"}
            },
        }), encoding="utf-8")
    return run_dir


def coach_report():
    drill = {
        "name": "Width under pressure",
        "evidence": "Wide usage stayed at 25% across both matches.",
        "setup": "Mark targets one racket width from each side wall.",
        "work": "Alternate straight drives to the two targets.",
        "dose": "4 sets of 10 shots.",
        "success_measure": "Land 7 of 10 shots in the target lane.",
        "match_application": "Use width on the first neutral ball.",
    }
    return {
        "headline": "Build repeatable width",
        "summary": "Your recent matches repeatedly show central targeting.",
        "trend_observations": [
            "Match 2 retained the same 25% wide usage as Match 1."
        ],
        "drills": [drill, {**drill, "name": "Height control"}],
        "next_match_focus": "Track whether 6 of your first 10 shots go wide.",
    }


def test_multi_match_ollama_request_uses_structured_coaching_schema(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "message": {"content": json.dumps(coach_report())}
            }).encode("utf-8")

    def fake_urlopen(request_obj, timeout):
        captured["request"] = request_obj
        captured["timeout"] = timeout
        return FakeResponse()

    app_module.POOLED_COACH_CACHE.clear()
    monkeypatch.setenv("OLLAMA_COACH_MODEL", "coach-test-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9999")
    monkeypatch.setattr(app_module.urllib.request, "urlopen", fake_urlopen)
    history = [{
        "match_order": 1,
        "run_id": "oldest",
        "metrics": {"player_number": 1, "total_wall_hits": 8},
    }]

    report, status = app_module.ollama_multi_match_coaching_feedback(
        history, {"player_number": None, "total_wall_hits": 8}
    )

    assert status == "ok"
    assert report == coach_report()
    body = json.loads(captured["request"].data)
    assert body["model"] == "coach-test-model"
    assert body["format"]["required"] == [
        "headline",
        "summary",
        "trend_observations",
        "drills",
        "next_match_focus",
    ]
    assert body["format"]["properties"]["drills"]["minItems"] == 2
    assert body["options"]["num_ctx"] == 16384
    assert "oldest_to_newest" in body["messages"][1]["content"]


def test_endpoint_does_not_substitute_rule_based_advice(runs_dir, client):
    write_run(runs_dir, "1785000000000", 4)

    body = client.get("/api/coach/advice").get_json()

    assert body["coach"] is None
    assert body["llm_status"] == "no_identified_sessions"
    assert "advice" not in body["me"]
    assert all("advice" not in player for player in body["players"])


def test_ollama_receives_identified_matches_oldest_to_newest(
    runs_dir, client, monkeypatch
):
    write_run(runs_dir, "1785000000001", 2, selected_player=1)
    write_run(runs_dir, "1785000000002", 3, selected_player=2)
    captured = {}

    def fake_ollama(history, pooled_analytics):
        captured["history"] = history
        captured["pooled_analytics"] = pooled_analytics
        return coach_report(), "ok"

    monkeypatch.setenv("COACH_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        app_module, "ollama_multi_match_coaching_feedback", fake_ollama
    )

    body = client.get("/api/coach/advice").get_json()

    assert body["coach"] == coach_report()
    assert body["coach_match_ordering"] == "oldest_to_newest"
    assert body["coach_match_order"] == [
        "1785000000001", "1785000000002",
    ]
    assert [match["match_order"] for match in captured["history"]] == [1, 2]
    assert [
        match["metrics"]["player_number"] for match in captured["history"]
    ] == [1, 2]
    assert captured["pooled_analytics"]["total_wall_hits"] == 5


def test_the_session_limit_takes_the_newest_runs(runs_dir, client):
    for offset in range(6):
        write_run(runs_dir, str(1785000000000 + offset), 2)

    body = client.get("/api/coach/advice?sessions=2").get_json()

    assert body["session_count"] == 2
    assert [session["run_id"] for session in body["sessions"]] == [
        "1785000000005", "1785000000004",
    ]


def test_identified_limit_takes_newest_identified_runs(
    runs_dir, client, monkeypatch
):
    for offset in range(4):
        write_run(
            runs_dir,
            str(1785000000000 + offset),
            2,
            selected_player=1 if offset != 2 else None,
        )
    monkeypatch.setenv("COACH_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        app_module,
        "ollama_multi_match_coaching_feedback",
        lambda history, pooled: (coach_report(), "ok"),
    )

    body = client.get("/api/coach/advice?sessions=2").get_json()

    assert [session["run_id"] for session in body["me"]["sessions"]] == [
        "1785000000003", "1785000000001",
    ]
    assert body["coach_match_order"] == [
        "1785000000001", "1785000000003",
    ]


def test_runs_without_front_wall_contact_do_not_consume_a_session_slot(
    runs_dir, client
):
    write_run(runs_dir, "1785000000000", 3)
    empty = runs_dir / "1785000000001"
    empty.mkdir()
    (empty / "detected_hits.json").write_text(
        json.dumps({"hits": []}), encoding="utf-8"
    )

    body = client.get("/api/coach/advice").get_json()

    assert [session["run_id"] for session in body["sessions"]] == [
        "1785000000000"
    ]


def test_no_runs_returns_an_empty_pool_without_calling_ollama(
    runs_dir, client, monkeypatch
):
    monkeypatch.setattr(
        app_module,
        "ollama_multi_match_coaching_feedback",
        lambda *_: pytest.fail("Ollama should not be called"),
    )

    body = client.get("/api/coach/advice").get_json()

    assert body["ok"] is True
    assert body["session_count"] == 0
    assert body["llm_status"] == "no_identified_sessions"


@pytest.mark.parametrize("value", ["0", "51", "many"])
def test_an_unusable_session_count_is_rejected(runs_dir, client, value):
    assert client.get(f"/api/coach/advice?sessions={value}").status_code == 400
