"""Coaching advice pooled across sessions.

A one-minute clip yields a handful of front-wall contacts — well under
coaching_advice.MIN_HITS_FOR_ADVICE — so per-run advice is nearly always "too
few shots to say anything", and the Training hub had nothing to show. Pooling
the recent sessions is what makes the page able to speak.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module
from coaching_advice import MIN_HITS_FOR_ADVICE


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def hit(index, player_number, zone=5, band="normal", side="center", y=0.5):
    """One front-wall contact, shaped the way detected_hits.json stores them."""
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
        "velocity": {"speed_before": {"mph": 30.0}, "speed_after": {"mph": 25.0},
                     "velocity_change": {"mph": 5.0}},
    }


def write_run(runs_dir, run_id, hits_per_player):
    """A completed run with `hits_per_player` contacts for each of P1 and P2."""
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
    (run_dir / "detected_hits.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_one_short_session_alone_cannot_produce_advice(runs_dir, client):
    """The state that motivated pooling: a real clip, and nothing to say."""
    short = MIN_HITS_FOR_ADVICE - 3
    write_run(runs_dir, "1785000000000", short)

    body = client.get("/api/coach/advice").get_json()

    player = body["players"][0]
    assert player["analytics"]["total_wall_hits"] == short
    assert player["advice"]["items"] == []
    assert "too few" in player["advice"]["note"]


def test_pooling_those_same_sessions_clears_the_advice_threshold(runs_dir, client):
    short = MIN_HITS_FOR_ADVICE - 3
    for offset in range(4):
        write_run(runs_dir, str(1785000000000 + offset), short)

    body = client.get("/api/coach/advice").get_json()

    assert body["ok"] is True
    assert body["session_count"] == 4
    for player in body["players"]:
        assert player["analytics"]["total_wall_hits"] == short * 4
        assert player["advice"]["items"], "pooled shots should clear the gate"


def test_every_advice_item_carries_the_full_four_stage_progression(runs_dir, client):
    for offset in range(4):
        write_run(runs_dir, str(1785000000000 + offset), MIN_HITS_FOR_ADVICE)

    body = client.get("/api/coach/advice").get_json()

    for player in body["players"]:
        for item in player["advice"]["items"]:
            stages = [step["stage"] for step in item["progression"]]
            assert stages == ["Solo", "Drills", "Conditioned games", "Matchplay"]
            assert all(step["text"].strip() for step in item["progression"])


def test_the_session_limit_takes_the_newest_runs(runs_dir, client):
    for offset in range(6):
        write_run(runs_dir, str(1785000000000 + offset), 2)

    body = client.get("/api/coach/advice?sessions=2").get_json()

    assert body["session_count"] == 2
    assert [session["run_id"] for session in body["sessions"]] == [
        "1785000000005", "1785000000004",
    ]


def test_runs_without_front_wall_contact_do_not_consume_a_session_slot(runs_dir, client):
    """Half-second aborted clips are common; they must not crowd out real ones."""
    write_run(runs_dir, "1785000000000", 3)
    empty = runs_dir / "1785000000001"
    empty.mkdir()
    (empty / "detected_hits.json").write_text(json.dumps({"hits": []}), encoding="utf-8")

    body = client.get("/api/coach/advice").get_json()

    assert [session["run_id"] for session in body["sessions"]] == ["1785000000000"]


def test_the_response_states_that_pooling_is_by_player_slot(runs_dir, client):
    """Attribution is per-clip, so Player 1 is a slot and not a person. The UI
    cannot state that caveat unless the payload carries it."""
    write_run(runs_dir, "1785000000000", 3)

    body = client.get("/api/coach/advice").get_json()

    assert "served first" in body["pooling_note"]


def test_no_runs_at_all_answers_with_an_empty_pool_rather_than_an_error(runs_dir, client):
    body = client.get("/api/coach/advice").get_json()

    assert body["ok"] is True
    assert body["session_count"] == 0
    assert [player["analytics"]["total_wall_hits"] for player in body["players"]] == [0, 0]


@pytest.mark.parametrize("value", ["0", "51", "many"])
def test_an_unusable_session_count_is_rejected(runs_dir, client, value):
    assert client.get(f"/api/coach/advice?sessions={value}").status_code == 400
