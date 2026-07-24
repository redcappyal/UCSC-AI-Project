import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    build_coaching_analytics,
    local_coaching_feedback,
    local_player_coaching_feedback,
    parse_llm_coaching_report,
    player_error_metrics,
)


def sample_llm_report():
    return {
        "summary": "Player 1 used more height while Player 2 attacked lower targets.",
        "players": {
            "1": {
                "observations": [
                    "Your average wall height was 8.3 ft.",
                    "Half of your shots used the middle targets.",
                ],
                "drill_name": "Width under pressure",
                "drill_instructions": "Hit 3 sets of 10 drives, alternating back corners.",
                "drill_goal": "Land at least 7 of 10 shots outside the middle targets.",
            },
            "2": {
                "observations": [
                    "Your low attacking rate was 50%.",
                    "Your unforced-error rate was 100% from one rally loss.",
                ],
                "drill_name": "Safe attacking targets",
                "drill_instructions": "Hit 3 sets of 8 low drives above the tin.",
                "drill_goal": "Complete 6 of 8 shots without an OUT call.",
            },
        },
    }


def test_parse_llm_coaching_report_validates_required_player_sections():
    report = sample_llm_report()

    assert parse_llm_coaching_report(json.dumps(report)) == report
    assert parse_llm_coaching_report("not json") is None
    del report["players"]["2"]["drill_goal"]
    assert parse_llm_coaching_report(json.dumps(report)) is None


def test_llm_coaching_feedback_requests_structured_player_report(monkeypatch):
    import app

    report = sample_llm_report()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": json.dumps(report),
                    }],
                }],
            }).encode("utf-8")

    def fake_urlopen(request_obj, timeout):
        captured["request"] = request_obj
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COACH_MODEL", "test-model")
    monkeypatch.setenv("COACH_LLM_PROVIDER", "openai")
    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)

    result, status = app.llm_coaching_feedback({"players": [{"player_number": 1}]})

    assert result == report
    assert status == "ok"
    assert captured["timeout"] == 20
    request_body = json.loads(captured["request"].data)
    assert request_body["model"] == "test-model"
    assert request_body["text"]["format"]["type"] == "json_schema"
    assert request_body["text"]["format"]["strict"] is True
    assert request_body["text"]["format"]["schema"]["required"] == ["summary", "players"]
    assert "test-key" not in captured["request"].data.decode("utf-8")


def test_ollama_coaching_feedback_uses_local_structured_output(monkeypatch):
    import app

    report = sample_llm_report()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "message": {"role": "assistant", "content": json.dumps(report)}
            }).encode("utf-8")

    def fake_urlopen(request_obj, timeout):
        captured["request"] = request_obj
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("COACH_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_COACH_MODEL", "test-local-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9999/")
    monkeypatch.setenv("OLLAMA_COACH_TIMEOUT_SECONDS", "45")
    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)

    result, status = app.llm_coaching_feedback({"players": [{"player_number": 1}]})

    assert result == report
    assert status == "ok"
    assert captured["timeout"] == 45
    assert captured["request"].full_url == "http://127.0.0.1:9999/api/chat"
    assert captured["request"].get_header("Authorization") is None
    request_body = json.loads(captured["request"].data)
    assert request_body["model"] == "test-local-model"
    assert request_body["stream"] is False
    assert request_body["think"] is False
    assert request_body["format"]["required"] == ["summary", "players"]
    assert request_body["options"]["num_ctx"] == 8192


def test_player_error_metrics_classifies_lost_rallies_and_calculates_percentage():
    rallies = [
        {
            "winner_player_number": 1,
            "last_player_number": 2,
            "last_call": "OUT",
        },
        {
            "winner_player_number": 2,
            "last_player_number": 2,
            "last_call": "IN",
        },
        {
            "winner_player_number": 2,
            "last_player_number": 1,
            "last_call": "OUT",
        },
        {
            "winner_player_number": None,
            "last_player_number": 1,
            "last_call": None,
        },
    ]

    assert player_error_metrics(rallies, 1) == {
        "unforced_errors": 1,
        "forced_errors": 1,
        "total_errors": 2,
        "average_rally_duration_seconds": None,
        "unforced_error_percentage": 50.0,
    }
    assert player_error_metrics(rallies, 2) == {
        "unforced_errors": 1,
        "forced_errors": 0,
        "total_errors": 1,
        "average_rally_duration_seconds": None,
        "unforced_error_percentage": 100.0,
    }


def test_player_error_metrics_has_no_percentage_without_a_rally_loss():
    assert player_error_metrics([], 1) == {
        "unforced_errors": 0,
        "forced_errors": 0,
        "total_errors": 0,
        "average_rally_duration_seconds": None,
        "unforced_error_percentage": None,
    }


def test_player_error_metrics_averages_duration_of_rallies_won_by_each_player():
    rallies = [
        {
            "winner_player_number": 1,
            "duration_seconds": 5.0,
            "last_player_number": 1,
            "last_call": "IN",
        },
        {
            "winner_player_number": 1,
            "start_time_seconds": 10.0,
            "end_time_seconds": 13.0,
            "last_player_number": 1,
            "last_call": "IN",
        },
        {
            "winner_player_number": 2,
            "start_time_seconds": 20.0,
            "end_time_seconds": 26.0,
            "last_player_number": 2,
            "last_call": "IN",
        },
        {
            "winner_player_number": 2,
            "start_time_seconds": 30.0,
            "end_time_seconds": 29.0,
            "last_player_number": 2,
            "last_call": "IN",
        },
    ]

    assert player_error_metrics(rallies, 1)["average_rally_duration_seconds"] == 4.0
    assert player_error_metrics(rallies, 2)["average_rally_duration_seconds"] == 6.0


def test_coaching_analytics_summarizes_target_and_speed_data():
    payload = {
        "hits": [
            {
                "event_type": "wall",
                "player_number": 1,
                "call": "IN",
                "target_zone": {"zone": 4},
                "wall_diagram": {"x": 0.50, "y": 0.25},
                "velocity": {
                    "speed_before": {"mph": 52.0},
                    "speed_after": {"mph": 31.0},
                    "velocity_change": {"mph": 24.0},
                },
            },
            {
                "event_type": "wall",
                "player_number": 2,
                "call": "OUT",
                "target_zone": {"zone": 5},
                "wall_diagram": {"x": 0.45, "y": 0.75},
                "velocity": {
                    "speed_before": {"mph": 48.0},
                    "speed_after": {"mph": 29.0},
                    "velocity_change": {"mph": 22.0},
                },
            },
            {
                "event_type": "floor",
                "call": None,
                "court_position_ft": {"x": 10.5, "y": 15.0},
            },
        ],
        "target_zones": {
            "total_wall_hits": 2,
            "zones": [
                {"zone": 1, "count": 0, "percentage": 0.0},
                {"zone": 2, "count": 0, "percentage": 0.0},
                {"zone": 3, "count": 0, "percentage": 0.0},
                {"zone": 4, "count": 1, "percentage": 50.0},
                {"zone": 5, "count": 1, "percentage": 50.0},
            ],
            "common_zones": [
                {"zone": 4, "count": 1, "percentage": 50.0},
                {"zone": 5, "count": 1, "percentage": 50.0},
            ],
            "missing_zones": [
                {"zone": 1, "count": 0, "percentage": 0.0},
                {"zone": 2, "count": 0, "percentage": 0.0},
                {"zone": 3, "count": 0, "percentage": 0.0},
            ],
        },
        "target_zones_by_player": {
            "1": {
                "total_wall_hits": 1,
                "zones": [
                    {"zone": 1, "count": 0, "percentage": 0.0},
                    {"zone": 2, "count": 0, "percentage": 0.0},
                    {"zone": 3, "count": 0, "percentage": 0.0},
                    {"zone": 4, "count": 1, "percentage": 100.0},
                    {"zone": 5, "count": 0, "percentage": 0.0},
                ],
                "common_zones": [{"zone": 4, "count": 1, "percentage": 100.0}],
                "missing_zones": [
                    {"zone": 1, "count": 0, "percentage": 0.0},
                    {"zone": 2, "count": 0, "percentage": 0.0},
                    {"zone": 3, "count": 0, "percentage": 0.0},
                    {"zone": 5, "count": 0, "percentage": 0.0},
                ],
            },
            "2": {
                "total_wall_hits": 1,
                "zones": [
                    {"zone": 1, "count": 0, "percentage": 0.0},
                    {"zone": 2, "count": 0, "percentage": 0.0},
                    {"zone": 3, "count": 0, "percentage": 0.0},
                    {"zone": 4, "count": 0, "percentage": 0.0},
                    {"zone": 5, "count": 1, "percentage": 100.0},
                ],
                "common_zones": [{"zone": 5, "count": 1, "percentage": 100.0}],
                "missing_zones": [
                    {"zone": 1, "count": 0, "percentage": 0.0},
                    {"zone": 2, "count": 0, "percentage": 0.0},
                    {"zone": 3, "count": 0, "percentage": 0.0},
                    {"zone": 4, "count": 0, "percentage": 0.0},
                ],
            },
        },
        "floor_zones": {
            "total_floor_bounces": 1,
            "common_zones": [{"zone": "middle", "count": 1, "percentage": 100.0}],
            "missing_zones": [],
        },
        "rallies": [
            {
                "winner_player_number": 1,
                "start_time_seconds": 1.0,
                "end_time_seconds": 5.5,
                "last_player_number": 2,
                "last_call": "OUT",
            },
            {
                "winner_player_number": 2,
                "duration_seconds": 7.0,
                "last_player_number": 2,
                "last_call": "IN",
            },
        ],
    }

    analytics = build_coaching_analytics(payload)

    assert analytics["total_wall_hits"] == 2
    assert analytics["total_floor_bounces"] == 1
    assert analytics["center_target_rate"] == 100.0
    assert analytics["side_target_rate"] == 0.0
    assert analytics["average_incoming_speed_mph"] == 50.0
    assert analytics["average_exit_speed_mph"] == 30.0
    assert analytics["average_velocity_change_mph"] == 23.0
    assert analytics["average_wall_height_ft"] == pytest.approx(8.3, abs=0.1)
    assert analytics["in_count"] == 1
    assert analytics["out_count"] == 1
    assert analytics["players"][0]["player_number"] == 1
    assert analytics["players"][0]["total_wall_hits"] == 1
    assert analytics["players"][0]["common_target_zones"][0]["zone"] == 4
    assert analytics["players"][0]["unforced_errors"] == 0
    assert analytics["players"][0]["forced_errors"] == 1
    assert analytics["players"][0]["average_rally_duration_seconds"] == 4.5
    assert analytics["players"][0]["unforced_error_percentage"] == 0.0
    assert analytics["players"][1]["player_number"] == 2
    assert analytics["players"][1]["total_wall_hits"] == 1
    assert analytics["players"][1]["common_target_zones"][0]["zone"] == 5
    assert analytics["players"][1]["unforced_errors"] == 1
    assert analytics["players"][1]["forced_errors"] == 0
    assert analytics["players"][1]["average_rally_duration_seconds"] == 7.0
    assert analytics["players"][1]["unforced_error_percentage"] == 100.0

    feedback = local_coaching_feedback(analytics)
    assert "Player 1" in feedback
    assert "Player 2" in feedback
    assert "zone 4" in feedback
    assert "middle of the wall" in feedback


def test_coaching_analytics_splits_each_player_by_won_and_lost_rallies():
    def wall_hit(player_number, rally_number, zone, y, incoming_mph):
        return {
            "event_type": "wall",
            "player_number": player_number,
            "rally_number": rally_number,
            "call": "IN",
            "target_zone": {"zone": zone},
            "wall_diagram": {"x": 0.5, "y": y},
            "velocity": {
                "speed_before": {"mph": incoming_mph},
                "speed_after": {"mph": incoming_mph - 10},
            },
        }

    payload = {
        "hits": [
            wall_hit(1, 1, 4, 0.2, 50),
            wall_hit(2, 1, 1, 0.7, 30),
            wall_hit(1, 2, 5, 0.8, 32),
            wall_hit(2, 2, 4, 0.3, 48),
        ],
        "rallies": [
            {
                "rally_number": 1,
                "winner_player_number": 1,
                "duration_seconds": 8.0,
                "last_player_number": 1,
                "last_call": "IN",
            },
            {
                "rally_number": 2,
                "winner_player_number": 2,
                "duration_seconds": 4.0,
                "last_player_number": 2,
                "last_call": "IN",
            },
        ],
    }

    analytics = build_coaching_analytics(payload)
    player_1 = analytics["players"][0]["rally_outcome_analytics"]
    player_2 = analytics["players"][1]["rally_outcome_analytics"]

    assert player_1["winning"]["rally_count"] == 1
    assert player_1["winning"]["total_wall_hits"] == 1
    assert player_1["winning"]["common_target_zones"][0]["zone"] == 4
    assert player_1["winning"]["average_incoming_speed_mph"] == 50.0
    assert player_1["winning"]["average_rally_duration_seconds"] == 8.0
    assert player_1["winning"]["average_wall_height_ft"] > player_1["losing"]["average_wall_height_ft"]
    assert player_1["losing"]["common_target_zones"][0]["zone"] == 5
    assert player_1["losing"]["average_incoming_speed_mph"] == 32.0
    assert player_1["losing"]["average_rally_duration_seconds"] == 4.0

    assert player_2["winning"]["average_incoming_speed_mph"] == 48.0
    assert player_2["losing"]["average_incoming_speed_mph"] == 30.0
    assert player_2["winning"]["average_wall_height_ft"] > player_2["losing"]["average_wall_height_ft"]

    feedback = local_player_coaching_feedback(analytics["players"][0])
    assert "in won rallies versus" in feedback
    assert "average shot height" in feedback


def test_coaching_analytics_excludes_serves_from_all_shot_metrics():
    payload = {
        "hits": [
            {
                "event_type": "wall",
                "player_number": 1,
                "rally_number": 1,
                "rally_hit_sequence": 1,
                "is_serve": True,
                "call": "IN",
                "target_zone": {"zone": 4},
                "wall_diagram": {"x": 0.5, "y": 0.1},
                "velocity": {
                    "speed_before": {"mph": 100},
                    "speed_after": {"mph": 80},
                },
            },
            {
                "event_type": "wall",
                "player_number": 1,
                "rally_number": 1,
                "rally_hit_sequence": 3,
                "call": "IN",
                "target_zone": {"zone": 5},
                "wall_diagram": {"x": 0.5, "y": 0.8},
                "velocity": {
                    "speed_before": {"mph": 20},
                    "speed_after": {"mph": 10},
                },
            },
        ],
        "rallies": [{
            "rally_number": 1,
            "winner_player_number": 1,
            "duration_seconds": 6.0,
            "last_player_number": 1,
            "last_call": "IN",
        }],
    }

    analytics = build_coaching_analytics(payload)
    player = analytics["players"][0]
    winning = player["rally_outcome_analytics"]["winning"]

    assert analytics["total_wall_hits"] == 1
    assert analytics["average_incoming_speed_mph"] == 20.0
    assert analytics["common_target_zones"][0]["zone"] == 5
    assert player["total_wall_hits"] == 1
    assert player["average_incoming_speed_mph"] == 20.0
    assert winning["total_wall_hits"] == 1
    assert winning["average_incoming_speed_mph"] == 20.0
    assert winning["average_rally_duration_seconds"] == 6.0


def test_rally_outcome_analytics_matches_legacy_hits_using_rally_time_bounds():
    payload = {
        "hits": [{
            "event_type": "wall",
            "player_number": 1,
            "timestamp_seconds": 2.0,
            "call": "IN",
            "target_zone": {"zone": 4},
            "wall_diagram": {"x": 0.5, "y": 0.3},
        }],
        "rallies": [{
            "rally_number": 3,
            "winner_player_number": 1,
            "start_time_seconds": 1.0,
            "end_time_seconds": 5.0,
            "last_player_number": 1,
            "last_call": "IN",
        }],
    }

    analytics = build_coaching_analytics(payload)

    winning = analytics["players"][0]["rally_outcome_analytics"]["winning"]
    assert winning["rally_count"] == 1
    assert winning["total_wall_hits"] == 1
    assert winning["average_rally_duration_seconds"] == 4.0


def test_coach_route_returns_local_feedback_without_waiting_for_llm(tmp_path, monkeypatch):
    import app

    run_id = "coach-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    run_dir.joinpath("detected_hits.json").write_text(
        """
        {
          "hits": [],
          "target_zones": {"total_wall_hits": 0, "zones": []},
          "floor_zones": {"total_floor_bounces": 0}
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("COACH_LLM_PROVIDER", "openai")

    def unexpected_llm_call(analytics):
        raise AssertionError("The instant local route must not call the LLM")

    monkeypatch.setattr(app, "llm_coaching_feedback", unexpected_llm_call)

    client = app.app.test_client()
    response = client.get(f"/api/runs/{run_id}/coach")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["feedback_source"] == "local"
    assert payload["llm_status"] == "pending"
    assert payload["analytics"]["total_wall_hits"] == 0
    assert sorted(payload["player_feedback"]) == ["1", "2"]
    assert "Player 1" in payload["player_feedback"]["1"]
    assert "Player 2" in payload["player_feedback"]["2"]
    assert payload["player_feedback_source"] == "local"


def test_coach_route_displays_llm_feedback_and_drills_for_each_player(tmp_path, monkeypatch):
    import app

    run_id = "coach-llm-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    run_dir.joinpath("detected_hits.json").write_text(
        """
        {
          "hits": [],
          "target_zones": {"total_wall_hits": 0, "zones": []},
          "floor_zones": {"total_floor_bounces": 0}
        }
        """,
        encoding="utf-8",
    )
    report = sample_llm_report()
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("COACH_LLM_PROVIDER", "openai")
    monkeypatch.setattr(app, "llm_coaching_feedback", lambda analytics: (report, "ok"))

    client = app.app.test_client()
    response = client.post(f"/api/runs/{run_id}/coach/llm")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["feedback"] == report["summary"]
    assert payload["feedback_source"] == "llm"
    assert payload["player_feedback_source"] == "llm"
    assert "Your average wall height was 8.3 ft." in payload["player_feedback"]["1"]
    assert "Drill: Width under pressure" in payload["player_feedback"]["1"]
    assert "Goal: Land at least 7 of 10" in payload["player_feedback"]["1"]
    assert "Drill: Safe attacking targets" in payload["player_feedback"]["2"]
