import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import build_coaching_analytics, local_coaching_feedback


def test_coaching_analytics_summarizes_target_and_speed_data():
    payload = {
        "hits": [
            {
                "event_type": "wall",
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
        "floor_zones": {
            "total_floor_bounces": 1,
            "common_zones": [{"zone": "middle", "count": 1, "percentage": 100.0}],
            "missing_zones": [],
        },
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

    feedback = local_coaching_feedback(analytics)
    assert "zone 4" in feedback
    assert "middle of the wall" in feedback


def test_coach_route_returns_local_feedback(tmp_path, monkeypatch):
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
    monkeypatch.setattr(app, "llm_coaching_feedback", lambda analytics: (None, "missing_api_key"))

    client = app.app.test_client()
    response = client.get(f"/api/runs/{run_id}/coach")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["feedback_source"] == "local"
    assert payload["llm_status"] == "missing_api_key"
    assert payload["analytics"]["total_wall_hits"] == 0
