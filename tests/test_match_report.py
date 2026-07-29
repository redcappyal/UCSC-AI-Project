"""report-v1 assembly: everything a run found, and everything it could not.

The report is where the analysis ladder becomes visible. Its hardest
requirement is not completeness but *tolerance*: runs recorded before any of
the analysis tiers existed are still on disk and must still render. A legacy
run has no probe, no capabilities, no rally timeline -- and the difference
between "this tier found nothing" and "this run predates the tier" has to
survive all the way to the page, because they look identical once flattened
to an empty list.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from match_report import build_report


def _run_dir(tmp_path, run_id, job, detected_hits=None):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    if detected_hits is not None:
        (run_dir / "detected_hits.json").write_text(
            json.dumps(detected_hits), encoding="utf-8"
        )
    return run_dir


FULL_JOB = {
    "run_id": "1790000000000",
    "status": "complete",
    "fps": 60.0,
    "start_frame": 0,
    "end_frame": 599,
    "probe": {"fps": 60.0, "width": 1920, "height": 1080,
              "frame_count": 600, "duration_s": 10.0,
              "sharpness": 120.0, "has_audio": True},
    "capabilities": {
        "rally_structure": {"enabled": True, "reason": None},
        "player_movement": {"enabled": True, "reason": None},
        "ball_tracking": {"enabled": True, "reason": None},
        "line_calls": {"enabled": True, "reason": None},
    },
    "detection_coverage": 0.42,
    "rally_timeline": {"schema": "rally-timeline-v1", "rallies": [
        {"start_s": 1.0, "end_s": 6.0, "impact_count": 5,
         "source": "audio+motion", "confidence": 1.0}],
        "gap_s": 5.0, "audio_available": True, "agrees_with_hits": True},
    "players_v2": {"schema": "players-v2", "backend": "stub",
                   "player_a": {"distance_ft": 40.0},
                   "player_b": {"distance_ft": 35.0}},
    "hits": [{"frame": 100}],
    "target_zones": {"cells": []},
}

LEGACY_JOB = {
    "run_id": "1780000000000",
    "status": "complete",
    "fps": 60.0,
    "start_frame": 0,
    "end_frame": 599,
    "hits": [{"frame": 100}],
    "target_zones": {"cells": []},
    "player_assignment": {"rallies": []},
}


def test_a_full_run_reports_every_tier(tmp_path):
    report = build_report(_run_dir(tmp_path, "1790000000000", FULL_JOB))

    assert report["schema"] == "report-v1"
    assert report["run_id"] == "1790000000000"
    assert report["capabilities"]["ball_tracking"]["enabled"] is True
    assert report["rally_timeline"]["rallies"]
    assert report["players_v2"]["backend"] == "stub"
    assert report["detection_coverage"] == 0.42


def test_created_ms_comes_from_the_run_directory_name(tmp_path):
    """Run dirs are epoch-ms; that is the only creation time that survives."""
    report = build_report(_run_dir(tmp_path, "1790000000000", FULL_JOB))

    assert report["created_ms"] == 1790000000000


def test_a_legacy_run_renders_rather_than_raising(tmp_path):
    """~38 runs on disk predate every one of these keys."""
    report = build_report(_run_dir(tmp_path, "1780000000000", LEGACY_JOB))

    assert report["schema"] == "report-v1"
    assert report["capabilities"] is None
    assert report["probe"] is None
    assert report["rally_timeline"] is None
    assert report["players_v2"] is None


def test_a_legacy_run_says_it_is_legacy_rather_than_empty(tmp_path):
    """"No capabilities" and "analyzed before capabilities existed" are
    different facts, and a reader cannot tell them apart from an empty card."""
    report = build_report(_run_dir(tmp_path, "1780000000000", LEGACY_JOB))

    assert report["legacy"] is True
    assert "before" in report["legacy_reason"].lower()


def test_a_current_run_is_not_flagged_legacy(tmp_path):
    report = build_report(_run_dir(tmp_path, "1790000000000", FULL_JOB))

    assert report["legacy"] is False
    assert report["legacy_reason"] is None


def test_a_rally_only_run_reports_the_ball_tier_off_with_its_reason(tmp_path):
    job = dict(FULL_JOB)
    job["capabilities"] = dict(
        FULL_JOB["capabilities"],
        ball_tracking={"enabled": False, "reason": "needs >=50 fps (got 30)"},
        line_calls={"enabled": False, "reason": "ball tracking off"},
    )
    job["hits"] = []
    job.pop("detection_coverage")

    report = build_report(_run_dir(tmp_path, "1790000000001", job))

    assert report["capabilities"]["ball_tracking"]["enabled"] is False
    assert "50 fps" in report["capabilities"]["ball_tracking"]["reason"]
    assert report["shots"] is None
    assert report["rally_timeline"]["rallies"]


def test_video_facts_come_through_for_the_header(tmp_path):
    report = build_report(_run_dir(tmp_path, "1790000000000", FULL_JOB))

    assert report["video"]["fps"] == 60.0
    assert report["video"]["width"] == 1920
    assert report["video"]["duration_s"] == pytest.approx(10.0, abs=0.1)


def test_shots_carry_the_coach_analytics_when_one_is_injected(tmp_path):
    run_dir = _run_dir(tmp_path, "1790000000000", FULL_JOB,
                       detected_hits={"hits": [{"frame": 100}]})

    report = build_report(run_dir, coach_builder=lambda d: {"summary": "ok"})

    assert report["shots"]["coach"] == {"summary": "ok"}


def test_a_failing_coach_builder_costs_the_coaching_not_the_report(tmp_path):
    """Coaching is the most fragile part of the chain -- it can reach an LLM.

    A report that 500s because the narration failed would take the rally and
    movement tiers down with it, and those are computed and correct.
    """
    run_dir = _run_dir(tmp_path, "1790000000000", FULL_JOB,
                       detected_hits={"hits": [{"frame": 100}]})

    def explode(_):
        raise RuntimeError("ollama is not running")

    report = build_report(run_dir, coach_builder=explode)

    assert report["shots"] is not None
    assert report["shots"]["coach"] is None
    assert "ollama" in report["shots"]["coach_error"]


def test_a_missing_job_raises_so_the_endpoint_can_404(tmp_path):
    empty = tmp_path / "1790000000009"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        build_report(empty)


def test_tiers_enabled_lists_what_actually_ran(tmp_path):
    from match_report import tiers_enabled

    assert tiers_enabled(FULL_JOB) == [
        "rally_structure", "player_movement", "ball_tracking", "line_calls",
    ]
    assert tiers_enabled(LEGACY_JOB) == []


# --- endpoints --------------------------------------------------------------


def _client_with_runs(tmp_path, monkeypatch, runs):
    import app as app_module

    root = tmp_path / "ui_runs"
    root.mkdir()
    monkeypatch.setattr(app_module, "RUNS_DIR", root)
    for run_id, job in runs.items():
        _run_dir(root, run_id, job)
    return app_module.app.test_client()


def test_report_endpoint_serves_the_report(tmp_path, monkeypatch):
    client = _client_with_runs(tmp_path, monkeypatch,
                              {"1790000000000": FULL_JOB})

    body = client.get("/api/runs/1790000000000/report").get_json()

    assert body["ok"] is True
    assert body["report"]["schema"] == "report-v1"
    assert body["report"]["tiers_enabled"]


def test_report_endpoint_404s_on_an_unknown_run(tmp_path, monkeypatch):
    client = _client_with_runs(tmp_path, monkeypatch, {})

    assert client.get("/api/runs/nope/report").status_code == 404


def test_report_endpoint_serves_a_legacy_run_rather_than_500ing(
    tmp_path, monkeypatch
):
    client = _client_with_runs(tmp_path, monkeypatch,
                              {"1780000000000": LEGACY_JOB})

    response = client.get("/api/runs/1780000000000/report")

    assert response.status_code == 200
    assert response.get_json()["report"]["legacy"] is True


def test_the_runs_index_says_which_tiers_each_run_ran(tmp_path, monkeypatch):
    """A run list where every row looks the same is not a list worth reading."""
    client = _client_with_runs(tmp_path, monkeypatch, {
        "1790000000000": FULL_JOB,
        "1780000000000": LEGACY_JOB,
    })

    runs = {row["run_id"]: row for row in client.get("/api/runs").get_json()["runs"]}

    assert "ball_tracking" in runs["1790000000000"]["tiers_enabled"]
    assert runs["1780000000000"]["tiers_enabled"] == []


def test_the_runs_index_is_newest_first(tmp_path, monkeypatch):
    client = _client_with_runs(tmp_path, monkeypatch, {
        "1790000000000": FULL_JOB,
        "1780000000000": LEGACY_JOB,
    })

    ids = [row["run_id"] for row in client.get("/api/runs").get_json()["runs"]]

    assert ids == ["1790000000000", "1780000000000"]


def test_report_carries_ball_backend(tmp_path):
    run_dir = _run_dir(tmp_path, "1790000000010", {
        "run_id": "1790000000010", "status": "complete",
        "capabilities": {},
        "ball_backend": {"backend": "local", "name": "crosscourt-wasb-416",
                         "version": 1, "artifact_sha256": "abc",
                         "device": "cuda"},
    })
    report = build_report(run_dir)
    assert report["ball_backend"]["backend"] == "local"
    assert report["ball_backend"]["name"] == "crosscourt-wasb-416"


def test_report_ball_backend_none_for_legacy_runs(tmp_path):
    run_dir = _run_dir(tmp_path, "1790000000011", {
        "run_id": "1790000000011", "status": "complete",
    })
    report = build_report(run_dir)
    assert report["ball_backend"] is None
