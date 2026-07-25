"""Bounce corrections: the /api/runs/<id>/corrections endpoint (schema v2).

A correction records the human's view of one detected hit: the hit type
(wall / side_wall / floor / racket / none), the corrected ball position,
bounce timing, and — for front-wall hits only — the IN/OUT call.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """An isolated run directory the corrections endpoint can see.

    These tests used to write into the real RUNS_DIR under fixed run ids and
    clean up with `shutil.rmtree(..., ignore_errors=True)`. On Windows that
    rmtree fails while the app still holds `corrections.json` open, and
    `ignore_errors` swallows the failure: the surviving file made the next
    run's first POST answer `count: 2`, so the suite alternated pass and fail
    run over run. Redirecting RUNS_DIR at `tmp_path` deletes the shared state
    instead of depending on cleanup having worked — the same approach
    test_pipeline.py and test_coaching_analytics.py already take.
    """
    import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)
    directory = tmp_path / "corr-test-run"
    directory.mkdir()
    return directory


def post(client, run_id, payload):
    return client.post(f"/api/runs/{run_id}/corrections", json=payload)


def wall_payload(frame=42, **overrides):
    payload = {
        "frame": frame,
        "corrected": {
            "type": "wall", "call": "OUT",
            "ball": {"x": 100.0, "y": 200.0},
            "frame_is_bounce": True, "frame": None,
        },
        "predicted": {
            "type": "wall", "call": "IN", "source": "impact_estimate",
            "margin_px": 3.5, "ball": {"x": 98.0, "y": 205.0},
        },
    }
    payload.update(overrides)
    return payload


def test_correction_roundtrip_and_upsert(run_dir):
    import app as app_module

    client = app_module.app.test_client()
    response = post(client, run_dir.name, wall_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["count"] == 1
    entry = body["correction"]
    assert entry["corrected"]["call"] == "OUT"
    # Server-derived agreement: right type, wrong call, right timing.
    assert entry["agrees"] == {"type": True, "call": False, "frame": True}

    # A second frame, submitted out of order, lands sorted on disk.
    response = post(client, run_dir.name, {
        "frame": 7,
        "corrected": {"type": "floor", "ball": {"x": 400, "y": 700},
                      "frame_is_bounce": False, "frame": 9},
        "predicted": {"type": "floor", "ball": {"x": 402, "y": 690}},
    })
    entry = response.get_json()["correction"]
    assert entry["agrees"] == {"type": True, "call": None, "frame": False}
    assert entry["corrected"]["frame"] == 9

    stored = json.loads((run_dir / "corrections.json").read_text())
    assert stored["schema_version"] == "corrections-v2"
    assert [c["frame"] for c in stored["corrections"]] == [7, 42]
    assert all("recorded_at" in c for c in stored["corrections"])

    # Re-correcting a frame replaces its entry rather than duplicating it.
    response = post(client, run_dir.name, {
        "frame": 42, "corrected": {"type": "none"},
        "predicted": {"type": "wall", "call": "IN"},
    })
    assert response.get_json()["count"] == 2
    stored = json.loads((run_dir / "corrections.json").read_text())
    by_frame = {c["frame"]: c for c in stored["corrections"]}
    assert by_frame[42]["corrected"]["type"] == "none"

    # Fetchable through the generic run-file route (the UI's load path).
    fetched = client.get(f"/api/runs/{run_dir.name}/corrections.json")
    assert fetched.status_code == 200
    assert len(fetched.get_json()["corrections"]) == 2


def test_none_correction_carries_no_position_or_timing(run_dir):
    import app as app_module

    client = app_module.app.test_client()
    response = post(client, run_dir.name, {
        "frame": 12, "corrected": {"type": "none"},
        "predicted": {"type": "wall", "call": "OUT", "margin_px": -8.0,
                      "ball": {"x": 50, "y": 60}},
    })
    assert response.status_code == 200
    corrected = response.get_json()["correction"]["corrected"]
    assert corrected == {"type": "none", "call": None, "ball": None,
                         "frame_is_bounce": None, "frame": None}
    agrees = response.get_json()["correction"]["agrees"]
    assert agrees == {"type": False, "call": None, "frame": None}


def test_get_corrections_empty_and_populated(run_dir):
    import app as app_module

    client = app_module.app.test_client()
    # A run with nothing recorded answers with an empty list, not a 404.
    response = client.get(f"/api/runs/{run_dir.name}/corrections")
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True, "schema_version": None, "corrections": []}

    post(client, run_dir.name, wall_payload(frame=3))
    response = client.get(f"/api/runs/{run_dir.name}/corrections")
    body = response.get_json()
    assert body["schema_version"] == "corrections-v2"
    assert [c["frame"] for c in body["corrections"]] == [3]

    assert client.get("/api/runs/no-such-run/corrections").status_code == 404


def test_correction_undo_removes_entry(run_dir):
    import app as app_module

    client = app_module.app.test_client()
    post(client, run_dir.name, wall_payload(frame=10))
    response = post(client, run_dir.name, {"frame": 10, "corrected": None})
    body = response.get_json()
    assert body["count"] == 0
    assert body["correction"] is None
    stored = json.loads((run_dir / "corrections.json").read_text())
    assert stored["corrections"] == []


def test_correction_validation(run_dir):
    import app as app_module

    client = app_module.app.test_client()

    def rejected(payload):
        return post(client, run_dir.name, payload).status_code == 400

    # Frame problems.
    assert rejected(wall_payload(frame="not-a-number"))
    bad = wall_payload()
    del bad["frame"]
    assert rejected(bad)

    # Type problems.
    assert rejected({"frame": 5, "corrected": {"type": "ceiling"}})
    assert rejected({"frame": 5, "corrected": {}})
    assert rejected({"frame": 5, "corrected": "IN"})   # v1-style scalar

    # Call rules: wall requires one, non-wall forbids one.
    bad = wall_payload()
    bad["corrected"]["call"] = None
    assert rejected(bad)
    bad = wall_payload()
    bad["corrected"]["call"] = "LET"
    assert rejected(bad)
    assert rejected({"frame": 5, "corrected": {
        "type": "floor", "call": "IN", "ball": {"x": 1, "y": 2},
        "frame_is_bounce": True}})

    # Ball rules.
    bad = wall_payload()
    bad["corrected"]["ball"] = None
    assert rejected(bad)
    bad = wall_payload()
    bad["corrected"]["ball"] = {"x": "wide", "y": 2}
    assert rejected(bad)
    assert rejected({"frame": 5, "corrected": {
        "type": "none", "ball": {"x": 1, "y": 2}}})

    # Timing rules.
    bad = wall_payload()
    bad["corrected"]["frame_is_bounce"] = "yes"
    assert rejected(bad)
    bad = wall_payload()
    bad["corrected"]["frame_is_bounce"] = False   # ...but no frame given
    assert rejected(bad)
    bad = wall_payload()
    bad["corrected"]["frame"] = 43                # frame given but bounce=true
    assert rejected(bad)

    # Malformed predicted snapshots are client bugs, not user input.
    bad = wall_payload()
    bad["predicted"] = {"call": "MAYBE"}
    assert rejected(bad)
    bad = wall_payload()
    bad["predicted"] = {"type": "none"}   # detector never predicts "none"
    assert rejected(bad)

    # Case is normalized rather than rejected.
    payload = wall_payload(frame=5)
    payload["corrected"]["type"] = "WALL"
    payload["corrected"]["call"] = "out"
    response = post(client, run_dir.name, payload)
    assert response.status_code == 200
    corrected = response.get_json()["correction"]["corrected"]
    assert (corrected["type"], corrected["call"]) == ("wall", "OUT")

    assert post(client, "no-such-run",
                wall_payload()).status_code == 404
