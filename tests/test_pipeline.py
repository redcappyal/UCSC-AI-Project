"""Pipeline checks that need no model: windows, persistence, dedup, routes."""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import job_runner
from job_runner import audio_hits_from_candidates, refine_segments_for_audio_candidates, refine_segments_for_hits


def test_refine_windows_merge_and_clamp():
    segments = refine_segments_for_hits(
        [{"hit_frame": 100}, {"hit_frame": 110}, {"hit_frame": 300}], 0, 1000, 3
    )
    assert segments == [(88, 122, 1), (288, 312, 1)]


def test_refine_windows_clamped_to_clip():
    segments = refine_segments_for_hits([{"hit_frame": 5}], 0, 1000, 1)
    assert segments == [(0, 17, 1)]


def test_audio_candidate_windows_merge_and_emit_hits():
    candidates = [
        {"frame": 100, "window_start_frame": 96, "window_end_frame": 104, "score": 12.5},
        {"frame": 110, "window_start_frame": 106, "window_end_frame": 114, "score": 8.0},
    ]
    assert refine_segments_for_audio_candidates(candidates, 0, 200) == [(92, 118, 1)]

    hits = audio_hits_from_candidates(candidates, 50.0)
    assert hits[0]["source"] == "audio"
    assert hits[0]["call"] == "AUDIO"
    assert hits[0]["timestamp_seconds"] == 2.0
    assert hits[0]["window_start_seconds"] == 96 / 50.0


def test_judge_hits_skips_line_call_for_racket_events(tmp_path):
    from job_runner import judge_hits

    results = {
        60: {
            "source_frame": 60,
            "timestamp_seconds": "2.000000",
            "detected": "True",
            "x_center": "900.000",
            "y_center": "180.000",
        }
    }
    signals = {"audio_score": 6.0, "audio_rms": 0.05, "audio_offset_s": 0.01,
               "ball_size_px": 30.0, "size_ratio": 1.5, "gap_prev_s": None, "gap_next_s": 1.2}
    hit = {
        "hit_frame": 60,
        "timestamp_seconds": 2.0,
        "dv_magnitude": 400.0,
        "after_gap": False,
        "event_type": "racket",
        "wall_score": -0.8,
        "signals": signals,
    }

    judged = judge_hits(tmp_path, results, [hit], audio_available=True)

    entry = judged[0]
    # Verdicts apply to front-wall hits only: no call for racket events.
    assert entry["call"] is None
    assert entry["reason"] == "classified_as_racket"
    assert entry["margin_px"] is None and entry["judge_source"] is None
    assert entry["event_type"] == "racket"
    assert entry["wall_score"] == -0.8
    assert entry["signals"] == signals

    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    assert payload["audio_available"] is True
    assert payload["hits"][0]["call"] is None
    assert payload["target_zones"]["total_wall_hits"] == 0


def test_judge_hits_wall_events_judged_as_before(tmp_path):
    import json as json_module

    from job_runner import judge_hits

    (tmp_path / "calibration.json").write_text(json_module.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
        ]
    }))
    results = {
        60: {
            "source_frame": 60,
            "timestamp_seconds": "2.000000",
            "detected": "True",
            "x_center": "900.000",
            "y_center": "180.000",
        }
    }
    hit = {
        "hit_frame": 60,
        "timestamp_seconds": 2.0,
        "dv_magnitude": 400.0,
        "speed_before": 400.0,
        "speed_after": 380.0,
        "after_gap": False,
        "event_type": "wall",
        "wall_score": 0.9,
        "signals": {"audio_score": 24.0},
    }

    judged = judge_hits(tmp_path, results, [hit], audio_available=True)

    entry = judged[0]
    assert entry["call"] == "IN"
    assert entry["judge_source"] == "detected_center"
    assert entry["event_type"] == "wall"
    assert entry["wall_diagram"]["x"] == 0.45
    assert entry["target_zone"]["zone"] == 4

    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    # This is the rally's first front-wall contact, so it remains judgeable
    # but is excluded from coaching target metrics as a serve.
    assert payload["target_zones"]["total_wall_hits"] == 0
    assert payload["target_zones"]["layout"] == "front_wall_9_target"
    assert payload["target_zones"]["zones"][3]["count"] == 0
    assert payload["target_zones"]["common_zones"] == []


def test_first_front_wall_hit_in_each_rally_uses_service_line(tmp_path, monkeypatch):
    from job_runner import judge_hits

    monkeypatch.setenv("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS", "5")
    tmp_path.joinpath("calibration.json").write_text(json.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 100], [1000, 100]]},
            {"name": "service_line_top_edge", "endpoints": [[0, 400], [1000, 400]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [1000, 700]]},
        ]
    }))
    results = {
        10: {
            "source_frame": 10, "timestamp_seconds": "1.0", "detected": "True",
            "x_center": "500", "y_center": "250",
        },
        20: {
            "source_frame": 20, "timestamp_seconds": "2.0", "detected": "True",
            "x_center": "500", "y_center": "500",
        },
        100: {
            "source_frame": 100, "timestamp_seconds": "10.0", "detected": "True",
            "x_center": "500", "y_center": "500",
        },
    }
    detected = [
        {
            "hit_frame": frame,
            "timestamp_seconds": float(results[frame]["timestamp_seconds"]),
            "event_type": "wall",
        }
        for frame in (10, 20, 100)
    ]

    judged = judge_hits(tmp_path, results, detected)

    assert [hit["call"] for hit in judged] == ["IN", "IN", "OUT"]
    assert [hit.get("is_serve", False) for hit in judged] == [True, False, True]
    assert judged[0]["reason"] == "between_out_and_service_lines"
    assert judged[1]["reason"] == "between_lines"
    assert judged[2]["reason"] == "below_or_on_service_line"
    assert judged[2]["standard_call"] == "IN"
    assert judged[2]["margin_px"] == -100

    payload = json.loads(tmp_path.joinpath("detected_hits.json").read_text())
    rallies = payload["rallies"]
    assert [rally["serve_call"] for rally in rallies] == ["IN", "OUT"]
    assert [rally["winner_player_number"] for rally in rallies] == [2, 1]
    assert rallies[1]["winner_reason"] == "serve_out"
    assert judged[2]["player_number"] == 2
    assert payload["target_zones"]["total_wall_hits"] == 1


def test_judge_endpoint_preserves_serve_specific_call(tmp_path, monkeypatch):
    import app as app_module
    from job_runner import judge_hits

    run_id = "serve-judge-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    run_dir.joinpath("calibration.json").write_text(json.dumps({
        "frame_width": 1000,
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 100], [1000, 100]]},
            {"name": "service_line_top_edge", "endpoints": [[0, 400], [1000, 400]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [1000, 700]]},
        ],
    }))
    results = {
        10: {
            "source_frame": 10, "timestamp_seconds": "1.0", "detected": "True",
            "x_center": "500", "y_center": "500",
        }
    }
    judge_hits(run_dir, results, [{
        "hit_frame": 10, "timestamp_seconds": 1.0, "event_type": "wall",
    }])
    run_dir.joinpath("ball_coordinates.csv").write_text(
        "source_frame,detected,x_center,y_center\n10,True,500,500\n"
    )
    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)
    app_module.BALL_POSITIONS_CACHE.pop(run_id, None)
    app_module.RUN_HITS_CACHE.pop(run_id, None)

    response = app_module.app.test_client().post(
        "/api/judge", json={"run_id": run_id, "frame": 10}
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["is_serve"] is True
    assert payload["standard_call"] == "IN"
    assert payload["call"] == "OUT"
    assert payload["reason"] == "below_or_on_service_line"
    assert payload["service_y"] == 400


def test_judge_hits_clamps_horizontal_overflow_into_regular_shot(tmp_path):
    import json as json_module

    from job_runner import judge_hits

    (tmp_path / "calibration.json").write_text(json_module.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [1100, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [1100, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [200, 50]},
                    {"id": "top_right", "tap_px": [1000, 80]},
                    {"id": "bottom_right", "tap_px": [940, 760]},
                    {"id": "bottom_left", "tap_px": [260, 730]},
                ]
            }
        },
    }))
    results = {
        60: {
            "source_frame": 60,
            "timestamp_seconds": "2.000000",
            "detected": "True",
            "x_center": "160.000",
            "y_center": "400.000",
        }
    }
    hit = {
        "hit_frame": 60,
        "timestamp_seconds": 2.0,
        "dv_magnitude": 400.0,
        "speed_before": 400.0,
        "speed_after": 380.0,
        "after_gap": False,
        "event_type": "wall",
    }

    judged = judge_hits(tmp_path, results, [hit])

    entry = judged[0]
    assert entry["call"] == "IN"
    assert entry["reason"] == "between_lines_x_clamped"
    assert entry["judge_source"] == "detected_center_x_clamped"
    assert entry["judge_point"]["x_clamped"] is True
    assert entry["wall_diagram"]["x"] == 0.0
    assert entry["target_zone"] is not None
    assert entry["player_number"] == 1
    assert entry["rally_number"] == 1

    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    # This one-hit fixture is a serve, so it is deliberately excluded from
    # aggregate target-zone rates even though it has a valid target_zone.
    assert payload["target_zones"]["total_wall_hits"] == 0
    assert payload["player_assignment"]["assigned_front_wall_hit_count"] == 1
    assert payload["player_assignment"]["rally_count"] == 1


def test_target_zone_layout_matches_front_wall_sketch():
    """3x3 grid: columns left/center/right are 1-3, 4-6, 7-9; within a column
    the rows run lob, driving height, low."""
    from job_runner import target_zone_for_diagram

    # left column
    assert target_zone_for_diagram({"x": 0.08, "y": 0.10})["zone"] == 1
    assert target_zone_for_diagram({"x": 0.08, "y": 0.50})["zone"] == 2
    assert target_zone_for_diagram({"x": 0.08, "y": 0.76})["zone"] == 3
    # center column — the lob/normal split now applies here too
    assert target_zone_for_diagram({"x": 0.50, "y": 0.10})["zone"] == 4
    assert target_zone_for_diagram({"x": 0.50, "y": 0.50})["zone"] == 5
    assert target_zone_for_diagram({"x": 0.50, "y": 0.80})["zone"] == 6
    # right column
    assert target_zone_for_diagram({"x": 0.92, "y": 0.10})["zone"] == 7
    assert target_zone_for_diagram({"x": 0.92, "y": 0.50})["zone"] == 8
    assert target_zone_for_diagram({"x": 0.92, "y": 0.95})["zone"] == 9


def test_target_zone_bands_are_the_same_height_in_every_column():
    """The old layout gave the center one tall band while the sides had two,
    so "high" meant a different contact height depending on where it landed."""
    from job_runner import target_zone_for_diagram

    for y, row in ((0.10, 0), (0.50, 1), (0.90, 2)):
        bands = {target_zone_for_diagram({"x": x, "y": y})["band"]
                 for x in (0.08, 0.50, 0.92)}
        assert len(bands) == 1, f"row {row} disagreed across columns: {bands}"
    assert target_zone_for_diagram({"x": 0.5, "y": 0.10})["band"] == "lob"
    assert target_zone_for_diagram({"x": 0.5, "y": 0.50})["band"] == "normal"
    assert target_zone_for_diagram({"x": 0.5, "y": 0.90})["band"] == "low"


def test_target_zone_summary_excludes_serves_from_metrics():
    from job_runner import build_target_zone_summary

    hits = [
        {
            "event_type": "wall",
            "is_serve": True,
            "rally_hit_sequence": 1,
            "target_zone": {"zone": 4},
        },
        {
            "event_type": "wall",
            "rally_hit_sequence": 2,
            "target_zone": {"zone": 5},
        },
    ]

    summary = build_target_zone_summary(hits)

    assert summary["total_wall_hits"] == 1
    assert summary["zones"][3]["count"] == 0
    assert summary["zones"][4]["count"] == 1


def test_target_zones_are_split_by_players_within_one_rally(monkeypatch):
    from job_runner import assign_front_wall_hit_players, build_player_target_zone_summaries

    monkeypatch.setenv("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS", "5")
    hits = [
        {"frame": 10, "timestamp_seconds": 1.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 1}},
        {"frame": 20, "timestamp_seconds": 2.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 4}},
        {"frame": 30, "timestamp_seconds": 3.0, "event_type": "floor", "target_zone": {"zone": 5}},
        {"frame": 40, "timestamp_seconds": 4.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 5}},
        {"frame": 50, "timestamp_seconds": 5.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 2}},
    ]

    assignment = assign_front_wall_hit_players(hits)
    summaries = build_player_target_zone_summaries(hits)

    assert [hit.get("player_number") for hit in hits if hit.get("event_type") == "wall"] == [1, 2, 1, 2]
    assert assignment["rally_count"] == 1
    assert hits[0]["front_wall_sequence"] == 1
    assert hits[1]["front_wall_sequence"] == 2
    assert summaries["1"]["total_wall_hits"] == 1
    assert summaries["1"]["zones"][0]["count"] == 0
    assert summaries["1"]["zones"][4]["count"] == 1
    assert summaries["2"]["total_wall_hits"] == 2
    assert summaries["2"]["zones"][1]["count"] == 1
    assert summaries["2"]["zones"][3]["count"] == 1


def test_player_assignment_resets_each_rally_from_previous_winner(monkeypatch):
    from job_runner import assign_front_wall_hit_players

    monkeypatch.setenv("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS", "5")
    hits = [
        {"frame": 10, "timestamp_seconds": 1.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 1}},
        {"frame": 20, "timestamp_seconds": 2.0, "event_type": "wall", "call": "OUT", "target_zone": {"zone": 4}},
        {"frame": 80, "timestamp_seconds": 8.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 6}},
        {"frame": 90, "timestamp_seconds": 9.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 7}},
        {"frame": 150, "timestamp_seconds": 15.0, "event_type": "wall", "call": "IN", "target_zone": {"zone": 8}},
    ]

    assignment = assign_front_wall_hit_players(hits)

    assert assignment["rally_count"] == 3
    assert [rally["server_player_number"] for rally in assignment["rallies"]] == [1, 1, 2]
    assert [rally["winner_player_number"] for rally in assignment["rallies"]] == [1, 2, 2]
    assert [rally["duration_seconds"] for rally in assignment["rallies"]] == [1.0, 1.0, 0.0]
    assert [hit["rally_number"] for hit in hits] == [1, 1, 2, 2, 3]
    assert [hit["player_number"] for hit in hits] == [1, 2, 1, 2, 2]


def test_job_restart_recovery(tmp_path, monkeypatch):
    import job_runner

    monkeypatch.setattr(job_runner, "RUNS_DIR", tmp_path)
    run_id = "restart-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id, "run_dir": str(run_dir), "status": "running",
    }))
    job = job_runner.get_job(run_id)
    assert job["status"] == "failed"
    assert "restarted" in job["error"]

    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id, "run_dir": str(run_dir), "status": "complete", "rows": 5,
    }))
    job = job_runner.get_job(run_id)
    assert job["status"] == "complete" and job["rows"] == 5


def test_update_job_persists_atomically(tmp_path):
    import job_runner

    run_id = "persist-test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    try:
        job_runner.update_job(run_id, run_dir=str(run_dir), status="queued", note="x")
        on_disk = json.loads((run_dir / "job.json").read_text())
        assert on_disk["status"] == "queued" and on_disk["note"] == "x"
    finally:
        with job_runner.JOBS_LOCK:
            job_runner.JOBS.pop(run_id, None)


def test_upload_dedup_and_track_validation(runs_dir):
    # runs_dir is here for its redirect, not its value: it moves BY_HASH_DIR
    # off the real upload store, which is what makes "exactly one file matches
    # this hash" a fact about these two uploads rather than about whatever
    # earlier runs left behind.
    import app as app_module

    client = app_module.app.test_client()

    health = client.get("/api/health").get_json()
    assert health["ok"] is True and "version" in health

    response = client.post("/api/track", data={})
    assert response.status_code == 400

    response = client.post("/api/track", data={
        "video_id": "deadbeef", "calibration_json": "{}",
        "start_time": "0", "end_time": "5",
    })
    assert response.status_code == 404

    payload = b"fake video bytes for hashing"
    ids = []
    for _ in range(2):
        response = client.post(
            "/api/upload",
            data={"video_file": (io.BytesIO(payload), "clip.mp4")},
            content_type="multipart/form-data",
        )
        ids.append(response.get_json()["video_id"])

    assert ids[0] == ids[1]
    matches = list(app_module.BY_HASH_DIR.glob(f"{ids[0]}.*"))
    assert len(matches) == 1
    assert app_module.video_path_for_id(ids[0]) == matches[0]

    assert client.get("/api/track/status/does-not-exist").status_code == 404


def test_track_rejection_leaves_no_run_directory(tmp_path, monkeypatch):
    """A refused /api/track must not create the run it refused to start.

    The run directory used to be made before the video was looked up, so every
    rejection left an empty run behind: they accumulate in the runs list, and
    the test suite grew one per invocation.
    """
    import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)
    client = app_module.app.test_client()

    response = client.post("/api/track", data={
        "video_id": "deadbeef", "calibration_json": "{}",
        "start_time": "0", "end_time": "5",
    })
    assert response.status_code == 404
    assert list(tmp_path.iterdir()) == []


def test_track_rejects_a_clip_past_the_end_without_making_a_run(tmp_path, monkeypatch):
    """Same guarantee for the checks that only run once the video is readable.

    `video_info` and the clip-window comparison sit after the video is
    resolved, so they are early returns the 404 case above never reaches.
    """
    import app as app_module
    import cv2
    import numpy as np

    video_path = tmp_path / "tiny.mp4"
    # mp4v, not avc1: Linux opencv-python-headless ships no H.264 encoder,
    # and a failed VideoWriter drops no file rather than raising.
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    assert video_path.exists(), "VideoWriter produced no file (codec unavailable?)"

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(app_module, "RUNS_DIR", runs_dir)
    # Resolving the id here keeps the upload branch (and the real UPLOADS_DIR)
    # out of a test about run-directory creation.
    monkeypatch.setattr(app_module, "video_path_for_id", lambda _id: video_path)
    client = app_module.app.test_client()

    # 0.4s of video; a window starting at 100s lands past the last frame.
    response = client.post("/api/track", data={
        "video_id": "whatever", "calibration_json": "{}",
        "start_time": "100", "end_time": "101",
    })
    assert response.status_code == 400
    assert list(runs_dir.iterdir()) == []


def test_ground_truth_save_and_fetch_roundtrip(runs_dir):
    import app as app_module

    client = app_module.app.test_client()
    run_id = "gt-route-test"
    (runs_dir / run_id).mkdir(parents=True, exist_ok=True)

    response = client.post(f"/api/runs/{run_id}/ground_truth", json={
        "events": [
            {"frame": 60, "type": "wall"},
            {"frame": 20, "type": "racket"},
        ],
    })
    assert response.status_code == 200
    assert response.get_json()["count"] == 2

    fetched = client.get(f"/api/runs/{run_id}/ground_truth.json").get_json()
    assert fetched["tolerance_frames"] == 1
    # Events come back sorted by frame regardless of submitted order.
    assert [e["frame"] for e in fetched["events"]] == [20, 60]

    response = client.post(f"/api/runs/{run_id}/ground_truth", json={
        "events": [{"frame": 5, "type": "volley_boast"}],
    })
    assert response.status_code == 400

    response = client.post("/api/runs/no-such-run/ground_truth", json={"events": []})
    assert response.status_code == 404


def test_label_run_creation_from_uploaded_video(tmp_path, runs_dir):
    import app as app_module
    import cv2
    import numpy as np

    video_path = tmp_path / "tiny.mp4"
    # mp4v, not avc1: Linux opencv-python-headless ships no H.264 encoder,
    # and a failed VideoWriter drops no file rather than raising.
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    assert video_path.exists(), "VideoWriter produced no file (codec unavailable?)"

    client = app_module.app.test_client()

    response = client.post("/api/label_runs", json={"video_id": "0000dead0000"})
    assert response.status_code == 404
    assert client.post("/api/label_runs", json={}).status_code == 400

    with video_path.open("rb") as f:
        video_id = client.post(
            "/api/upload",
            data={"video_file": (f, "tiny.mp4")},
            content_type="multipart/form-data",
        ).get_json()["video_id"]

    data = client.post("/api/label_runs", json={"video_id": video_id}).get_json()
    assert data["run_id"] == f"label-{video_id[:12]}"
    assert data["label_only"] is True
    assert data["start_frame"] == 0 and data["end_frame"] == 11
    assert abs(data["fps"] - 30.0) < 0.1

    run_dir = runs_dir / data["run_id"]
    meta = json.loads((run_dir / "label_run.json").read_text())
    assert meta["video_path"].endswith(f"{video_id}.mp4")

    # ground truth saves into the label-only run like any other run
    response = client.post(f"/api/runs/{data['run_id']}/ground_truth", json={
        "events": [{"frame": 6, "type": "wall"}],
    })
    assert response.status_code == 200


def _attribution_hit(t_s, call="IN", is_serve=False):
    return {
        "timestamp_seconds": t_s,
        "frame": int(t_s * 60),
        "event_type": "wall",
        "target_zone": {"zone": 4, "side": "center", "x": 0.5, "y": 0.5},
        "call": call,
        "is_serve": is_serve,
    }


def _three_rally_hits():
    # Rally 1: t=10.0 (serve), 11.0, 12.0 ; rally 2: t=30.0, 31.0 ;
    # rally 3: t=60.0. Gaps >> RALLY_GAP defaults.
    return (
        [_attribution_hit(10.0, is_serve=True), _attribution_hit(11.0),
         _attribution_hit(12.0)]
        + [_attribution_hit(30.0, is_serve=True), _attribution_hit(31.0)]
        + [_attribution_hit(60.0, is_serve=True)]
    )


def test_assign_without_resolver_uses_a_first_then_winner_chain():
    hits = _three_rally_hits()
    assignment = job_runner.assign_front_wall_hit_players(hits)
    assert assignment["method"] == "rally_gap_a_first_winner_chain"
    assert assignment["observed_serve_count"] == 0
    rallies = assignment["rallies"]
    assert [rally["server_player_number"] for rally in rallies] == [1, 1, 2]
    assert [rally["server_track"] for rally in rallies] == ["A", "A", "B"]
    for rally in rallies:
        assert rally["server_source"] == "propagated"
        assert rally["winner_crosscheck_agrees"] is None
    # Rally 1 has three hits, so A takes the last IN contact and serves rally
    # 2. Rally 2 has two, so B takes the last IN contact and serves rally 3.
    assert [rally["winner_player_number"] for rally in rallies] == [1, 2, 2]
    assert [hit["player_number"] for hit in hits] == [1, 2, 1, 1, 2, 2]


def test_assign_with_resolver_observed_servers_and_next_serve_winners():
    hits = _three_rally_hits()
    by_first_hit_t = {10.0: "A", 30.0: "B", 60.0: "B"}

    def resolver(rally_hits):
        return by_first_hit_t.get(float(rally_hits[0]["timestamp_seconds"]))

    assignment = job_runner.assign_front_wall_hit_players(hits, serve_resolver=resolver)
    assert assignment["method"] == "rally_gap_observed_serves"
    assert assignment["observed_serve_count"] == 3
    rallies = assignment["rallies"]
    # Anchor: rally 1 observed "A" -> A=player 1 (propagated first server).
    assert rallies[0]["server_player_number"] == 1
    assert rallies[0]["server_track"] == "A"
    assert rallies[0]["server_source"] == "observed"
    # Rally 2 served by B -> player 2; so rally 1 winner = 2 via next_serve.
    assert rallies[1]["server_player_number"] == 2
    assert rallies[0]["winner_player_number"] == 2
    assert rallies[0]["winner_source"] == "next_serve"
    # est winner of rally 1 was player 1 (last IN hit) -> crosscheck disagrees.
    assert rallies[0]["winner_crosscheck_agrees"] is False
    # Back-filled winner_reason must explain the *actual* winner (next-serve
    # observation), not stay frozen on the est explanation for the player who
    # was overridden -- otherwise the payload self-contradicts (winner=2 but
    # reason talks about player 1's last hit).
    assert rallies[0]["winner_reason"] == "next_rally_serve_observed"
    # Rally 3 also served by B: winner of rally 2 = player 2 via next_serve.
    # Rally 2's LAST hit (t=31.0, IN) was by the receiver (player 1), so the
    # est winner is 1 and the cross-check disagrees.
    assert rallies[1]["winner_player_number"] == 2
    assert rallies[1]["winner_source"] == "next_serve"
    assert rallies[1]["winner_crosscheck_agrees"] is False
    assert rallies[1]["winner_reason"] == "next_rally_serve_observed"
    # Last rally: est only, never back-filled -> keeps its est winner_reason.
    assert rallies[2]["winner_source"] in ("est", None)
    assert rallies[2]["winner_reason"] in (
        "last_front_wall_hit_in_winner",
        "last_front_wall_hit_out",
        "serve_out",
        "last_front_wall_hit_unjudged",
    )
    # Per-hit alternation from the observed servers.
    rally2_hits = [h for h in hits if h.get("rally_number") == 2]
    assert [h["player_number"] for h in rally2_hits] == [2, 1]


def test_assign_with_partial_resolver_falls_back_to_propagation():
    hits = _three_rally_hits()

    def resolver(rally_hits):
        # Only rally 2 observed.
        return "B" if float(rally_hits[0]["timestamp_seconds"]) == 30.0 else None

    assignment = job_runner.assign_front_wall_hit_players(hits, serve_resolver=resolver)
    rallies = assignment["rallies"]
    assert rallies[0]["server_source"] == "propagated"
    assert rallies[1]["server_source"] == "observed"
    # Anchor is rally 2: its propagated server is the est winner of rally 1
    # (player 1), so B -> player 1 there.
    assert rallies[1]["server_player_number"] == 1
    assert assignment["observed_serve_count"] == 1


def test_safe_resolver_swallows_exceptions_and_falls_back_to_propagation():
    """job_runner._safe_resolver is the one hole review found left in the
    person-layer degradation invariant: build_person_pass already swallows
    detector-load failures and PersonFramePass.observe already swallows
    detect() failures, but a resolver built from their output (samples_by_
    track, ball rows) ran unguarded inside assign_front_wall_hit_players --
    an exception there would fail the whole tracking job. Wrapping it must
    make assign_front_wall_hit_players behave exactly as if no resolver had
    been supplied at all: every rally propagated, nothing observed."""
    hits = _three_rally_hits()

    def raising_resolver(rally_hits):
        raise RuntimeError("boom")

    safe_resolver = job_runner._safe_resolver(raising_resolver)
    assignment = job_runner.assign_front_wall_hit_players(
        hits, serve_resolver=safe_resolver
    )
    assert assignment["method"] == "rally_gap_a_first_winner_chain"
    assert assignment["observed_serve_count"] == 0
    assert [rally["server_track"] for rally in assignment["rallies"]] == [
        "A", "A", "B",
    ]
    for rally in assignment["rallies"]:
        assert rally["server_source"] == "propagated"


def test_judge_hits_threads_resolver_through_unconditional_assign_without_service_line(tmp_path):
    # Regression: judge_hits calls assign_front_wall_hit_players twice — an
    # unconditional call whose result seeds player_assignment, and a second
    # call inside `if top_line is not None and service_line is not None`
    # that overwrites it once serve-line judging has run. A run with no
    # service_line calibration line never enters that second branch, so the
    # unconditional call is the ONLY one that determines the final
    # player_assignment. It must receive serve_resolver too, or observed
    # attribution silently vanishes for every un-service-line-calibrated run.
    (tmp_path / "calibration.json").write_text(json.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
        ]
        # No service_line_top_edge -> service_line stays None in judge_hits.
    }))
    results = {
        60: {
            "source_frame": 60,
            "timestamp_seconds": "2.000000",
            "detected": "True",
            "x_center": "900.000",
            "y_center": "180.000",
        }
    }
    hit = {
        "hit_frame": 60,
        "timestamp_seconds": 2.0,
        "dv_magnitude": 400.0,
        "speed_before": 400.0,
        "speed_after": 380.0,
        "after_gap": False,
        "event_type": "wall",
        "wall_score": 0.9,
        "signals": {"audio_score": 24.0},
    }

    def resolver(rally_hits):
        return "A"

    judged = job_runner.judge_hits(tmp_path, results, [hit], serve_resolver=resolver)

    entry = judged[0]
    assert entry["call"] == "IN"
    assert entry["rally_number"] == 1
    assert entry["server_player_number"] in (1, 2)

    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    assert payload["player_assignment"]["method"] == "rally_gap_observed_serves"
    assert payload["player_assignment"]["observed_serve_count"] == 1
    assert payload["rallies"][0]["server_track"] == "A"
    assert payload["rallies"][0]["server_source"] == "observed"
