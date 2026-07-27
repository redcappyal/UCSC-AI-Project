"""player_attribution: serve resolver, identity confidence, players_v1."""

import pytest

from player_tracker import TrackSample
from player_attribution import (
    MAX_SERVE_TRACK_GAP_S,
    SERVE_LOOKBACK_S,
    build_players_v1,
    build_serve_resolver,
    rally_identity_confidences,
    serve_crop_target,
)


def sample(t_s, x, y, coasted=False):
    return TrackSample(t_s=t_s, frame_idx=int(t_s * 60), foot_px=(x, y),
                       bbox=(x, y - 50.0, 40.0, 100.0),
                       confidence=0.0 if coasted else 0.9, coasted=coasted)


def ball_row(frame, t_s, x, y):
    return {"source_frame": frame, "timestamp_seconds": f"{t_s:.6f}",
            "detected": True, "x_center": str(x), "y_center": str(y)}


def hit(t_s):
    return {"timestamp_seconds": t_s, "frame": int(t_s * 60)}


def test_resolver_picks_track_nearest_serve_ball():
    samples = {
        "A": [sample(t, 300, 500) for t in (9.0, 9.5, 10.0)],
        "B": [sample(t, 900, 500) for t in (9.0, 9.5, 10.0)],
    }
    # Ball detected near A 0.4 s before the first front-wall hit at t=10.4.
    rows = [ball_row(600, 10.0, 320, 480)]
    resolver = build_serve_resolver(samples, rows)
    assert resolver([hit(10.4), hit(11.5)]) == "A"


def test_resolver_none_without_ball_or_fresh_samples():
    samples = {"A": [sample(9.9, 300, 500)], "B": [sample(9.9, 900, 500)]}
    resolver = build_serve_resolver(samples, [])
    assert resolver([hit(10.4)]) is None  # no ball rows at all

    stale = {"A": [sample(1.0, 300, 500)], "B": [sample(1.0, 900, 500)]}
    rows = [ball_row(600, 10.0, 320, 480)]
    resolver = build_serve_resolver(stale, rows)
    assert resolver([hit(10.4)]) is None  # both tracks stale at ball time

    coasting = {"A": [sample(10.0, 300, 500, coasted=True)],
                "B": [sample(1.0, 900, 500)]}
    resolver = build_serve_resolver(coasting, rows)
    assert resolver([hit(10.4)]) is None  # coasted samples don't vote


def test_resolver_ignores_ball_rows_outside_lookback():
    samples = {"A": [sample(10.0, 300, 500)], "B": [sample(10.0, 900, 500)]}
    rows = [ball_row(60, 1.0, 320, 480)]  # far outside SERVE_LOOKBACK_S
    resolver = build_serve_resolver(samples, rows)
    assert resolver([hit(10.4)]) is None


def test_rally_identity_confidences_window_before_rally():
    rallies = [
        {"rally_number": 1, "start_time_seconds": 10.0, "end_time_seconds": 20.0},
        {"rally_number": 2, "start_time_seconds": 30.0, "end_time_seconds": 40.0},
        {"rally_number": 3, "start_time_seconds": 50.0, "end_time_seconds": 60.0},
    ]
    # Two ambiguous events during the break before rally 2, none before 3.
    confidences = rally_identity_confidences([21.0, 25.0], rallies)
    assert confidences[1] is None            # no break precedes rally 1
    assert confidences[2] == pytest.approx(0.0)
    assert confidences[3] == pytest.approx(1.0)


def test_build_players_v1_shape():
    assignment = {
        "method": "rally_gap_observed_serves",
        "rally_count": 2,
        "rallies": [
            {"rally_number": 1, "server_player_number": 1, "server_track": "A",
             "server_source": "observed", "winner_player_number": 2,
             "winner_source": "next_serve", "winner_crosscheck_agrees": True,
             "start_time_seconds": 10.0, "end_time_seconds": 20.0},
            {"rally_number": 2, "server_player_number": 2, "server_track": "B",
             "server_source": "observed", "winner_player_number": None,
             "winner_source": None, "winner_crosscheck_agrees": None,
             "start_time_seconds": 30.0, "end_time_seconds": 40.0},
        ],
    }
    block = build_players_v1(assignment, {"updates": 100, "ambiguous_assignments": 3},
                             detector_backend="rfdetr",
                             serve_crop_relpath="players/serve_rally1.jpg")
    assert block["attribution_backend"] == "observed"
    assert block["detector_backend"] == "rfdetr"
    assert block["serve_crop"] == "players/serve_rally1.jpg"
    assert block["player_names"] == {"A": None, "B": None}
    assert block["tracker"] == {"updates": 100, "ambiguous_assignments": 3}
    assert [r["rally_number"] for r in block["rallies"]] == [1, 2]
    scores = [r["score_after"] for r in block["rallies"]]
    assert scores[0] == {"1": 0, "2": 1}   # player 2 won rally 1
    assert scores[1] == {"1": 0, "2": 1}   # rally 2 winner unknown -> carried

    assumed = build_players_v1({"method": "rally_gap_server_alternation",
                                "rally_count": 0, "rallies": []},
                               None, detector_backend="none")
    assert assumed["attribution_backend"] == "assumed"


def test_serve_crop_target_uses_rally1_observed_server():
    assignment = {"rallies": [
        {"rally_number": 1, "server_track": "A", "server_source": "observed",
         "start_time_seconds": 10.0},
    ]}
    samples = {"A": [sample(9.8, 300, 500), sample(10.6, 310, 500)],
               "B": [sample(9.8, 900, 500)]}
    frame_idx, chosen = serve_crop_target(assignment, samples)
    assert chosen.foot_px == (300.0, 500.0)
    assert frame_idx == chosen.frame_idx

    no_obs = {"rallies": [{"rally_number": 1, "server_track": None,
                           "server_source": "propagated",
                           "start_time_seconds": 10.0}]}
    assert serve_crop_target(no_obs, samples) is None


def test_write_track_samples_round_trip(tmp_path):
    import json
    import job_runner

    samples = {"A": [sample(1.0, 300, 500)], "B": []}
    assert job_runner.write_track_samples(tmp_path, samples, [2.5]) is True
    payload = json.loads((tmp_path / "players" / "track_samples.json").read_text())
    assert payload["schema"] == "player-tracks-v1"
    assert payload["ambiguity_times"] == [2.5]
    entry = payload["tracks"]["A"][0]
    assert entry == {"t_s": 1.0, "frame_idx": 60, "foot_px": [300.0, 500.0],
                     "bbox": [300.0, 450.0, 40.0, 100.0],
                     "confidence": 0.9, "coasted": False}


def test_write_track_samples_unwritable_target_returns_false(tmp_path):
    import job_runner

    # run_dir itself is a plain file, so `Path(run_dir) / "players"` can
    # never be created -- mkdir(parents=True) raises NotADirectoryError (an
    # OSError subclass). Must return False, not raise.
    run_dir = tmp_path / "not_a_directory"
    run_dir.write_text("i am a file, not a run dir")
    samples = {"A": [sample(1.0, 300, 500)], "B": []}
    assert job_runner.write_track_samples(run_dir, samples, [2.5]) is False
