"""A serve called OUT has to end its rally.

Gap segmentation only sees time, so a serve fault followed by a quick re-serve
used to land in one rally. The winner was then decided by the fault while
last_player_number pointed at a later contact, so a real error was attributed
to the wrong player and counted as forced rather than unforced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_runner import assign_front_wall_hit_players, split_rallies_on_serve_fault


def hit(frame, call="IN", is_serve=False):
    return {
        "frame": frame,
        "hit_frame": frame,
        "timestamp_seconds": frame / 60.0,
        "event_type": "wall",
        "call": call,
        "is_serve": is_serve,
        # assignability needs a placed contact (see
        # is_player_assignable_front_wall_hit)
        "target_zone": {"zone": 5, "side": "center", "band": "normal"},
        "wall_diagram": {"x": 0.5, "y": 0.5},
    }


def frames(rally):
    return [h["frame"] for h in rally]


def test_serve_fault_is_cut_into_its_own_rally():
    rally = [hit(100, "OUT"), hit(200), hit(300), hit(400)]

    assert [frames(r) for r in split_rallies_on_serve_fault([rally])] == [
        [100], [200, 300, 400],
    ]


def test_back_to_back_serve_faults_each_end_a_rally():
    rally = [hit(100, "OUT"), hit(200, "OUT"), hit(300)]

    assert [frames(r) for r in split_rallies_on_serve_fault([rally])] == [
        [100], [200], [300],
    ]


def test_a_clean_serve_leaves_the_rally_intact():
    rally = [hit(100), hit(200), hit(300, "OUT")]

    assert [frames(r) for r in split_rallies_on_serve_fault([rally])] == [
        [100, 200, 300],
    ]


def test_a_lone_serve_fault_is_left_alone():
    assert [frames(r) for r in split_rallies_on_serve_fault([[hit(100, "OUT")]])] == [[100]]


def test_out_later_in_the_rally_does_not_split():
    """Only the first contact is a serve; a drive hit out ends the rally
    naturally at its last hit and needs no cut."""
    rally = [hit(100), hit(200, "OUT"), hit(300)]

    assert [frames(r) for r in split_rallies_on_serve_fault([rally])] == [
        [100, 200, 300],
    ]


def test_serve_fault_is_charged_to_the_server_as_an_unforced_error(monkeypatch):
    """End to end: the fault becomes its own rally, the server loses it, and
    the error lands on the server rather than the opponent."""
    monkeypatch.setenv("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS", "5")
    # Player A serves out at frame 100; play resumes 3s later, inside the gap
    # threshold, so this used to merge into a single rally.
    hits = [hit(100, "OUT", is_serve=True), hit(280), hit(400), hit(520, "OUT")]

    assignment = assign_front_wall_hit_players(hits)
    rallies = assignment["rallies"]

    assert len(rallies) == 2, "the serve fault must end its rally"
    fault = rallies[0]
    assert fault["start_frame"] == fault["end_frame"] == 100
    assert fault["last_call"] == "OUT"
    assert fault["server_player_number"] == 1
    assert fault["server_track"] == "A"
    assert fault["last_player_number"] == 1, "the fault belongs to the server"
    assert fault["winner_player_number"] == 2, "the opponent wins a serve fault"
    followup = rallies[1]
    assert followup["server_player_number"] == 2, "the fault winner serves next"
    assert followup["server_track"] == "B"
    assert hits[1]["player_number"] == 2, "the next rally starts with its server"

    import app
    errors = app.player_error_metrics(rallies, player_number=1)
    assert errors["unforced_errors"] >= 1, "a serve fault is an unforced error"


def test_clean_serve_still_yields_one_rally(monkeypatch):
    monkeypatch.setenv("PLAYER_ASSIGNMENT_RALLY_GAP_SECONDS", "5")
    hits = [hit(100, is_serve=True), hit(280), hit(400)]

    assignment = assign_front_wall_hit_players(hits)

    assert len(assignment["rallies"]) == 1
    assert [h["rally_hit_sequence"] for h in hits] == [1, 2, 3]
