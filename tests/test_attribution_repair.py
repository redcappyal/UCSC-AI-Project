"""Parity repair and attribution provenance (spec §4).

The winner back-fill computes a cross-check -- est winner (from rally parity)
vs observed winner (the next rally's observed server) -- and used to discard
the answer. A disagreement is not noise: the est winner is a pure function of
the rally's parity, so it can only disagree if the parity is wrong. These
tests pin down when that gets repaired and when it only gets flagged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_runner import assign_front_wall_hit_players


def hit(t_s, call="IN", is_serve=False):
    return {
        "timestamp_seconds": t_s,
        "frame": int(t_s * 60),
        "event_type": "wall",
        "target_zone": {"zone": 4, "side": "center", "x": 0.5, "y": 0.5},
        "call": call,
        "is_serve": is_serve,
    }


def three_rally_hits():
    # Rally 1: t=10 (serve), 11, 12 ; rally 2: t=30 (serve), 31 ;
    # rally 3: t=60 (serve). Gaps >> the inferred rally gap.
    return (
        [hit(10.0, is_serve=True), hit(11.0), hit(12.0)]
        + [hit(30.0, is_serve=True), hit(31.0)]
        + [hit(60.0, is_serve=True)]
    )


def resolver_from(by_first_hit_t):
    def resolver(rally_hits):
        return by_first_hit_t.get(float(rally_hits[0]["timestamp_seconds"]))

    return resolver


def rally_hits_of(hits, rally_number):
    return [h for h in hits if h.get("rally_number") == rally_number]


def test_propagated_rally_with_disagreeing_crosscheck_is_repaired():
    hits = three_rally_hits()
    # Rally 1 observed on A (the anchor -> A = player 1), rally 2 unobserved,
    # rally 3 observed on A again. Rally 2's est winner is player 2 (its last
    # hit was the receiver's, called IN) but rally 3's observed server is
    # player 1 -- so rally 2's parity, propagated by alternation, is wrong.
    assignment = assign_front_wall_hit_players(
        hits, serve_resolver=resolver_from({10.0: "A", 60.0: "A"})
    )
    rallies = assignment["rallies"]
    repaired = rallies[1]

    assert repaired["server_source"] == "propagated"
    assert repaired["attribution_state"] == "repaired"
    assert repaired["parity_repaired"] is True
    # The diagnostic that triggered the repair must survive it -- recomputing
    # it post-flip would erase the only evidence anything was wrong.
    assert repaired["winner_crosscheck_agrees"] is False
    # Propagation made player 1 the server; the observation says player 2.
    assert repaired["server_player_number"] == 2
    assert repaired["last_player_number"] == 1
    assert repaired["winner_player_number"] == 1
    assert repaired["winner_source"] == "next_serve"
    assert repaired["winner_reason"] == "next_rally_serve_observed"

    rally2 = rally_hits_of(hits, 2)
    assert [h["player_number"] for h in rally2] == [2, 1]
    assert [h["server_player_number"] for h in rally2] == [2, 2]

    # Neighbours are untouched: rally 1 has no next *observed* serve to check
    # against (rally 2's is propagated), rally 3 has no next rally at all.
    assert rallies[0]["attribution_state"] == "observed"
    assert rallies[0]["winner_crosscheck_agrees"] is None
    assert rallies[2]["attribution_state"] == "observed"
    assert all(not r["parity_repaired"] for r in (rallies[0], rallies[2]))


def test_repair_does_not_rewrite_server_source():
    """index.html's attributionAnchor() keys on server_source == 'observed'
    to decide which rally's serve crop names the players. Promoting a repaired
    rally to "observed" would hand it that anchor on the strength of an
    inference, not a sighting."""
    hits = three_rally_hits()
    assignment = assign_front_wall_hit_players(
        hits, serve_resolver=resolver_from({10.0: "A", 60.0: "A"})
    )
    assert assignment["rallies"][1]["parity_repaired"] is True
    assert assignment["rallies"][1]["server_source"] == "propagated"
    assert assignment["observed_serve_count"] == 2


def test_observed_rally_with_disagreeing_crosscheck_is_flagged_not_repaired():
    """Flipping an observed serve would discard a direct sighting of who
    served, and if the true cause was a missed mid-rally hit the flip would
    only move the error from the end of the rally to its start."""
    hits = three_rally_hits()
    assignment = assign_front_wall_hit_players(
        hits, serve_resolver=resolver_from({10.0: "A", 30.0: "B", 60.0: "B"})
    )
    rallies = assignment["rallies"]

    for index in (0, 1):
        assert rallies[index]["server_source"] == "observed"
        assert rallies[index]["winner_crosscheck_agrees"] is False
        assert rallies[index]["attribution_state"] == "conflict"
        assert rallies[index]["parity_repaired"] is False

    # Parity untouched: rally 1 still served by player 1, rally 2 by player 2.
    assert rallies[0]["server_player_number"] == 1
    assert rallies[1]["server_player_number"] == 2
    assert [h["player_number"] for h in rally_hits_of(hits, 1)] == [1, 2, 1]
    assert [h["player_number"] for h in rally_hits_of(hits, 2)] == [2, 1]


def test_propagated_rally_without_disagreement_is_assumed():
    hits = three_rally_hits()
    assignment = assign_front_wall_hit_players(hits)

    for rally in assignment["rallies"]:
        assert rally["server_source"] == "propagated"
        assert rally["attribution_state"] == "assumed"
        assert rally["parity_repaired"] is False
        assert rally["winner_crosscheck_agrees"] is None
    # Pure alternation, exactly as before the repair existed.
    assert [h["player_number"] for h in rally_hits_of(hits, 1)] == [1, 2, 1]


def test_agreeing_observed_rally_stays_observed():
    hits = three_rally_hits()
    # A, A, B: each rally's est winner is the next rally's observed server.
    assignment = assign_front_wall_hit_players(
        hits, serve_resolver=resolver_from({10.0: "A", 30.0: "A", 60.0: "B"})
    )
    rallies = assignment["rallies"]

    for index in (0, 1):
        assert rallies[index]["winner_crosscheck_agrees"] is True
        assert rallies[index]["attribution_state"] == "observed"
        assert rallies[index]["parity_repaired"] is False
    # Last rally is never back-filled, so it is never cross-checked.
    assert rallies[2]["winner_crosscheck_agrees"] is None
    assert rallies[2]["attribution_state"] == "observed"


def test_repair_leaves_later_rallies_alone():
    """No cascade. The back-fill fires only when rally N+1's serve is
    observed, so rally N+1's server is read from the track map, never from
    rally N's winner -- a repair cannot propagate forward."""
    hits = (
        [hit(10.0, is_serve=True), hit(11.0), hit(12.0)]
        + [hit(30.0, is_serve=True), hit(31.0)]
        + [hit(60.0, is_serve=True), hit(61.0)]
        + [hit(90.0, is_serve=True)]
    )
    assignment = assign_front_wall_hit_players(
        hits, serve_resolver=resolver_from({10.0: "A", 60.0: "A", 90.0: "B"})
    )
    rallies = assignment["rallies"]

    assert rallies[1]["parity_repaired"] is True
    assert rallies[1]["server_player_number"] == 2
    # Rally 3 was observed on A before the repair and still is after it.
    assert rallies[2]["server_source"] == "observed"
    assert rallies[2]["server_player_number"] == 1
    assert rallies[2]["parity_repaired"] is False
    assert [h["player_number"] for h in rally_hits_of(hits, 3)] == [1, 2]
