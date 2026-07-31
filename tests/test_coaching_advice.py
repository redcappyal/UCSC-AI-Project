"""Metric -> drill-progression mapping for the per-player coaching report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coaching_advice import MIN_HITS_FOR_ADVICE, MAX_ADVICE_ITEMS, player_advice


def player(**overrides):
    """A player with every metric inside its healthy band, so a test only has
    to push the one metric it cares about out of range."""
    base = {
        "total_wall_hits": 40,
        "average_wall_height_ft": 7.0,
        "high_target_rate": 15.0,
        "mid_target_rate": 60.0,
        "low_target_rate": 25.0,
        "side_target_rate": 65.0,
        "average_incoming_speed_mph": 40.0,
        "missing_target_zones": [],
        "unforced_errors": 1,
        "total_errors": 10,
        "unforced_error_percentage": 10.0,
    }
    base.update(overrides)
    return base


def metrics(result):
    return [item["metric"] for item in result["items"]]


def find(result, metric):
    return next(item for item in result["items"] if item["metric"] == metric)


def test_healthy_player_gets_no_drills():
    result = player_advice(player())

    assert result["items"] == []
    assert "No clear weakness" in result["note"]


def test_thin_sample_withholds_advice():
    result = player_advice(player(total_wall_hits=MIN_HITS_FOR_ADVICE - 1,
                                  average_wall_height_ft=13.0))

    assert result["items"] == []
    assert "too few" in result["note"]


def test_thin_sample_note_agrees_in_number():
    assert "1 front-wall shot was analyzed" in player_advice(
        player(total_wall_hits=1))["note"]
    assert "0 front-wall shots were analyzed" in player_advice(
        player(total_wall_hits=0))["note"]


def test_every_item_carries_the_full_four_stage_progression():
    result = player_advice(player(average_wall_height_ft=12.0,
                                  side_target_rate=20.0,
                                  unforced_error_percentage=80.0,
                                  unforced_errors=8))

    assert result["items"]
    for item in result["items"]:
        stages = [step["stage"] for step in item["progression"]]
        assert stages == ["Solo", "Drills", "Conditioned games", "Matchplay"]
        assert all(step["text"].strip() for step in item["progression"])
        assert item["issue"].strip()


def test_high_average_height_suggests_lower_shots():
    result = player_advice(player(average_wall_height_ft=12.0))

    item = find(result, "average_wall_height_ft")
    assert "predictable" in item["issue"]
    assert "kills" in item["progression"][0]["text"]
    assert "short game" in item["progression"][2]["text"]


def test_low_average_height_suggests_lifting():
    result = player_advice(player(average_wall_height_ft=3.0))

    item = find(result, "average_wall_height_ft")
    assert "tin" in item["issue"]
    assert "lifting" in item["progression"][0]["text"]
    assert "service line" in item["progression"][2]["text"]


def test_not_enough_length_suggests_rotating_drives_and_deep_game():
    result = player_advice(player(mid_target_rate=5.0))

    item = find(result, "mid_target_rate")
    assert "length" in item["issue"]
    assert "Rotating drives" in item["progression"][0]["text"]
    assert "deep game" in item["progression"][2]["text"]


def test_no_lob_suggests_lifting_not_length():
    """The top band is lob height (above ~10.6 ft), not driving height, so it
    must coach lifting rather than length."""
    result = player_advice(player(high_target_rate=0.0))

    item = find(result, "high_target_rate")
    assert "lob height" in item["issue"]
    assert "lobs" in item["progression"][0]["text"]
    assert "Boast and lob" in item["progression"][1]["text"]


def test_lob_advice_is_suppressed_when_the_player_is_already_too_low():
    """A player hugging the tin already gets a lifting card; a second one
    about the lob would just repeat it."""
    result = player_advice(player(average_wall_height_ft=3.0, high_target_rate=0.0))

    assert "average_wall_height_ft" in metrics(result)
    assert "high_target_rate" not in metrics(result)


def test_no_short_threat_suggests_short_shots_and_short_game():
    result = player_advice(player(low_target_rate=2.0))

    item = find(result, "low_target_rate")
    assert "drops and kills" in item["progression"][0]["text"]
    assert "Boast and drive" in item["progression"][1]["text"]
    assert "short game" in item["progression"][2]["text"]


def test_narrow_game_suggests_straight_drives():
    result = player_advice(player(side_target_rate=20.0))

    item = find(result, "side_target_rate")
    assert "middle" in item["issue"]
    assert "Rotating drives" in item["progression"][0]["text"]
    assert "straight" in item["progression"][1]["text"]


def test_fast_and_slow_pace_give_opposite_advice():
    fast = find(player_advice(player(average_incoming_speed_mph=90.0)),
                "average_incoming_speed_mph")
    slow = find(player_advice(player(average_incoming_speed_mph=10.0)),
                "average_incoming_speed_mph")

    assert "lobs" in fast["progression"][0]["text"]
    assert "kills and stuns" in slow["progression"][0]["text"]
    assert fast["issue"] != slow["issue"]


def test_unforced_errors_price_the_risk_in_a_conditioned_game():
    result = player_advice(player(unforced_errors=8, total_errors=10,
                                  unforced_error_percentage=80.0))

    item = find(result, "unforced_error_percentage")
    assert "two points" in item["progression"][2]["text"]
    # Errors lose points directly, so they outrank stylistic advice.
    assert metrics(result)[0] == "unforced_error_percentage"


def test_missing_zones_are_named_and_drilled_by_where_they_sit():
    low = player_advice(player(missing_target_zones=[{"zone": 3}, {"zone": 9}]))
    lob = player_advice(player(missing_target_zones=[{"zone": 1}, {"zone": 7}]))

    low_item = find(low, "missing_target_zones")
    assert "zone 3" in low_item["issue"] and "low on the left" in low_item["issue"]
    assert "Boast and drive" in low_item["progression"][1]["text"]

    lob_item = find(lob, "missing_target_zones")
    assert "lob height on the left" in lob_item["issue"]
    assert "Boast and lob" in lob_item["progression"][1]["text"]


def test_advice_is_capped_and_ordered_by_priority():
    result = player_advice(player(
        average_wall_height_ft=13.0,
        high_target_rate=1.0,
        low_target_rate=1.0,
        side_target_rate=5.0,
        average_incoming_speed_mph=95.0,
        missing_target_zones=[{"zone": 3}],
        unforced_errors=9, total_errors=10, unforced_error_percentage=90.0,
    ))

    assert len(result["items"]) == MAX_ADVICE_ITEMS
    priorities = [item["priority"] for item in result["items"]]
    assert priorities == sorted(priorities, reverse=True)


def test_small_but_usable_sample_is_flagged_as_a_pointer():
    # The hedge is provenance, so it rides in the disclosure field, not in the
    # note the page renders as body copy.
    result = player_advice(player(total_wall_hits=8, average_wall_height_ft=13.0))

    assert result["items"]
    assert result["note"] is None
    assert "pointer" in result["low_sample_note"]


def test_missing_metrics_do_not_crash():
    result = player_advice({"total_wall_hits": 30})

    assert isinstance(result["items"], list)


def test_empty_player_is_handled():
    assert player_advice({})["items"] == []
    assert player_advice(None)["items"] == []
