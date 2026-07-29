"""Per-player movement statistics in court feet.

Tier 2 of the analysis ladder. These are the numbers a player would actually
change their training over -- distance covered, how much of the rally was
spent on the T, whether they live in the front or the back -- so the tests
here are mostly about the statistics being *right in feet*, not merely
present.

Two properties get disproportionate attention because getting them wrong is
invisible in the output. Samples outside a rally must not count (walking to
pick up the ball is not court movement, and it would inflate every distance),
and coasted samples -- positions the tracker guessed while the detector saw
nothing -- must be reported as reduced coverage rather than silently averaged
in as if they were observations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from movement_stats import (
    CourtSample,
    GRID_X,
    GRID_Y,
    T_POINT_FT,
    T_RADIUS_FT,
    movement_stats,
)


def _rally(start, end):
    return {"start_s": start, "end_s": end}


def _still_at(x, y, start=0.0, end=10.0, step=0.1, coasted=False):
    samples = []
    time = start
    while time <= end + 1e-9:
        samples.append(CourtSample(time, x, y, coasted))
        time += step
    return samples


def test_a_player_standing_on_the_t_spends_all_their_time_there():
    samples = _still_at(*T_POINT_FT)

    stats = movement_stats(samples, [_rally(0.0, 10.0)])

    assert stats["t_time_pct"] == pytest.approx(1.0)
    assert stats["distance_ft"] == pytest.approx(0.0, abs=0.01)


def test_a_player_in_the_back_corner_never_reads_as_on_the_t():
    samples = _still_at(1.0, 30.0)

    stats = movement_stats(samples, [_rally(0.0, 10.0)])

    assert stats["t_time_pct"] == 0.0
    assert stats["front_pct"] == 0.0
    assert stats["back_pct"] == pytest.approx(1.0)


def _shuttle(jitter=0.0, seed=1):
    """Four 10 ft legs up and down the court -- 40 ft of known path."""
    import random

    rng = random.Random(seed)
    samples = []
    time = 0.0
    for leg in range(4):
        for step in range(11):               # 1 ft per 0.1 s = 10 ft/s
            y = 20.0 + step if leg % 2 == 0 else 30.0 - step
            samples.append(CourtSample(
                round(time, 4),
                10.5 + rng.gauss(0, jitter),
                y + rng.gauss(0, jitter),
                False,
            ))
            time += 0.1
    return samples, time


def test_distance_covered_matches_the_path_walked():
    """A shuttle of known length, within the accuracy this actually has.

    Distance is the headline movement number: wrong by a constant factor,
    nothing in the report looks odd, it just quietly misreports how hard
    someone worked. So the tolerance here is the measured accuracy, not an
    aspiration -- 10% covers the ~7.5% systematic under-count that centered
    smoothing costs on sharp direction changes (see the table at
    movement_stats.SMOOTH_WINDOW_S).
    """
    samples, duration = _shuttle()

    stats = movement_stats(samples, [_rally(0.0, duration)])

    assert stats["distance_ft"] == pytest.approx(40.0, rel=0.10)


def test_distance_under_reads_rather_than_over_reads_on_sharp_turns():
    """Pins the direction of the known bias, so a change of sign is caught.

    Smoothing rounds corners, so the error is an under-count. The failure that
    would matter is drifting the other way: an over-count means the report is
    crediting a player with distance nobody ran, which is the kind of number
    that flatters and cannot be noticed.
    """
    samples, duration = _shuttle()

    stats = movement_stats(samples, [_rally(0.0, duration)])

    assert stats["distance_ft"] < 40.0


def test_distance_does_not_diverge_when_positions_are_noisy():
    """The reason smoothing is there at all.

    Jitter is a random walk, so an unsmoothed path length has no ceiling --
    measured at 104% error with a foot of positional noise. Smoothed, the same
    input stays within a sane band.
    """
    samples, duration = _shuttle(jitter=0.5, seed=3)

    stats = movement_stats(samples, [_rally(0.0, duration)])

    assert stats["distance_ft"] == pytest.approx(40.0, rel=0.15)


def test_speed_is_reported_and_clamped_to_something_a_human_can_run():
    """A tracker identity swap teleports a player across the court.

    Unclamped, one bad frame turns into a 200 ft/s "sprint" and drags the
    average with it -- a number that is obviously absurd to a reader but
    perfectly plausible once averaged into a season summary.
    """
    samples = [
        CourtSample(0.0, 1.0, 1.0, False),
        CourtSample(0.02, 20.0, 31.0, False),   # ~35 ft in 20 ms
        CourtSample(0.04, 20.2, 31.0, False),
    ]

    stats = movement_stats(samples, [_rally(0.0, 1.0)])

    assert stats["p95_speed_ftps"] <= 30.0


def test_samples_outside_every_rally_are_excluded():
    """Walking to fetch the ball between points is not court movement."""
    in_rally = _still_at(10.5, 20.0, start=0.0, end=5.0)
    between = [
        CourtSample(6.0 + i * 0.1, 1.0 + i, 30.0, False) for i in range(20)
    ]

    stats = movement_stats(in_rally + between, [_rally(0.0, 5.0)])

    assert stats["distance_ft"] == pytest.approx(0.0, abs=0.01)


def test_hit_derived_first_and_last_hit_fields_bound_movement():
    samples = (
        _still_at(10.5, 20.0, start=0.0, end=1.9)
        + _still_at(10.5, 20.0, start=2.0, end=4.0)
        + _still_at(1.0, 30.0, start=4.1, end=6.0)
    )
    rallies = [{
        "start_time_seconds": 2.0,
        "end_time_seconds": 4.0,
    }]

    stats = movement_stats(samples, rallies)

    assert stats["distance_ft"] == pytest.approx(0.0, abs=0.01)
    assert stats["front_pct"] == 0.0
    assert stats["back_pct"] == pytest.approx(1.0)


def test_distance_never_bridges_the_gap_between_hit_bounded_rallies():
    samples = (
        _still_at(1.0, 20.0, start=0.0, end=1.0)
        + _still_at(20.0, 20.0, start=10.0, end=11.0)
    )
    rallies = [
        {"start_time_seconds": 0.0, "end_time_seconds": 1.0},
        {"start_time_seconds": 10.0, "end_time_seconds": 11.0},
    ]

    stats = movement_stats(samples, rallies)

    assert stats["distance_ft"] == pytest.approx(0.0, abs=0.01)
    assert stats["avg_speed_ftps"] == pytest.approx(0.0, abs=0.01)


def test_heatmap_is_normalised_and_peaks_where_the_player_was():
    samples = _still_at(1.0, 1.0)          # front-left corner

    stats = movement_stats(samples, [_rally(0.0, 10.0)])
    heatmap = stats["heatmap"]

    assert len(heatmap) == GRID_Y
    assert all(len(row) == GRID_X for row in heatmap)
    assert sum(sum(row) for row in heatmap) == pytest.approx(1.0)
    assert heatmap[0][0] == pytest.approx(1.0)


def test_a_player_with_no_samples_gets_zeros_not_a_crash():
    """A run where the detector never saw one player still has to render."""
    stats = movement_stats([], [_rally(0.0, 10.0)])

    assert stats["distance_ft"] == 0.0
    assert stats["sample_coverage"] == 0.0
    assert sum(sum(row) for row in stats["heatmap"]) == 0.0


def test_coverage_reports_how_much_of_the_rally_was_actually_observed():
    """Coasted positions are guesses, and must not read as observations.

    This is the movement tier's version of detection_coverage: a distance
    computed from a quarter of the rally looks exactly like one computed from
    all of it, and only coverage tells them apart.
    """
    observed = _still_at(10.5, 20.0, start=0.0, end=2.5)
    guessed = _still_at(10.5, 20.0, start=2.6, end=10.0, coasted=True)

    stats = movement_stats(observed + guessed, [_rally(0.0, 10.0)])

    assert 0.0 < stats["sample_coverage"] < 0.5


def test_front_and_back_split_covers_everything():
    samples = (
        _still_at(10.5, 5.0, start=0.0, end=2.0)
        + _still_at(10.5, 28.0, start=2.1, end=6.0)
    )

    stats = movement_stats(samples, [_rally(0.0, 6.0)])

    assert stats["front_pct"] + stats["back_pct"] == pytest.approx(1.0)
    assert stats["front_pct"] < stats["back_pct"]


def test_the_t_radius_is_a_named_constant_not_a_literal():
    """Every threshold this tier introduces has to be tunable by eval."""
    assert T_RADIUS_FT == 6.0
