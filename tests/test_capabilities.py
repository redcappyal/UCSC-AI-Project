"""Capability gating: which analyses a given clip can honestly support.

The contract these tests pin is design-spec Principle 3: an analysis that
cannot run on a recording is *disabled with a stated reason*, never silently
degraded and never reported as empty success. So every assertion here is
either "this tier is off" or "and here is the reason a human will read".

Reason strings are asserted by substring, not by equality -- the wording is
allowed to improve, the fact that it names the failing gate and the measured
value is not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capabilities as cap

PROBE_60 = {
    "fps": 60.0,
    "width": 1920,
    "height": 1080,
    "frame_count": 600,
    "duration_s": 10.0,
    "sharpness": 120.0,
    "has_audio": True,
}
PROBE_30 = dict(PROBE_60, fps=30.0)


def test_qualified_footage_enables_every_tier():
    caps = cap.compute_capabilities(PROBE_60, court_solved=True)

    assert all(tier["enabled"] for tier in caps.values())
    assert all(tier["reason"] is None for tier in caps.values())


def test_sub_50_fps_footage_still_enables_ball_tracking():
    caps = cap.compute_capabilities(PROBE_30, court_solved=True)

    assert caps["ball_tracking"]["enabled"] is True
    assert caps["ball_tracking"]["reason"] is None
    assert caps["line_calls"]["enabled"] is True


def test_a_slow_clip_still_gets_rally_structure_and_movement():
    """The whole point of the ladder: tier N never requires tier N+1.

    30 fps camera-roll footage is the common case, and it is exactly the case
    the old pipeline turned into an empty successful run.
    """
    caps = cap.compute_capabilities(PROBE_30, court_solved=True)

    assert caps["rally_structure"]["enabled"] is True
    assert caps["player_movement"]["enabled"] is True


def test_no_floor_solve_disables_movement_but_not_ball_or_line_calls():
    caps = cap.compute_capabilities(PROBE_60, court_solved=False)

    assert caps["player_movement"]["enabled"] is False
    assert caps["ball_tracking"]["enabled"] is True
    assert caps["line_calls"]["enabled"] is True
    assert caps["rally_structure"]["enabled"] is True
    assert "court" in caps["player_movement"]["reason"]
    assert caps["ball_tracking"]["reason"] is None


def test_rally_structure_is_enabled_even_without_audio():
    """Audio absence is runtime information, not a capability input.

    Whether the audio channel actually yielded transients is reported by the
    rally timeline itself; gating the tier here would disable rally structure
    on footage where frame motion alone segments it fine.
    """
    caps = cap.compute_capabilities(
        dict(PROBE_60, has_audio=False), court_solved=True
    )

    assert caps["rally_structure"]["enabled"] is True
    assert caps["rally_structure"]["reason"] is None


def test_ball_tier_gated_by_width():
    caps = cap.compute_capabilities(
        dict(PROBE_60, width=1280, height=720), court_solved=True
    )

    assert caps["ball_tracking"]["enabled"] is False
    assert "1600" in caps["ball_tracking"]["reason"]


def test_ball_tier_gated_by_blur():
    caps = cap.compute_capabilities(
        dict(PROBE_60, sharpness=5.0), court_solved=True
    )

    assert caps["ball_tracking"]["enabled"] is False
    assert "blur" in caps["ball_tracking"]["reason"].lower()


def test_unmeasured_sharpness_disables_the_ball_tier_saying_so():
    """"Could not measure" must not read as "measured and passed".

    probe_video returns None when no frame decoded. Treating that as a pass
    would run the expensive ball stages on a clip nothing could read.
    """
    caps = cap.compute_capabilities(
        dict(PROBE_60, sharpness=None), court_solved=True
    )

    assert caps["ball_tracking"]["enabled"] is False
    assert "measure" in caps["ball_tracking"]["reason"].lower()


def test_every_failing_gate_is_named_not_just_the_first():
    """A card names every actual disqualifier, not just the first."""
    caps = cap.compute_capabilities(
        dict(PROBE_60, width=1280, sharpness=5.0), court_solved=True
    )

    reason = caps["ball_tracking"]["reason"]
    assert "1600" in reason
    assert "blur" in reason.lower()


def test_line_calls_follow_the_ball_tier():
    """Line calls judge ball hits, so they cannot outlive the tier producing them."""
    caps = cap.compute_capabilities(
        dict(PROBE_60, width=1280), court_solved=True
    )

    assert caps["line_calls"]["enabled"] is False
    assert "ball" in caps["line_calls"]["reason"].lower()


def test_movement_gated_by_a_frame_rate_too_low_to_track_people():
    caps = cap.compute_capabilities(
        dict(PROBE_60, fps=5.0), court_solved=True
    )

    assert caps["player_movement"]["enabled"] is False
    assert "12" in caps["player_movement"]["reason"]


def test_reason_is_none_exactly_when_enabled():
    for probe, solved in (
        (PROBE_60, True), (PROBE_30, True), (PROBE_60, False),
        (dict(PROBE_60, sharpness=None), True),
    ):
        for name, tier in cap.compute_capabilities(probe, court_solved=solved).items():
            assert (tier["reason"] is None) == tier["enabled"], name


def test_result_carries_a_schema_version():
    """Reports render these; a stored run must say which shape it stored."""
    caps = cap.compute_capabilities(PROBE_60, court_solved=True)

    assert cap.CAPABILITIES_SCHEMA == "capabilities-v1"
    assert set(caps) == {
        "rally_structure", "player_movement", "ball_tracking", "line_calls",
    }


def test_a_missing_probe_disables_the_measured_tiers_rather_than_assuming():
    """Legacy runs have no probe. They must not be assumed qualified."""
    caps = cap.compute_capabilities(None, court_solved=True)

    assert caps["rally_structure"]["enabled"] is True
    assert caps["ball_tracking"]["enabled"] is False
    assert "probe" in caps["ball_tracking"]["reason"].lower()
