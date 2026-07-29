"""Rally structure from audio and motion, with the ball nowhere in sight.

This is the module the analysis ladder stands on. Today rallies are segmented
from front-wall *hits* (job_runner.segment_front_wall_hits_into_rallies), so
rally counts, lengths and tempo inherit the ball detector's recall -- which the
standing baseline puts near 35%. A rally count computed from a third of the
events reads exactly as plausible as a correct one, which is the failure mode
CLAUDE.md names as worse in a coaching product than a missed line call.

Audio transients and frame-motion energy are available on any clip at any
frame rate. So the tests here feed the segmenter deterministic synthetic
series and never a detection: if any assertion below could only be satisfied
by knowing where the ball was, the module has been built wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rally_segmenter import MIN_RALLY_S, segment_rallies


def _motion(spans, duration=60.0, dt=0.2, hi=8.0, lo=0.4):
    """A motion-energy series that is `hi` inside `spans` and `lo` outside."""
    ts = np.arange(0.0, duration, dt)
    energy = np.full(ts.shape, lo)
    for start, end in spans:
        energy[(ts >= start) & (ts <= end)] = hi
    return list(zip(ts.tolist(), energy.tolist()))


def test_two_rallies_from_impacts_and_motion():
    impacts = [5.0, 5.8, 6.9, 8.0, 9.2, 30.0, 30.7, 31.9, 33.0]

    rallies = segment_rallies(impacts, _motion([(4.5, 9.5), (29.5, 33.5)]), 60.0)

    assert len(rallies) == 2
    assert rallies[0]["start_s"] <= 5.0
    assert rallies[0]["end_s"] >= 9.2
    assert rallies[0]["impact_count"] == 5
    assert rallies[0]["source"] == "audio+motion"


def test_motion_only_rally_low_confidence():
    """No audio at all still yields structure, flagged as less trustworthy."""
    rallies = segment_rallies([], _motion([(10.0, 16.0)], duration=30.0), 30.0)

    assert len(rallies) == 1
    assert rallies[0]["source"] == "motion"
    assert rallies[0]["confidence"] == 0.3


def test_short_blips_dropped():
    """A single knock is not a rally; a door closing must not become one."""
    assert segment_rallies([12.0], _motion([(12.0, 12.6)], duration=30.0), 30.0) == []


def test_rallies_are_sorted_and_non_overlapping():
    """Downstream code indexes and sums these; overlaps double-count time."""
    impacts = [5.0, 6.0, 7.0, 8.0, 20.0, 21.0, 22.0, 23.0, 40.0, 41.0, 42.0, 43.0]
    spans = [(4.5, 8.5), (19.5, 23.5), (39.5, 43.5)]

    rallies = segment_rallies(impacts, _motion(spans), 60.0)

    starts = [r["start_s"] for r in rallies]
    assert starts == sorted(starts)
    for earlier, later in zip(rallies, rallies[1:]):
        assert earlier["end_s"] <= later["start_s"]


def test_rallies_stay_inside_the_clip():
    """Padding must not invent time before 0 or past the end of the video."""
    impacts = [0.2, 0.9, 1.8, 2.9]

    rallies = segment_rallies(impacts, _motion([(0.0, 3.2)], duration=4.0), 4.0)

    assert rallies
    assert rallies[0]["start_s"] >= 0.0
    assert rallies[-1]["end_s"] <= 4.0


def test_silence_produces_no_rallies_rather_than_one_long_one():
    """"Nothing happened" and "the whole clip was a rally" are different."""
    ts = np.arange(0.0, 30.0, 0.2)
    flat = list(zip(ts.tolist(), np.full(ts.shape, 0.5).tolist()))

    assert segment_rallies([], flat, 30.0) == []


def test_impacts_with_no_motion_signal_still_segment():
    """Motion can be unusable -- a locked-off camera on a dark court.

    Impacts alone must still yield rallies, or tier 1 depends on both inputs
    when the design says it needs either.
    """
    ts = np.arange(0.0, 60.0, 0.2)
    flat = list(zip(ts.tolist(), np.full(ts.shape, 1.0).tolist()))
    impacts = [5.0, 6.0, 7.0, 8.0, 9.0, 30.0, 31.0, 32.0, 33.0]

    rallies = segment_rallies(impacts, flat, 60.0)

    assert len(rallies) == 2
    assert all(r["source"] == "audio" for r in rallies)


def test_no_inputs_at_all_is_empty_not_an_error():
    assert segment_rallies([], [], 30.0) == []


def test_confidence_rises_with_impact_count_and_is_capped():
    short = segment_rallies(
        [5.0, 6.0, 7.0], _motion([(4.5, 7.5)], duration=20.0), 20.0
    )
    long = segment_rallies(
        [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
        _motion([(4.5, 9.5)], duration=20.0), 20.0,
    )

    assert short and long
    assert short[0]["confidence"] < long[0]["confidence"]
    assert long[0]["confidence"] <= 1.0


def test_every_rally_is_at_least_the_minimum_length():
    impacts = [5.0, 5.4, 5.9, 6.4, 20.0, 20.4, 20.9, 21.4]

    rallies = segment_rallies(impacts, _motion([(4.5, 6.9), (19.5, 21.9)]), 40.0)

    for rally in rallies:
        assert rally["end_s"] - rally["start_s"] >= MIN_RALLY_S


def test_a_long_pause_splits_one_run_of_impacts_into_two_rallies():
    """The gap threshold is the whole definition of a rally boundary."""
    impacts = [5.0, 5.8, 6.6, 7.4, 25.0, 25.8, 26.6, 27.4]

    rallies = segment_rallies(impacts, _motion([(4.5, 7.9), (24.5, 27.9)]), 40.0)

    assert len(rallies) == 2


def test_the_module_never_reaches_for_a_ball():
    """The ladder's first principle, enforced rather than documented.

    Tier 1 must work where per-frame ball detection is hopeless. The moment
    this module *can* read a hit, something will make it, and rally structure
    silently re-acquires the detector's recall.

    Imports are the binding constraint, not prose -- the docstring is allowed
    to say "ball" in order to explain why there isn't one.
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "rally_segmenter.py").read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    pipeline_modules = {
        "job_runner", "detect_wall_hits", "judge_call", "inference_engine",
        "tracking_common", "bounce_gb_model_detector", "classify_events",
        "ball_detector", "ball_model", "court_model", "app",
    }
    assert not (imported & pipeline_modules), sorted(imported & pipeline_modules)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_a_zero_length_clip_is_empty_not_an_error(duration):
    assert segment_rallies([1.0, 2.0], [], duration) == []


# --- the motion threshold ---------------------------------------------------


def test_threshold_survives_the_median_being_a_plateau():
    """MAD is exactly zero whenever most samples share the median value.

    That is the ordinary case for a match clip, not a pathological one: idle
    time dominates, so the median IS the idle level and more than half the
    deviations are zero. A median+3*MAD threshold therefore lands exactly on
    the idle floor -- and with a strict > comparison, finds nothing at all.
    Every rally in the clip disappears and the run reports "no rallies", which
    is indistinguishable from a clip where nobody played.
    """
    from rally_segmenter import motion_threshold

    idle_dominated = [0.4] * 254 + [8.0] * 46

    threshold = motion_threshold(idle_dominated)

    assert threshold is not None
    assert 0.4 < threshold < 8.0


def test_threshold_is_none_when_nothing_rises_above_the_floor():
    from rally_segmenter import motion_threshold

    assert motion_threshold([0.5] * 100) is None


def test_threshold_uses_mad_when_it_is_usable():
    """Continuous real energy has a non-zero MAD; that path is the default."""
    from rally_segmenter import MOTION_MAD_MULTIPLIER, motion_threshold
    import statistics

    energies = [float(value % 7) + 0.5 for value in range(200)]
    median = statistics.median(energies)
    mad = statistics.median([abs(energy - median) for energy in energies])
    assert mad > 0, "fixture must exercise the MAD path"

    assert motion_threshold(energies) == median + MOTION_MAD_MULTIPLIER * mad


# --- motion energy ----------------------------------------------------------


def test_motion_energy_is_zero_between_identical_frames():
    """A locked-off camera on a still court must read as no motion at all."""
    import cv2
    from rally_segmenter import motion_energy_step

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    small, _ = motion_energy_step(None, frame)
    _, energy = motion_energy_step(small, frame)

    assert energy == 0.0


def test_motion_energy_rises_with_how_much_of_the_frame_changed():
    from rally_segmenter import motion_energy_step

    base = np.full((480, 640, 3), 120, dtype=np.uint8)
    small_change = base.copy()
    small_change[0:40, 0:40] = 255
    big_change = base.copy()
    big_change[0:400, 0:600] = 255

    reference, _ = motion_energy_step(None, base)
    _, little = motion_energy_step(reference, small_change)
    _, lots = motion_energy_step(reference, big_change)

    assert 0.0 < little < lots


def test_the_first_frame_has_no_predecessor_so_no_energy():
    """Seeding must not invent a spike at frame zero that reads as a rally."""
    from rally_segmenter import motion_energy_step

    _, energy = motion_energy_step(None, np.zeros((480, 640, 3), dtype=np.uint8))

    assert energy == 0.0


def test_motion_energy_is_resolution_independent():
    """1080p and 4K of the same scene must not produce different rally counts."""
    from rally_segmenter import motion_energy_step

    def energy_at(height, width):
        base = np.full((height, width, 3), 120, dtype=np.uint8)
        moved = base.copy()
        moved[: height // 2, : width // 2] = 255
        reference, _ = motion_energy_step(None, base)
        return motion_energy_step(reference, moved)[1]

    assert abs(energy_at(480, 640) - energy_at(1080, 1440)) < 2.0


# --- the timeline -----------------------------------------------------------


def _hit_rally(number, start, end):
    return {"rally_number": number,
            "start_time_seconds": start, "end_time_seconds": end}


def test_timeline_reports_audio_unavailable_without_pretending_it_was_silent():
    """None impacts means the audio could not be read; [] means it was quiet.

    Collapsing them would report a clip with no audio track as one where
    nobody hit anything.
    """
    from rally_segmenter import build_rally_timeline

    timeline = build_rally_timeline(None, _motion([(10.0, 16.0)], duration=30.0),
                                    30.0, None)

    assert timeline["audio_available"] is False
    assert timeline["rallies"]


def test_timeline_agrees_when_every_hit_rally_lands_inside_one():
    from rally_segmenter import build_rally_timeline

    impacts = [5.0, 5.8, 6.9, 8.0, 9.2, 30.0, 30.7, 31.9, 33.0]
    timeline = build_rally_timeline(
        impacts, _motion([(4.5, 9.5), (29.5, 33.5)]), 60.0,
        {"rallies": [_hit_rally(1, 5.0, 9.2), _hit_rally(2, 30.0, 33.0)]},
    )

    assert timeline["agrees_with_hits"] is True


def test_timeline_disagrees_when_a_hit_rally_falls_outside_every_span():
    from rally_segmenter import build_rally_timeline

    impacts = [5.0, 5.8, 6.9, 8.0, 9.2]
    timeline = build_rally_timeline(
        impacts, _motion([(4.5, 9.5)], duration=60.0), 60.0,
        {"rallies": [_hit_rally(1, 5.0, 9.2), _hit_rally(2, 48.0, 52.0)]},
    )

    assert timeline["agrees_with_hits"] is False


def test_agreement_is_unknown_rather_than_true_when_there_are_no_hit_rallies():
    """With the ball tier off there is nothing to agree with.

    Reporting True would claim corroboration that never happened.
    """
    from rally_segmenter import build_rally_timeline

    timeline = build_rally_timeline(
        [5.0, 6.0, 7.0, 8.0], _motion([(4.5, 8.5)], duration=30.0), 30.0, None
    )

    assert timeline["agrees_with_hits"] is None


def test_timeline_carries_the_gap_it_used():
    """The gap is inferred per clip, so a reader cannot reconstruct it."""
    from rally_segmenter import build_rally_timeline

    timeline = build_rally_timeline(
        [5.0, 6.0, 7.0, 8.0, 30.0, 31.0, 32.0, 33.0],
        _motion([(4.5, 8.5), (29.5, 33.5)]), 60.0, None,
    )

    assert timeline["gap_s"] > 0


def test_a_rally_that_dips_below_threshold_is_not_shattered_into_fragments():
    """Real motion energy is spiky, not a plateau.

    Measured on the repo's own 5-minute clip: 12.3% of samples sit above the
    threshold, but as 98 fragments whose MEDIAN duration is 0.00 s and whose
    longest is 1.83 s. Requiring strictly contiguous samples therefore found
    zero rallies in a real match -- every fragment fell under MIN_RALLY_S.

    A rally is sustained activity, which in a noisy signal means "active most
    of the time", not "active at every sample". Short dips are bridged before
    duration is measured.
    """
    ts = np.arange(0.0, 40.0, 0.2)
    energy = np.full(ts.shape, 0.4)
    # One 12-second rally, interrupted by three sub-second dips.
    energy[(ts >= 10.0) & (ts <= 22.0)] = 8.0
    for dip in (13.0, 16.5, 19.0):
        energy[(ts >= dip) & (ts <= dip + 0.6)] = 0.4
    series = list(zip(ts.tolist(), energy.tolist()))

    rallies = segment_rallies([], series, 40.0)

    assert len(rallies) == 1, [
        (round(r["start_s"], 1), round(r["end_s"], 1)) for r in rallies
    ]
    assert rallies[0]["end_s"] - rallies[0]["start_s"] > 10.0


def test_bridging_never_merges_two_genuinely_separate_rallies():
    """The bridge must stay well under the shortest gap between rallies.

    MIN_GAP_S is 4 s. If the bridge approached that, two rallies separated by
    a normal between-point pause would fuse into one and the rally count would
    silently halve.
    """
    from rally_segmenter import MIN_GAP_S, MOTION_BRIDGE_S

    assert MOTION_BRIDGE_S < MIN_GAP_S / 2.0

    rallies = segment_rallies(
        [], _motion([(5.0, 12.0), (20.0, 27.0)], duration=40.0), 40.0
    )

    assert len(rallies) == 2
