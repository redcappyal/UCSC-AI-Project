"""Scoring for the rally-boundary eval axis.

An analysis with no eval axis is an opinion. The rally segmenter's thresholds
are all first guesses, and this is the scorer that turns "the boundaries look
about right" into a number that can move.

Matching is on rally *midpoints* within a tolerance, not on exact edges: the
predicted and labeled structures come from different signals, and their
borders legitimately differ by a second either way. What is being scored is
whether the same rallies were found.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_rally_boundaries import cluster_hit_frames_into_rallies, score_rallies


def _rally(start, end):
    return {"start_s": start, "end_s": end}


def test_exact_match_scores_perfect():
    rallies = [_rally(5.0, 10.0), _rally(20.0, 25.0)]

    score = score_rallies(rallies, rallies)

    assert score["f1"] == 1.0
    assert score["tp"] == 2
    assert score["fp"] == 0
    assert score["fn"] == 0


def test_one_missed_rally_costs_recall():
    labeled = [_rally(5.0, 10.0), _rally(20.0, 25.0), _rally(40.0, 45.0)]
    predicted = [_rally(5.0, 10.0), _rally(20.0, 25.0)]

    score = score_rallies(predicted, labeled)

    assert score["recall"] == 2 / 3
    assert score["precision"] == 1.0
    assert score["fn"] == 1


def test_an_invented_rally_costs_precision():
    labeled = [_rally(5.0, 10.0)]
    predicted = [_rally(5.0, 10.0), _rally(30.0, 35.0)]

    score = score_rallies(predicted, labeled)

    assert score["precision"] == 0.5
    assert score["recall"] == 1.0
    assert score["fp"] == 1


def test_a_near_miss_inside_tolerance_still_counts():
    """Edges differ by design; only gross misplacement should be penalised."""
    score = score_rallies([_rally(5.8, 10.8)], [_rally(5.0, 10.0)], tol_s=1.5)

    assert score["tp"] == 1


def test_a_miss_outside_tolerance_does_not_count():
    score = score_rallies([_rally(12.0, 17.0)], [_rally(5.0, 10.0)], tol_s=1.5)

    assert score["tp"] == 0
    assert score["fp"] == 1
    assert score["fn"] == 1


def test_each_label_is_claimed_at_most_once():
    """Splitting one rally into three must cost precision, not earn credit.

    Without one-to-one matching, over-segmentation scores as several true
    positives and the failure mode reads as a perfect result. Here a single
    10 s rally is split into three consecutive pieces: only the piece whose
    midpoint lands near the label's can claim it.
    """
    labeled = [_rally(5.0, 15.0)]
    predicted = [_rally(5.0, 8.0), _rally(8.5, 11.5), _rally(12.0, 15.0)]

    score = score_rallies(predicted, labeled)

    assert score["tp"] == 1
    assert score["fp"] == 2


def test_two_predictions_inside_tolerance_still_claim_only_one_label():
    """The one-to-one rule itself, with the geometry made trivial."""
    labeled = [_rally(5.0, 15.0)]
    predicted = [_rally(8.0, 12.0), _rally(8.2, 12.2)]

    score = score_rallies(predicted, labeled)

    assert score["tp"] == 1
    assert score["fp"] == 1


def test_empty_prediction_against_labels_scores_zero_not_an_error():
    score = score_rallies([], [_rally(5.0, 10.0)], tol_s=1.5)

    assert score["f1"] == 0.0
    assert score["fn"] == 1


def test_nothing_predicted_and_nothing_labeled_is_perfect_not_undefined():
    """A clip with no rallies, correctly reported as having none."""
    score = score_rallies([], [])

    assert score["f1"] == 1.0


def test_count_delta_reports_over_and_under_segmentation_signed():
    over = score_rallies([_rally(1, 2), _rally(5, 6), _rally(9, 10)],
                         [_rally(1, 2)])
    under = score_rallies([_rally(1, 2)],
                          [_rally(1, 2), _rally(5, 6), _rally(9, 10)])

    assert over["count_delta"] == 2
    assert under["count_delta"] == -2


# --- silver labels from human hit frames ------------------------------------


def test_human_hit_frames_cluster_into_rallies():
    """Silver labels come from human-labeled hits, clustered by their gaps.

    Not from detector output -- that would score the segmenter against the
    very recall problem it exists to route around.
    """
    # Two bursts at 60 fps: frames 60-300 (1-5 s) and 1800-2100 (30-35 s).
    frames = [60, 120, 180, 240, 300, 1800, 1860, 1920, 1980, 2100]

    rallies = cluster_hit_frames_into_rallies(frames, fps=60.0)

    assert len(rallies) == 2
    assert rallies[0]["start_s"] == 1.0
    assert rallies[1]["end_s"] == 35.0


def test_clustering_needs_a_frame_rate_to_mean_anything():
    """Frame numbers without an fps are anonymous integers.

    The README makes this point about the label CSVs themselves: without the
    sidecar recording fps and the video sha, the labels index into nothing.
    """
    import pytest

    with pytest.raises(ValueError):
        cluster_hit_frames_into_rallies([60, 120], fps=0)


def test_a_lone_hit_is_not_a_rally():
    assert cluster_hit_frames_into_rallies([60], fps=60.0) == []
