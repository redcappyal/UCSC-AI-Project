"""The bounce classifier's train/test split must not leak across time.

Feature rows are per-frame with overlapping context windows, so a random
split puts near-duplicate neighbouring frames on both sides and inflates the
reported test score. These tests pin the chronological behaviour.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_bounce_classifier import chronological_split


def features_over_frames(frame_count):
    return pd.DataFrame(
        {
            "frame": list(range(frame_count)),
            "is_wall_hit": [1 if frame % 10 == 0 else 0 for frame in range(frame_count)],
        }
    )


def test_every_test_frame_comes_after_every_train_frame():
    train, test, cut_frame = chronological_split(
        features_over_frames(100), test_size=0.25, embargo_frames=0
    )

    assert train["frame"].max() < test["frame"].min()
    assert cut_frame == 75


def test_embargo_drops_the_band_straddling_the_cut():
    train, test, cut_frame = chronological_split(
        features_over_frames(100), test_size=0.25, embargo_frames=5
    )

    # Rows whose context window would span the cut belong to neither side.
    assert train["frame"].max() == cut_frame - 6
    assert test["frame"].min() == cut_frame
    assert not set(train["frame"]) & set(test["frame"])


def test_split_is_order_independent():
    shuffled = features_over_frames(100).sample(frac=1.0, random_state=0)

    train, test, cut_frame = chronological_split(
        shuffled, test_size=0.25, embargo_frames=0
    )

    assert cut_frame == 75
    assert train["frame"].max() < test["frame"].min()


def test_split_is_deterministic():
    first = chronological_split(features_over_frames(100), 0.25, 3)
    second = chronological_split(features_over_frames(100), 0.25, 3)

    assert list(first[0]["frame"]) == list(second[0]["frame"])
    assert list(first[1]["frame"]) == list(second[1]["frame"])


def test_empty_side_is_rejected():
    with pytest.raises(RuntimeError, match="empty"):
        chronological_split(features_over_frames(10), test_size=1.0, embargo_frames=0)
