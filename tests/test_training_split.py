"""The bounce classifier uses the restored random stratified train/test split."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_bounce_classifier import split_training_rows


def features_over_frames(frame_count):
    return pd.DataFrame(
        {
            "frame": list(range(frame_count)),
            "is_wall_hit": [1 if frame % 10 == 0 else 0 for frame in range(frame_count)],
        }
    )


def test_split_keeps_requested_sizes():
    train, test, y_train, y_test = split_training_rows(
        features_over_frames(100),
        random_seed=7,
        test_size=0.25,
    )

    assert len(train) == 75
    assert len(test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25


def test_split_is_stratified_when_each_class_has_enough_rows():
    train, test, y_train, y_test = split_training_rows(
        features_over_frames(100),
        random_seed=7,
        test_size=0.25,
    )

    assert int(y_train.sum()) == 7
    assert int(y_test.sum()) == 3
    assert train["is_wall_hit"].mean() == y_train.mean()
    assert test["is_wall_hit"].mean() == y_test.mean()


def test_split_is_deterministic_for_same_seed():
    first = split_training_rows(features_over_frames(100), random_seed=7, test_size=0.25)
    second = split_training_rows(features_over_frames(100), random_seed=7, test_size=0.25)

    assert list(first[0]["frame"]) == list(second[0]["frame"])
    assert list(first[1]["frame"]) == list(second[1]["frame"])


def test_split_changes_with_different_seed():
    first = split_training_rows(features_over_frames(100), random_seed=7, test_size=0.25)
    second = split_training_rows(features_over_frames(100), random_seed=11, test_size=0.25)

    assert list(first[1]["frame"]) != list(second[1]["frame"])


def test_split_is_not_chronological():
    train, test, _, _ = split_training_rows(
        features_over_frames(100),
        random_seed=7,
        test_size=0.25,
    )

    assert train["frame"].max() > test["frame"].min()
