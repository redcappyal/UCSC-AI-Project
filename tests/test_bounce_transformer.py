import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from bounce_transformer import (
    BounceTransformerClassifier,
    temporal_feature_columns,
    transformer_feature_columns,
)
from train_bounce_classifier import MODEL_FEATURE_COLUMNS


def synthetic_features(row_count=24, context=1):
    rng = np.random.default_rng(7)
    columns = transformer_feature_columns(context, MODEL_FEATURE_COLUMNS)
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(columns))).astype(np.float32),
        columns=columns,
    )
    for offset in range(-context, context + 1):
        features[f"t{offset:+d}_detected"] = 1.0
        features[f"t{offset:+d}_confidence"] = rng.uniform(
            0.4,
            0.95,
            size=row_count,
        )
    return features


def tiny_classifier(context=1):
    return BounceTransformerClassifier(
        context=context,
        global_feature_columns=MODEL_FEATURE_COLUMNS,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=8,
        epochs=1,
        validation_fraction=0.0,
        patience=0,
        device="cpu",
        verbose=False,
    )


def test_temporal_feature_columns_keep_frame_tokens_together():
    columns = temporal_feature_columns(context=1)

    assert columns[:6] == [
        "t-1_detected",
        "t-1_confidence",
        "t-1_x",
        "t-1_y",
        "t-1_width",
        "t-1_height",
    ]
    assert columns[-1] == "t+1_height"


def test_transformer_classifier_predicts_and_survives_joblib_round_trip(tmp_path):
    features = synthetic_features()
    labels = np.asarray([0, 1] * 12)
    classifier = tiny_classifier().fit(features, labels)

    before = classifier.predict_proba(features.iloc[:5])
    artifact_path = tmp_path / "bounce_transformer.pkl"
    joblib.dump(
        {
            "model": classifier,
            "feature_columns": classifier.feature_columns,
            "hit_threshold": 0.25,
        },
        artifact_path,
    )
    loaded = joblib.load(artifact_path)
    after = loaded["model"].predict_proba(features.iloc[:5])

    assert before.shape == (5, 2)
    assert np.allclose(before.sum(axis=1), 1.0)
    assert np.allclose(before, after)
    assert loaded["model"].classes_.tolist() == [0, 1]
