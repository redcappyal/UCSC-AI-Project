"""Train the temporal transformer bounce classifier from feature CSVs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from bounce_transformer import (
    BounceTransformerClassifier,
    transformer_feature_columns,
)
from train_bounce_classifier import (
    MODEL_FEATURE_COLUMNS,
    RUNTIME_EVAL_MIN_GAP_FRAMES,
    RUNTIME_EVAL_WALL_VISIT_GAP_FRAMES,
    evaluate_model_predictions,
    load_geometry,
    split_training_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_OUTPUT = ROOT / "bounce_transformer_model.pkl"
DEFAULT_COMBINED_FEATURES_OUTPUT = ROOT / "bounce_transformer_features.csv"


def named_path(value):
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("source name cannot be empty")
    return name, Path(path_text.strip())


def load_feature_tables(feature_specs):
    tables = []
    sources = []
    for source_name, path in feature_specs:
        if not path.exists():
            raise FileNotFoundError(f"Feature CSV was not found: {path}")
        table = pd.read_csv(path)
        if "is_wall_hit" not in table.columns:
            raise ValueError(f"{path} does not contain is_wall_hit.")
        table = table.copy()
        table["source_video"] = source_name
        tables.append(table)
        sources.append(
            {
                "name": source_name,
                "path": str(path),
                "rows": len(table),
                "positives": int(table["is_wall_hit"].sum()),
            }
        )
        print(
            f"Loaded {len(table)} row(s) from {source_name} "
            f"({int(table['is_wall_hit'].sum())} positive): {path}",
            flush=True,
        )
    return pd.concat(tables, ignore_index=True, sort=False), sources


def infer_context(features):
    offsets = []
    for column in features.columns:
        if not column.startswith("t") or "_" not in column:
            continue
        prefix = column.split("_", 1)[0]
        try:
            offsets.append(abs(int(prefix[1:])))
        except ValueError:
            continue
    if not offsets:
        raise ValueError("Feature CSVs do not contain temporal t+/-N columns.")
    return max(offsets)


def calibration_map(specs):
    geometries = {}
    for source_name, path in specs:
        geometry = load_geometry(path)
        if geometry is None:
            raise FileNotFoundError(
                f"Calibration for {source_name!r} was not found: {path}"
            )
        geometries[source_name] = geometry
    return geometries


def model_from_args(args, context, *, epochs=None, validation_fraction=None):
    return BounceTransformerClassifier(
        context=context,
        global_feature_columns=MODEL_FEATURE_COLUMNS,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.feedforward_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs if epochs is None else epochs,
        validation_fraction=(
            args.validation_fraction
            if validation_fraction is None
            else validation_fraction
        ),
        patience=args.patience,
        random_seed=args.random_seed,
        device=args.device,
        verbose=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a temporal transformer bounce classifier from one or more "
            "named feature CSVs."
        )
    )
    parser.add_argument(
        "--features",
        action="append",
        type=named_path,
        required=True,
        metavar="SOURCE=CSV",
        help="Feature CSV to include. Repeat once per source video.",
    )
    parser.add_argument(
        "--calibration",
        action="append",
        type=named_path,
        default=[],
        metavar="SOURCE=JSON",
        help="Optional per-source calibration used for app-style evaluation.",
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--combined-features-output",
        type=Path,
        default=DEFAULT_COMBINED_FEATURES_OUTPUT,
    )
    parser.add_argument("--hit-threshold", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--no-runtime-eval-filters",
        action="store_true",
        help="Evaluate raw threshold predictions without app-style grouping.",
    )
    parser.add_argument(
        "--no-refit-all",
        action="store_true",
        help="Save the train-split model instead of refitting on every row.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.d_model % args.nhead != 0:
        raise ValueError("--d-model must be divisible by --nhead.")

    features, source_summaries = load_feature_tables(args.features)
    context = infer_context(features)
    feature_columns = transformer_feature_columns(context, MODEL_FEATURE_COLUMNS)
    missing_columns = [
        column for column in feature_columns if column not in features.columns
    ]
    if missing_columns:
        print(
            "Feature inputs are missing columns; filling with 0.0: "
            + ", ".join(missing_columns),
            flush=True,
        )
        for column in missing_columns:
            features[column] = 0.0

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if features["is_wall_hit"].astype(int).nunique() < 2:
        raise RuntimeError("Training requires both positive and negative rows.")

    args.combined_features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.combined_features_output, index=False)
    print(
        f"Combined {len(features)} row(s) from {len(source_summaries)} video(s) "
        f"into {args.combined_features_output}.",
        flush=True,
    )

    train_rows, test_rows, y_train, y_test = split_training_rows(
        features,
        args.random_seed,
        test_size=args.test_size,
    )
    x_train = train_rows[feature_columns]
    x_test = test_rows[feature_columns]
    training_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(np.float32)

    print(
        f"Training transformer with context +/-{context}: "
        f"{len(x_train)} train row(s), {len(x_test)} test row(s), "
        f"{len(feature_columns)} input feature(s).",
        flush=True,
    )
    evaluation_model = model_from_args(args, context)
    evaluation_model.fit(x_train, y_train.to_numpy(), sample_weight=training_weights)
    probabilities = evaluation_model.predict_proba(x_test)[:, 1]

    runtime_eval_config = {
        "enabled": not args.no_runtime_eval_filters,
        "geometry": None,
        "geometry_by_source": calibration_map(args.calibration),
        "spatial_filter": True,
        "spatial_filter_mode": "sidewall",
        "collapse_wall_area": True,
        "wall_visit_gap": RUNTIME_EVAL_WALL_VISIT_GAP_FRAMES,
        "min_gap": RUNTIME_EVAL_MIN_GAP_FRAMES,
    }
    evaluation = evaluate_model_predictions(
        test_rows.reset_index(drop=True),
        y_test.reset_index(drop=True),
        probabilities,
        args.hit_threshold,
        runtime_eval_config,
    )
    print(f"Hit threshold: {args.hit_threshold:.3f}")
    print("Confusion matrix:")
    print(evaluation["confusion_matrix"])
    print(
        classification_report(
            y_test,
            evaluation["predictions"],
            digits=3,
            zero_division=0,
        )
    )

    model_to_save = evaluation_model
    refit_epochs = len(evaluation_model.training_history_)
    if not args.no_refit_all:
        print(
            f"Refitting the transformer on all {len(features)} rows for "
            f"{refit_epochs} epoch(s)...",
            flush=True,
        )
        all_weights = compute_sample_weight(
            class_weight="balanced",
            y=features["is_wall_hit"].astype(int),
        ).astype(np.float32)
        model_to_save = model_from_args(
            args,
            context,
            epochs=refit_epochs,
            validation_fraction=0.0,
        )
        model_to_save.fit(
            features[feature_columns],
            features["is_wall_hit"].astype(int).to_numpy(),
            sample_weight=all_weights,
        )

    artifact = {
        "model": model_to_save,
        "model_type": "bounce_transformer",
        "feature_columns": feature_columns,
        "positive_label": "is_wall_hit",
        "hit_threshold": float(args.hit_threshold),
        "context": context,
        "transformer_config": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "refit_epochs": refit_epochs,
        },
        "training_sources": source_summaries,
        "metrics": evaluation["metrics"],
        "raw_metrics": evaluation["raw_metrics"],
        "runtime_evaluation": evaluation["runtime_eval"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)
    print(f"Saved transformer artifact to {args.model_output}", flush=True)
    print(
        "Metrics: "
        + json.dumps(
            {
                "precision": evaluation["metrics"]["precision"],
                "recall": evaluation["metrics"]["recall"],
                "f1": evaluation["metrics"]["f1"],
                "f2": evaluation["metrics"]["f2"],
                "accuracy": evaluation["metrics"]["accuracy"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
