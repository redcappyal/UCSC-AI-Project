import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_BALL_CSV = Path(__file__).with_name("ball_coordinates.csv")
DEFAULT_HIT_CSV = Path(__file__).with_name("wall_hits.csv")
DEFAULT_MODEL_OUTPUT = Path(__file__).with_name("wall_hit_model.pkl")
DEFAULT_DATASET_OUTPUT = Path(__file__).with_name("wall_hit_dataset.npz")
FRAME_COLUMNS = ("source_frame", "frame", "hit_frame")
WINDOW_RADIUS = 10
FEATURE_COLUMNS = ("x_center", "y_center")
FEATURE_FIELDS = (*FEATURE_COLUMNS, "detected")


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_ball_positions(csv_path):
    positions = {}

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"source_frame", "detected", *FEATURE_COLUMNS}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            frame = int(row["source_frame"])
            detected = parse_bool(row["detected"])

            if detected and row["x_center"] and row["y_center"]:
                x = float(row["x_center"])
                y = float(row["y_center"])
            else:
                x = 0.0
                y = 0.0

            positions[frame] = {
                "detected": 1.0 if detected else 0.0,
                "x_center": x,
                "y_center": y,
            }

    return positions


def load_hit_frames(csv_path):
    hit_frames = set()

    with csv_path.open(newline="") as csv_file:
        sample = csv_file.read(1024)
        csv_file.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False

        if has_header:
            reader = csv.DictReader(csv_file)
            frame_column = next(
                (column for column in FRAME_COLUMNS if column in (reader.fieldnames or [])),
                None,
            )

            if frame_column is None:
                expected = ", ".join(FRAME_COLUMNS)
                raise ValueError(
                    f"{csv_path} must contain one of these columns: {expected}"
                )

            for row in reader:
                if row.get(frame_column, "").strip():
                    hit_frames.add(int(row[frame_column]))
        else:
            reader = csv.reader(csv_file)
            for row in reader:
                if row and row[0].strip():
                    hit_frames.add(int(row[0]))

    return hit_frames


def build_feature_vector(positions, center_frame, window_radius):
    features = []

    for offset in range(-window_radius, window_radius + 1):
        row = positions.get(center_frame + offset)

        if row is None:
            features.extend([0.0, 0.0, 0.0])
            continue

        features.extend([row["x_center"], row["y_center"], row["detected"]])

    return features


def build_feature_names(window_radius):
    feature_names = []

    for offset in range(-window_radius, window_radius + 1):
        for field in FEATURE_FIELDS:
            feature_names.append(f"frame_{offset:+d}_{field}")

    return np.array(feature_names)


def build_dataset(positions, hit_frames, window_radius):
    X = []
    y = []
    frames = []

    for frame in sorted(positions):
        X.append(build_feature_vector(positions, frame, window_radius))
        y.append(1 if frame in hit_frames else 0)
        frames.append(frame)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8), np.array(frames)


def oversample_minority_class(X, y, random_state):
    rng = np.random.default_rng(random_state)
    class_counts = np.bincount(y, minlength=2)

    if class_counts[0] == class_counts[1]:
        return X, y

    minority_class = int(np.argmin(class_counts))
    majority_count = int(np.max(class_counts))
    minority_indices = np.where(y == minority_class)[0]
    extra_count = majority_count - len(minority_indices)
    extra_indices = rng.choice(minority_indices, size=extra_count, replace=True)
    all_indices = np.concatenate([np.arange(len(y)), extra_indices])
    rng.shuffle(all_indices)

    return X[all_indices], y[all_indices]


def train_model(X, y, test_size, random_state):
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)

    if positive_count == 0:
        raise ValueError("No positive hit labels were found. Check the hit-frame CSV.")

    if negative_count == 0:
        raise ValueError("No negative labels were found. The hit-frame CSV labels every row.")

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_state,
        ),
    )

    min_class_count = min(positive_count, negative_count)
    can_split_both_classes = (
        min_class_count * test_size >= 1
        and min_class_count * (1 - test_size) >= 1
    )

    if not can_split_both_classes:
        X_train, y_train = oversample_minority_class(X, y, random_state)
        model.fit(X_train, y_train)
        report = (
            "Skipped test evaluation because there are not enough examples "
            "for both classes to appear in both train and test sets."
        )
        return model, report

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, y_train = oversample_minority_class(X_train, y_train, random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["miss", "hit"],
        zero_division=0,
    )
    return model, report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build frame-window features and train a wall-hit classifier."
    )
    parser.add_argument(
        "--ball-csv",
        type=Path,
        default=DEFAULT_BALL_CSV,
        help=f"CSV from modelEval.py. Defaults to {DEFAULT_BALL_CSV.name}.",
    )
    parser.add_argument(
        "--hit-csv",
        type=Path,
        default=DEFAULT_HIT_CSV,
        help=(
            "CSV containing hit frames. Accepts source_frame, frame, or hit_frame "
            f"column. Defaults to {DEFAULT_HIT_CSV.name}."
        ),
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=WINDOW_RADIUS,
        help="Number of frames before and after the current frame to include.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT,
        help=f"Where to save X, y, and frame numbers. Defaults to {DEFAULT_DATASET_OUTPUT.name}.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help=f"Where to save the trained model. Defaults to {DEFAULT_MODEL_OUTPUT.name}.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of examples reserved for test evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model initialization.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only build and save X/y, without training the classifier.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    positions = load_ball_positions(args.ball_csv)
    hit_frames = load_hit_frames(args.hit_csv)
    X, y, frames = build_dataset(positions, hit_frames, args.window_radius)
    feature_names = build_feature_names(args.window_radius)

    np.savez_compressed(
        args.dataset_output,
        X=X,
        y=y,
        frames=frames,
        feature_names=feature_names,
    )

    unmatched_hit_frames = hit_frames - set(frames.tolist())

    print(f"Loaded ball positions: {len(positions)}")
    print(f"Loaded hit frames: {len(hit_frames)}")
    print(f"Built X shape: {X.shape}")
    print(f"Built y shape: {y.shape}")
    print(f"Positive hit labels: {int(y.sum())}")
    print(f"Saved dataset: {args.dataset_output}")

    if unmatched_hit_frames:
        preview = ", ".join(str(frame) for frame in sorted(unmatched_hit_frames)[:10])
        print(f"Warning: hit frame(s) not found in ball CSV: {preview}")

    if args.no_train:
        return

    model, report = train_model(X, y, args.test_size, args.random_state)

    with args.model_output.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(report)
    print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()
