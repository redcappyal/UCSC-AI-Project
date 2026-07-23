"""CLI: download the Roboflow dataset and train YOLO11n for on-device Core ML.

The production cloud model stays RF-DETR; this trains the *phone* model on the
same labels. Train anywhere with a GPU; export to Core ML on the Mac
(ios/MODEL.md). ultralytics/roboflow import lazily so the test env stays light.
"""

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))


def build_train_kwargs(data_yaml, imgsz=960, epochs=100, batch=-1,
                       name="ball-yolo11n", device=None):
    """Ultralytics train() kwargs. imgsz=960 matches the pipeline's inference
    width — the ball is small in frame and 640 measurably hurts recall."""
    kwargs = {
        "data": str(data_yaml), "imgsz": imgsz, "epochs": epochs,
        "batch": batch, "name": name, "cache": True,
    }
    if device is not None:
        kwargs["device"] = device
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",
                        default=os.environ.get("ROBOFLOW_WORKSPACE"),
                        help="Roboflow workspace slug")
    parser.add_argument("--project", default="ai-squash-line-tracker")
    parser.add_argument("--dataset-version", type=int, required=True,
                        help="Roboflow DATASET version (not the model version)")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (same key the app uses).")
    if not args.workspace:
        raise SystemExit("Pass --workspace or set ROBOFLOW_WORKSPACE in .env.")

    from roboflow import Roboflow  # lazy
    from ultralytics import YOLO   # lazy

    dataset = (
        Roboflow(api_key=api_key)
        .workspace(args.workspace)
        .project(args.project)
        .version(args.dataset_version)
        .download("yolov11")
    )
    data_yaml = Path(dataset.location) / "data.yaml"

    model = YOLO(args.model)
    results = model.train(**build_train_kwargs(
        data_yaml, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device))
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    print("next: score it with yolo_model_eval.py, then export via ios/MODEL.md")


if __name__ == "__main__":
    main()
