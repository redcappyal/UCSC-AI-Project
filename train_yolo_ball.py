"""DEPRECATED: Ultralytics YOLO11n trainer. Do not ship anything it produces.

Ultralytics is AGPL-3.0, which is incompatible with shipping CrossCourt on the
App Store *and*, via section 13's network clause, with serving inference from
the Flask pipeline — so keeping it server-side is not a workaround either. The
supported path is YOLOX (Apache-2.0): see `yolox_ball_exp.py` and ios/MODEL.md.

Kept for reference only, because the phone model's history runs through it.

Originally: download the Roboflow dataset and train YOLO11n for on-device Core
ML. The production cloud model stays RF-DETR; this trained the *phone* model on
the same labels. ultralytics/roboflow import lazily so the test env stays light.
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
                       name="ball-yolo11n", device=None, cache=False):
    """Ultralytics train() kwargs. imgsz=960 matches the pipeline's inference
    width — the ball is small in frame and 640 measurably hurts recall.

    `cache` defaults OFF: RAM caching costs roughly imgsz^2 * 3 bytes per
    image (~30 GB for the 11.5k-image v3 dataset at 960 px), which thrashes
    or OOMs a normal box. It was a safe default only while the dataset was
    v1-sized (~1.4k images). Pass "ram"/"disk" deliberately if you have the
    headroom; the dataloader's workers handle JPEG decode fine without it.
    """
    kwargs = {
        "data": str(data_yaml), "imgsz": imgsz, "epochs": epochs,
        "batch": batch, "name": name, "cache": cache,
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
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 auto-sizes to available VRAM (CUDA)")
    parser.add_argument("--device", default=None,
                        help='e.g. "0" for the first CUDA GPU, "mps", "cpu"')
    parser.add_argument("--cache", choices=("off", "ram", "disk"), default="off",
                        help="image cache; see build_train_kwargs (default off)")
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
        batch=args.batch, device=args.device,
        cache=False if args.cache == "off" else args.cache))
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    print("next: score it with yolo_model_eval.py, then export via ios/MODEL.md")


if __name__ == "__main__":
    main()
