"""train_yolo_ball: training-kwargs builder (no ultralytics/roboflow needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_yolo_ball import build_train_kwargs


def test_defaults_pin_imgsz_960_and_name():
    kwargs = build_train_kwargs("data/data.yaml")
    assert kwargs == {
        "data": "data/data.yaml", "imgsz": 960, "epochs": 100,
        "batch": -1, "name": "ball-yolo11n", "cache": False,
    }


def test_device_only_included_when_set():
    assert "device" not in build_train_kwargs("d.yaml", device=None)
    assert build_train_kwargs("d.yaml", device="0")["device"] == "0"


def test_cache_defaults_off_and_is_opt_in():
    # RAM caching the 11.5k-image v3 dataset at 960 px wants ~30 GB; the
    # default must not silently do that on a normal box.
    assert build_train_kwargs("d.yaml")["cache"] is False
    assert build_train_kwargs("d.yaml", cache="disk")["cache"] == "disk"
