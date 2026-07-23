"""yolo_model_eval: ultralytics boxes -> the pipeline's prediction dicts.

Module import and the adapter must work without ultralytics installed —
the test env is requirements-test.txt only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking_common import select_ball_prediction
from yolo_model_eval import yolo_boxes_to_predictions


def test_boxes_convert_to_center_size_dicts():
    rows = [(100.0, 200.0, 110.0, 212.0, 0.9, 0)]
    names = {0: "ball"}
    predictions = yolo_boxes_to_predictions(rows, names)
    assert predictions == [{
        "x": 105.0, "y": 206.0, "width": 10.0, "height": 12.0,
        "confidence": 0.9, "class": "ball",
    }]


def test_ball_class_wins_selection_over_other_classes():
    rows = [
        (0.0, 0.0, 50.0, 50.0, 0.95, 1),      # e.g. "player"
        (100.0, 200.0, 110.0, 212.0, 0.6, 0), # "ball"
    ]
    names = {0: "ball", 1: "player"}
    selected = select_ball_prediction(yolo_boxes_to_predictions(rows, names))
    assert selected["class"] == "ball"
    assert selected["confidence"] == 0.6


def test_unknown_class_index_stringifies():
    predictions = yolo_boxes_to_predictions([(0, 0, 2, 2, 0.5, 7)], {})
    assert predictions[0]["class"] == "7"
