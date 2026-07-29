import csv
import json
import sys

import cv2
import numpy as np

import racket_model_eval
from inference_engine import infer_frame_predictions
from racket_model_eval import (
    ServerlessRacketModel,
    csv_rows_for_frame,
    draw_predictions,
    filter_predictions,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def run_workflow(
        self,
        *,
        workspace_name,
        workflow_id,
        images,
        use_cache,
    ):
        self.calls.append(
            {
                "workspace_name": workspace_name,
                "workflow_id": workflow_id,
                "images": images,
                "use_cache": use_cache,
            }
        )
        return {
            "predictions": [
                {
                    "class": "racket",
                    "confidence": 0.82,
                    "x": 20,
                    "y": 10,
                    "width": 8,
                    "height": 6,
                },
                {
                    "class": "racket",
                    "confidence": 0.12,
                    "x": 40,
                    "y": 20,
                    "width": 10,
                    "height": 8,
                },
            ]
        }


def test_serverless_predictions_are_normalized_scaled_and_filtered():
    client = FakeClient()
    model = ServerlessRacketModel(
        client,
        "alvins-workspace-vzjhh",
        "racketdetection-2",
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    predictions = infer_frame_predictions(model, frame, 0.30, max_width=100)
    predictions = filter_predictions(predictions, 0.30)

    assert client.calls[0]["images"]["image"].shape == (50, 100, 3)
    assert client.calls[0]["workspace_name"] == "alvins-workspace-vzjhh"
    assert client.calls[0]["workflow_id"] == "racketdetection-2"
    assert client.calls[0]["use_cache"] is True
    assert predictions == [
        {
            "class": "racket",
            "class_name": "racket",
            "confidence": 0.82,
            "x": 40.0,
            "y": 20.0,
            "width": 16.0,
            "height": 12.0,
        }
    ]


def test_annotation_and_csv_support_multiple_detections_per_frame():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    predictions = [
        {
            "class": "racket",
            "confidence": 0.91,
            "x": 40,
            "y": 30,
            "width": 20,
            "height": 10,
        },
        {
            "class": "racket",
            "confidence": 0.76,
            "x": 85,
            "y": 45,
            "width": 16,
            "height": 22,
        },
    ]

    annotated = draw_predictions(frame, predictions)
    rows = csv_rows_for_frame(60, 30.0, predictions)

    assert np.any(annotated != frame)
    assert len(rows) == 2
    assert rows[0]["source_frame"] == 60
    assert rows[0]["timestamp_seconds"] == "2.000000"
    assert rows[0]["detection_index"] == 0
    assert rows[1]["detection_index"] == 1


def test_csv_writes_explicit_missing_detection_row():
    rows = csv_rows_for_frame(15, 30.0, [])

    assert rows == [
        {
            "source_frame": 15,
            "timestamp_seconds": "0.500000",
            "detected": False,
            "detection_index": "",
            "class_name": "",
            "confidence": "",
            "x_center": "",
            "y_center": "",
            "width": "",
            "height": "",
            "x_min": "",
            "y_min": "",
            "x_max": "",
            "y_max": "",
        }
    ]


def test_main_annotates_a_tiny_video_with_injected_serverless_model(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    csv_path = tmp_path / "detections.csv"
    metadata_path = tmp_path / "detections.metadata.json"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (0, 40, 80):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    model = ServerlessRacketModel(
        FakeClient(),
        "alvins-workspace-vzjhh",
        "racketdetection-2",
    )
    monkeypatch.setattr(
        racket_model_eval,
        "create_serverless_model",
        lambda api_url, api_key, workspace_name, workflow_id: model,
    )
    monkeypatch.setenv("RACKET_ROBOFLOW_API_KEY", "complete-test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "racket_model_eval.py",
            "--video",
            str(input_path),
            "--output-video",
            str(output_path),
            "--csv",
            str(csv_path),
            "--metadata-json",
            str(metadata_path),
            "--confidence",
            "0.30",
            "--inference-width",
            "0",
        ],
    )

    racket_model_eval.main()

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_capture = cv2.VideoCapture(str(output_path))
    output_frame_count = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_capture.release()

    assert len(rows) == 3
    assert all(row["detected"] == "True" for row in rows)
    assert metadata["processed_frames"] == 3
    assert metadata["detection_count"] == 3
    assert output_frame_count == 3
