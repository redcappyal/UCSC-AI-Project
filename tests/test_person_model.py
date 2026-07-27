"""person_model: RF-DETR adapter, backend gating, crop helper.

No rfdetr import may happen at test time — the adapter math and gating are
exercised pure, and detection loading is stubbed."""

import numpy as np
import pytest

import person_model
from person_model import (
    PERSON_CONFIDENCE_THRESHOLD,
    PersonDetection,
    keypoints_result_to_detections,
)


def test_adapter_converts_xyxy_to_center_boxes_and_keypoints():
    xyxy = np.array([[100.0, 200.0, 180.0, 420.0]])
    det_conf = np.array([0.9])
    kp_xy = np.zeros((1, 17, 2)) + 5.0
    kp_conf = np.ones((1, 17)) * 0.7
    detections = keypoints_result_to_detections(xyxy, det_conf, kp_xy, kp_conf)
    assert len(detections) == 1
    det = detections[0]
    assert det.x == pytest.approx(140.0)
    assert det.y == pytest.approx(310.0)
    assert det.width == pytest.approx(80.0)
    assert det.height == pytest.approx(220.0)
    assert det.confidence == pytest.approx(0.9)
    assert len(det.keypoints) == 17
    assert det.keypoints[0] == (5.0, 5.0, 0.7)
    assert det.foot_px == (pytest.approx(140.0), pytest.approx(420.0))


def test_adapter_filters_below_threshold_and_handles_missing_keypoints():
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 60.0]])
    det_conf = np.array([PERSON_CONFIDENCE_THRESHOLD - 0.1,
                         PERSON_CONFIDENCE_THRESHOLD + 0.1])
    detections = keypoints_result_to_detections(xyxy, det_conf, None, None)
    assert len(detections) == 1
    assert detections[0].keypoints == ()


def test_available_backend_is_none_when_rfdetr_missing(monkeypatch):
    monkeypatch.setattr(person_model, "_import_rfdetr", lambda: None)
    assert person_model.available_backend() == "none"
    assert person_model.load_person_detector() is None


def test_available_backend_env_kill_switch(monkeypatch):
    monkeypatch.setenv("PERSON_DETECTOR", "none")
    assert person_model.available_backend() == "none"


def test_save_person_crop(tmp_path):
    import cv2
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), i * 10, dtype=np.uint8)
        frame[100:200, 140:180] = 255
        writer.write(frame)
    writer.release()

    det = PersonDetection(x=160.0, y=150.0, width=40.0, height=100.0,
                          confidence=0.9, keypoints=())
    out_path = tmp_path / "crop.jpg"
    assert person_model.save_person_crop(video, 5, det, out_path) is True
    crop = cv2.imread(str(out_path))
    assert crop is not None
    assert crop.shape[0] > 100  # padded beyond the raw bbox height
