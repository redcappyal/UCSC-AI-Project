import numpy as np
import pytest

import court_model
from court_model import CameraModel
from synthetic3d import make_camera


def test_project_ray_roundtrip():
    camera = make_camera()
    point = np.array([6.0, 12.0, 3.0])
    u, v = camera.project(point)
    origin, direction = camera.ray((u, v))
    # The ray from the camera through that pixel must pass through the point.
    to_point = point - origin
    to_point /= np.linalg.norm(to_point)
    assert np.allclose(direction, to_point, atol=1e-9)


def test_project_center_pixel():
    camera = make_camera(center=(960.0, 540.0), look_at=(10.5, 0.0, 7.0))
    # A point straight along the optical axis lands on the principal point.
    u, v = camera.project((10.5, 0.0, 7.0))
    assert u == pytest.approx(960.0, abs=1e-6)
    assert v == pytest.approx(540.0, abs=1e-6)


def test_project_behind_camera_raises():
    camera = make_camera(position=(10.5, 30.0, 7.0))
    with pytest.raises(ValueError):
        camera.project((10.5, 33.0, 7.0))  # behind the back-wall camera


def test_serialization_roundtrip():
    camera = make_camera()
    restored = CameraModel.from_dict(camera.to_dict())
    assert np.allclose(restored.rotation, camera.rotation)
    assert np.allclose(restored.camera_center_ft, camera.camera_center_ft)
    assert restored.focal_px == camera.focal_px
    u1, v1 = camera.project((5.0, 10.0, 2.0))
    u2, v2 = restored.project((5.0, 10.0, 2.0))
    assert (u1, v1) == pytest.approx((u2, v2))


def test_projection_matrix_agrees_with_project():
    camera = make_camera()
    point = np.array([3.0, 20.0, 1.0, 1.0])
    projected = camera.projection_matrix() @ point
    u, v = projected[0] / projected[2], projected[1] / projected[2]
    assert (u, v) == pytest.approx(camera.project(point[:3]))


def _synthetic_calibration(camera, frame_size=(1920, 1080)):
    """Project the real landmark/line geometry through a known camera."""
    landmarks = []
    for mark in court_model.FLOOR_LANDMARKS:
        if mark["optional"]:
            continue
        x_ft, y_ft = mark["court_ft"]
        u, v = camera.project((x_ft, y_ft, 0.0))
        landmarks.append({"id": mark["id"], "court_ft": [x_ft, y_ft],
                          "refined_px": [u, v]})
    lines = []
    for name, height in (("out_line_lower_edge", court_model.OUT_LINE_HEIGHT_FT),
                         ("tin_top_edge", court_model.TIN_TOP_HEIGHT_FT)):
        left = camera.project((0.0, 0.0, height))
        right = camera.project((court_model.COURT_WIDTH_FT, 0.0, height))
        lines.append({"name": name, "endpoints": [list(left), list(right)]})
    return {
        "schema": "squash-calibration-v2",
        "frame_width": frame_size[0], "frame_height": frame_size[1],
        "lines": lines,
        "planes": {"floor": {"landmarks": landmarks}},
    }


def test_camera_correspondences_extracts_floor_and_wall():
    camera = make_camera()
    calibration = _synthetic_calibration(camera)
    image_px, court_xyz = court_model._camera_correspondences(calibration)
    assert len(image_px) == len(court_xyz) == 7 + 4  # 7 required landmarks + 4 line ends
    heights = sorted(set(round(z, 4) for z in court_xyz[:, 2]))
    assert heights == [0.0, round(court_model.TIN_TOP_HEIGHT_FT, 4),
                       court_model.OUT_LINE_HEIGHT_FT]


def test_init_camera_from_floor_recovers_pose():
    camera = make_camera(focal_px=1500.0, position=(10.5, 29.0, 7.0),
                         look_at=(10.5, 0.0, 4.0))
    calibration = _synthetic_calibration(camera)
    result = court_model._init_camera_from_floor(calibration, (1920, 1080))
    assert result is not None
    focal, rotation, center = result
    assert focal == pytest.approx(1500.0, rel=0.05)
    assert np.allclose(center, camera.camera_center_ft, atol=0.75)
    assert np.allclose(rotation, camera.rotation, atol=0.05)
