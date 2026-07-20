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
