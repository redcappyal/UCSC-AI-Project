"""Synthetic camera + trajectory generators shared by the 3D test files."""
import numpy as np

from court_model import CameraModel


def make_camera(focal_px=1600.0, center=(960.0, 540.0),
                position=(10.5, 30.0, 7.0), look_at=(10.5, 0.0, 5.0),
                frame_size=(1920, 1080)):
    """Back-wall-mounted synthetic camera looking toward the front wall.

    `frame_size` is the pixel space `focal_px`/`center` are expressed in --
    the same thing a real solve records, so synthetic models can be scaled
    (court_model.scale_camera_model) like solved ones.
    """
    position = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # image y grows downward
    rotation = np.vstack([right, down, forward])  # world -> camera rows
    return CameraModel(
        focal_px=float(focal_px),
        center_px=(float(center[0]), float(center[1])),
        rotation=rotation,
        camera_center_ft=position,
        distortion=None,
        fit_rms_px=0.0,
        point_count=0,
        frame_width=float(frame_size[0]),
        frame_height=float(frame_size[1]),
    )
