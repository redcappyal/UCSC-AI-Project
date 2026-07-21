"""Row-shaped synthetic data for engine-level 3D tests."""
import numpy as np

from ballistic import GRAVITY_VEC


def make_bounce_rows(camera, fps=60.0):
    """CSV-row dicts for a drive that bounces off the front wall.
    Returns (rows, bounce_frame)."""
    position = np.array([10.0, 24.0, 4.0])
    velocity = np.array([2.0, -70.0, 6.0])
    dt = 1.0 / fps
    rows, bounce_frame = [], None
    for frame in range(40):
        u, v = camera.project(position)
        rows.append({
            "source_frame": frame, "timestamp_seconds": frame * dt,
            "detected": "true", "x_center": u, "y_center": v,
            "width": 8.0, "height": 8.0,
        })
        velocity_next = velocity + GRAVITY_VEC * dt
        position_next = position + velocity * dt + 0.5 * GRAVITY_VEC * dt * dt
        if position_next[1] <= 0.0 and bounce_frame is None:
            bounce_frame = frame + 1
            position_next[1] = -position_next[1]
            velocity_next[1] = -0.7 * velocity_next[1]
        position, velocity = position_next, velocity_next
    return rows, bounce_frame
