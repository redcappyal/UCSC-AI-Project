"""Geometry and schema tests for the floor-plane homography (court_model)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import court_model
from court_model import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    FLOOR_LANDMARKS,
    apply_homography,
    build_floor_zone_summary,
    distort_point,
    fit_homography,
    floor_zone_for_point,
    invert_homography,
    load_floor_calibration,
    undistort_point,
)
from judge_call import load_calibration_lines


V1_CALIBRATION = {
    "lines": [
        {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
        {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
    ]
}


class SyntheticCamera:
    """Pinhole camera on the back wall: centered (x=10.5 ft), 7 ft high,
    pitched down, wide lens — the recommended mount for this app."""

    def __init__(self, pitch_deg=20.0, focal_px=1600.0, center=(1920.0, 1080.0)):
        self.focal = focal_px
        self.cx, self.cy = center
        theta = np.radians(pitch_deg)
        # Right-handed frame: X = court x, Yp = distance from BACK wall
        # (camera looks along +Yp), Z = up. Court y = 32 - Yp.
        self.position = np.array([COURT_WIDTH_FT / 2.0, 0.0, 7.0])
        x_cam = np.array([1.0, 0.0, 0.0])
        z_cam = np.array([0.0, np.cos(theta), -np.sin(theta)])
        y_cam = np.cross(z_cam, x_cam)
        self.rotation = np.vstack([x_cam, y_cam, z_cam])

    def project_court_point(self, x_ft, y_ft, z_ft=0.0):
        world = np.array([x_ft, COURT_LENGTH_FT - y_ft, z_ft])
        cam = self.rotation @ (world - self.position)
        assert cam[2] > 0, "point must be in front of the camera"
        return (
            self.focal * cam[0] / cam[2] + self.cx,
            self.focal * cam[1] / cam[2] + self.cy,
        )


def landmark_correspondences(camera):
    court = [mark["court_ft"] for mark in FLOOR_LANDMARKS]
    image = [camera.project_court_point(x, y) for x, y in court]
    return image, court


def make_v2_calibration(camera=None, **overrides):
    camera = camera or SyntheticCamera()
    image, court = landmark_correspondences(camera)
    homography, _ = fit_homography(court, image)
    floor_plane = {
        "landmarks": [
            {
                "id": mark["id"],
                "court_ft": mark["court_ft"],
                "tap_px": list(pixel),
                "refined_px": list(pixel),
                "method": "line_intersection",
                "skipped": False,
            }
            for mark, pixel in zip(FLOOR_LANDMARKS, image)
        ],
        "homography_image_from_court": homography.tolist(),
        "fit_rms_px": 0.0,
    }
    floor_plane.update(overrides)
    return {
        "schema": "squash-calibration-v2",
        "frame_width": 3840,
        "frame_height": 2160,
        "lines": V1_CALIBRATION["lines"],
        "court_units": "feet",
        "planes": {"floor": floor_plane},
        "distortion": None,
    }


def test_synthetic_camera_round_trip():
    camera = SyntheticCamera()
    image, court = landmark_correspondences(camera)
    homography, residuals = fit_homography(court, image)
    assert residuals.max() < 1e-6

    inverse = invert_homography(homography)
    for x_ft, y_ft in [(3.0, 5.0), (10.5, 18.0), (18.5, 30.0), (0.5, 0.5)]:
        pixel = camera.project_court_point(x_ft, y_ft)
        mapped = apply_homography(inverse, pixel)
        assert abs(mapped[0] - x_ft) < 1e-6
        assert abs(mapped[1] - y_ft) < 1e-6


def test_noise_robustness():
    camera = SyntheticCamera()
    image, court = landmark_correspondences(camera)
    rng = np.random.default_rng(7)
    noisy = np.asarray(image) + rng.normal(0.0, 1.0, size=(len(image), 2))
    homography, residuals = fit_homography(court, noisy.tolist())
    rms = float(np.sqrt(np.mean(residuals**2)))
    assert 0.0 < rms < 5.0

    inverse = invert_homography(homography)
    for x_ft, y_ft in [(10.5, 18.0), (10.5, 25.0), (5.0, 20.0)]:
        pixel = camera.project_court_point(x_ft, y_ft)
        mapped = apply_homography(inverse, pixel)
        assert abs(mapped[0] - x_ft) < 0.2
        assert abs(mapped[1] - y_ft) < 0.2


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])

    collinear = [[0, 0], [1, 1], [2, 2], [3, 3]]
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(ValueError):
        fit_homography(collinear, square)
    with pytest.raises(ValueError):
        fit_homography(square, collinear)


def test_distortion_round_trip():
    for k1 in (0.1, -0.1, 0.02):
        distortion = {
            "model": "division_k1",
            "k1": k1,
            "center_px": [1920.0, 1080.0],
            "norm_px": 1920.0,
        }
        for point in [(100.0, 50.0), (1920.0, 1080.0), (3500.0, 2000.0)]:
            distorted = distort_point(point, distortion)
            recovered = undistort_point(distorted, distortion)
            assert abs(recovered[0] - point[0]) < 1e-6
            assert abs(recovered[1] - point[1]) < 1e-6

    assert undistort_point((12.5, 8.25), None) == (12.5, 8.25)
    assert distort_point((12.5, 8.25), None) == (12.5, 8.25)


def test_distorted_calibration_round_trip():
    """FloorMap must undo lens distortion before applying the homography."""
    camera = SyntheticCamera()
    distortion = {
        "model": "division_k1",
        "k1": -0.08,
        "center_px": [1920.0, 1080.0],
        "norm_px": 1920.0,
    }
    calibration = make_v2_calibration(camera)
    calibration["distortion"] = distortion
    # Simulate what the real lens does: stored pixels are distorted observations.
    for landmark in calibration["planes"]["floor"]["landmarks"]:
        landmark["refined_px"] = list(distort_point(landmark["refined_px"], distortion))
    floor_map = load_floor_calibration(calibration)
    assert floor_map is not None and floor_map.source == "refit"

    pixel = distort_point(camera.project_court_point(6.0, 22.0), distortion)
    x_ft, y_ft = floor_map.image_to_court(*pixel)
    assert abs(x_ft - 6.0) < 1e-6
    assert abs(y_ft - 22.0) < 1e-6


def test_load_floor_calibration_v1_returns_none():
    assert load_floor_calibration(V1_CALIBRATION) is None
    assert load_floor_calibration({}) is None
    assert load_floor_calibration(None) is None
    assert load_floor_calibration({"planes": {}}) is None


def test_load_floor_calibration_v2_refit_and_back_compat():
    calibration = make_v2_calibration()
    floor_map = load_floor_calibration(calibration)
    assert floor_map is not None
    assert floor_map.source == "refit"
    assert floor_map.landmark_count == len(FLOOR_LANDMARKS)
    assert floor_map.fit_rms_px < 1e-6

    camera = SyntheticCamera()
    pixel = camera.project_court_point(10.5, 18.0)
    x_ft, y_ft = floor_map.image_to_court(*pixel)
    assert abs(x_ft - 10.5) < 1e-6
    assert abs(y_ft - 18.0) < 1e-6

    # v1 front-wall parsing must still work on the same dict.
    top_line, bottom_line = load_calibration_lines(calibration)
    assert top_line.left.y == 100
    assert bottom_line.left.y == 700


def test_load_floor_calibration_falls_back_to_stored_matrix():
    calibration = make_v2_calibration()
    # Keep only 3 landmarks (too few to refit) but leave the stored matrix.
    calibration["planes"]["floor"]["landmarks"] = calibration["planes"]["floor"][
        "landmarks"
    ][:3]
    floor_map = load_floor_calibration(calibration)
    assert floor_map is not None
    assert floor_map.source == "stored_matrix"

    camera = SyntheticCamera()
    pixel = camera.project_court_point(4.0, 12.0)
    x_ft, y_ft = floor_map.image_to_court(*pixel)
    assert abs(x_ft - 4.0) < 1e-6
    assert abs(y_ft - 12.0) < 1e-6


def test_load_floor_calibration_corrupt_returns_none():
    calibration = make_v2_calibration()
    calibration["planes"]["floor"]["landmarks"] = calibration["planes"]["floor"][
        "landmarks"
    ][:3]
    calibration["planes"]["floor"]["homography_image_from_court"] = [[1, 2], [3, 4]]
    assert load_floor_calibration(calibration) is None

    calibration = make_v2_calibration()
    calibration["planes"]["floor"]["landmarks"] = []
    calibration["planes"]["floor"]["homography_image_from_court"] = None
    assert load_floor_calibration(calibration) is None

    calibration = make_v2_calibration()
    calibration["distortion"] = {"model": "mystery", "k1": 1.0, "center_px": [0, 0]}
    assert load_floor_calibration(calibration) is None


def test_skipped_landmarks_are_ignored():
    calibration = make_v2_calibration()
    landmarks = calibration["planes"]["floor"]["landmarks"]
    landmarks[0]["skipped"] = True
    landmarks[0]["refined_px"] = [9999.0, 9999.0]  # garbage that must not be used
    floor_map = load_floor_calibration(calibration)
    assert floor_map is not None
    assert floor_map.source == "refit"
    assert floor_map.landmark_count == len(FLOOR_LANDMARKS) - 1
    assert floor_map.fit_rms_px < 1e-6


def test_floor_zone_for_point():
    t_zone = floor_zone_for_point(10.5, 18.0)
    assert t_zone["behind_short_line"] is True
    assert t_zone["side"] == "right"
    assert t_zone["in_left_service_box"] is False

    left_box = floor_zone_for_point(2.0, 20.0)
    assert left_box["in_left_service_box"] is True
    assert left_box["in_right_service_box"] is False
    assert left_box["side"] == "left"

    right_box = floor_zone_for_point(19.0, 23.0)
    assert right_box["in_right_service_box"] is True

    front = floor_zone_for_point(10.0, 2.0)
    assert front["row"] == 0
    assert front["behind_short_line"] is False

    clamped = floor_zone_for_point(-5.0, 99.0)
    assert clamped["x_ft"] == 0.0
    assert clamped["y_ft"] == COURT_LENGTH_FT
    assert 1 <= clamped["zone"] <= court_model.FLOOR_ZONE_ROWS * court_model.FLOOR_ZONE_COLUMNS


def test_build_floor_zone_summary():
    hits = [
        {"event_type": "floor", "floor_zone": floor_zone_for_point(2.0, 30.0)},
        {"event_type": "floor", "floor_zone": floor_zone_for_point(2.5, 31.0)},
        {"event_type": "floor", "floor_zone": floor_zone_for_point(19.0, 4.0)},
        {"event_type": "wall", "target_zone": {"zone": 1}},  # ignored
        {"event_type": "floor"},  # no mapping -> ignored
    ]
    summary = build_floor_zone_summary(hits)
    assert summary["total_floor_bounces"] == 3
    assert summary["rows"] == court_model.FLOOR_ZONE_ROWS
    assert summary["common_zones"][0]["count"] == 2
    counted = sum(zone["count"] for zone in summary["zones"])
    assert counted == 3
    assert all(zone["count"] == 0 for zone in summary["missing_zones"])


def test_judge_hits_maps_floor_bounces_to_court(tmp_path):
    import json

    from job_runner import judge_hits

    camera = SyntheticCamera()
    calibration = make_v2_calibration(camera)
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))

    bounce_pixel = camera.project_court_point(4.0, 28.0)  # back-left drive
    wall_pixel = (900.0, 180.0)
    results = {
        60: {
            "source_frame": 60,
            "timestamp_seconds": "2.000000",
            "detected": "True",
            "x_center": f"{wall_pixel[0]:.3f}",
            "y_center": f"{wall_pixel[1]:.3f}",
        },
        90: {
            "source_frame": 90,
            "timestamp_seconds": "3.000000",
            "detected": "True",
            "x_center": f"{bounce_pixel[0]:.3f}",
            "y_center": f"{bounce_pixel[1]:.3f}",
        },
    }
    hits = [
        {
            "hit_frame": 60,
            "timestamp_seconds": 2.0,
            "dv_magnitude": 400.0,
            "after_gap": False,
            "event_type": "wall",
        },
        {
            "hit_frame": 90,
            "timestamp_seconds": 3.0,
            "dv_magnitude": 300.0,
            "after_gap": False,
            "event_type": "floor",
            "impact_x": bounce_pixel[0],
            "impact_y": bounce_pixel[1],
            "impact_frame": 90,
        },
    ]

    judged = judge_hits(tmp_path, results, hits)

    wall_entry, floor_entry = judged
    # Front-wall judging is untouched by the floor plane.
    assert wall_entry["call"] == "IN"
    assert "court_position_ft" not in wall_entry
    # Floor bounce gains metric court coordinates + zone.
    assert floor_entry["call"] is None
    assert abs(floor_entry["court_position_ft"]["x"] - 4.0) < 0.01
    assert abs(floor_entry["court_position_ft"]["y"] - 28.0) < 0.01
    assert floor_entry["floor_zone"]["side"] == "left"
    assert floor_entry["floor_zone"]["behind_short_line"] is True

    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    assert payload["floor_zones"]["total_floor_bounces"] == 1
    assert payload["target_zones"]["total_wall_hits"] == 1


def test_judge_hits_v1_calibration_has_no_floor_fields(tmp_path):
    import json

    from job_runner import judge_hits

    (tmp_path / "calibration.json").write_text(json.dumps(V1_CALIBRATION))
    results = {
        90: {
            "source_frame": 90,
            "timestamp_seconds": "3.000000",
            "detected": "True",
            "x_center": "1000.000",
            "y_center": "1500.000",
        }
    }
    hit = {
        "hit_frame": 90,
        "timestamp_seconds": 3.0,
        "dv_magnitude": 300.0,
        "after_gap": False,
        "event_type": "floor",
    }
    judged = judge_hits(tmp_path, results, [hit])
    assert "court_position_ft" not in judged[0]
    assert "floor_zone" not in judged[0]
    payload = json.loads((tmp_path / "detected_hits.json").read_text())
    assert "floor_zones" not in payload


def test_court_model_endpoint_and_floor_validation():
    import app as app_module

    client = app_module.app.test_client()
    model = client.get("/api/court-model").get_json()
    assert model["ok"] is True
    assert model["width_ft"] == 21.0
    assert model["landmarks"][0]["id"] == "short_line_left"

    # Valid floor plane passes validation untouched.
    good = make_v2_calibration()
    assert app_module.validate_floor_calibration(good) is None
    assert "floor" in good["planes"]

    # A corrupt floor plane is stripped with a warning, not an error.
    bad = make_v2_calibration()
    bad["planes"]["floor"]["landmarks"] = []
    bad["planes"]["floor"]["homography_image_from_court"] = None
    warning = app_module.validate_floor_calibration(bad)
    assert warning is not None
    assert "floor" not in bad["planes"]

    # v1 calibrations produce no warning.
    assert app_module.validate_floor_calibration(dict(V1_CALIBRATION)) is None


def test_court_model_public_is_json_safe():
    import json

    payload = court_model.court_model_public()
    encoded = json.loads(json.dumps(payload))
    assert encoded["width_ft"] == 21.0
    assert len(encoded["landmarks"]) == len(FLOOR_LANDMARKS)
    assert encoded["landmarks"][0]["id"] == "short_line_left"
    assert len(encoded["wireframe"]) == len(court_model.FLOOR_WIREFRAME)
