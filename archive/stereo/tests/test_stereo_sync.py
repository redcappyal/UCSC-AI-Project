# ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
# Not imported by any runtime module, not collected by pytest, not linted.
# Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
"""Phase 5: the CSV adapter and the offline clock-offset refiner.

The refiner is the only genuinely new algorithm in Phase 5, and it is fully
testable without footage: project a known 3D trajectory through a known camera
pair, shift one stream by a known offset, and demand the offset back.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import stereo_sync
from stereo_engine import TrackSample
from synthetic3d import make_camera
from test_stereo_track import make_fin_pair, sample_camera, simulate_front_wall_shot


def shifted(samples, offset_s):
    """Re-stamp a stream as a peer clock running `offset_s` fast would."""
    return [TrackSample(t_s=s.t_s - offset_s, px=s.px) for s in samples]


def write_csv(path, samples, fps, undetected_frames=()):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_frame", "timestamp_seconds", "detected",
                        "x_center", "y_center"])
        writer.writeheader()
        for index, sample in enumerate(samples):
            detected = index not in undetected_frames
            writer.writerow({
                "source_frame": int(round(sample.t_s * fps)),
                "timestamp_seconds": f"{sample.t_s:.6f}",
                "detected": "true" if detected else "false",
                "x_center": f"{sample.px[0]:.4f}" if detected else "",
                "y_center": f"{sample.px[1]:.4f}" if detected else "",
            })


# --- Task 3: the CSV adapter ------------------------------------------------

def test_csv_round_trips_to_track_samples(tmp_path):
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    path = tmp_path / "ball_coordinates.csv"
    write_csv(path, samples, fps=60.0)

    loaded = stereo_sync.track_samples_from_csv(path, fps=60.0)
    assert len(loaded) == len(samples)
    assert loaded[0].t_s == float(f"{samples[0].t_s:.6f}")
    assert loaded[0].px[0] == float(f"{samples[0].px[0]:.4f}")


def test_undetected_rows_are_skipped(tmp_path):
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    path = tmp_path / "ball_coordinates.csv"
    write_csv(path, samples, fps=60.0, undetected_frames={0, 3, 7})

    loaded = stereo_sync.track_samples_from_csv(path, fps=60.0)
    assert len(loaded) == len(samples) - 3


def test_offset_is_added_to_every_timestamp(tmp_path):
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    path = tmp_path / "ball_coordinates.csv"
    write_csv(path, samples, fps=60.0)

    plain = stereo_sync.track_samples_from_csv(path, fps=60.0)
    shifted_load = stereo_sync.track_samples_from_csv(path, fps=60.0, offset_s=0.25)
    for a, b in zip(plain, shifted_load):
        assert b.t_s == a.t_s + 0.25


def test_samples_come_back_time_sorted(tmp_path):
    """The CSV is frame-ordered today; nothing guarantees it stays that way."""
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    path = tmp_path / "ball_coordinates.csv"
    write_csv(path, list(reversed(samples)), fps=60.0)

    loaded = stereo_sync.track_samples_from_csv(path, fps=60.0)
    times = [s.t_s for s in loaded]
    assert times == sorted(times)


def test_all_undetected_yields_empty_not_an_error(tmp_path):
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)[:5]
    path = tmp_path / "ball_coordinates.csv"
    write_csv(path, samples, fps=60.0, undetected_frames=set(range(5)))

    assert stereo_sync.track_samples_from_csv(path, fps=60.0) == []


def test_missing_timestamp_falls_back_to_frame_and_fps(tmp_path):
    """Older CSVs predate timestamp_seconds; frame/fps still locates them."""
    path = tmp_path / "ball_coordinates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_frame", "detected", "x_center", "y_center"])
        writer.writeheader()
        for frame in range(4):
            writer.writerow({"source_frame": frame * 30, "detected": "true",
                             "x_center": "100", "y_center": "200"})

    loaded = stereo_sync.track_samples_from_csv(path, fps=60.0)
    assert [s.t_s for s in loaded] == [0.0, 0.5, 1.0, 1.5]


# --- Task 4: the offset refiner ---------------------------------------------

def paired_streams(offset_s, fps=60.0, noise_px=0.0, seed=0):
    """Two eyes on one ball, with camera b's clock running `offset_s` fast."""
    left, right = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=fps)
    samples_b = sample_camera(states, right, fps=fps, phase_s=1.0 / (2 * fps))
    if noise_px:
        rng = np.random.default_rng(seed)
        samples_a = [TrackSample(t_s=s.t_s, px=tuple(np.array(s.px) + rng.normal(0, noise_px, 2)))
                     for s in samples_a]
        samples_b = [TrackSample(t_s=s.t_s, px=tuple(np.array(s.px) + rng.normal(0, noise_px, 2)))
                     for s in samples_b]
    return left, samples_a, right, shifted(samples_b, offset_s)


def test_recovers_a_known_offset():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.012)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)

    assert report["refined"]["accepted"] is True
    assert abs(report["refined"]["offset_s"] - 0.012) < 0.001
    assert report["cost"]["at_refined"] < report["cost"]["at_seed"]
    assert report["search"]["argmin_interior"] is True


def test_recovers_a_negative_offset():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=-0.019)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
    assert abs(report["refined"]["offset_s"] + 0.019) < 0.001


def test_a_good_seed_is_polished_not_discarded():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.012)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b,
                                       seed_s=0.011, coarse_range_s=0.005)
    assert abs(report["refined"]["offset_s"] - 0.012) < 0.001
    assert report["seed"]["offset_s"] == 0.011


def test_no_bias_at_zero_offset():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.0)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
    assert abs(report["refined"]["offset_s"]) < 0.001


def test_is_deterministic():
    """No RNG, no dict ordering, no wall clock — same input, same bytes."""
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.008)
    first = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
    second = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
    assert first == second


def test_degrades_monotonically_with_pixel_noise():
    """Measured: 0.00000 / 0.00010 / 0.00030 s of offset error at 0/1/3 px.

    The spec expects offline refinement to beat the live 2 ms budget; it does
    so by a wide margin, and stays under 1 ms even at an absurd 12 px.
    """
    errors, costs = [], []
    for noise in (0.0, 1.0, 3.0, 12.0):
        model_a, samples_a, model_b, samples_b = paired_streams(
            offset_s=0.012, noise_px=noise, seed=7)
        report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
        assert report["refined"]["accepted"] is True, f"rejected a real pair at {noise} px"
        errors.append(abs(report["refined"]["offset_s"] - 0.012))
        costs.append(report["cost"]["at_refined"])
    assert errors == sorted(errors), f"noise should not help: {errors}"
    assert costs == sorted(costs), f"cost should rise with noise: {costs}"
    assert max(errors) < 0.002, "must stay inside the live 2 ms budget"
    # Every real case sits far below the disagreement gate.
    assert max(costs) < stereo_sync.MAX_AGREEING_GAP_FT / 5


def test_true_offset_outside_the_search_range_is_flagged():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.040)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b,
                                       seed_s=0.0, coarse_range_s=0.010)
    # The grid cannot reach 0.040, so the minimum sits on the edge. Saying so
    # is the point — a confident wrong offset is worse than an honest miss.
    assert report["search"]["argmin_interior"] is False
    assert report["refined"]["accepted"] is False
    assert report["refined"]["offset_s"] == report["seed"]["offset_s"]


def test_unrelated_streams_are_rejected():
    model_a, samples_a, model_b, _ = paired_streams(offset_s=0.0)
    rng = np.random.default_rng(3)
    noise = [TrackSample(t_s=s.t_s, px=tuple(rng.uniform(0, 1900, 2))) for s in samples_a]
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, noise, seed_s=0.0)
    assert report["refined"]["accepted"] is False
    assert report["refined"]["reason"]


def test_empty_overlap_is_rejected_not_raised():
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.0)
    far = [TrackSample(t_s=s.t_s + 1000.0, px=s.px) for s in samples_b]
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, far, seed_s=0.0)
    assert report["refined"]["accepted"] is False
    assert report["timeline"]["count"] == 0


def test_no_samples_at_all_is_rejected_not_raised():
    model_a, samples_a, model_b, _ = paired_streams(offset_s=0.0)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, [], seed_s=0.0)
    assert report["refined"]["accepted"] is False


def test_report_is_json_serialisable():
    """It is written verbatim into sync_report.json."""
    import json
    model_a, samples_a, model_b, samples_b = paired_streams(offset_s=0.012)
    report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
    assert json.loads(json.dumps(report)) == report


def test_half_width_widens_with_noise():
    """The uncertainty proxy has to actually track uncertainty."""
    widths = []
    for noise in (0.0, 3.0):
        model_a, samples_a, model_b, samples_b = paired_streams(
            offset_s=0.012, noise_px=noise, seed=11)
        report = stereo_sync.refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0)
        widths.append(report["half_width_s"])
    assert widths[1] >= widths[0]


def test_seed_from_manifest_prefers_the_clap_anchor():
    manifest = {"clap_anchor_s": 0.0071, "offset_series": [0.004, 0.009, 0.005]}
    assert stereo_sync.seed_offset_from_manifest(manifest) == (0.0071, "clap_anchor")


def test_seed_falls_back_to_the_offset_series_median():
    manifest = {"offset_series": [0.004, 0.009, 0.005]}
    offset, source = stereo_sync.seed_offset_from_manifest(manifest)
    assert offset == 0.005 and source == "offset_series_median"


def test_seed_tolerates_a_junk_manifest():
    for manifest in (None, {}, {"clap_anchor_s": "soon"}, {"offset_series": []},
                     {"offset_series": ["x"]}, {"unrecognised": 1}):
        assert stereo_sync.seed_offset_from_manifest(manifest) == (0.0, "none")
