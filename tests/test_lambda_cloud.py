import json
import subprocess
import sys
from pathlib import Path

import lambda_cloud

# Trimmed real response shape (live API, 2026-07-30).
FIXTURE = {
    "data": {
        "gpu_1x_a6000": {
            "instance_type": {"name": "gpu_1x_a6000", "price_cents_per_hour": 109},
            "regions_with_capacity_available": [],
        },
        "gpu_1x_a10": {
            "instance_type": {"name": "gpu_1x_a10", "price_cents_per_hour": 129},
            "regions_with_capacity_available": [
                {"name": "us-east-1"}, {"name": "us-west-1"},
            ],
        },
        "gpu_1x_a100": {
            "instance_type": {"name": "gpu_1x_a100", "price_cents_per_hour": 199},
            "regions_with_capacity_available": [{"name": "europe-central-1"}],
        },
        "gpu_1x_a100_sxm4": {
            "instance_type": {"name": "gpu_1x_a100_sxm4", "price_cents_per_hour": 199},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        },
        "gpu_1x_h100_pcie": {
            "instance_type": {"name": "gpu_1x_h100_pcie", "price_cents_per_hour": 329},
            "regions_with_capacity_available": [{"name": "us-west-3"}],
        },
        "gpu_8x_a100": {
            "instance_type": {"name": "gpu_8x_a100", "price_cents_per_hour": 1592},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        },
    }
}

PREFS = ["gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100", "gpu_1x_a100_sxm4"]


def test_launch_constants_in_common_sh_match_this_files_prefs():
    """Pin the two constants that actually reach a launch POST.

    Every other test here feeds `pick_instance_type` the PREFS list above, so all
    of them would still pass if `common.sh` named a different GPU or a higher cap
    — the logic is covered, the values that ride it into `POST
    /instance-operations/launch` are not. This is the last unguarded hop between
    the spec and a box that bills.
    """
    common_sh = (
        Path(lambda_cloud.__file__).parent / "scripts" / "lambda" / "common.sh"
    ).read_text(encoding="utf-8")

    assert 'PREFER_TYPES="%s"' % ",".join(PREFS) in common_sh, (
        "scripts/lambda/common.sh no longer launches "
        + " -> ".join(PREFS)
        + " in that exact order. Preference order is authoritative — "
        "pick_instance_type never looks past it and never substitutes — so a "
        "rename or a reorder here silently changes which GPU gets rented. If the "
        "change is deliberate, update the design spec and this list together."
    )
    assert "PRICE_CAP_CENTS=200" in common_sh, (
        "scripts/lambda/common.sh no longer caps launches at $2.00/hr. That cap is "
        "the spec's hard limit and the only thing standing between a typo and a "
        "pricier box (an 8x A100 is $15.92/hr on the same endpoint). Raising it is "
        "the account owner's call, not a code change."
    )


def test_pick_prefers_earlier_entries_with_us_capacity():
    # a6000 has no capacity -> a10 (which does) wins over the a100s.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 200) == (
        "gpu_1x_a10", "us-east-1",
    )


def test_pick_takes_first_preference_when_available():
    payload = json.loads(json.dumps(FIXTURE))
    payload["data"]["gpu_1x_a6000"]["regions_with_capacity_available"] = [
        {"name": "us-west-1"},
    ]
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) == (
        "gpu_1x_a6000", "us-west-1",
    )


def test_pick_ignores_non_us_regions():
    # gpu_1x_a100 has only europe capacity; sxm4 has us-east-1.
    payload = json.loads(json.dumps(FIXTURE))
    payload["data"]["gpu_1x_a10"]["regions_with_capacity_available"] = []
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) == (
        "gpu_1x_a100_sxm4", "us-east-1",
    )


def test_pick_respects_price_cap():
    # Cap below every candidate -> None, even with capacity present.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 100) is None


def test_pick_returns_none_when_nothing_available():
    payload = json.loads(json.dumps(FIXTURE))
    for entry in payload["data"].values():
        entry["regions_with_capacity_available"] = []
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) is None


def test_pick_never_selects_types_outside_preferences():
    # h100 has US capacity and would fit a high cap; it is not in PREFS.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 400) == (
        "gpu_1x_a10", "us-east-1",
    )


def test_availability_rows_sorted_by_price_single_gpu_only():
    rows = lambda_cloud.availability_rows(FIXTURE)
    names = [name for name, _, _ in rows]
    assert names == [
        "gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100", "gpu_1x_a100_sxm4",
        "gpu_1x_h100_pcie",
    ]
    a10 = rows[1]
    assert a10[1] == 1.29
    assert a10[2] == ["us-east-1", "us-west-1"]


def test_cli_pick_type_success(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "pick-type",
         "--prefer", ",".join(PREFS), "--cap-cents", "200"],
        input=json.dumps(FIXTURE), capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "gpu_1x_a10 us-east-1"


def test_cli_pick_type_no_capacity_exit_2():
    payload = json.loads(json.dumps(FIXTURE))
    for entry in payload["data"].values():
        entry["regions_with_capacity_available"] = []
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "pick-type",
         "--prefer", ",".join(PREFS), "--cap-cents", "200"],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "gpu_1x_a6000" in proc.stderr  # availability table printed


def test_pick_prefers_earlier_preference_over_cheaper_later_one():
    # Preference order beats price: a6000 (first, 199) wins over sxm4 (later, 109).
    payload = json.loads(json.dumps(FIXTURE))
    payload["data"]["gpu_1x_a6000"]["instance_type"]["price_cents_per_hour"] = 199
    payload["data"]["gpu_1x_a6000"]["regions_with_capacity_available"] = [
        {"name": "us-west-1"},
    ]
    payload["data"]["gpu_1x_a100_sxm4"]["instance_type"]["price_cents_per_hour"] = 109
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) == (
        "gpu_1x_a6000", "us-west-1",
    )


def test_pick_returns_none_when_only_unlisted_types_have_capacity():
    # h100 is the only single-GPU type left with capacity, and fits the 400 cap.
    payload = json.loads(json.dumps(FIXTURE))
    for name in PREFS:
        payload["data"][name]["regions_with_capacity_available"] = []
    assert payload["data"]["gpu_1x_h100_pcie"]["regions_with_capacity_available"] == [
        {"name": "us-west-3"},
    ]
    assert lambda_cloud.pick_instance_type(payload, PREFS, 400) is None


def _make_run(ui_runs, run_id, video_name, with_calibration=True):
    run_dir = ui_runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(
        json.dumps({"video_path": f"/anywhere/by-hash/{video_name}"}),
        encoding="utf-8",
    )
    if with_calibration:
        (run_dir / "calibration.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_find_calibration_newest_matching_run_wins(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    digest = lambda_cloud.file_sha256(clip)
    ui_runs = tmp_path / "ui_runs"
    _make_run(ui_runs, "1753900000000", f"{digest}.mp4")
    newest = _make_run(ui_runs, "1753990000000", f"{digest}.mp4")
    _make_run(ui_runs, "1753995000000", "0123deadbeef.mp4")  # other video, newer
    found = lambda_cloud.find_default_calibration(clip, ui_runs)
    assert found == newest / "calibration.json"


def test_find_calibration_skips_runs_without_calibration(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    digest = lambda_cloud.file_sha256(clip)
    ui_runs = tmp_path / "ui_runs"
    older = _make_run(ui_runs, "1753900000000", f"{digest}.mp4")
    _make_run(ui_runs, "1753990000000", f"{digest}.mp4", with_calibration=False)
    assert lambda_cloud.find_default_calibration(clip, ui_runs) == older / "calibration.json"


def test_find_calibration_none_when_no_match(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    ui_runs = tmp_path / "ui_runs"
    _make_run(ui_runs, "1753900000000", "0123deadbeef.mp4")
    assert lambda_cloud.find_default_calibration(clip, ui_runs) is None


def test_cli_find_calibration_exit_2_when_missing(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "find-calibration",
         str(clip), "--ui-runs", str(tmp_path / "ui_runs")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "--calibration" in proc.stderr
