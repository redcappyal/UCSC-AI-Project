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
