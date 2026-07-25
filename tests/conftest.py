"""Fixtures shared across test modules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A RUNS_DIR holding only the runs the calling test creates.

    Tests that exercise run-backed endpoints used to build their runs inside
    the real RUNS_DIR under fixed ids and clean up with
    `shutil.rmtree(..., ignore_errors=True)`. Two things went wrong with that.
    On Windows the rmtree fails whenever a handle is still open and
    `ignore_errors` swallows the failure, so debris survived into later runs —
    that is what made test_corrections.py alternate pass and fail. And any
    assertion about the *newest* calibration, or about there being none, was
    really an assertion about what every other test had left behind: the
    /api/calibration/latest 404 case had to ask whether a stray calibration
    existed and settle for "the endpoint answered" when one did.

    Redirecting RUNS_DIR at `tmp_path` removes the shared state instead of
    depending on cleanup having worked, which is also what lets those tests
    assert outright.
    """
    import app as app_module

    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path)
    return tmp_path
