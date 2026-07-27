"""Cancelling an abandoned tracking run.

Only one job holds TRACKING_JOB_SEMAPHORE, so a run the user backed out of has
to actually stop — otherwise it keeps the semaphore and blocks every later
track until it finishes on its own.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import job_runner
from job_runner import (
    JobCancelled,
    cancel_requested,
    clear_cancel,
    raise_if_cancelled,
    request_cancel,
)


@pytest.fixture(autouse=True)
def clean_cancel_registry():
    job_runner.JOB_CANCELS.clear()
    yield
    job_runner.JOB_CANCELS.clear()


def test_a_run_is_not_cancelled_by_default():
    assert cancel_requested("run-1") is False
    raise_if_cancelled("run-1")          # must not raise


def test_request_then_observe():
    request_cancel("run-1")

    assert cancel_requested("run-1") is True
    with pytest.raises(JobCancelled):
        raise_if_cancelled("run-1")


def test_cancel_is_scoped_to_one_run():
    request_cancel("run-1")

    assert cancel_requested("run-2") is False
    raise_if_cancelled("run-2")


def test_cancel_can_be_requested_before_the_job_starts():
    """A queued job sits behind the semaphore; cancelling it then must still
    register, so it can bail out the moment it is scheduled."""
    request_cancel("queued-run")

    assert cancel_requested("queued-run") is True


def test_clear_cancel_releases_the_flag():
    request_cancel("run-1")
    clear_cancel("run-1")

    assert cancel_requested("run-1") is False
    assert "run-1" not in job_runner.JOB_CANCELS


def test_repeated_cancels_are_harmless():
    request_cancel("run-1")
    request_cancel("run-1")

    assert cancel_requested("run-1") is True


def test_clearing_an_unknown_run_is_harmless():
    clear_cancel("never-seen")          # must not raise


def test_progress_callback_aborts_the_job_once_cancelled():
    """The per-frame callback is where a running job notices the flag."""
    seen = []

    def on_frame(frame_idx):
        raise_if_cancelled("run-1")
        seen.append(frame_idx)

    on_frame(1)
    on_frame(2)
    request_cancel("run-1")
    with pytest.raises(JobCancelled):
        on_frame(3)

    assert seen == [1, 2], "frames after the cancel must not be processed"


def test_cancel_flag_is_visible_across_threads():
    """The UI thread sets the flag; the tracking thread reads it."""
    observed = {}
    started = threading.Event()

    def worker():
        started.set()
        for _ in range(2000):
            if cancel_requested("run-1"):
                observed["stopped"] = True
                return
            threading.Event().wait(0.001)
        observed["stopped"] = False

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait(timeout=2)
    request_cancel("run-1")
    thread.join(timeout=5)

    assert observed.get("stopped") is True
