"""Synthetic-WAV tests for audio impact-peak detection (no model, no network)."""

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_events import (
    detect_audio_candidates_from_file,
    extract_audio_candidates,
    match_audio_peak,
)

SAMPLE_RATE = 16000
FPS = 30.0


def write_wav(path, seconds, bursts):
    """16-bit mono WAV: quiet noise floor plus (time, amplitude) bursts."""
    total = int(seconds * SAMPLE_RATE)
    samples = [0.001 * math.sin(2 * math.pi * 50 * i / SAMPLE_RATE) for i in range(total)]
    for burst_time, amplitude in bursts:
        start = int(burst_time * SAMPLE_RATE)
        for i in range(start, min(total, start + int(0.02 * SAMPLE_RATE))):
            samples[i] = amplitude * math.sin(2 * math.pi * 1500 * i / SAMPLE_RATE)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(
            b"".join(struct.pack("<h", int(sample * 32767)) for sample in samples)
        )


def test_detects_loud_and_soft_bursts_at_expected_frames(tmp_path):
    audio_path = tmp_path / "clip.wav"
    write_wav(audio_path, 4.0, [(1.0, 0.9), (3.0, 0.15)])

    candidates = detect_audio_candidates_from_file(audio_path, 0, int(4.0 * FPS), FPS, 8)

    assert len(candidates) == 2
    frames = sorted(candidate["frame"] for candidate in candidates)
    assert abs(frames[0] - 1.0 * FPS) <= 1
    assert abs(frames[1] - 3.0 * FPS) <= 1

    by_time = sorted(candidates, key=lambda candidate: candidate["time_seconds"])
    assert by_time[0]["score"] > by_time[1]["score"]
    assert by_time[0]["rms"] > by_time[1]["rms"]


def test_min_separation_suppresses_twin_burst(tmp_path):
    audio_path = tmp_path / "twin.wav"
    write_wav(audio_path, 3.0, [(1.0, 0.9), (1.05, 0.8)])

    candidates = detect_audio_candidates_from_file(audio_path, 0, int(3.0 * FPS), FPS, 8)

    assert len(candidates) == 1
    assert abs(candidates[0]["time_seconds"] - 1.0) < 0.06


def test_extract_returns_none_for_file_without_audio(tmp_path):
    silent_failure = tmp_path / "video.mp4"
    silent_failure.write_bytes(b"not a real container")

    assert extract_audio_candidates(silent_failure, 0, 100, FPS) is None
    assert extract_audio_candidates(tmp_path / "missing.mp4", 0, 100, FPS) is None


def test_extract_returns_list_when_audio_present(tmp_path):
    audio_path = tmp_path / "clip.wav"
    write_wav(audio_path, 2.0, [(1.0, 0.9)])

    candidates = extract_audio_candidates(audio_path, 0, int(2.0 * FPS), FPS)
    assert isinstance(candidates, list)
    assert len(candidates) == 1


def test_match_audio_peak_tolerance():
    candidates = [
        {"time_seconds": 1.0, "score": 20.0, "rms": 0.5},
        {"time_seconds": 2.0, "score": 10.0, "rms": 0.1},
    ]

    assert match_audio_peak(1.05, candidates, 0.10)["time_seconds"] == 1.0
    assert match_audio_peak(1.5, candidates, 0.10) is None
    assert match_audio_peak(1.95, candidates, 0.10)["time_seconds"] == 2.0
    assert match_audio_peak(1.0, [], 0.10) is None
    assert match_audio_peak(1.0, None, 0.10) is None
