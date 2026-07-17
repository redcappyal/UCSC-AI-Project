"""Audio impact-peak detection shared by the runtime pipeline and training CLI.

Peaks corroborate trajectory-detected events (the classification vote) and
drive recall: unmatched peaks widen the refine-pass tracking windows and
rescue hits the trajectory thresholds reject (audio_rescue_hits in
detect_wall_hits). Peaks never move a trajectory-detected event.
"""

import math
from pathlib import Path

import numpy as np


def audio_to_mono_float(audio_path):
    audio_path = Path(audio_path)
    try:
        import av
    except ImportError:
        av = None

    if av is not None:
        try:
            container = av.open(str(audio_path))
        except Exception:
            container = None  # unreadable container: try the WAV/afconvert path
        if container is not None:
            # A readable container with no audio stream is definitive - don't
            # waste an afconvert attempt on it.
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                container.close()
                raise RuntimeError(f"{audio_path} has no audio stream")
            try:
                chunks = []
                sample_rate = int(stream.rate or 0)
                for frame in container.decode(stream):
                    frame_samples = frame.to_ndarray()
                    if frame_samples.ndim == 2:
                        frame_samples = frame_samples.mean(axis=0)
                    chunks.append(frame_samples.astype(np.float32))
                    if not sample_rate:
                        sample_rate = int(frame.sample_rate)
                container.close()
                if chunks and sample_rate:
                    return sample_rate, np.concatenate(chunks)
            except Exception:
                pass  # decode failed: fall through to the WAV/afconvert path

    import shutil
    import subprocess
    import tempfile

    from scipy.io import wavfile

    cleanup_path = None
    source_path = audio_path

    if audio_path.suffix.lower() != ".wav":
        afconvert = shutil.which("afconvert")
        if afconvert is None:
            raise RuntimeError(
                f"{audio_path} is not a WAV file, and afconvert is not available. "
                "Convert the audio to WAV first or pass --audio-candidates."
            )
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cleanup_path = Path(temp_file.name)
        temp_file.close()
        subprocess.run(
            [afconvert, "-f", "WAVE", "-d", "LEI16", str(audio_path), str(cleanup_path)],
            check=True,
        )
        source_path = cleanup_path

    try:
        sample_rate, samples = wavfile.read(source_path)
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    if np.issubdtype(samples.dtype, np.integer):
        scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / float(scale)
    else:
        samples = samples.astype(np.float32)

    return sample_rate, samples


def percentile(sorted_values, p):
    if len(sorted_values) == 0:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(math.floor((len(sorted_values) - 1) * p))))
    return float(sorted_values[index])


def detect_audio_candidates_from_file(
    audio_path,
    start_frame,
    end_frame,
    fps,
    max_peaks,
    log=None,
    threshold_db_above_median=10.0,
):
    if log:
        log(f"Analyzing audio file for impact peaks: {audio_path}")
    sample_rate, samples = audio_to_mono_float(audio_path)
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    window_size = max(256, int(round(sample_rate * 0.012)))
    hop = max(128, int(round(sample_rate * 0.005)))
    start_sample = max(0, int(math.floor(max(0.0, start_seconds - 0.5) * sample_rate)))
    end_sample = min(len(samples) - window_size, int(math.ceil((end_seconds + 0.5) * sample_rate)))

    if end_sample <= start_sample or len(samples) < window_size:
        if log:
            log("Audio is too short for peak detection in the selected frame range.")
        return []

    starts = np.arange(start_sample, end_sample + 1, hop, dtype=np.int64)
    squared = samples.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    rms = np.sqrt((cumulative[starts + window_size] - cumulative[starts]) / window_size)
    db = 20 * np.log10(rms + 1e-7)
    times = (starts + window_size / 2) / sample_rate

    selected_range = (times >= start_seconds) & (times <= end_seconds)
    if selected_range.sum() < 3:
        if log:
            log("Audio has too few analysis windows in the selected frame range.")
        return []

    sorted_db = np.sort(db)
    median = percentile(sorted_db, 0.5)
    p90 = percentile(sorted_db, 0.9)
    threshold = max(median + threshold_db_above_median, p90 + 2.0)
    local_peaks = []
    for index in range(1, len(db) - 1):
        if not selected_range[index]:
            continue
        if db[index] < threshold:
            continue
        if db[index] < db[index - 1] or db[index] < db[index + 1]:
            continue
        local_peaks.append(
            {
                "time_seconds": float(times[index]),
                "score": float(db[index] - median),
                "rms": float(rms[index]),
            }
        )

    local_peaks.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    min_separation_seconds = 0.12
    for peak in local_peaks:
        if any(abs(peak["time_seconds"] - item["time_seconds"]) < min_separation_seconds for item in selected):
            continue
        selected.append(peak)
        if len(selected) >= max_peaks:
            break

    half_window_seconds = 0.08
    candidates = []
    for peak in sorted(selected, key=lambda item: item["time_seconds"]):
        frame = int(round(peak["time_seconds"] * fps))
        candidates.append(
            {
                "frame": frame,
                "time_seconds": peak["time_seconds"],
                "window_start_frame": int(round((peak["time_seconds"] - half_window_seconds) * fps)),
                "window_end_frame": int(round((peak["time_seconds"] + half_window_seconds) * fps)),
                "score": peak["score"],
                "rms": peak["rms"],
            }
        )

    if log:
        log(
            f"Audio peak detection found {len(candidates)} candidate(s) "
            f"(threshold {threshold:.2f} dB, median {median:.2f} dB)."
        )
    return candidates


def extract_audio_candidates(
    video_path, start_frame, end_frame, fps, max_peaks=32, threshold_db_above_median=6.0
):
    """Peak candidates from the video's audio track, or None when the file has
    no decodable audio — callers can distinguish 'no audio available' from
    'audio present but silent' (an empty list).

    Uses a lower dB gate than the training default: real match recordings are
    heavily compressed (far mic / AGC), leaving wall impacts under 10 dB above
    the clip median. The classification vote does the discriminating.
    """
    try:
        return detect_audio_candidates_from_file(
            video_path,
            start_frame,
            end_frame,
            fps,
            max_peaks,
            threshold_db_above_median=threshold_db_above_median,
        )
    except Exception:
        return None


# --- Repeating-waveform impact windows (fusion engine, method 1) ---------
#
# Ball impacts recur with nearly the same waveform shape all match long while
# their loudness varies with distance and AGC. Peaks are therefore picked by
# local prominence (not absolute dB) and grouped by a volume-normalized
# spectral fingerprint; only sounds that repeat become impact windows.

FINGERPRINT_SECONDS = 0.064
FINGERPRINT_BANDS = 24
FINGERPRINT_LOW_HZ = 150.0
FINGERPRINT_HIGH_HZ = 8000.0


def _rms_envelope(sample_rate, samples, start_seconds, end_seconds):
    """(times, rms, db) analysis windows covering [start, end] plus margin."""
    window_size = max(256, int(round(sample_rate * 0.012)))
    hop = max(128, int(round(sample_rate * 0.005)))
    start_sample = max(0, int(math.floor(max(0.0, start_seconds - 0.5) * sample_rate)))
    end_sample = min(len(samples) - window_size, int(math.ceil((end_seconds + 0.5) * sample_rate)))
    if end_sample <= start_sample or len(samples) < window_size:
        return None

    starts = np.arange(start_sample, end_sample + 1, hop, dtype=np.int64)
    squared = samples.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    rms = np.sqrt((cumulative[starts + window_size] - cumulative[starts]) / window_size)
    db = 20 * np.log10(rms + 1e-7)
    times = (starts + window_size / 2) / sample_rate
    return times, rms, db


def transient_peaks(
    sample_rate,
    samples,
    start_seconds,
    end_seconds,
    max_peaks=64,
    min_separation_seconds=0.12,
    prominence_db=4.0,
):
    """Local envelope maxima ranked by prominence above a sliding local floor.

    Prominence, not absolute level, so a quiet impact under compressed match
    audio still surfaces (see audio-classification tuning history).
    """
    from scipy.ndimage import median_filter

    envelope = _rms_envelope(sample_rate, samples, start_seconds, end_seconds)
    if envelope is None:
        return []
    times, rms, db = envelope

    hop_seconds = float(times[1] - times[0]) if len(times) > 1 else 0.005
    floor_size = max(3, int(round(0.5 / hop_seconds)) | 1)
    prominence = db - median_filter(db, size=floor_size, mode="nearest")

    peaks = [
        {
            "time_seconds": float(times[i]),
            "prominence_db": float(prominence[i]),
            "rms": float(rms[i]),
        }
        for i in range(1, len(db) - 1)
        if start_seconds <= times[i] <= end_seconds
        and prominence[i] >= prominence_db
        and db[i] >= db[i - 1]
        and db[i] >= db[i + 1]
    ]
    peaks.sort(key=lambda item: item["prominence_db"], reverse=True)

    selected = []
    for peak in peaks:
        if any(
            abs(peak["time_seconds"] - kept["time_seconds"]) < min_separation_seconds
            for kept in selected
        ):
            continue
        selected.append(peak)
        if len(selected) >= max_peaks:
            break
    selected.sort(key=lambda item: item["time_seconds"])
    return selected


def waveform_fingerprint(sample_rate, samples, time_seconds):
    """Volume-normalized log-band spectral envelope around one transient."""
    length = int(round(FINGERPRINT_SECONDS * sample_rate))
    start = int(round((time_seconds - FINGERPRINT_SECONDS / 2) * sample_rate))
    clip = samples[max(0, start) : max(0, start) + length]
    if len(clip) < length // 2:
        return None
    windowed = clip.astype(np.float64) * np.hanning(len(clip))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(clip), 1.0 / sample_rate)

    edges = np.geomspace(FINGERPRINT_LOW_HZ, FINGERPRINT_HIGH_HZ, FINGERPRINT_BANDS + 1)
    bands = np.empty(FINGERPRINT_BANDS)
    for index in range(FINGERPRINT_BANDS):
        mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        bands[index] = np.sqrt(np.mean(spectrum[mask] ** 2)) if mask.any() else 0.0

    bands = np.log1p(bands)
    norm = float(np.linalg.norm(bands))
    if norm < 1e-9:
        return None
    return bands / norm


def cluster_fingerprints(fingerprints, similarity_threshold=0.90):
    """Greedy clustering by cosine similarity to the cluster's mean shape."""
    clusters = []
    for index, fingerprint in enumerate(fingerprints):
        best = None
        best_similarity = similarity_threshold
        for cluster in clusters:
            mean = cluster["sum"] / np.linalg.norm(cluster["sum"])
            similarity = float(fingerprint @ mean)
            if similarity >= best_similarity:
                best, best_similarity = cluster, similarity
        if best is None:
            clusters.append({"sum": fingerprint.copy(), "members": [index]})
        else:
            best["sum"] += fingerprint
            best["members"].append(index)
    return [cluster["members"] for cluster in clusters]


def repeating_impact_windows(
    sample_rate,
    samples,
    start_frame,
    end_frame,
    fps,
    min_repeats=3,
    similarity_threshold=0.90,
    half_window_seconds=0.08,
):
    """Impact windows for every transient whose waveform shape repeats.

    Returns one window dict per repeating peak, tagged with its cluster:
    {frame, time_seconds, window_start_frame, window_end_frame, cluster_id,
    cluster_size, score (prominence dB), rms}.
    """
    peaks = transient_peaks(sample_rate, samples, start_frame / fps, end_frame / fps)
    fingerprinted = [
        (peak, fingerprint)
        for peak in peaks
        if (fingerprint := waveform_fingerprint(sample_rate, samples, peak["time_seconds"]))
        is not None
    ]
    if not fingerprinted:
        return []

    clusters = cluster_fingerprints(
        [fingerprint for _, fingerprint in fingerprinted], similarity_threshold
    )

    windows = []
    for cluster_id, members in enumerate(clusters):
        if len(members) < min_repeats:
            continue
        for index in members:
            peak, _ = fingerprinted[index]
            time_seconds = peak["time_seconds"]
            windows.append(
                {
                    "frame": int(round(time_seconds * fps)),
                    "time_seconds": time_seconds,
                    "window_start_frame": int(round((time_seconds - half_window_seconds) * fps)),
                    "window_end_frame": int(round((time_seconds + half_window_seconds) * fps)),
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "score": peak["prominence_db"],
                    "rms": peak["rms"],
                }
            )
    windows.sort(key=lambda item: item["time_seconds"])
    return windows


def extract_repeating_audio_windows(video_path, start_frame, end_frame, fps, **kwargs):
    """Repeating-waveform impact windows from a video's audio track, or None
    when the file has no decodable audio (same contract as
    extract_audio_candidates)."""
    try:
        sample_rate, samples = audio_to_mono_float(video_path)
        return repeating_impact_windows(
            sample_rate, samples, start_frame, end_frame, fps, **kwargs
        )
    except Exception:
        return None


def match_audio_peak(event_time_seconds, candidates, tolerance_seconds=0.10):
    """Nearest candidate by |time_seconds - event_time| within tolerance, or None."""
    best = None
    best_offset = None
    for candidate in candidates or []:
        offset = abs(candidate["time_seconds"] - event_time_seconds)
        if offset > tolerance_seconds:
            continue
        if best is None or offset < best_offset:
            best = candidate
            best_offset = offset
    return best
