"""
Detector logic for ROOT cosmic-muon decay reconstruction.

This module analyzes one acquisition event at a time.

Detector strategy
-----------------
1. Estimate the baseline and noise from the pre-trigger region.
2. Validate the prompt trigger in both detector channels.
3. Search channel 2 for a delayed negative pulse.
4. Apply a channel 1 coincidence veto.
5. Return the accepted or rejected candidate properties.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths

from src.config import (
    BASELINE_END_US,
    CH1_TO_CH2_MAX_RATIO,
    CH1_VETO_SIGMA,
    COINCIDENCE_WINDOW_NS,
    DECAY_SEARCH_END_US,
    DECAY_SEARCH_START_US,
    DELAYED_HEIGHT_SIGMA,
    DELAYED_PROMINENCE_SIGMA,
    MIN_PEAK_DISTANCE_NS,
    MIN_PEAK_WIDTH_NS,
    ONE_CANDIDATE_PER_EVENT,
    TRIGGER_END_US,
    TRIGGER_MIN_SIGMA,
    TRIGGER_START_US,
)


def robust_noise_std(values: np.ndarray) -> float:
    """
    Estimate baseline noise using the median absolute deviation.

    The median absolute deviation is less sensitive to isolated spikes
    than the ordinary standard deviation.

    Parameters
    ----------
    values : numpy.ndarray
        Baseline samples.

    Returns
    -------
    float
        Robust estimate of the baseline noise standard deviation.

    Raises
    ------
    ValueError
        If a positive finite noise estimate cannot be obtained.
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot estimate noise from an empty array.")

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    noise_std = 1.4826 * mad

    if not np.isfinite(noise_std) or noise_std <= 0:
        noise_std = float(np.std(values))

    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("Could not estimate a positive baseline noise.")

    return float(noise_std)


def baseline_and_noise(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the waveform baseline and noise.

    The estimate is obtained from samples before ``BASELINE_END_US``.

    Parameters
    ----------
    time_us : numpy.ndarray
        Time samples in microseconds.
    signal : numpy.ndarray
        Voltage samples.

    Returns
    -------
    tuple of float
        ``(baseline, noise_std)``.
    """

    time_us = np.asarray(time_us, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if time_us.shape != signal.shape:
        raise ValueError("Time and signal arrays must have matching shapes.")

    baseline_mask = time_us < BASELINE_END_US

    if not np.any(baseline_mask):
        raise ValueError("No samples were found in the pre-trigger baseline region.")

    baseline_samples = signal[baseline_mask]

    baseline = float(np.median(baseline_samples))
    noise_std = robust_noise_std(baseline_samples)

    return baseline, noise_std


def invert_negative_pulses(
    signal: np.ndarray,
    baseline: float,
) -> np.ndarray:
    """
    Subtract the baseline and invert negative detector pulses.

    The acquisition pulses are negative. After inversion, physical
    pulses appear as positive peaks.

    Parameters
    ----------
    signal : numpy.ndarray
        Raw waveform voltage samples.
    baseline : float
        Estimated waveform baseline.

    Returns
    -------
    numpy.ndarray
        Baseline-subtracted and inverted waveform.
    """

    signal = np.asarray(signal, dtype=float)

    return -(signal - baseline)


def maximum_in_window(
    time_us: np.ndarray,
    signal: np.ndarray,
    start_us: float,
    end_us: float,
) -> float:
    """
    Return the maximum signal value inside a time window.

    Parameters
    ----------
    time_us : numpy.ndarray
        Time samples in microseconds.
    signal : numpy.ndarray
        Waveform samples.
    start_us : float
        Beginning of the search window.
    end_us : float
        End of the search window.

    Returns
    -------
    float
        Maximum value in the selected window. Returns ``nan`` when the
        window contains no samples.
    """

    mask = (time_us >= start_us) & (time_us <= end_us)

    if not np.any(mask):
        return float("nan")

    return float(np.max(signal[mask]))


def validate_initial_trigger(
    time_us: np.ndarray,
    channel1_inverted: np.ndarray,
    channel2_inverted: np.ndarray,
    channel1_noise: float,
    channel2_noise: float,
) -> tuple[bool, float, float]:
    """
    Check for a strong prompt trigger pulse in both channels.

    Parameters
    ----------
    time_us : numpy.ndarray
        Time samples in microseconds.
    channel1_inverted : numpy.ndarray
        Baseline-corrected and inverted channel 1 waveform.
    channel2_inverted : numpy.ndarray
        Baseline-corrected and inverted channel 2 waveform.
    channel1_noise : float
        Channel 1 baseline noise.
    channel2_noise : float
        Channel 2 baseline noise.

    Returns
    -------
    tuple
        ``(trigger_valid, channel1_trigger, channel2_trigger)``.
    """

    channel1_trigger = maximum_in_window(
        time_us,
        channel1_inverted,
        TRIGGER_START_US,
        TRIGGER_END_US,
    )

    channel2_trigger = maximum_in_window(
        time_us,
        channel2_inverted,
        TRIGGER_START_US,
        TRIGGER_END_US,
    )

    trigger_valid = (
        np.isfinite(channel1_trigger)
        and np.isfinite(channel2_trigger)
        and channel1_trigger >= TRIGGER_MIN_SIGMA * channel1_noise
        and channel2_trigger >= TRIGGER_MIN_SIGMA * channel2_noise
    )

    return (
        bool(trigger_valid),
        channel1_trigger,
        channel2_trigger,
    )


def find_delayed_channel2_peaks(
    time_us: np.ndarray,
    channel2_inverted: np.ndarray,
    channel2_noise: float,
) -> list[dict]:
    """
    Find delayed pulse candidates in channel 2.

    Parameters
    ----------
    time_us : numpy.ndarray
        Time samples in microseconds.
    channel2_inverted : numpy.ndarray
        Baseline-corrected and inverted channel 2 waveform.
    channel2_noise : float
        Channel 2 baseline noise.

    Returns
    -------
    list of dict
        Candidate pulse properties, ordered by decreasing prominence.
    """

    search_mask = (time_us >= DECAY_SEARCH_START_US) & (time_us <= DECAY_SEARCH_END_US)

    search_time = time_us[search_mask]
    search_signal = channel2_inverted[search_mask]

    if len(search_time) < 3:
        return []

    sample_interval_us = float(np.median(np.diff(search_time)))

    if not np.isfinite(sample_interval_us) or sample_interval_us <= 0:
        raise ValueError("Invalid waveform sample interval.")

    sample_interval_ns = sample_interval_us * 1000.0

    minimum_distance_samples = max(
        1,
        int(round(MIN_PEAK_DISTANCE_NS / sample_interval_ns)),
    )

    minimum_width_samples = max(
        1.0,
        MIN_PEAK_WIDTH_NS / sample_interval_ns,
    )

    peaks, properties = find_peaks(
        search_signal,
        height=DELAYED_HEIGHT_SIGMA * channel2_noise,
        prominence=(DELAYED_PROMINENCE_SIGMA * channel2_noise),
        distance=minimum_distance_samples,
        width=minimum_width_samples,
    )

    if len(peaks) == 0:
        return []

    widths_samples = peak_widths(
        search_signal,
        peaks,
        rel_height=0.5,
    )[0]

    candidates = []

    for peak_number, peak_index in enumerate(peaks):
        amplitude = float(search_signal[peak_index])

        candidate = {
            "decay_time_us": float(search_time[peak_index]),
            "channel2_amplitude_v": amplitude,
            "channel2_prominence_v": float(properties["prominences"][peak_number]),
            "channel2_width_ns": float(
                widths_samples[peak_number] * sample_interval_ns
            ),
            "channel2_snr": float(amplitude / channel2_noise),
        }

        candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (candidate["channel2_prominence_v"]),
        reverse=True,
    )

    if ONE_CANDIDATE_PER_EVENT:
        return candidates[:1]

    return candidates


def evaluate_channel1_veto(
    time_us: np.ndarray,
    channel1_inverted: np.ndarray,
    channel1_noise: float,
    channel2_candidate_time_us: float,
    channel2_candidate_amplitude_v: float,
) -> dict:
    """
    Check for a simultaneous delayed pulse in channel 1.

    A sufficiently strong channel 1 pulse occurring near the selected
    channel 2 candidate may indicate a through-going particle or an
    accidental coincidence rather than a stopped-muon decay.

    Returns
    -------
    dict
        Coincidence-veto measurements and decision.
    """

    half_window_us = COINCIDENCE_WINDOW_NS / 1000.0

    window_start_us = channel2_candidate_time_us - half_window_us

    window_end_us = channel2_candidate_time_us + half_window_us

    channel1_amplitude = maximum_in_window(
        time_us,
        channel1_inverted,
        window_start_us,
        window_end_us,
    )

    if not np.isfinite(channel1_amplitude):
        channel1_amplitude = 0.0

    channel1_snr = channel1_amplitude / channel1_noise

    if channel2_candidate_amplitude_v > 0:
        amplitude_ratio = channel1_amplitude / channel2_candidate_amplitude_v
    else:
        amplitude_ratio = float("inf")

    veto_by_sigma = channel1_snr >= CH1_VETO_SIGMA

    veto_by_ratio = amplitude_ratio >= CH1_TO_CH2_MAX_RATIO

    vetoed = veto_by_sigma and veto_by_ratio

    return {
        "channel1_veto_amplitude_v": float(channel1_amplitude),
        "channel1_veto_snr": float(channel1_snr),
        "channel1_to_channel2_ratio": float(amplitude_ratio),
        "veto_by_sigma": bool(veto_by_sigma),
        "veto_by_ratio": bool(veto_by_ratio),
        "vetoed": bool(vetoed),
    }


def analyze_event(
    file_name: str,
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> list[dict]:
    """
    Analyze one complete ROOT acquisition event.

    Parameters
    ----------
    file_name : str
        Name of the source ROOT file.
    event_index : int
        Event entry index inside the ROOT tree.
    time : numpy.ndarray
        Time samples in seconds.
    channel1 : numpy.ndarray
        Channel 1 waveform.
    channel2 : numpy.ndarray
        Channel 2 waveform.

    Returns
    -------
    list of dict
        Event-selection results.

        The list is empty when the trigger is valid but no delayed
        channel 2 candidate is found.

        A rejected result is returned for malformed events, invalid
        triggers, or candidates rejected by the channel 1 veto.
    """

    time = np.asarray(time, dtype=float)
    channel1 = np.asarray(channel1, dtype=float)
    channel2 = np.asarray(channel2, dtype=float)

    if not (len(time) == len(channel1) == len(channel2)):
        return [
            {
                "file": file_name,
                "event_index": event_index,
                "accepted": False,
                "rejection_reason": "mismatched_array_lengths",
            }
        ]

    if len(time) < 3:
        return [
            {
                "file": file_name,
                "event_index": event_index,
                "accepted": False,
                "rejection_reason": "insufficient_samples",
            }
        ]

    time_us = time * 1e6

    (
        channel1_baseline,
        channel1_noise,
    ) = baseline_and_noise(
        time_us,
        channel1,
    )

    (
        channel2_baseline,
        channel2_noise,
    ) = baseline_and_noise(
        time_us,
        channel2,
    )

    channel1_inverted = invert_negative_pulses(
        channel1,
        channel1_baseline,
    )

    channel2_inverted = invert_negative_pulses(
        channel2,
        channel2_baseline,
    )

    (
        trigger_valid,
        channel1_trigger_amplitude,
        channel2_trigger_amplitude,
    ) = validate_initial_trigger(
        time_us,
        channel1_inverted,
        channel2_inverted,
        channel1_noise,
        channel2_noise,
    )

    if not trigger_valid:
        return [
            {
                "file": file_name,
                "event_index": event_index,
                "accepted": False,
                "rejection_reason": "invalid_initial_trigger",
                "channel1_trigger_amplitude_v": channel1_trigger_amplitude,
                "channel2_trigger_amplitude_v": channel2_trigger_amplitude,
                "channel1_noise_std_v": channel1_noise,
                "channel2_noise_std_v": channel2_noise,
            }
        ]

    delayed_candidates = find_delayed_channel2_peaks(
        time_us,
        channel2_inverted,
        channel2_noise,
    )

    if not delayed_candidates:
        return []

    results = []

    for candidate in delayed_candidates:
        veto = evaluate_channel1_veto(
            time_us=time_us,
            channel1_inverted=channel1_inverted,
            channel1_noise=channel1_noise,
            channel2_candidate_time_us=(candidate["decay_time_us"]),
            channel2_candidate_amplitude_v=(candidate["channel2_amplitude_v"]),
        )

        accepted = not veto["vetoed"]

        rejection_reason = "" if accepted else "simultaneous_channel1_pulse"

        result = {
            "file": file_name,
            "event_index": event_index,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "decay_time_us": candidate["decay_time_us"],
            "channel2_amplitude_v": candidate["channel2_amplitude_v"],
            "channel2_prominence_v": candidate["channel2_prominence_v"],
            "channel2_width_ns": candidate["channel2_width_ns"],
            "channel2_snr": candidate["channel2_snr"],
            "channel1_veto_amplitude_v": veto["channel1_veto_amplitude_v"],
            "channel1_veto_snr": veto["channel1_veto_snr"],
            "channel1_to_channel2_ratio": veto["channel1_to_channel2_ratio"],
            "veto_by_sigma": veto["veto_by_sigma"],
            "veto_by_ratio": veto["veto_by_ratio"],
            "channel1_trigger_amplitude_v": channel1_trigger_amplitude,
            "channel2_trigger_amplitude_v": channel2_trigger_amplitude,
            "channel1_baseline_v": channel1_baseline,
            "channel2_baseline_v": channel2_baseline,
            "channel1_noise_std_v": channel1_noise,
            "channel2_noise_std_v": channel2_noise,
        }

        results.append(result)

    return results
