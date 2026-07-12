"""
Detector module for cosmic muon lifetime reconstruction.

This module identifies:

1. The trigger time (t0) from the coincidence channel (CH2).
2. A delayed decay pulse (t1) in the scintillator signal (CH1).

Detector strategy (v2):

- Trigger is defined by the maximum of CH2.
- An early-time veto suppresses prompt electronics features.
- Both positive and negative pulse candidates are considered.
- The candidate with the highest prominence is selected.
"""

import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
from src.config import (
    MAX_LIFETIME,
    MIN_DELAY,
    MIN_HEIGHT,
    MIN_PROMINENCE,
    MIN_WIDTH,
)

def find_pulse_candidates(waveform):
    """
    Detect positive and negative pulse candidates in a waveform.

    Parameters
    ----------
    waveform : array-like
        Waveform segment searched for delayed decay pulses.

    Returns
    -------
    list of dict
        Candidate pulse properties.
    """

    waveform = np.asarray(waveform)
    candidates = []

    # Positive pulse candidates
    peaks, properties = scipy_find_peaks(
        waveform,
        height=MIN_HEIGHT,
        prominence=MIN_PROMINENCE,
        width=MIN_WIDTH,
    )

    for index, peak_index in enumerate(peaks):
        candidates.append({
            "index": int(peak_index),
            "polarity": "positive",
            "height": float(properties["peak_heights"][index]),
            "prominence": float(properties["prominences"][index]),
            "width": float(properties["widths"][index]),
        })

    # Negative pulse candidates
    peaks, properties = scipy_find_peaks(
        -waveform,
        height=MIN_HEIGHT,
        prominence=MIN_PROMINENCE,
        width=MIN_WIDTH,
    )

    for index, peak_index in enumerate(peaks):
        candidates.append({
            "index": int(peak_index),
            "polarity": "negative",
            "height": float(properties["peak_heights"][index]),
            "prominence": float(properties["prominences"][index]),
            "width": float(properties["widths"][index]),
        })

    return candidates




def find_peaks(time, ch1, ch2):
    """
    Reconstruct a muon decay event.

    Parameters
    ----------
    time : array-like
        Time samples.
    ch1 : array-like
        Analog scintillator waveform.
    ch2 : array-like
        Coincidence trigger waveform.

    Returns
    -------
    tuple
        (t0, t1)

        t0 : trigger time
        t1 : reconstructed decay time

        Returns (t0, None) if no valid decay pulse is found.
    """

    time = np.asarray(time)
    ch1 = np.asarray(ch1)
    ch2 = np.asarray(ch2)

    # ------------------------------------------------------------------
    # Trigger reconstruction
    # ------------------------------------------------------------------

    t0_idx = np.argmax(ch2)
    t0 = time[t0_idx]

    # ------------------------------------------------------------------
    # Ignore prompt region after trigger
    # ------------------------------------------------------------------

    search_start = np.searchsorted(time, t0 + MIN_DELAY)

    if search_start >= len(ch1):
        return t0, None

    waveform = ch1[search_start:]

    candidates = find_pulse_candidates(waveform)
    

    if not candidates:
        return t0, None

    # ------------------------------------------------------------------
    # Select the most prominent pulse
    # ------------------------------------------------------------------

    best = max(candidates, key=lambda c: c["prominence"])

    t1 = time[search_start + best["index"]]

    # ------------------------------------------------------------------
    # Basic physics sanity checks
    # ------------------------------------------------------------------

    lifetime = t1 - t0

    if lifetime <= 0:
        return t0, None

    if lifetime > MAX_LIFETIME:
        return t0, None

    return t0, t1