from collections.abc import Iterable
from pathlib import Path
from src.detector import find_peaks
from src.io import load_waveforms
from src.lifetime import compute_lifetime


def calculate_lifetimes(files: Iterable[Path]) -> list[float]:
    """
    Calculate valid muon lifetimes from oscilloscope CSV files.

    Parameters
    ----------
    files : iterable of pathlib.Path
        CSV files containing TIME, CH1, and CH2 data.

    Returns
    -------
    list of float
        Valid muon lifetimes.
    """

    lifetimes = []

    for file_path in files:
        time, ch1, ch2 = load_waveforms(file_path)

        t0, t1 = find_peaks(time, ch1, ch2)
        tau = compute_lifetime(t0, t1)

        if tau is not None:
            lifetimes.append(tau)

    return lifetimes
