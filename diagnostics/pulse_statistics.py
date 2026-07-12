import csv

import numpy as np

from src.config import (
    DATA_DIR,
    FILE_PATTERN,
    MAX_LIFETIME,
    MIN_DELAY,
    PULSE_STATISTICS_PATH,
)
from src.detector import find_pulse_candidates
from src.io import load_waveforms


CSV_COLUMNS = [
    "file",
    "tau_us",
    "polarity",
    "height",
    "prominence",
    "width",
]


def analyze_pulse_file(file_path):
    """
    Extract the selected delayed-pulse properties from one waveform file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to an oscilloscope CSV file.

    Returns
    -------
    list or None
        CSV row containing the pulse measurements, or None if no valid
        delayed pulse is found.
    """

    time, ch1, ch2 = load_waveforms(file_path)

    t0_index = np.argmax(ch2)
    t0 = time[t0_index]

    search_start = np.searchsorted(
        time,
        t0 + MIN_DELAY,
    )

    if search_start >= len(ch1):
        return None

    waveform = ch1[search_start:]
    candidates = find_pulse_candidates(waveform)

    if not candidates:
        return None

    best_candidate = max(
        candidates,
        key=lambda candidate: candidate["prominence"],
    )

    t1 = time[search_start + best_candidate["index"]]
    lifetime_s = t1 - t0

    if lifetime_s <= 0 or lifetime_s > MAX_LIFETIME:
        return None

    return [
        file_path.name,
        lifetime_s * 1e6,
        best_candidate["polarity"],
        best_candidate["height"],
        best_candidate["prominence"],
        best_candidate["width"],
    ]


def collect_pulse_statistics(files):
    """Collect valid pulse measurements from multiple waveform files."""

    rows = []

    for file_path in files:
        row = analyze_pulse_file(file_path)

        if row is not None:
            rows.append(row)

    return rows


def save_pulse_statistics(rows) -> None:
    """Save pulse statistics to CSV."""

    PULSE_STATISTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PULSE_STATISTICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def main() -> None:
    """Generate and save delayed-pulse statistics."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    rows = collect_pulse_statistics(files)

    save_pulse_statistics(rows)

    print(
        f"Saved {len(rows)} events to "
        f"{PULSE_STATISTICS_PATH}"
    )


if __name__ == "__main__":
    main()