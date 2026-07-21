import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DATA_DIR,
    DIAGNOSTIC_FILE_LIMIT,
    DIAGNOSTIC_X_MAX_OFFSET,
    DIAGNOSTIC_X_MIN_OFFSET,
    DIAGNOSTIC_Y_MAX,
    DIAGNOSTIC_Y_MIN,
    DIAGNOSTICS_DIR,
    FILE_PATTERN,
    MAX_LIFETIME,
    MIN_DELAY,
)
from src.detector import find_pulse_candidates
from src.io import load_waveforms


def plot_detector_event(
    file_path,
    output_path,
) -> bool:
    """
    Create a diagnostic plot for one oscilloscope waveform.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input CSV file.
    output_path : pathlib.Path
        Path where the diagnostic figure will be saved.

    Returns
    -------
    bool
        True if the plot was created, otherwise False.
    """

    time, ch1, ch2 = load_waveforms(file_path)

    t0_index = np.argmax(ch2)
    t0 = time[t0_index]

    search_start = np.searchsorted(
        time,
        t0 + MIN_DELAY,
    )

    if search_start >= len(ch1):
        return False

    waveform = ch1[search_start:]
    candidates = find_pulse_candidates(waveform)

    valid_candidates = []

    for candidate in candidates:
        candidate_time = time[search_start + candidate["index"]]

        lifetime_s = candidate_time - t0

        if 0 < lifetime_s <= MAX_LIFETIME:
            valid_candidates.append(candidate)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        time * 1e6,
        ch1,
        label="CH1 waveform",
    )

    ax.axvline(
        t0 * 1e6,
        linestyle="--",
        label="Trigger time",
    )

    ax.axvline(
        time[search_start] * 1e6,
        linestyle=":",
        label="Search start",
    )

    if valid_candidates:
        candidate_indices = np.asarray(
            [search_start + candidate["index"] for candidate in valid_candidates]
        )

        ax.scatter(
            time[candidate_indices] * 1e6,
            ch1[candidate_indices],
            marker="x",
            label="Pulse candidates",
        )

        best_candidate = max(
            valid_candidates,
            key=lambda candidate: candidate["prominence"],
        )

        best_index = search_start + best_candidate["index"]
        best_time = time[best_index]
        best_value = ch1[best_index]
        lifetime_us = (best_time - t0) * 1e6

        ax.scatter(
            best_time * 1e6,
            best_value,
            s=100,
            marker="o",
            label="Selected pulse",
        )

        title = (
            f"{file_path.name} | "
            f"τ = {lifetime_us:.3f} μs | "
            f"{best_candidate['polarity']}"
        )
    else:
        title = f"{file_path.name} | No valid decay candidate"

    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Voltage (V)")

    ax.set_xlim(
        (t0 + DIAGNOSTIC_X_MIN_OFFSET) * 1e6,
        (t0 + DIAGNOSTIC_X_MAX_OFFSET) * 1e6,
    )

    ax.set_ylim(
        DIAGNOSTIC_Y_MIN,
        DIAGNOSTIC_Y_MAX,
    )

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def main() -> None:
    """Generate detector diagnostic plots for a sample of input files."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    selected_files = files[:DIAGNOSTIC_FILE_LIMIT]

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_created = 0

    for index, file_path in enumerate(selected_files):
        output_path = DIAGNOSTICS_DIR / f"diagnostic_{index:03d}.png"

        created = plot_detector_event(
            file_path,
            output_path,
        )

        if created:
            plots_created += 1

    print(f"Saved {plots_created} diagnostic plots to: " f"{DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
