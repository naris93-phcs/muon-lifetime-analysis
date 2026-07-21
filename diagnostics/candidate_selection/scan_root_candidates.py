from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.signal import find_peaks, peak_widths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"

RESULTS_DIR = PROJECT_ROOT / "results" / "root_candidate_scan"

CANDIDATES_CSV = RESULTS_DIR / "candidate_pulses.csv"
HISTOGRAM_PATH = RESULTS_DIR / "candidate_time_histogram.png"


# ---------------------------------------------------------------------
# Initial exploratory parameters
# ---------------------------------------------------------------------

# We exclude the trigger and its immediate recovery region.
SEARCH_START_US = 0.20

# The waveform ends near 9.4 μs.
SEARCH_END_US = 9.00

# A pulse must stand out from the baseline noise by this many sigma.
MIN_HEIGHT_SIGMA = 5.0
MIN_PROMINENCE_SIGMA = 5.0

# Prevent multiple nearby samples from being counted as separate pulses.
MIN_DISTANCE_NS = 40.0

# Require the pulse to have at least a small physical width.
MIN_WIDTH_NS = 2.0

# Limit used only for quick testing.
# Set to None to scan every event in the selected file.
MAX_EVENTS = None


def find_root_files(directory: Path) -> list[Path]:
    """Return all ROOT files found in the data directory."""

    return sorted(directory.glob("*.root"))


def robust_noise_std(values: np.ndarray) -> float:
    """
    Estimate the baseline noise using the median absolute deviation.

    This is less sensitive to occasional spikes than the ordinary
    standard deviation.
    """

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    noise_std = 1.4826 * mad

    if noise_std <= 0:
        noise_std = float(np.std(values))

    return float(noise_std)


def calculate_baseline_and_noise(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate baseline and noise from the pre-trigger region.
    """

    baseline_mask = time_us < -0.10

    if not np.any(baseline_mask):
        raise ValueError("No pre-trigger samples available for baseline estimation.")

    pre_trigger = signal[baseline_mask]

    baseline = float(np.median(pre_trigger))
    noise_std = robust_noise_std(pre_trigger)

    return baseline, noise_std


def scan_channel(
    time_us: np.ndarray,
    signal: np.ndarray,
    channel_name: str,
) -> list[dict]:
    """
    Find delayed negative pulses in one waveform channel.

    The detector signals are negative, so after baseline subtraction
    we multiply by -1. Real negative pulses then appear as positive peaks.
    """

    baseline, noise_std = calculate_baseline_and_noise(
        time_us,
        signal,
    )

    corrected_signal = signal - baseline
    inverted_signal = -corrected_signal

    search_mask = (time_us >= SEARCH_START_US) & (time_us <= SEARCH_END_US)

    search_time = time_us[search_mask]
    search_signal = inverted_signal[search_mask]

    if len(search_time) < 3:
        return []

    sample_interval_us = float(np.median(np.diff(search_time)))

    sample_interval_ns = sample_interval_us * 1000.0

    minimum_distance_samples = max(
        1,
        int(MIN_DISTANCE_NS / sample_interval_ns),
    )

    minimum_width_samples = max(
        1,
        MIN_WIDTH_NS / sample_interval_ns,
    )

    minimum_height = MIN_HEIGHT_SIGMA * noise_std
    minimum_prominence = MIN_PROMINENCE_SIGMA * noise_std

    peaks, properties = find_peaks(
        search_signal,
        height=minimum_height,
        prominence=minimum_prominence,
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

    for candidate_number, peak_index in enumerate(peaks):
        pulse_time_us = float(search_time[peak_index])
        amplitude_v = float(search_signal[peak_index])

        prominence_v = float(properties["prominences"][candidate_number])

        width_ns = float(widths_samples[candidate_number] * sample_interval_ns)

        signal_to_noise = amplitude_v / noise_std if noise_std > 0 else np.nan

        candidates.append(
            {
                "channel": channel_name,
                "pulse_time_us": pulse_time_us,
                "amplitude_v": amplitude_v,
                "prominence_v": prominence_v,
                "width_ns": width_ns,
                "baseline_v": baseline,
                "noise_std_v": noise_std,
                "signal_to_noise": signal_to_noise,
            }
        )

    return candidates


def load_tree_arrays(
    file_path: Path,
) -> tuple[ak.Array, ak.Array, ak.Array]:
    """
    Load time, channel1, and channel2 from the newest t1 cycle.
    """

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]

        number_of_events = tree.num_entries

        if MAX_EVENTS is not None:
            number_of_events = min(
                number_of_events,
                MAX_EVENTS,
            )

        print(f"TTree: {tree.name}")
        print(f"Events to scan: {number_of_events}")

        arrays = tree.arrays(
            ["time", "channel1", "channel2"],
            entry_start=0,
            entry_stop=number_of_events,
            library="ak",
        )

    return (
        arrays["time"],
        arrays["channel1"],
        arrays["channel2"],
    )


def scan_root_file(
    file_path: Path,
) -> pd.DataFrame:
    """
    Scan every event in one ROOT file for delayed pulse candidates.
    """

    time_events, channel1_events, channel2_events = load_tree_arrays(file_path)

    all_candidates = []

    number_of_events = len(time_events)

    for event_index in range(number_of_events):
        if event_index % 100 == 0:
            print(f"Scanning event " f"{event_index}/{number_of_events}")

        time = ak.to_numpy(time_events[event_index])
        channel1 = ak.to_numpy(channel1_events[event_index])
        channel2 = ak.to_numpy(channel2_events[event_index])

        if not (len(time) == len(channel1) == len(channel2)):
            print(
                f"Skipping malformed event {event_index}: "
                "array lengths do not match."
            )
            continue

        time_us = time * 1e6

        channel1_candidates = scan_channel(
            time_us,
            channel1,
            "channel1",
        )

        channel2_candidates = scan_channel(
            time_us,
            channel2,
            "channel2",
        )

        for candidate in channel1_candidates + channel2_candidates:
            candidate["file"] = file_path.name
            candidate["event_index"] = event_index

            all_candidates.append(candidate)

    columns = [
        "file",
        "event_index",
        "channel",
        "pulse_time_us",
        "amplitude_v",
        "prominence_v",
        "width_ns",
        "baseline_v",
        "noise_std_v",
        "signal_to_noise",
    ]

    return pd.DataFrame(
        all_candidates,
        columns=columns,
    )


def print_scan_summary(
    candidates: pd.DataFrame,
) -> None:
    """Print a compact summary of the candidate scan."""

    print()
    print("=" * 72)
    print("CANDIDATE SCAN SUMMARY")
    print("=" * 72)

    if candidates.empty:
        print("No delayed pulse candidates were found.")
        print("=" * 72)
        return

    unique_events = candidates["event_index"].nunique()

    print(f"Candidate pulses: {len(candidates)}")
    print(f"Events containing candidates: {unique_events}")
    print()

    print("Candidates by channel:")
    print(candidates["channel"].value_counts())
    print()

    print("Pulse-time summary:")
    print(candidates["pulse_time_us"].describe().round(4))

    early_candidates = candidates[
        (candidates["pulse_time_us"] >= 0.35) & (candidates["pulse_time_us"] <= 0.65)
    ]

    print()
    print("Candidates between 0.35 and 0.65 μs: " f"{len(early_candidates)}")

    print()
    print("Strongest candidates:")
    print(
        candidates.sort_values(
            "prominence_v",
            ascending=False,
        )[
            [
                "event_index",
                "channel",
                "pulse_time_us",
                "amplitude_v",
                "prominence_v",
                "signal_to_noise",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("=" * 72)


def save_candidate_histogram(
    candidates: pd.DataFrame,
) -> None:
    """Create a histogram of all delayed candidate times."""

    if candidates.empty:
        return

    plt.figure(figsize=(11, 6))

    for channel_name, channel_data in candidates.groupby("channel"):
        plt.hist(
            channel_data["pulse_time_us"],
            bins=90,
            alpha=0.6,
            label=channel_name,
        )

    plt.axvspan(
        0.35,
        0.65,
        alpha=0.15,
        label="0.35–0.65 μs region",
    )

    plt.xlabel("Delayed pulse time (μs)")
    plt.ylabel("Candidate count")
    plt.title("Delayed pulse candidates\n" "Initial exploratory ROOT scan")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=150,
    )
    plt.show()
    plt.close()

    print(f"Saved histogram: {HISTOGRAM_PATH}")


def main() -> None:
    """Select and scan one ROOT acquisition file."""

    root_files = find_root_files(ROOT_DATA_DIR)

    if not root_files:
        raise FileNotFoundError(f"No ROOT files were found in:\n" f"{ROOT_DATA_DIR}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Available ROOT files:")
    print("-" * 72)

    for index, file_path in enumerate(root_files):
        size_mb = file_path.stat().st_size / (1024**2)

        print(f"[{index}] {file_path.name} " f"({size_mb:.2f} MB)")

    print()

    try:
        file_index = int(input("Select ROOT file index: "))
    except ValueError as error:
        raise ValueError("The ROOT file index must be a whole number.") from error

    if file_index < 0 or file_index >= len(root_files):
        raise IndexError(
            f"ROOT file index must be between 0 and " f"{len(root_files) - 1}."
        )

    selected_file = root_files[file_index]

    print()
    print(f"Scanning: {selected_file.name}")
    print("-" * 72)

    candidates = scan_root_file(selected_file)

    candidates.to_csv(
        CANDIDATES_CSV,
        index=False,
    )

    print_scan_summary(candidates)
    save_candidate_histogram(candidates)

    print()
    print(f"Saved candidate table: {CANDIDATES_CSV}")


if __name__ == "__main__":
    main()
