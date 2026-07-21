from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.signal import find_peaks, peak_widths

DEBUG_PRINTED = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "root_decay_selection"
)

ACCEPTED_CSV = RESULTS_DIR / "accepted_candidates.csv"
REJECTED_CSV = RESULTS_DIR / "rejected_candidates.csv"
HISTOGRAM_PATH = RESULTS_DIR / "accepted_decay_times.png"

ACCEPTED_PLOTS_DIR = RESULTS_DIR / "accepted_examples"
REJECTED_PLOTS_DIR = RESULTS_DIR / "rejected_examples"


# =====================================================================
# Initial exploratory physics cuts
# =====================================================================

# Pre-trigger region used to estimate baseline and noise.
BASELINE_END_US = -0.10

# Region around t = 0 where the initial event is expected.
TRIGGER_START_US = -0.10
TRIGGER_END_US = 0.20

# Delayed-pulse search region.
# Starting at 0.8 μs safely excludes the strong ~0.5 μs population.
DECAY_SEARCH_START_US = 0.80
DECAY_SEARCH_END_US = 9.00

# Pulse-search requirements in units of baseline noise.
DELAYED_HEIGHT_SIGMA = 6.0
DELAYED_PROMINENCE_SIGMA = 6.0

# Require a reasonably strong initial pulse in both channels.
TRIGGER_MIN_SIGMA = 20.0

# Peaks closer than this are not treated as independent.
MIN_PEAK_DISTANCE_NS = 40.0

# Minimum FWHM-like width of a delayed pulse.
MIN_PEAK_WIDTH_NS = 2.0

# Window around the CH2 delayed pulse used for the CH1 veto.
COINCIDENCE_WINDOW_NS = 40.0

# CH1 veto threshold.
# A simultaneous CH1 pulse larger than this many noise sigmas causes
# rejection as possible through-going / accidental background.
CH1_VETO_SIGMA = 8.0

# In addition to the sigma threshold, reject when the simultaneous CH1
# pulse is a substantial fraction of the selected CH2 pulse.
CH1_TO_CH2_MAX_RATIO = 0.20

# Keep only the strongest delayed CH2 pulse from each event.
ONE_CANDIDATE_PER_EVENT = True

# Number of accepted and rejected diagnostic plots to save.
NUMBER_OF_ACCEPTED_PLOTS = 20
NUMBER_OF_REJECTED_PLOTS = 15

# Read this many events at a time.
# This makes the script suitable for the larger ROOT files later.
CHUNK_SIZE = 100

# For a quick test, set an integer such as 500.
# Leave as None to scan the complete selected ROOT file.
MAX_EVENTS = 20


def find_root_files(directory: Path) -> list[Path]:
    """Return all ROOT files in the acquisition-data directory."""

    return sorted(directory.glob("*.root"))


def robust_noise_std(values: np.ndarray) -> float:
    """
    Estimate noise using the median absolute deviation.

    This is more robust than an ordinary standard deviation when a few
    abnormal samples are present.
    """

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
    """Calculate baseline and noise from the pre-trigger region."""

    mask = time_us < BASELINE_END_US

    if not np.any(mask):
        raise ValueError(
            "No samples were found in the pre-trigger baseline region."
        )

    baseline_samples = signal[mask]

    baseline = float(np.median(baseline_samples))
    global DEBUG_PRINTED

    if np.std(baseline_samples) == 0 and not DEBUG_PRINTED:
        DEBUG_PRINTED = True

        print("\n----- ZERO NOISE -----")
        print("Samples:", len(baseline_samples))
        print("Unique values:", len(np.unique(baseline_samples)))
        print("First 20 values:", baseline_samples[:20])

    noise_std = robust_noise_std(baseline_samples)

    return baseline, noise_std


def invert_negative_pulses(
    signal: np.ndarray,
    baseline: float,
) -> np.ndarray:
    """
    Subtract the baseline and invert detector pulses.

    The acquisition pulses are negative. After inversion, physical
    negative pulses appear as positive peaks.
    """

    return -(signal - baseline)


def maximum_in_window(
    time_us: np.ndarray,
    signal: np.ndarray,
    start_us: float,
    end_us: float,
) -> float:
    """Return the maximum signal value inside a time window."""

    mask = (
        (time_us >= start_us)
        & (time_us <= end_us)
    )

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
    Check whether both channels contain a strong initial trigger pulse.
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
    """Find delayed pulse candidates in channel 2."""

    search_mask = (
        (time_us >= DECAY_SEARCH_START_US)
        & (time_us <= DECAY_SEARCH_END_US)
    )

    search_time = time_us[search_mask]
    search_signal = channel2_inverted[search_mask]

    if len(search_time) < 3:
        return []

    sample_interval_us = float(
        np.median(np.diff(search_time))
    )

    if sample_interval_us <= 0:
        raise ValueError("Invalid waveform sample interval.")

    sample_interval_ns = sample_interval_us * 1000.0

    minimum_distance_samples = max(
        1,
        int(round(
            MIN_PEAK_DISTANCE_NS / sample_interval_ns
        )),
    )

    minimum_width_samples = max(
        1.0,
        MIN_PEAK_WIDTH_NS / sample_interval_ns,
    )

    peaks, properties = find_peaks(
        search_signal,
        height=DELAYED_HEIGHT_SIGMA * channel2_noise,
        prominence=DELAYED_PROMINENCE_SIGMA * channel2_noise,
        distance=minimum_distance_samples,
        width=minimum_width_samples,
    )

    if len(peaks) == 0:
        return []

    widths = peak_widths(
        search_signal,
        peaks,
        rel_height=0.5,
    )[0]

    candidates = []

    for peak_number, peak_index in enumerate(peaks):
        candidate = {
            "decay_time_us": float(
                search_time[peak_index]
            ),
            "channel2_amplitude_v": float(
                search_signal[peak_index]
            ),
            "channel2_prominence_v": float(
                properties["prominences"][peak_number]
            ),
            "channel2_width_ns": float(
                widths[peak_number] * sample_interval_ns
            ),
            "channel2_snr": float(
                search_signal[peak_index] / channel2_noise
            ),
        }

        candidates.append(candidate)

    candidates.sort(
        key=lambda item: item["channel2_prominence_v"],
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
    Check for a simultaneous pulse in channel 1.

    A substantial channel1 pulse at the same delayed time indicates a
    possible second through-going particle or accidental coincidence.
    """

    half_window_us = (
        COINCIDENCE_WINDOW_NS / 1000.0
    )

    window_start = (
        channel2_candidate_time_us - half_window_us
    )

    window_end = (
        channel2_candidate_time_us + half_window_us
    )

    channel1_amplitude = maximum_in_window(
        time_us,
        channel1_inverted,
        window_start,
        window_end,
    )

    if not np.isfinite(channel1_amplitude):
        channel1_amplitude = 0.0

    channel1_snr = (
        channel1_amplitude / channel1_noise
    )

    amplitude_ratio = (
        channel1_amplitude / channel2_candidate_amplitude_v
        if channel2_candidate_amplitude_v > 0
        else float("inf")
    )

    veto_by_sigma = (
        channel1_snr >= CH1_VETO_SIGMA
    )

    veto_by_ratio = (
        amplitude_ratio >= CH1_TO_CH2_MAX_RATIO
    )

    vetoed = veto_by_sigma and veto_by_ratio

    return {
        "channel1_veto_amplitude_v": channel1_amplitude,
        "channel1_veto_snr": channel1_snr,
        "channel1_to_channel2_ratio": amplitude_ratio,
        "veto_by_sigma": veto_by_sigma,
        "veto_by_ratio": veto_by_ratio,
        "vetoed": vetoed,
    }


def analyze_event(
    file_name: str,
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> list[dict]:
    """Analyze one complete acquisition event."""

    if not (
        len(time)
        == len(channel1)
        == len(channel2)
    ):
        return [
            {
                "file": file_name,
                "event_index": event_index,
                "accepted": False,
                "rejection_reason": "mismatched_array_lengths",
            }
        ]

    time_us = time * 1e6

    channel1_baseline, channel1_noise = baseline_and_noise(
        time_us,
        channel1,
    )

    channel2_baseline, channel2_noise = baseline_and_noise(
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
                "channel1_trigger_amplitude_v":
                    channel1_trigger_amplitude,
                "channel2_trigger_amplitude_v":
                    channel2_trigger_amplitude,
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
            channel2_candidate_time_us=(
                candidate["decay_time_us"]
            ),
            channel2_candidate_amplitude_v=(
                candidate["channel2_amplitude_v"]
            ),
        )

        accepted = not veto["vetoed"]

        rejection_reason = (
            ""
            if accepted
            else "simultaneous_channel1_pulse"
        )

        result = {
            "file": file_name,
            "event_index": event_index,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "decay_time_us": candidate["decay_time_us"],
            "channel2_amplitude_v":
                candidate["channel2_amplitude_v"],
            "channel2_prominence_v":
                candidate["channel2_prominence_v"],
            "channel2_width_ns":
                candidate["channel2_width_ns"],
            "channel2_snr":
                candidate["channel2_snr"],
            "channel1_veto_amplitude_v":
                veto["channel1_veto_amplitude_v"],
            "channel1_veto_snr":
                veto["channel1_veto_snr"],
            "channel1_to_channel2_ratio":
                veto["channel1_to_channel2_ratio"],
            "channel1_trigger_amplitude_v":
                channel1_trigger_amplitude,
            "channel2_trigger_amplitude_v":
                channel2_trigger_amplitude,
            "channel1_baseline_v":
                channel1_baseline,
            "channel2_baseline_v":
                channel2_baseline,
            "channel1_noise_std_v":
                channel1_noise,
            "channel2_noise_std_v":
                channel2_noise,
        }

        results.append(result)

    return results


def scan_root_file(
    file_path: Path,
) -> pd.DataFrame:
    """
    Scan one ROOT file in chunks.

    The complete file is never loaded into memory at once.
    """

    all_results = []

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]

        total_events = tree.num_entries

        if MAX_EVENTS is not None:
            total_events = min(
                total_events,
                MAX_EVENTS,
            )

        print(f"TTree: {tree.name}")
        print(f"Events to analyze: {total_events}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print()

        for chunk_start in range(
            0,
            total_events,
            CHUNK_SIZE,
        ):
            chunk_stop = min(
                chunk_start + CHUNK_SIZE,
                total_events,
            )

            print(
                f"Reading events "
                f"{chunk_start}–{chunk_stop - 1}"
            )

            arrays = tree.arrays(
                [
                    "time",
                    "channel1",
                    "channel2",
                ],
                entry_start=chunk_start,
                entry_stop=chunk_stop,
                library="ak",
            )

            number_in_chunk = len(
                arrays["time"]
            )

            for local_index in range(
                number_in_chunk
            ):
                event_index = (
                    chunk_start + local_index
                )

                time = ak.to_numpy(
                    arrays["time"][local_index]
                )

                channel1 = ak.to_numpy(
                    arrays["channel1"][local_index]
                )

                channel2 = ak.to_numpy(
                    arrays["channel2"][local_index]
                )

                try:
                    event_results = analyze_event(
                        file_name=file_path.name,
                        event_index=event_index,
                        time=time,
                        channel1=channel1,
                        channel2=channel2,
                    )
                except Exception as error:
                    print(
                        f"Skipping event {event_index}: "
                        f"{error}"
                    )
                    continue

                all_results.extend(
                    event_results
                )

    return pd.DataFrame(all_results)


def load_event(
    file_path: Path,
    event_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one event for diagnostic plotting."""

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]

        event = tree.arrays(
            [
                "time",
                "channel1",
                "channel2",
            ],
            entry_start=event_index,
            entry_stop=event_index + 1,
            library="ak",
        )

    return (
        ak.to_numpy(event["time"][0]),
        ak.to_numpy(event["channel1"][0]),
        ak.to_numpy(event["channel2"][0]),
    )


def plot_selection_example(
    file_path: Path,
    row: pd.Series,
    output_directory: Path,
) -> Path:
    """Create a diagnostic plot for one accepted or rejected event."""

    event_index = int(row["event_index"])
    decay_time_us = float(row["decay_time_us"])

    time, channel1, channel2 = load_event(
        file_path,
        event_index,
    )

    time_us = time * 1e6

    channel1_baseline, _ = baseline_and_noise(
        time_us,
        channel1,
    )

    channel2_baseline, _ = baseline_and_noise(
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

    local_mask = (
        (time_us >= decay_time_us - 0.50)
        & (time_us <= decay_time_us + 0.80)
    )

    post_trigger_mask = (
        (time_us >= 0.15)
        & (time_us <= 9.10)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = (
        "accepted"
        if bool(row["accepted"])
        else "rejected"
    )

    output_path = (
        output_directory
        / (
            f"{file_path.stem}"
            f"_event_{event_index:05d}"
            f"_{status}"
            f"_{decay_time_us:.3f}us.png"
        )
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
    )

    axes[0].plot(
        time_us[post_trigger_mask],
        channel1_inverted[post_trigger_mask],
        label="Channel 1",
        linewidth=0.9,
    )

    axes[0].plot(
        time_us[post_trigger_mask],
        channel2_inverted[post_trigger_mask],
        label="Channel 2",
        linewidth=0.9,
        alpha=0.85,
    )

    axes[0].axvline(
        decay_time_us,
        linestyle="--",
        linewidth=1.3,
        label=f"Candidate: {decay_time_us:.3f} μs",
    )

    axes[0].axvspan(
        0.35,
        0.65,
        alpha=0.15,
        label="Excluded early region",
    )

    axes[0].set_xlabel("Time after trigger (μs)")
    axes[0].set_ylabel("Inverted voltage (V)")
    axes[0].set_title("Complete post-trigger waveform")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time_us[local_mask],
        channel1_inverted[local_mask],
        label="Channel 1",
        linewidth=1,
    )

    axes[1].plot(
        time_us[local_mask],
        channel2_inverted[local_mask],
        label="Channel 2",
        linewidth=1,
        alpha=0.85,
    )

    axes[1].axvline(
        decay_time_us,
        linestyle="--",
        linewidth=1.3,
        label="Selected delayed pulse",
    )

    axes[1].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Inverted voltage (V)")
    axes[1].set_title("Local coincidence-veto window")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    rejection_reason = str(
        row.get("rejection_reason", "")
    )

    figure.suptitle(
        (
            f"{status.upper()} — event {event_index}\n"
            f"t = {decay_time_us:.3f} μs | "
            f"CH2 SNR = {row['channel2_snr']:.1f} | "
            f"CH1 veto SNR = {row['channel1_veto_snr']:.1f} | "
            f"ratio = {row['channel1_to_channel2_ratio']:.3f}"
            + (
                f" | reason: {rejection_reason}"
                if rejection_reason
                else ""
            )
        ),
        fontsize=12,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    return output_path


def save_decay_histogram(
    accepted: pd.DataFrame,
) -> None:
    """Save the first histogram of accepted decay-like times."""

    if accepted.empty:
        return

    plt.figure(figsize=(11, 6))

    plt.hist(
        accepted["decay_time_us"],
        bins=np.arange(
            DECAY_SEARCH_START_US,
            DECAY_SEARCH_END_US + 0.25,
            0.25,
        ),
    )

    plt.xlabel("Delayed-pulse time (μs)")
    plt.ylabel("Accepted candidate count")
    plt.title(
        "Decay-like delayed pulses\n"
        "Initial physics-aware selection"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=150,
    )

    plt.show()
    plt.close()

    print(f"Saved histogram: {HISTOGRAM_PATH}")


def print_summary(
    all_results: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
) -> None:
    """Print selection and validation information."""

    print()
    print("=" * 78)
    print("PHYSICS-AWARE SELECTION SUMMARY")
    print("=" * 78)

    delayed_results = all_results[
        all_results["decay_time_us"].notna()
    ] if "decay_time_us" in all_results else pd.DataFrame()

    print(
        f"Events with delayed CH2 candidates: "
        f"{delayed_results['event_index'].nunique()}"
        if not delayed_results.empty
        else "Events with delayed CH2 candidates: 0"
    )

    print(
        f"Accepted decay-like candidates: "
        f"{len(accepted)}"
    )

    print(
        f"Rejected by CH1 coincidence veto: "
        f"{len(rejected)}"
    )

    if not accepted.empty:
        print()
        print("Accepted decay-time summary:")
        print(
            accepted["decay_time_us"]
            .describe()
            .round(4)
        )

        print()
        print("Strongest accepted candidates:")

        columns = [
            "event_index",
            "decay_time_us",
            "channel2_amplitude_v",
            "channel2_snr",
            "channel1_veto_snr",
            "channel1_to_channel2_ratio",
        ]

        print(
            accepted.sort_values(
                "channel2_prominence_v",
                ascending=False,
            )[columns]
            .head(15)
            .to_string(index=False)
        )

    if not rejected.empty:
        print()
        print("Strongest vetoed candidates:")

        columns = [
            "event_index",
            "decay_time_us",
            "channel2_amplitude_v",
            "channel1_veto_amplitude_v",
            "channel1_veto_snr",
            "channel1_to_channel2_ratio",
        ]

        print(
            rejected.sort_values(
                "channel1_veto_amplitude_v",
                ascending=False,
            )[columns]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("Known-event validation:")
    print("-" * 78)

    known_events = [
        466,
        799,
        1620,
        1663,
        2137,
    ]

    for event_index in known_events:
        accepted_match = accepted[
            accepted["event_index"] == event_index
        ]

        rejected_match = rejected[
            rejected["event_index"] == event_index
        ]

        if not accepted_match.empty:
            status = "ACCEPTED"
        elif not rejected_match.empty:
            status = "REJECTED"
        else:
            status = "NO DELAYED CH2 CANDIDATE"

        print(
            f"Event {event_index:4d}: {status}"
        )

    print("=" * 78)


def main() -> None:
    """Run physics-aware selection on one ROOT acquisition file."""

    root_files = find_root_files(
        ROOT_DATA_DIR
    )

    if not root_files:
        raise FileNotFoundError(
            f"No ROOT files were found in:\n"
            f"{ROOT_DATA_DIR}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Available ROOT files:")
    print("-" * 78)

    for index, file_path in enumerate(root_files):
        size_mb = (
            file_path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"[{index}] {file_path.name} "
            f"({size_mb:.2f} MB)"
        )

    print()

    try:
        file_index = int(
            input("Select ROOT file index: ")
        )
    except ValueError as error:
        raise ValueError(
            "ROOT file index must be a whole number."
        ) from error

    if file_index < 0 or file_index >= len(root_files):
        raise IndexError(
            f"ROOT file index must be between "
            f"0 and {len(root_files) - 1}."
        )

    selected_file = root_files[file_index]

    print()
    print(f"Analyzing: {selected_file.name}")
    print("-" * 78)
    

    all_results = scan_root_file(
        selected_file
    )

    if all_results.empty:
        print("No analysis results were produced.")
        return

    if "decay_time_us" not in all_results.columns:
        print("No delayed pulse candidates were found.")
        return

    candidate_results = all_results[
        all_results["decay_time_us"].notna()
    ].copy()

    accepted = candidate_results[
        candidate_results["accepted"] == True
    ].copy()

    rejected = candidate_results[
        candidate_results["accepted"] == False
    ].copy()

    accepted = accepted.sort_values(
        "channel2_prominence_v",
        ascending=False,
    )

    rejected = rejected.sort_values(
        "channel1_veto_amplitude_v",
        ascending=False,
    )

    accepted.to_csv(
        ACCEPTED_CSV,
        index=False,
    )

    rejected.to_csv(
        REJECTED_CSV,
        index=False,
    )

    print_summary(
        all_results,
        accepted,
        rejected,
    )

    save_decay_histogram(
        accepted
    )

    print()
    print("Creating accepted-event plots...")

    for _, row in accepted.head(
        NUMBER_OF_ACCEPTED_PLOTS
    ).iterrows():
        plot_selection_example(
            selected_file,
            row,
            ACCEPTED_PLOTS_DIR,
        )

    print("Creating rejected-event plots...")

    for _, row in rejected.head(
        NUMBER_OF_REJECTED_PLOTS
    ).iterrows():
        plot_selection_example(
            selected_file,
            row,
            REJECTED_PLOTS_DIR,
        )

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)
    print(f"Accepted candidates:\n{ACCEPTED_CSV}")
    print()
    print(f"Rejected candidates:\n{REJECTED_CSV}")
    print()
    print(f"Accepted plots:\n{ACCEPTED_PLOTS_DIR}")
    print()
    print(f"Rejected plots:\n{REJECTED_PLOTS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()