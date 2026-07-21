from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"

SCAN_RESULTS_DIR = PROJECT_ROOT / "results" / "root_candidate_scan"

CANDIDATES_CSV = SCAN_RESULTS_DIR / "candidate_pulses.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "decay_candidate_plots"


# ---------------------------------------------------------------------
# Candidate-selection settings
# ---------------------------------------------------------------------

# Region dominated by the repeated feature seen near 0.5 μs.
ARTIFACT_START_US = 0.35
ARTIFACT_END_US = 0.65

# Avoid the trigger and its immediate recovery region.
MIN_CANDIDATE_TIME_US = 0.80

# The waveform ends near 9.4 μs.
MAX_CANDIDATE_TIME_US = 9.00

# Plot this many strongest unique events.
NUMBER_OF_EVENTS_TO_PLOT = 15

# Minimum candidate strength relative to baseline noise.
MIN_SIGNAL_TO_NOISE = 8.0

# Baseline is estimated from the pre-trigger region.
BASELINE_END_US = -0.10

# Window shown around the selected candidate pulse.
LOCAL_WINDOW_BEFORE_US = 0.50
LOCAL_WINDOW_AFTER_US = 0.80


def find_root_files(directory: Path) -> list[Path]:
    """Return all ROOT files inside the raw-data directory."""

    return sorted(directory.glob("*.root"))


def calculate_baseline(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> float:
    """Estimate the waveform baseline from pre-trigger samples."""

    baseline_mask = time_us < BASELINE_END_US

    if not np.any(baseline_mask):
        raise ValueError("No pre-trigger samples were found for baseline estimation.")

    return float(np.median(signal[baseline_mask]))


def load_candidate_table(csv_path: Path) -> pd.DataFrame:
    """Load and validate the candidate table created by the scanner."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Candidate CSV was not found:\n{csv_path}\n\n"
            "Run diagnostics/scan_root_candidates.py first."
        )

    candidates = pd.read_csv(csv_path)

    required_columns = {
        "file",
        "event_index",
        "channel",
        "pulse_time_us",
        "amplitude_v",
        "prominence_v",
        "signal_to_noise",
    }

    missing_columns = required_columns - set(candidates.columns)

    if missing_columns:
        raise ValueError(
            "Candidate CSV is missing the following columns: "
            f"{sorted(missing_columns)}"
        )

    return candidates


def select_interesting_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select strong delayed candidates outside the repeated 0.5 μs region.

    Only one candidate is retained per file/event combination.
    If an event has several peaks, the strongest one is kept.
    """

    artifact_mask = (candidates["pulse_time_us"] >= ARTIFACT_START_US) & (
        candidates["pulse_time_us"] <= ARTIFACT_END_US
    )

    time_mask = (candidates["pulse_time_us"] >= MIN_CANDIDATE_TIME_US) & (
        candidates["pulse_time_us"] <= MAX_CANDIDATE_TIME_US
    )

    strength_mask = candidates["signal_to_noise"] >= MIN_SIGNAL_TO_NOISE

    selected = candidates[(~artifact_mask) & time_mask & strength_mask].copy()

    selected = selected.sort_values(
        by=[
            "prominence_v",
            "signal_to_noise",
        ],
        ascending=False,
    )

    selected = selected.drop_duplicates(
        subset=[
            "file",
            "event_index",
        ],
        keep="first",
    )

    selected = selected.head(NUMBER_OF_EVENTS_TO_PLOT)

    return selected.reset_index(drop=True)


def load_event(
    file_path: Path,
    event_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one waveform event from the newest t1 cycle."""

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]

        if event_index < 0 or event_index >= tree.num_entries:
            raise IndexError(
                f"Event {event_index} is outside the valid range "
                f"0–{tree.num_entries - 1} for {file_path.name}."
            )

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

    time = ak.to_numpy(event["time"][0])
    channel1 = ak.to_numpy(event["channel1"][0])
    channel2 = ak.to_numpy(event["channel2"][0])

    if not (len(time) == len(channel1) == len(channel2)):
        raise ValueError(f"Array lengths do not match in event {event_index}.")

    return time, channel1, channel2


def plot_candidate_event(
    file_path: Path,
    event_index: int,
    candidate_channel: str,
    candidate_time_us: float,
    candidate_prominence_v: float,
    signal_to_noise: float,
) -> Path:
    """
    Plot one candidate event using three views:

    1. Full waveform.
    2. Full post-trigger baseline-subtracted waveform.
    3. Local zoom around the selected delayed pulse.
    """

    time, channel1, channel2 = load_event(
        file_path,
        event_index,
    )

    time_us = time * 1e6

    channel1_baseline = calculate_baseline(
        time_us,
        channel1,
    )

    channel2_baseline = calculate_baseline(
        time_us,
        channel2,
    )

    # Detector pulses are negative.
    # Inversion makes them appear as positive peaks.
    channel1_inverted = -(channel1 - channel1_baseline)

    channel2_inverted = -(channel2 - channel2_baseline)

    post_trigger_mask = (time_us >= 0.15) & (time_us <= 9.10)

    local_mask = (time_us >= candidate_time_us - LOCAL_WINDOW_BEFORE_US) & (
        time_us <= candidate_time_us + LOCAL_WINDOW_AFTER_US
    )

    if not np.any(local_mask):
        raise ValueError("No samples were found around the candidate pulse.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / (
        f"{file_path.stem}"
        f"_event_{event_index:05d}"
        f"_{candidate_channel}"
        f"_{candidate_time_us:.3f}us.png"
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(13, 12),
    )

    # -----------------------------------------------------------------
    # Panel 1: complete raw waveform
    # -----------------------------------------------------------------

    axes[0].plot(
        time_us,
        channel1,
        label="Channel 1",
        linewidth=0.9,
    )

    axes[0].plot(
        time_us,
        channel2,
        label="Channel 2",
        linewidth=0.9,
        alpha=0.85,
    )

    axes[0].axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="Trigger reference",
    )

    axes[0].axvline(
        candidate_time_us,
        linestyle="--",
        linewidth=1.3,
        label=(f"Candidate: " f"{candidate_time_us:.3f} μs"),
    )

    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Raw voltage (V)")
    axes[0].set_title("Complete raw waveform")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # -----------------------------------------------------------------
    # Panel 2: complete post-trigger region
    # -----------------------------------------------------------------

    axes[1].plot(
        time_us[post_trigger_mask],
        channel1_inverted[post_trigger_mask],
        label="Channel 1 inverted",
        linewidth=0.9,
    )

    axes[1].plot(
        time_us[post_trigger_mask],
        channel2_inverted[post_trigger_mask],
        label="Channel 2 inverted",
        linewidth=0.9,
        alpha=0.85,
    )

    axes[1].axvspan(
        ARTIFACT_START_US,
        ARTIFACT_END_US,
        alpha=0.15,
        label="Repeated 0.5 μs region",
    )

    axes[1].axvline(
        candidate_time_us,
        linestyle="--",
        linewidth=1.3,
        label="Selected candidate",
    )

    axes[1].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[1].set_xlabel("Time after trigger (μs)")
    axes[1].set_ylabel("Baseline-subtracted inverted voltage (V)")
    axes[1].set_title("Post-trigger waveform — negative pulses shown upward")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # -----------------------------------------------------------------
    # Panel 3: local candidate zoom
    # -----------------------------------------------------------------

    axes[2].plot(
        time_us[local_mask],
        channel1_inverted[local_mask],
        label="Channel 1 inverted",
        linewidth=1,
    )

    axes[2].plot(
        time_us[local_mask],
        channel2_inverted[local_mask],
        label="Channel 2 inverted",
        linewidth=1,
        alpha=0.85,
    )

    axes[2].axvline(
        candidate_time_us,
        linestyle="--",
        linewidth=1.3,
        label=(f"{candidate_channel}: " f"{candidate_time_us:.3f} μs"),
    )

    axes[2].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[2].set_xlabel("Time (μs)")
    axes[2].set_ylabel("Baseline-subtracted inverted voltage (V)")
    axes[2].set_title("Local zoom around delayed-pulse candidate")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        (
            f"Delayed-pulse candidate — event {event_index}\n"
            f"{file_path.name} | "
            f"{candidate_channel} | "
            f"t = {candidate_time_us:.3f} μs | "
            f"prominence = {candidate_prominence_v:.4f} V | "
            f"SNR = {signal_to_noise:.1f}"
        ),
        fontsize=13,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)

    return output_path


def save_selected_candidates(
    selected_candidates: pd.DataFrame,
) -> Path:
    """Save the final ranked list used for plotting."""

    output_path = OUTPUT_DIR / "selected_delayed_candidates.csv"

    selected_candidates.to_csv(
        output_path,
        index=False,
    )

    return output_path


def print_selected_candidates(
    selected_candidates: pd.DataFrame,
) -> None:
    """Print the ranked candidate list."""

    print()
    print("=" * 82)
    print("SELECTED DELAYED-PULSE CANDIDATES")
    print("=" * 82)

    if selected_candidates.empty:
        print("No candidates passed the current selection criteria.")
        print("=" * 82)
        return

    display_columns = [
        "file",
        "event_index",
        "channel",
        "pulse_time_us",
        "amplitude_v",
        "prominence_v",
        "signal_to_noise",
    ]

    print(
        selected_candidates[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("=" * 82)


def main() -> None:
    """Select and plot the strongest delayed-pulse candidates."""

    root_files = find_root_files(ROOT_DATA_DIR)

    if not root_files:
        raise FileNotFoundError(
            f"No ROOT acquisition files were found in:\n" f"{ROOT_DATA_DIR}"
        )

    root_files_by_name = {file_path.name: file_path for file_path in root_files}

    candidates = load_candidate_table(CANDIDATES_CSV)

    selected_candidates = select_interesting_candidates(candidates)

    print_selected_candidates(selected_candidates)

    if selected_candidates.empty:
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_csv_path = save_selected_candidates(selected_candidates)

    print()
    print("Creating diagnostic plots...")
    print("-" * 82)

    created_plots = []

    for row_number, candidate in selected_candidates.iterrows():
        file_name = str(candidate["file"])

        if file_name not in root_files_by_name:
            print(f"Skipping candidate: ROOT file not found: " f"{file_name}")
            continue

        file_path = root_files_by_name[file_name]

        event_index = int(candidate["event_index"])

        candidate_channel = str(candidate["channel"])

        candidate_time_us = float(candidate["pulse_time_us"])

        candidate_prominence_v = float(candidate["prominence_v"])

        signal_to_noise = float(candidate["signal_to_noise"])

        print(
            f"[{row_number + 1}/"
            f"{len(selected_candidates)}] "
            f"Event {event_index} | "
            f"{candidate_channel} | "
            f"{candidate_time_us:.3f} μs"
        )

        try:
            output_path = plot_candidate_event(
                file_path=file_path,
                event_index=event_index,
                candidate_channel=candidate_channel,
                candidate_time_us=candidate_time_us,
                candidate_prominence_v=candidate_prominence_v,
                signal_to_noise=signal_to_noise,
            )
        except Exception as error:
            print(f"Could not plot event {event_index}: " f"{error}")
            continue

        created_plots.append(output_path)

    print()
    print("=" * 82)
    print("CANDIDATE PLOTTING COMPLETE")
    print("=" * 82)
    print(f"Candidates selected: " f"{len(selected_candidates)}")
    print(f"Plots created: " f"{len(created_plots)}")
    print(f"Selected-candidate table:\n" f"{selected_csv_path}")
    print(f"Plot directory:\n" f"{OUTPUT_DIR}")
    print("=" * 82)


if __name__ == "__main__":
    main()
