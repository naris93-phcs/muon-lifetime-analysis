from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"
RESULTS_DIR = PROJECT_ROOT / "results" / "root_diagnostics"


def find_root_files(directory: Path) -> list[Path]:
    """Return all ROOT files inside the given directory."""
    return sorted(directory.glob("*.root"))


def load_event(
    file_path: Path,
    event_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one waveform event from the latest version of the t1 TTree.

    Returns:
        time:
            Time values in seconds.
        channel1:
            Channel 1 voltage values.
        channel2:
            Channel 2 voltage values.
    """

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]

        if event_index < 0 or event_index >= tree.num_entries:
            raise IndexError(
                f"Event index must be between 0 and "
                f"{tree.num_entries - 1}."
            )

        event = tree.arrays(
            ["time", "channel1", "channel2"],
            entry_start=event_index,
            entry_stop=event_index + 1,
            library="ak",
        )

    time = ak.to_numpy(event["time"][0])
    channel1 = ak.to_numpy(event["channel1"][0])
    channel2 = ak.to_numpy(event["channel2"][0])

    if not (
        len(time) == len(channel1) == len(channel2)
    ):
        raise ValueError(
            "Time, channel1, and channel2 do not have the same length."
        )

    return time, channel1, channel2


def calculate_baseline(
    time_microseconds: np.ndarray,
    signal: np.ndarray,
) -> float:
    """
    Estimate the baseline using the pre-trigger region.

    We use samples before -0.1 microseconds so that the main trigger pulse
    does not affect the baseline estimate.
    """

    baseline_mask = time_microseconds < -0.1

    if not np.any(baseline_mask):
        raise ValueError(
            "No pre-trigger samples were found for baseline estimation."
        )

    return float(np.median(signal[baseline_mask]))


def plot_full_event(
    file_path: Path,
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> None:
    """Plot the full waveform with both channels on the same graph."""

    time_microseconds = time * 1e6

    output_path = (
        RESULTS_DIR
        / f"event_{event_index:05d}_full.png"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_microseconds,
        channel1,
        label="Channel 1",
        linewidth=1,
    )

    plt.plot(
        time_microseconds,
        channel2,
        label="Channel 2",
        linewidth=1,
        alpha=0.8,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="t = 0",
    )

    plt.xlabel("Time (μs)")
    plt.ylabel("Voltage (V)")
    plt.title(
        f"ROOT waveform event {event_index}\n"
        f"{file_path.name}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved full waveform: {output_path}")


def plot_separate_channels(
    file_path: Path,
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> None:
    """Plot channel 1 and channel 2 on separate axes."""

    time_microseconds = time * 1e6

    output_path = (
        RESULTS_DIR
        / f"event_{event_index:05d}_separate.png"
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].plot(
        time_microseconds,
        channel1,
        linewidth=1,
    )
    axes[0].axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="t = 0",
    )
    axes[0].set_ylabel("Channel 1 (V)")
    axes[0].set_title("Channel 1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time_microseconds,
        channel2,
        linewidth=1,
    )
    axes[1].axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="t = 0",
    )
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Channel 2 (V)")
    axes[1].set_title("Channel 2")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        f"ROOT waveform event {event_index}\n"
        f"{file_path.name}"
    )
    figure.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved separate-channel plot: {output_path}")


def plot_post_trigger_zoom(
    file_path: Path,
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> None:
    """
    Plot the post-trigger region after baseline subtraction.

    The main trigger pulse is excluded so that smaller delayed pulses
    become visible.
    """

    time_microseconds = time * 1e6

    channel1_baseline = calculate_baseline(
        time_microseconds,
        channel1,
    )
    channel2_baseline = calculate_baseline(
        time_microseconds,
        channel2,
    )

    channel1_corrected = channel1 - channel1_baseline
    channel2_corrected = channel2 - channel2_baseline

    zoom_mask = (
        (time_microseconds >= 0.15)
        & (time_microseconds <= 9.0)
    )

    if not np.any(zoom_mask):
        raise ValueError(
            "No samples were found inside the post-trigger zoom range."
        )

    output_path = (
        RESULTS_DIR
        / f"event_{event_index:05d}_post_trigger.png"
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].plot(
        time_microseconds[zoom_mask],
        channel1_corrected[zoom_mask],
        linewidth=1,
    )
    axes[0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_ylabel("CH1 − baseline (V)")
    axes[0].set_title("Channel 1 post-trigger")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        time_microseconds[zoom_mask],
        channel2_corrected[zoom_mask],
        linewidth=1,
    )
    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("CH2 − baseline (V)")
    axes[1].set_title("Channel 2 post-trigger")
    axes[1].grid(alpha=0.3)

    figure.suptitle(
        f"Post-trigger zoom — event {event_index}\n"
        f"{file_path.name}"
    )
    figure.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved post-trigger zoom: {output_path}")


def print_event_summary(
    event_index: int,
    time: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> None:
    """Print basic numerical information about the event."""

    time_microseconds = time * 1e6

    sample_intervals = np.diff(time)

    if len(sample_intervals) == 0:
        raise ValueError(
            "The waveform does not contain enough samples."
        )

    mean_sample_interval_ps = (
        np.mean(sample_intervals) * 1e12
    )

    channel1_baseline = calculate_baseline(
        time_microseconds,
        channel1,
    )
    channel2_baseline = calculate_baseline(
        time_microseconds,
        channel2,
    )

    print()
    print("=" * 70)
    print("EVENT SUMMARY")
    print("=" * 70)
    print(f"Event: {event_index}")
    print(f"Samples: {len(time)}")
    print(
        f"Time range: "
        f"{time_microseconds[0]:.3f} to "
        f"{time_microseconds[-1]:.3f} μs"
    )
    print(
        f"Sample interval: "
        f"{mean_sample_interval_ps:.3f} ps"
    )
    print()
    print(f"CH1 baseline: {channel1_baseline:.6f} V")
    print(f"CH1 minimum:  {channel1.min():.6f} V")
    print(f"CH1 maximum:  {channel1.max():.6f} V")
    print()
    print(f"CH2 baseline: {channel2_baseline:.6f} V")
    print(f"CH2 minimum:  {channel2.min():.6f} V")
    print(f"CH2 maximum:  {channel2.max():.6f} V")
    print("=" * 70)
    print()


def main() -> None:
    """Select one ROOT file and one event, then create diagnostic plots."""

    root_files = find_root_files(ROOT_DATA_DIR)

    if not root_files:
        raise FileNotFoundError(
            f"No ROOT files were found in:\n{ROOT_DATA_DIR}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Available ROOT files:")
    print("-" * 70)

    for index, file_path in enumerate(root_files):
        size_mb = file_path.stat().st_size / (1024 ** 2)
        print(
            f"[{index}] {file_path.name} "
            f"({size_mb:.2f} MB)"
        )

    print()

    try:
        file_index = int(
            input("Select ROOT file index: ")
        )
        event_index = int(
            input("Select event index: ")
        )
    except ValueError as error:
        raise ValueError(
            "Both selections must be whole numbers."
        ) from error

    if file_index < 0 or file_index >= len(root_files):
        raise IndexError(
            f"ROOT file index must be between 0 and "
            f"{len(root_files) - 1}."
        )

    selected_file = root_files[file_index]

    time, channel1, channel2 = load_event(
        selected_file,
        event_index,
    )

    print_event_summary(
        event_index,
        time,
        channel1,
        channel2,
    )

    plot_full_event(
        selected_file,
        event_index,
        time,
        channel1,
        channel2,
    )

    plot_separate_channels(
        selected_file,
        event_index,
        time,
        channel1,
        channel2,
    )

    plot_post_trigger_zoom(
        selected_file,
        event_index,
        time,
        channel1,
        channel2,
    )


if __name__ == "__main__":
    main()