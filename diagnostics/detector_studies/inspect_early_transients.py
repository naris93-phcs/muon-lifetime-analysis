from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

ROOT_DIR = PROJECT_ROOT / "data" / "root"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "early_transient_review"


EARLY_MIN_US = 0.80
EARLY_MAX_US = 1.20

ZOOM_MIN_US = 0.30
ZOOM_MAX_US = 1.50

SEARCH_CUT_US = 0.80

NUMBER_OF_EVENTS = 30
RANDOM_SEED = 42


def find_tree(root_file: uproot.ReadOnlyDirectory):
    """Find the first TTree containing the expected branches."""

    required_branches = {
        "time",
        "channel1",
        "channel2",
    }

    for key in root_file.keys():
        object_name = key.split(";")[0]

        try:
            candidate = root_file[object_name]
        except Exception:
            continue

        if not hasattr(candidate, "keys"):
            continue

        branch_names = set(candidate.keys())

        if required_branches.issubset(branch_names):
            return candidate

    raise ValueError("No compatible TTree with time, channel1 and channel2 was found.")


def load_early_candidates() -> pd.DataFrame:
    """Load a random sample of early accepted candidates."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Candidate CSV not found:\n{INPUT_CSV}")

    dataframe = pd.read_csv(INPUT_CSV)

    dataframe = dataframe[dataframe["accepted"].fillna(False)].copy()

    dataframe["decay_time_us"] = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    )

    early = dataframe[
        (dataframe["decay_time_us"] >= EARLY_MIN_US)
        & (dataframe["decay_time_us"] < EARLY_MAX_US)
    ].copy()

    early = early.dropna(
        subset=[
            "file",
            "event_index",
            "decay_time_us",
        ]
    )

    if early.empty:
        raise ValueError("No early accepted candidates were found.")

    sample_size = min(
        NUMBER_OF_EVENTS,
        len(early),
    )

    return early.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    )


def load_event(
    root_path: Path,
    event_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, CH1 and CH2 arrays for one event."""

    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found:\n{root_path}")

    with uproot.open(root_path) as root_file:
        tree = find_tree(root_file)

        arrays = tree.arrays(
            [
                "time",
                "channel1",
                "channel2",
            ],
            entry_start=event_index,
            entry_stop=event_index + 1,
            library="np",
        )

    time = np.asarray(
        arrays["time"][0],
        dtype=float,
    )

    channel1 = np.asarray(
        arrays["channel1"][0],
        dtype=float,
    )

    channel2 = np.asarray(
        arrays["channel2"][0],
        dtype=float,
    )

    minimum_length = min(
        len(time),
        len(channel1),
        len(channel2),
    )

    if minimum_length == 0:
        raise ValueError("One or more event arrays are empty.")

    time = time[:minimum_length]
    channel1 = channel1[:minimum_length]
    channel2 = channel2[:minimum_length]

    return time, channel1, channel2


def convert_time_to_microseconds(
    time: np.ndarray,
) -> np.ndarray:
    """
    Convert time to microseconds.

    ROOT files normally store time in seconds.
    """

    maximum_absolute_time = np.max(np.abs(time))

    if maximum_absolute_time < 1e-2:
        return time * 1e6

    return time


def find_local_extrema(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> tuple[float, float]:
    """Find the strongest negative excursion in the 0.3–0.8 μs region."""

    mask = (time_us >= ZOOM_MIN_US) & (time_us < SEARCH_CUT_US)

    if not np.any(mask):
        return float("nan"), float("nan")

    local_time = time_us[mask]
    local_signal = signal[mask]

    index = np.argmin(local_signal)

    return (
        float(local_time[index]),
        float(local_signal[index]),
    )


def plot_event(
    row: pd.Series,
    time_us: np.ndarray,
    channel1: np.ndarray,
    channel2: np.ndarray,
) -> None:
    """Plot one early event with candidate and transient markers."""

    file_name = str(row["file"])
    event_index = int(row["event_index"])
    decay_time_us = float(row["decay_time_us"])

    zoom_mask = (time_us >= ZOOM_MIN_US) & (time_us <= ZOOM_MAX_US)

    if not np.any(zoom_mask):
        raise ValueError("No samples were found in the requested zoom range.")

    transient_time, transient_value = find_local_extrema(
        time_us,
        channel2,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].plot(
        time_us[zoom_mask],
        channel1[zoom_mask],
        linewidth=1,
    )

    axes[0].axvline(
        SEARCH_CUT_US,
        linestyle="--",
        label="Delayed-search cut",
    )

    axes[0].axvline(
        decay_time_us,
        linestyle=":",
        linewidth=2,
        label="Accepted candidate",
    )

    axes[0].set_ylabel("CH1 voltage (V)")

    axes[0].grid(alpha=0.3)

    axes[0].legend()

    axes[1].plot(
        time_us[zoom_mask],
        channel2[zoom_mask],
        linewidth=1,
    )

    axes[1].axvline(
        SEARCH_CUT_US,
        linestyle="--",
        label="Delayed-search cut",
    )

    axes[1].axvline(
        decay_time_us,
        linestyle=":",
        linewidth=2,
        label=(f"Candidate: {decay_time_us:.3f} μs"),
    )

    if np.isfinite(transient_time):
        axes[1].scatter(
            [transient_time],
            [transient_value],
            marker="x",
            s=80,
            label=(f"Pre-cut minimum: " f"{transient_time:.3f} μs"),
        )

    axes[1].set_xlabel("Time after trigger (μs)")

    axes[1].set_ylabel("CH2 voltage (V)")

    axes[1].grid(alpha=0.3)

    axes[1].legend()

    figure.suptitle(f"{file_name} | event {event_index}\n" f"Early accepted candidate")

    figure.tight_layout()

    output_name = (
        f"{Path(file_name).stem}"
        f"_event_{event_index}"
        f"_time_{decay_time_us:.3f}us.png"
    )

    figure.savefig(
        OUTPUT_DIR / output_name,
        dpi=160,
    )

    plt.close(figure)


def main() -> None:
    """Inspect early candidate waveforms."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_early_candidates()

    print()
    print("=" * 72)
    print("EARLY TRANSIENT REVIEW")
    print("=" * 72)
    print(f"Candidates selected: {len(candidates)}")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print("-" * 72)

    successful = 0
    failed = 0

    for _, row in candidates.iterrows():
        file_name = str(row["file"])
        event_index = int(row["event_index"])

        root_path = ROOT_DIR / file_name

        try:
            time, channel1, channel2 = load_event(
                root_path=root_path,
                event_index=event_index,
            )

            time_us = convert_time_to_microseconds(time)

            plot_event(
                row=row,
                time_us=time_us,
                channel1=channel1,
                channel2=channel2,
            )

            successful += 1

            print(f"Saved: {file_name}, event {event_index}")

        except Exception as error:
            failed += 1

            print(f"FAILED: {file_name}, event {event_index}")
            print(f"Reason: {error}")

    print("-" * 72)
    print(f"Successful plots: {successful}")
    print(f"Failed events: {failed}")
    print("=" * 72)


if __name__ == "__main__":
    main()
