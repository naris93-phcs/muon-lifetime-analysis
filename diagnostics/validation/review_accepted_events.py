from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

# -------------------------------------------------------
# Project paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_FILE = PROJECT_ROOT / "data" / "root" / "CH_t12_141223_1.root"

CSV_FILE = PROJECT_ROOT / "results" / "root_decay_selection" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_decay_selection" / "contact_sheets"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------------
# Plot configuration
# -------------------------------------------------------

EVENTS_PER_PAGE = 30
COLUMNS = 5
ROWS = 6

PLOT_START_US = 0.15
PLOT_END_US = 9.00


def load_accepted_candidates(
    csv_path: Path,
) -> pd.DataFrame:
    """Load accepted candidate information from CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Accepted candidates CSV was not found:\n" f"{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "event_index",
        "decay_time_us",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The accepted-candidate CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "event_index",
            "decay_time_us",
        ]
    ).copy()

    dataframe["event_index"] = pd.to_numeric(
        dataframe["event_index"],
        errors="coerce",
    )

    dataframe["decay_time_us"] = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "event_index",
            "decay_time_us",
        ]
    )

    dataframe["event_index"] = dataframe["event_index"].astype(int)

    dataframe = dataframe.sort_values("decay_time_us").reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("No accepted candidates were found.")

    return dataframe


def baseline_subtract_and_invert(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """
    Estimate the pre-trigger baseline, subtract it,
    and invert negative detector pulses.
    """

    baseline_mask = time_us < -0.10

    if not np.any(baseline_mask):
        baseline = float(np.median(signal))
    else:
        baseline = float(np.median(signal[baseline_mask]))

    return -(signal - baseline)


def load_event(
    tree,
    event_index: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load one waveform event from the ROOT tree."""

    arrays = tree.arrays(
        [
            "time",
            "channel1",
            "channel2",
        ],
        entry_start=event_index,
        entry_stop=event_index + 1,
        library="ak",
    )

    if len(arrays["time"]) == 0:
        raise IndexError(f"Event {event_index} was not found.")

    time = ak.to_numpy(arrays["time"][0])

    channel1 = ak.to_numpy(arrays["channel1"][0])

    channel2 = ak.to_numpy(arrays["channel2"][0])

    return time, channel1, channel2


def create_contact_sheets(
    accepted: pd.DataFrame,
    root_file_path: Path,
) -> None:
    """Create multi-event review pages for accepted candidates."""

    if not root_file_path.exists():
        raise FileNotFoundError(f"ROOT file was not found:\n" f"{root_file_path}")

    number_of_pages = int(np.ceil(len(accepted) / EVENTS_PER_PAGE))

    with uproot.open(root_file_path) as root_file:

        tree = root_file["t1"]

        total_root_events = tree.num_entries

        print(f"ROOT file: {root_file_path.name}")
        print(f"ROOT events: {total_root_events}")
        print(f"Accepted candidates: {len(accepted)}")
        print(f"Contact-sheet pages: {number_of_pages}")
        print()

        for page_index in range(number_of_pages):
            figure, axes = plt.subplots(
                ROWS,
                COLUMNS,
                figsize=(20, 22),
                sharex=True,
            )

            axes = axes.flatten()

            page_start = page_index * EVENTS_PER_PAGE

            page_stop = min(
                page_start + EVENTS_PER_PAGE,
                len(accepted),
            )

            page_candidates = accepted.iloc[page_start:page_stop]

            for axis in axes:
                axis.axis("off")

            for (
                axis,
                (_, row),
            ) in zip(
                axes,
                page_candidates.iterrows(),
            ):
                event_index = int(row["event_index"])

                decay_time_us = float(row["decay_time_us"])

                if event_index < 0 or event_index >= total_root_events:
                    axis.axis("on")

                    axis.text(
                        0.5,
                        0.5,
                        (f"Invalid event\n" f"{event_index}"),
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )

                    continue

                try:
                    (
                        time,
                        channel1,
                        channel2,
                    ) = load_event(
                        tree,
                        event_index,
                    )
                except Exception as error:
                    axis.axis("on")

                    axis.text(
                        0.5,
                        0.5,
                        (f"Load error\n" f"Event {event_index}\n" f"{error}"),
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )

                    continue

                time_us = time * 1e6

                channel1_processed = baseline_subtract_and_invert(
                    time_us,
                    channel1,
                )

                channel2_processed = baseline_subtract_and_invert(
                    time_us,
                    channel2,
                )

                plot_mask = (time_us >= PLOT_START_US) & (time_us <= PLOT_END_US)

                axis.axis("on")

                axis.plot(
                    time_us[plot_mask],
                    channel1_processed[plot_mask],
                    linewidth=0.7,
                    label="CH1",
                )

                axis.plot(
                    time_us[plot_mask],
                    channel2_processed[plot_mask],
                    linewidth=0.7,
                    alpha=0.85,
                    label="CH2",
                )

                axis.axvline(
                    decay_time_us,
                    linestyle="--",
                    linewidth=1.1,
                )

                candidate_mask = np.abs(time_us - decay_time_us) <= 0.02

                if np.any(candidate_mask):
                    local_indices = np.where(candidate_mask)[0]

                    peak_index = local_indices[
                        np.argmax(channel2_processed[local_indices])
                    ]

                    axis.scatter(
                        time_us[peak_index],
                        channel2_processed[peak_index],
                        s=18,
                        zorder=4,
                    )

                axis.set_title(
                    (f"Event {event_index}\n" f"t = " f"{decay_time_us:.3f} μs"),
                    fontsize=9,
                )

                axis.set_xlim(
                    PLOT_START_US,
                    PLOT_END_US,
                )

                axis.grid(alpha=0.2)

                axis.tick_params(labelsize=7)

            figure.suptitle(
                (
                    "Accepted delayed-pulse candidates\n"
                    f"Page "
                    f"{page_index + 1}"
                    f"/{number_of_pages}"
                ),
                fontsize=18,
            )

            figure.supxlabel("Time after trigger (μs)")

            figure.supylabel("Baseline-subtracted inverted voltage (V)")

            figure.tight_layout(
                rect=[
                    0.03,
                    0.03,
                    1.00,
                    0.96,
                ]
            )

            output_path = OUTPUT_DIR / (f"accepted_page_" f"{page_index + 1:02d}.png")

            figure.savefig(
                output_path,
                dpi=180,
            )

            plt.close(figure)

            print(
                f"Saved page "
                f"{page_index + 1}"
                f"/{number_of_pages}: "
                f"{output_path.name}"
            )


def main() -> None:
    """Generate accepted-event contact sheets."""

    accepted = load_accepted_candidates(CSV_FILE)

    create_contact_sheets(
        accepted=accepted,
        root_file_path=ROOT_FILE,
    )

    print()
    print("=" * 72)
    print("CONTACT SHEETS COMPLETE")
    print("=" * 72)
    print(f"Saved to:\n{OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
