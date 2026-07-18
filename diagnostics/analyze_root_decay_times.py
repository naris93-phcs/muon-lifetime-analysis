from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "root_decay_selection"
    / "accepted_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "root_decay_selection"
)

HISTOGRAM_PATH = (
    OUTPUT_DIR
    / "accepted_decay_time_histogram.png"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "accepted_decay_time_summary.txt"
)


# Number of acquisition events in the ROOT file currently analyzed.
TOTAL_EVENTS_SCANNED = 2574

# Same lower time cut used by the selector.
DECAY_SEARCH_START_US = 0.80

# Upper edge of the waveform search region.
DECAY_SEARCH_END_US = 9.00

# Histogram bin width.
BIN_WIDTH_US = 0.25


def load_accepted_candidates(csv_path: Path) -> pd.DataFrame:
    """Load and validate accepted delayed-pulse candidates."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Accepted-candidate file was not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "event_index",
        "decay_time_us",
        "channel2_snr",
        "channel1_veto_snr",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The accepted-candidate CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=["decay_time_us"]
    ).copy()

    dataframe["decay_time_us"] = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["decay_time_us"]
    )

    dataframe = dataframe[
        (
            dataframe["decay_time_us"]
            >= DECAY_SEARCH_START_US
        )
        & (
            dataframe["decay_time_us"]
            <= DECAY_SEARCH_END_US
        )
    ].copy()

    dataframe = dataframe.sort_values(
        "decay_time_us"
    ).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError(
            "No valid accepted decay times remain after filtering."
        )

    return dataframe


def calculate_summary(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """Calculate basic statistics for accepted decay-like candidates."""

    decay_times = dataframe["decay_time_us"].to_numpy()

    accepted_candidates = len(dataframe)

    unique_accepted_events = dataframe[
        "event_index"
    ].nunique()

    acceptance_fraction = (
        unique_accepted_events / TOTAL_EVENTS_SCANNED
    )

    summary = {
        "total_events_scanned": TOTAL_EVENTS_SCANNED,
        "accepted_candidates": accepted_candidates,
        "unique_accepted_events": unique_accepted_events,
        "acceptance_fraction": acceptance_fraction,
        "acceptance_percent": acceptance_fraction * 100.0,
        "mean_us": float(np.mean(decay_times)),
        "median_us": float(np.median(decay_times)),
        "std_us": float(
            np.std(decay_times, ddof=1)
        ),
        "minimum_us": float(np.min(decay_times)),
        "maximum_us": float(np.max(decay_times)),
        "q25_us": float(
            np.percentile(decay_times, 25)
        ),
        "q75_us": float(
            np.percentile(decay_times, 75)
        ),
    }

    return summary


def format_summary(
    summary: dict[str, float],
) -> str:
    """Return a readable analysis summary."""

    lines = [
        "=" * 72,
        "ROOT DELAYED-PULSE ANALYSIS",
        "=" * 72,
        (
            "Events scanned             : "
            f"{summary['total_events_scanned']}"
        ),
        (
            "Accepted candidates        : "
            f"{summary['accepted_candidates']}"
        ),
        (
            "Unique accepted events     : "
            f"{summary['unique_accepted_events']}"
        ),
        (
            "Acceptance                 : "
            f"{summary['acceptance_percent']:.3f} %"
        ),
        "",
        "Accepted delay-time statistics",
        "-" * 72,
        (
            "Mean delay                 : "
            f"{summary['mean_us']:.4f} μs"
        ),
        (
            "Median delay               : "
            f"{summary['median_us']:.4f} μs"
        ),
        (
            "Standard deviation         : "
            f"{summary['std_us']:.4f} μs"
        ),
        (
            "Minimum delay              : "
            f"{summary['minimum_us']:.4f} μs"
        ),
        (
            "25th percentile            : "
            f"{summary['q25_us']:.4f} μs"
        ),
        (
            "75th percentile            : "
            f"{summary['q75_us']:.4f} μs"
        ),
        (
            "Maximum delay              : "
            f"{summary['maximum_us']:.4f} μs"
        ),
        "=" * 72,
    ]

    return "\n".join(lines)


def make_histogram(
    dataframe: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    """Create the first decay-time histogram without fitting."""

    decay_times = dataframe[
        "decay_time_us"
    ].to_numpy()

    bin_edges = np.arange(
        DECAY_SEARCH_START_US,
        DECAY_SEARCH_END_US
        + BIN_WIDTH_US,
        BIN_WIDTH_US,
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    counts, _, _ = axis.hist(
        decay_times,
        bins=bin_edges,
        edgecolor="black",
        linewidth=0.7,
        label=(
            f"Accepted candidates "
            f"(N = {len(decay_times)})"
        ),
    )

    axis.axvline(
        summary["mean_us"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Mean = "
            f"{summary['mean_us']:.3f} μs"
        ),
    )

    axis.axvline(
        summary["median_us"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Median = "
            f"{summary['median_us']:.3f} μs"
        ),
    )

    axis.set_xlabel(
        "Delayed-pulse time (μs)"
    )

    axis.set_ylabel(
        f"Candidates per {BIN_WIDTH_US:.2f} μs"
    )

    axis.set_title(
        "Accepted decay-like delayed pulses\n"
        "Physics-aware ROOT event selection"
    )

    axis.set_xlim(
        DECAY_SEARCH_START_US,
        DECAY_SEARCH_END_US,
    )

    axis.set_ylim(
        bottom=0,
        top=max(counts) * 1.15
        if len(counts) > 0
        else None,
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        HISTOGRAM_PATH,
        dpi=160,
    )

    print(
        f"\nSaved histogram:\n{HISTOGRAM_PATH}"
    )

    plt.show()
    plt.close(figure)


def main() -> None:
    """Analyze accepted ROOT delayed-pulse candidates."""

    print(
        f"Loading accepted candidates:\n{INPUT_CSV}\n"
    )

    dataframe = load_accepted_candidates(
        INPUT_CSV
    )

    summary = calculate_summary(
        dataframe
    )

    summary_text = format_summary(
        summary
    )

    print(summary_text)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        f"\nSaved summary:\n{SUMMARY_PATH}"
    )

    make_histogram(
        dataframe,
        summary,
    )


if __name__ == "__main__":
    main()