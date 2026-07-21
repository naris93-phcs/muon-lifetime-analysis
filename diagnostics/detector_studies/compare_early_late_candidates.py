from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "early_late_comparison"

SUMMARY_CSV = OUTPUT_DIR / "group_summary.csv"
FILE_COUNTS_CSV = OUTPUT_DIR / "file_counts.csv"


EARLY_MIN_US = 0.80
EARLY_MAX_US = 1.20

LATE_MIN_US = 2.00
LATE_MAX_US = 3.00


COMPARISON_COLUMNS = [
    "channel2_amplitude_v",
    "channel2_prominence_v",
    "channel2_width_ns",
    "channel2_snr",
    "channel1_veto_amplitude_v",
    "channel1_veto_snr",
    "channel1_to_channel2_ratio",
    "channel1_trigger_amplitude_v",
    "channel2_trigger_amplitude_v",
    "channel1_noise_std_v",
    "channel2_noise_std_v",
]


def load_candidates() -> pd.DataFrame:
    """Load accepted candidates from the full ROOT dataset."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV was not found:\n{INPUT_CSV}")

    dataframe = pd.read_csv(INPUT_CSV)

    required_columns = {
        "file",
        "event_index",
        "accepted",
        "decay_time_us",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError("Missing required columns: " f"{sorted(missing_columns)}")

    dataframe = dataframe[dataframe["accepted"] == True].copy()  # noqa: E712

    dataframe["decay_time_us"] = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(subset=["decay_time_us"])

    return dataframe


def select_groups(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select early and late accepted-candidate groups."""

    early = dataframe[
        (dataframe["decay_time_us"] >= EARLY_MIN_US)
        & (dataframe["decay_time_us"] < EARLY_MAX_US)
    ].copy()

    late = dataframe[
        (dataframe["decay_time_us"] >= LATE_MIN_US)
        & (dataframe["decay_time_us"] < LATE_MAX_US)
    ].copy()

    early["group"] = "early"
    late["group"] = "late"

    return early, late


def make_group_summary(
    early: pd.DataFrame,
    late: pd.DataFrame,
) -> pd.DataFrame:
    """Create numerical summaries for both time groups."""

    available_columns = [
        column
        for column in COMPARISON_COLUMNS
        if column in early.columns and column in late.columns
    ]

    rows = []

    for group_name, group_dataframe in [
        ("early", early),
        ("late", late),
    ]:
        for column in available_columns:
            values = pd.to_numeric(
                group_dataframe[column],
                errors="coerce",
            )

            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            rows.append(
                {
                    "group": group_name,
                    "variable": column,
                    "events": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )

    return pd.DataFrame(rows)


def make_file_counts(
    early: pd.DataFrame,
    late: pd.DataFrame,
) -> pd.DataFrame:
    """Count early and late candidates in each ROOT file."""

    early_counts = early.groupby("file").size().rename("early_events")

    late_counts = late.groupby("file").size().rename("late_events")

    counts = pd.concat(
        [early_counts, late_counts],
        axis=1,
    ).fillna(0)

    counts["early_events"] = counts["early_events"].astype(int)

    counts["late_events"] = counts["late_events"].astype(int)

    counts["total_selected_events"] = counts["early_events"] + counts["late_events"]

    counts["early_fraction"] = np.where(
        counts["total_selected_events"] > 0,
        counts["early_events"] / counts["total_selected_events"],
        np.nan,
    )

    counts = counts.sort_values(
        "early_fraction",
        ascending=False,
    )

    return counts.reset_index()


def plot_variable_comparison(
    early: pd.DataFrame,
    late: pd.DataFrame,
    column: str,
) -> None:
    """Plot normalized early and late distributions."""

    early_values = pd.to_numeric(
        early[column],
        errors="coerce",
    ).to_numpy()

    late_values = pd.to_numeric(
        late[column],
        errors="coerce",
    ).to_numpy()

    early_values = early_values[np.isfinite(early_values)]

    late_values = late_values[np.isfinite(late_values)]

    if len(early_values) == 0 or len(late_values) == 0:
        return

    combined = np.concatenate([early_values, late_values])

    lower_limit = np.quantile(
        combined,
        0.01,
    )

    upper_limit = np.quantile(
        combined,
        0.99,
    )

    if lower_limit >= upper_limit:
        return

    bins = np.linspace(
        lower_limit,
        upper_limit,
        41,
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        early_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=(f"Early: {EARLY_MIN_US:.2f}–" f"{EARLY_MAX_US:.2f} μs"),
    )

    axis.hist(
        late_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=(f"Late: {LATE_MIN_US:.2f}–" f"{LATE_MAX_US:.2f} μs"),
    )

    axis.set_xlabel(column.replace("_", " "))

    axis.set_ylabel("Normalized density")

    axis.set_title(f"Early versus late candidates\n{column}")

    axis.grid(alpha=0.3)

    axis.legend()

    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{column}_comparison.png"

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def plot_file_fractions(
    file_counts: pd.DataFrame,
) -> None:
    """Plot the early-event fraction for each ROOT file."""

    if file_counts.empty:
        return

    figure, axis = plt.subplots(figsize=(13, 7))

    axis.bar(
        file_counts["file"],
        file_counts["early_fraction"],
    )

    axis.set_xlabel("ROOT file")

    axis.set_ylabel("Early fraction within early + late samples")

    axis.set_title("Early-candidate fraction by ROOT file")

    axis.tick_params(
        axis="x",
        rotation=75,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "early_fraction_by_file.png",
        dpi=160,
    )

    plt.close(figure)


def print_summary(
    early: pd.DataFrame,
    late: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print the main comparison results."""

    print()
    print("=" * 72)
    print("EARLY–LATE ACCEPTED-CANDIDATE COMPARISON")
    print("=" * 72)

    print(f"Early interval: " f"{EARLY_MIN_US:.2f}–{EARLY_MAX_US:.2f} μs")

    print(f"Early events: {len(early)}")

    print(f"Late interval: " f"{LATE_MIN_US:.2f}–{LATE_MAX_US:.2f} μs")

    print(f"Late events: {len(late)}")

    print("-" * 72)

    pivot = summary.pivot(
        index="variable",
        columns="group",
        values="median",
    )

    if {
        "early",
        "late",
    }.issubset(pivot.columns):
        pivot["early_to_late_median_ratio"] = pivot["early"] / pivot["late"]

    print("\nMedian comparison:\n")
    print(pivot.to_string(float_format=lambda value: f"{value:.5g}"))

    print("=" * 72)


def main() -> None:
    """Run the early-versus-late comparison."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates()

    early, late = select_groups(candidates)

    if early.empty:
        raise ValueError("No early events were found.")

    if late.empty:
        raise ValueError("No late events were found.")

    summary = make_group_summary(
        early,
        late,
    )

    file_counts = make_file_counts(
        early,
        late,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    file_counts.to_csv(
        FILE_COUNTS_CSV,
        index=False,
    )

    print_summary(
        early,
        late,
        summary,
    )

    for column in COMPARISON_COLUMNS:
        if column in early.columns and column in late.columns:
            plot_variable_comparison(
                early,
                late,
                column,
            )

    plot_file_fractions(file_counts)

    print(f"\nSaved comparison results in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
