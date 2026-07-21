from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    calculate_summary,
    estimate_truncated_exponential_lifetime,
)
from src.config import (
    ACCEPTED_CANDIDATES_PATH,
    FIT_MAX_US,
    RESULTS_DIR,
)

VALIDATION_DIR = RESULTS_DIR / "validation"

PER_FILE_STABILITY_PATH = VALIDATION_DIR / "per_file_fit_range_stability.csv"

TMIN_150_SUMMARY_PATH = VALIDATION_DIR / "per_file_tau_at_tmin_1_50.csv"

REFERENCE_FILE = "CH_t12_141223_1.root"

REFERENCE_FILE_TABLE_PATH = VALIDATION_DIR / "CH_t12_141223_1_validation.csv"

REFERENCE_FILE_PLOT_PATH = VALIDATION_DIR / "CH_t12_141223_1_validation.png"

T_MIN_VALUES_US = (
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
)


def lower_cut_mle(
    lifetimes_us: np.ndarray,
    t_min_us: float,
) -> tuple[float, float, int]:
    """
    Calculate the traditional exponential MLE with only
    a lower time threshold.

    For an exponential distribution beginning at t_min:

        tau_hat = mean(t - t_min)

    The approximate statistical uncertainty is:

        sigma_tau = tau_hat / sqrt(N)
    """

    lifetimes_us = np.asarray(
        lifetimes_us,
        dtype=float,
    )

    selected = lifetimes_us[np.isfinite(lifetimes_us) & (lifetimes_us >= t_min_us)]

    event_count = selected.size

    if event_count == 0:
        return np.nan, np.nan, 0

    tau_mle_us = float(np.mean(selected - t_min_us))

    tau_error_us = float(tau_mle_us / np.sqrt(event_count))

    return (
        tau_mle_us,
        tau_error_us,
        int(event_count),
    )


def read_accepted_candidates() -> pd.DataFrame:
    """
    Load and validate the accepted-candidate table.
    """

    if not ACCEPTED_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            "Accepted-candidate CSV was not found:\n"
            f"{ACCEPTED_CANDIDATES_PATH}\n\n"
            "Run main.py first."
        )

    dataframe = pd.read_csv(ACCEPTED_CANDIDATES_PATH)

    required_columns = {
        "source_file",
        "decay_time_us",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Accepted-candidate CSV is missing " f"columns: {sorted(missing_columns)}"
        )

    dataframe["decay_time_us"] = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "source_file",
            "decay_time_us",
        ]
    )

    return dataframe


def calculate_per_file_stability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate lifetime estimates for every ROOT file
    and lower fit threshold.
    """

    rows: list[dict] = []

    grouped_files = dataframe.groupby(
        "source_file",
        sort=True,
    )

    for source_file, file_dataframe in grouped_files:
        lifetimes_us = file_dataframe["decay_time_us"].to_numpy(dtype=float)

        for t_min_us in T_MIN_VALUES_US:
            truncated_summary = calculate_summary(
                lifetimes_us=lifetimes_us,
                t_min_us=t_min_us,
                t_max_us=FIT_MAX_US,
            )

            (
                lower_tau_us,
                lower_error_us,
                lower_event_count,
            ) = lower_cut_mle(
                lifetimes_us=lifetimes_us,
                t_min_us=t_min_us,
            )

            rows.append(
                {
                    "file": source_file,
                    "t_min_us": t_min_us,
                    "t_max_us": FIT_MAX_US,
                    "truncated_events": (truncated_summary["events_used"]),
                    "truncated_tau_us": (truncated_summary["tau_mle_us"]),
                    "truncated_error_us": (truncated_summary["tau_error_us"]),
                    "lower_cut_events": (lower_event_count),
                    "lower_cut_tau_us": (lower_tau_us),
                    "lower_cut_error_us": (lower_error_us),
                }
            )

    return pd.DataFrame(rows)


def classify_run(
    row: pd.Series,
) -> str:
    """
    Assign a descriptive run-quality category.

    These labels are diagnostic only. They do not
    automatically remove data from the analysis.
    """

    events = int(row["truncated_events"])
    tau_us = float(row["truncated_tau_us"])
    error_us = float(row["truncated_error_us"])

    if events < 50:
        return "low_statistics"

    if not np.isfinite(tau_us):
        return "fit_failed"

    if error_us > 0.40:
        return "large_uncertainty"

    if tau_us < 1.30:
        return "early_time_excess"

    if 1.70 <= tau_us <= 2.50:
        return "reference_compatible_region"

    return "intermediate"


def make_reference_file_plot(
    validation_dataframe: pd.DataFrame,
) -> None:
    """
    Compare the old lower-cut estimator with the new
    finite-range truncated estimator.
    """

    reference_rows = validation_dataframe[
        validation_dataframe["file"] == REFERENCE_FILE
    ].copy()

    if reference_rows.empty:
        print(
            f"WARNING: {REFERENCE_FILE} was not " "found in the accepted-candidate CSV."
        )
        return

    reference_rows = reference_rows.sort_values("t_min_us")

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.errorbar(
        reference_rows["t_min_us"],
        reference_rows["lower_cut_tau_us"],
        yerr=(reference_rows["lower_cut_error_us"]),
        marker="o",
        linestyle="--",
        capsize=4,
        label="Lower-cut MLE",
    )

    axis.errorbar(
        reference_rows["t_min_us"],
        reference_rows["truncated_tau_us"],
        yerr=(reference_rows["truncated_error_us"]),
        marker="s",
        linestyle="-",
        capsize=4,
        label=(f"Truncated MLE " f"(tmax = {FIT_MAX_US:.1f} μs)"),
    )

    axis.axhline(
        2.197,
        linestyle=":",
        label=("Reference free-muon lifetime " "(2.197 μs)"),
    )

    axis.set_title("Same-File Lifetime Validation\n" f"{REFERENCE_FILE}")

    axis.set_xlabel("Lower fit threshold " r"$t_{\min}$ (μs)")

    axis.set_ylabel("Estimated lifetime τ (μs)")

    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        REFERENCE_FILE_PLOT_PATH,
        dpi=160,
    )

    plt.close(figure)


def print_reference_validation(
    validation_dataframe: pd.DataFrame,
) -> None:
    """
    Print the validation table for the historical
    diagnostic ROOT file.
    """

    reference_rows = validation_dataframe[
        validation_dataframe["file"] == REFERENCE_FILE
    ].copy()

    if reference_rows.empty:
        return

    print()
    print("=" * 78)
    print("Same-file validation")
    print("=" * 78)
    print(f"ROOT file: {REFERENCE_FILE}")
    print()

    print(
        reference_rows.to_string(
            index=False,
            columns=[
                "t_min_us",
                "truncated_events",
                "truncated_tau_us",
                "truncated_error_us",
                "lower_cut_events",
                "lower_cut_tau_us",
                "lower_cut_error_us",
            ],
            formatters={
                "t_min_us": "{:.2f}".format,
                "truncated_tau_us": ("{:.4f}".format),
                "truncated_error_us": ("{:.4f}".format),
                "lower_cut_tau_us": ("{:.4f}".format),
                "lower_cut_error_us": ("{:.4f}".format),
            },
        )
    )

    tmin_150_rows = reference_rows[
        np.isclose(
            reference_rows["t_min_us"],
            1.50,
        )
    ]

    if not tmin_150_rows.empty:
        row = tmin_150_rows.iloc[0]

        print()
        print("Historical comparison at tmin = 1.50 μs")
        print("-----------------------------------------")
        print("Old diagnostic result reported: " "approximately 2.079 μs")
        print(
            "Current lower-cut MLE         : "
            f"{row['lower_cut_tau_us']:.4f} "
            f"± {row['lower_cut_error_us']:.4f} μs"
        )
        print(
            "Current truncated MLE         : "
            f"{row['truncated_tau_us']:.4f} "
            f"± {row['truncated_error_us']:.4f} μs"
        )


def main() -> None:
    """
    Run the final per-file and same-file validation.
    """

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print("Muon Run Validation")
    print("=" * 78)
    print(f"Reading accepted candidates from:\n" f"{ACCEPTED_CANDIDATES_PATH}")

    accepted_dataframe = read_accepted_candidates()

    print(f"Accepted candidates loaded: " f"{len(accepted_dataframe)}")

    print(
        f"ROOT files represented     : "
        f"{accepted_dataframe['source_file'].nunique()}"
    )

    validation_dataframe = calculate_per_file_stability(accepted_dataframe)

    validation_dataframe.to_csv(
        PER_FILE_STABILITY_PATH,
        index=False,
    )

    reference_dataframe = validation_dataframe[
        validation_dataframe["file"] == REFERENCE_FILE
    ].copy()

    reference_dataframe.to_csv(
        REFERENCE_FILE_TABLE_PATH,
        index=False,
    )

    tmin_150_dataframe = validation_dataframe[
        np.isclose(
            validation_dataframe["t_min_us"],
            1.50,
        )
    ].copy()

    tmin_150_dataframe["quality_flag"] = tmin_150_dataframe.apply(
        classify_run,
        axis=1,
    )

    tmin_150_dataframe = tmin_150_dataframe.sort_values(
        [
            "quality_flag",
            "truncated_tau_us",
        ]
    )

    tmin_150_dataframe.to_csv(
        TMIN_150_SUMMARY_PATH,
        index=False,
    )

    make_reference_file_plot(validation_dataframe)

    print_reference_validation(validation_dataframe)

    print()
    print("=" * 78)
    print("Per-file results at tmin = 1.50 μs")
    print("=" * 78)

    print(
        tmin_150_dataframe.to_string(
            index=False,
            columns=[
                "file",
                "truncated_events",
                "truncated_tau_us",
                "truncated_error_us",
                "lower_cut_tau_us",
                "quality_flag",
            ],
            formatters={
                "truncated_tau_us": ("{:.4f}".format),
                "truncated_error_us": ("{:.4f}".format),
                "lower_cut_tau_us": ("{:.4f}".format),
            },
        )
    )

    print()
    print("=" * 78)
    print("Validation finished")
    print("=" * 78)
    print(f"Full per-file table : " f"{PER_FILE_STABILITY_PATH}")
    print(f"Reference table     : " f"{REFERENCE_FILE_TABLE_PATH}")
    print(f"Reference plot      : " f"{REFERENCE_FILE_PLOT_PATH}")
    print(f"tmin = 1.50 table  : " f"{TMIN_150_SUMMARY_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
