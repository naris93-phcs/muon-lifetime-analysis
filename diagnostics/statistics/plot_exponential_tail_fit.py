from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both direct execution and python -m diagnostics.statistics... .
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (  # noqa: E402 - direct-script path setup above
    estimate_truncated_exponential_lifetime,
    select_fit_range,
)
from src.config import (  # noqa: E402 - direct-script path setup above
    ACCEPTED_CANDIDATES_PATH,
    FIT_MAX_US,
    RESULTS_DIR,
    T_MIN_US,
)

INPUT_FILE = ACCEPTED_CANDIDATES_PATH
OUTPUT_DIR = RESULTS_DIR / "diagnostics"

OUTPUT_FILE = OUTPUT_DIR / "exponential_tail_fit.png"


TIME_COLUMN = "decay_time_us"

SEARCH_CUTOFF_US = T_MIN_US
FIT_START_US = 1.5
N_BINS = 80


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Accepted-candidates file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if TIME_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{TIME_COLUMN}' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    decay_times = pd.to_numeric(
        df[TIME_COLUMN],
        errors="coerce",
    ).dropna()

    decay_times = select_fit_range(
        decay_times.to_numpy(), SEARCH_CUTOFF_US, FIT_MAX_US
    )

    if decay_times.size == 0:
        raise ValueError("No valid decay times were found.")

    # Events used to determine the exponential tail.
    tail_times = decay_times[decay_times >= FIT_START_US]

    if tail_times.size == 0:
        raise ValueError(f"No events were found above {FIT_START_US} μs.")

    # Use the same finite-window likelihood as the production analysis.
    # mean(t - t_min) is biased downward when an upper cut is present.
    tau_mle_us, tau_uncertainty_us = estimate_truncated_exponential_lifetime(
        tail_times, FIT_START_US, FIT_MAX_US
    )

    histogram_counts, bin_edges = np.histogram(
        decay_times,
        bins=N_BINS,
        range=(SEARCH_CUTOFF_US, FIT_MAX_US),
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    histogram_max_us = FIT_MAX_US  # Acquisition/analysis cut, not sample maximum.

    # Normalize the fitted exponential to the number of observed
    # tail events within the histogram range.
    #
    # Integral from FIT_START_US to histogram_max_us:
    #
    # A * tau * [1 - exp(-(t_max - t0) / tau)]
    #
    # This is set equal to the observed number of tail events.
    normalization = tail_times.size / (
        tau_mle_us * (1.0 - np.exp(-(histogram_max_us - FIT_START_US) / tau_mle_us))
    )

    # Expected number of events in each histogram bin.
    expected_counts = (
        normalization
        * tau_mle_us
        * (
            np.exp(-(bin_edges[:-1] - FIT_START_US) / tau_mle_us)
            - np.exp(-(bin_edges[1:] - FIT_START_US) / tau_mle_us)
        )
    )

    fit_region = bin_centers >= FIT_START_US

    extrapolation_region = (bin_centers >= SEARCH_CUTOFF_US) & (
        bin_centers < FIT_START_US
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 7))

    plt.hist(
        decay_times,
        bins=bin_edges,
        edgecolor="black",
        alpha=0.75,
        label="Accepted candidates",
    )

    # Solid curve: region actually used for the tail fit.
    plt.plot(
        bin_centers[fit_region],
        expected_counts[fit_region],
        linewidth=2.5,
        label=(
            "Exponential tail model\n"
            f"${FIT_START_US:.1f} \\leq t \\leq {FIT_MAX_US:.1f}$ μs, "
            f"$\\tau={tau_mle_us:.3f}"
            f"\\pm{tau_uncertainty_us:.3f}$ μs"
        ),
    )

    # Dashed curve: backward extrapolation, not part of the fit.
    plt.plot(
        bin_centers[extrapolation_region],
        expected_counts[extrapolation_region],
        linestyle="--",
        linewidth=2.5,
        label="Backward extrapolation of tail model",
    )

    plt.axvline(
        SEARCH_CUTOFF_US,
        linestyle="--",
        linewidth=1.8,
        label=f"Search cutoff: {SEARCH_CUTOFF_US:.1f} μs",
    )

    plt.axvline(
        FIT_START_US,
        linestyle=":",
        linewidth=2.2,
        label=f"Tail-fit start: {FIT_START_US:.1f} μs",
    )

    plt.yscale("log")

    plt.xlabel("Accepted decay time (μs)")
    plt.ylabel("Number of events per bin")
    plt.title("Accepted Muon Decay Times with Exponential Tail Model")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
    )

    plt.show()

    print()
    print("Exponential tail-fit diagnostic")
    print("--------------------------------")
    print(f"All accepted events: {decay_times.size}")
    print(
        f"Tail events ({FIT_START_US:.1f} <= t <= {FIT_MAX_US:.1f} μs): "
        f"{tail_times.size}"
    )
    print(f"tau_MLE = {tau_mle_us:.4f} " f"± {tau_uncertainty_us:.4f} μs")
    print(f"Histogram bins: {N_BINS}")
    print(f"Plot saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
