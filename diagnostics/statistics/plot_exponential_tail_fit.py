from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "diagnostics"

OUTPUT_FILE = OUTPUT_DIR / "exponential_tail_fit.png"


TIME_COLUMN = "decay_time_us"

SEARCH_CUTOFF_US = 0.8
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

    decay_times = decay_times[decay_times >= SEARCH_CUTOFF_US].to_numpy()

    if decay_times.size == 0:
        raise ValueError("No valid decay times were found.")

    # Events used to determine the exponential tail.
    tail_times = decay_times[decay_times >= FIT_START_US]

    if tail_times.size == 0:
        raise ValueError(f"No events were found above {FIT_START_US} μs.")

    # Lower-truncated exponential MLE:
    # tau_hat = mean(t - t_min)
    tau_mle_us = np.mean(tail_times - FIT_START_US)

    tau_uncertainty_us = tau_mle_us / np.sqrt(tail_times.size)

    histogram_counts, bin_edges = np.histogram(
        decay_times,
        bins=N_BINS,
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    histogram_max_us = bin_edges[-1]

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
            f"$t \\geq {FIT_START_US:.1f}$ μs, "
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
    print(f"Tail events (t >= {FIT_START_US:.1f} μs): " f"{tail_times.size}")
    print(f"tau_MLE = {tau_mle_us:.4f} " f"± {tau_uncertainty_us:.4f} μs")
    print(f"Histogram bins: {N_BINS}")
    print(f"Plot saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
