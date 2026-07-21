from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

TIME_COLUMN = "decay_time_us"

SEARCH_CUTOFF_US = 0.8
FIT_START_US = 1.5


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Accepted-candidates file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if TIME_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{TIME_COLUMN}' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    decay_times = (
        pd.to_numeric(
            df[TIME_COLUMN],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    decay_times = decay_times[decay_times >= SEARCH_CUTOFF_US]

    if decay_times.size == 0:
        raise ValueError("No valid decay times were found.")

    early_mask = (decay_times >= SEARCH_CUTOFF_US) & (decay_times < FIT_START_US)

    tail_mask = decay_times >= FIT_START_US

    early_times = decay_times[early_mask]
    tail_times = decay_times[tail_mask]

    if tail_times.size == 0:
        raise ValueError(f"No tail events were found above {FIT_START_US} μs.")

    # Truncated exponential MLE for the tail:
    #
    # tau_hat = mean(t - t_min)
    tau_mle_us = np.mean(tail_times - FIT_START_US)

    tau_uncertainty_us = tau_mle_us / np.sqrt(tail_times.size)

    observed_early = early_times.size
    observed_tail = tail_times.size

    # For a truncated exponential fitted at FIT_START_US,
    # the expected early/tail ratio is obtained by extending
    # the same exponential backward into:
    #
    # SEARCH_CUTOFF_US <= t < FIT_START_US
    #
    # Early integral:
    #
    # ∫ exp[-(t - t0)/tau] dt
    #
    # Tail integral:
    #
    # ∫ from t0 to infinity exp[-(t - t0)/tau] dt = tau
    #
    delta_us = FIT_START_US - SEARCH_CUTOFF_US

    expected_early_to_tail_ratio = np.exp(delta_us / tau_mle_us) - 1.0

    expected_early = observed_tail * expected_early_to_tail_ratio

    excess_events = observed_early - expected_early

    observed_to_expected_ratio = observed_early / expected_early

    excess_fraction_of_observed = excess_events / observed_early

    # Simple counting significance using Poisson uncertainty
    # on the expected count only.
    #
    # This is a first diagnostic estimate, not a full profile-
    # likelihood significance.
    poisson_sigma = np.sqrt(expected_early)

    significance_sigma = excess_events / poisson_sigma

    print()
    print("Early-event excess diagnostic")
    print("-----------------------------")
    print(f"Early region: " f"{SEARCH_CUTOFF_US:.1f} <= t < " f"{FIT_START_US:.1f} μs")
    print(f"Tail region: t >= " f"{FIT_START_US:.1f} μs")
    print()
    print(f"All accepted events: {decay_times.size}")
    print(f"Observed early events: {observed_early}")
    print(f"Observed tail events: {observed_tail}")
    print()
    print(f"Tail tau_MLE = " f"{tau_mle_us:.4f} " f"± {tau_uncertainty_us:.4f} μs")
    print()
    print(f"Expected early events from tail model: " f"{expected_early:.1f}")
    print(f"Observed early events: " f"{observed_early}")
    print(f"Excess events: " f"{excess_events:.1f}")
    print(f"Observed / expected ratio: " f"{observed_to_expected_ratio:.3f}")
    print(
        f"Fraction of observed early events "
        f"above model expectation: "
        f"{100.0 * excess_fraction_of_observed:.2f}%"
    )
    print()
    print(f"Simple Poisson excess significance: " f"{significance_sigma:.2f} sigma")
    print()
    print("Note: the quoted significance is only a simple " "counting diagnostic.")
    print(
        "It does not yet include uncertainty in tau, "
        "normalization, or detector-systematic effects."
    )


if __name__ == "__main__":
    main()
