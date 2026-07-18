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

OUTPUT_CSV = (
    OUTPUT_DIR
    / "fit_range_stability.csv"
)

OUTPUT_PLOT = (
    OUTPUT_DIR
    / "fit_range_stability.png"
)


# Different lower fit limits to test.
FIT_MIN_VALUES_US = [
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
]

# Upper edge of the acquisition window.
FIT_MAX_US = 9.00

# Require at least this many events for a meaningful estimate.
MINIMUM_EVENTS = 10


def load_decay_times(csv_path: Path) -> np.ndarray:
    """Load accepted decay-like times from the selector output."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Accepted-candidate CSV was not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    if "decay_time_us" not in dataframe.columns:
        raise ValueError(
            "Column 'decay_time_us' was not found in the CSV."
        )

    decay_times = pd.to_numeric(
        dataframe["decay_time_us"],
        errors="coerce",
    ).dropna().to_numpy()

    decay_times = decay_times[
        np.isfinite(decay_times)
    ]

    if len(decay_times) == 0:
        raise ValueError(
            "No valid decay times were found."
        )

    return decay_times


def truncated_exponential_mean(
    tau: float,
    fit_width_us: float,
) -> float:
    """
    Expected value of x = t - t_min for an exponential distribution
    truncated to 0 <= x <= fit_width_us.

    E[x] = tau - L / (exp(L / tau) - 1)
    """

    return (
        tau
        - fit_width_us
        / np.expm1(fit_width_us / tau)
    )


def estimate_truncated_tau(
    shifted_times_us: np.ndarray,
    fit_width_us: float,
) -> float:
    """
    Estimate tau for an exponential distribution with both lower and
    upper truncation.

    The MLE equation is solved numerically:

        sample_mean
        =
        tau - L / (exp(L / tau) - 1)
    """

    sample_mean = float(
        np.mean(shifted_times_us)
    )

    if sample_mean <= 0:
        raise ValueError(
            "Shifted-time mean must be positive."
        )

    # For a distribution truncated to [0, L], the mean cannot exceed L/2
    # in the tau -> infinity limit.
    maximum_possible_mean = fit_width_us / 2.0

    if sample_mean >= maximum_possible_mean:
        return float("nan")

    lower_tau = 1e-6
    upper_tau = 1000.0

    for _ in range(200):
        middle_tau = (
            lower_tau + upper_tau
        ) / 2.0

        expected_mean = truncated_exponential_mean(
            middle_tau,
            fit_width_us,
        )

        if expected_mean < sample_mean:
            lower_tau = middle_tau
        else:
            upper_tau = middle_tau

    return (
        lower_tau + upper_tau
    ) / 2.0


def bootstrap_tau_uncertainty(
    selected_times_us: np.ndarray,
    fit_min_us: float,
    fit_max_us: float,
    number_of_bootstraps: int = 2000,
    seed: int = 42,
) -> float:
    """Estimate the statistical uncertainty using bootstrap resampling."""

    rng = np.random.default_rng(seed)

    fit_width_us = (
        fit_max_us - fit_min_us
    )

    shifted_times = (
        selected_times_us - fit_min_us
    )

    estimates = []

    for _ in range(number_of_bootstraps):
        resampled = rng.choice(
            shifted_times,
            size=len(shifted_times),
            replace=True,
        )

        tau_estimate = estimate_truncated_tau(
            resampled,
            fit_width_us,
        )

        if np.isfinite(tau_estimate):
            estimates.append(
                tau_estimate
            )

    if len(estimates) < 2:
        return float("nan")

    return float(
        np.std(estimates, ddof=1)
    )


def analyze_fit_range(
    decay_times_us: np.ndarray,
    fit_min_us: float,
    fit_max_us: float,
) -> dict:
    """Calculate lifetime estimates for one lower fit boundary."""

    selected = decay_times_us[
        (
            decay_times_us >= fit_min_us
        )
        & (
            decay_times_us <= fit_max_us
        )
    ]

    number_of_events = len(selected)

    if number_of_events < MINIMUM_EVENTS:
        return {
            "fit_min_us": fit_min_us,
            "fit_max_us": fit_max_us,
            "events": number_of_events,
            "simple_shifted_tau_us": float("nan"),
            "truncated_tau_us": float("nan"),
            "bootstrap_uncertainty_us": float("nan"),
        }

    shifted_times = (
        selected - fit_min_us
    )

    # This ignores the finite upper acquisition limit.
    simple_shifted_tau = float(
        np.mean(shifted_times)
    )

    fit_width_us = (
        fit_max_us - fit_min_us
    )

    truncated_tau = estimate_truncated_tau(
        shifted_times,
        fit_width_us,
    )

    uncertainty = bootstrap_tau_uncertainty(
        selected_times_us=selected,
        fit_min_us=fit_min_us,
        fit_max_us=fit_max_us,
    )

    return {
        "fit_min_us": fit_min_us,
        "fit_max_us": fit_max_us,
        "events": number_of_events,
        "simple_shifted_tau_us": simple_shifted_tau,
        "truncated_tau_us": truncated_tau,
        "bootstrap_uncertainty_us": uncertainty,
    }


def print_results(
    results: pd.DataFrame,
) -> None:
    """Print the fit-range scan in readable form."""

    print()
    print("=" * 94)
    print("FIT-RANGE STABILITY TEST")
    print("=" * 94)
    print(
        f"{'t_min (μs)':>10} "
        f"{'events':>10} "
        f"{'simple τ (μs)':>17} "
        f"{'truncated τ (μs)':>20} "
        f"{'bootstrap σ (μs)':>20}"
    )
    print("-" * 94)

    for _, row in results.iterrows():
        print(
            f"{row['fit_min_us']:10.2f} "
            f"{int(row['events']):10d} "
            f"{row['simple_shifted_tau_us']:17.4f} "
            f"{row['truncated_tau_us']:20.4f} "
            f"{row['bootstrap_uncertainty_us']:20.4f}"
        )

    print("=" * 94)


def plot_stability(
    results: pd.DataFrame,
) -> None:
    """Plot fitted lifetime against the lower fit boundary."""

    valid = results[
        np.isfinite(
            results["truncated_tau_us"]
        )
    ].copy()

    if valid.empty:
        print(
            "No valid lifetime estimates were available for plotting."
        )
        return

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    axis.errorbar(
        valid["fit_min_us"],
        valid["truncated_tau_us"],
        yerr=valid["bootstrap_uncertainty_us"],
        marker="o",
        linestyle="-",
        capsize=4,
        label="Truncated exponential MLE",
    )

    axis.axhline(
        2.197,
        linestyle="--",
        linewidth=1.5,
        label="Free-muon lifetime: 2.197 μs",
    )

    axis.set_xlabel(
        "Lower fit boundary $t_{min}$ (μs)"
    )

    axis.set_ylabel(
        "Estimated lifetime τ (μs)"
    )

    axis.set_title(
        "Muon-lifetime fit-range stability\n"
        "Accepted delayed-pulse candidates"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PLOT,
        dpi=160,
    )

    plt.show()
    plt.close(figure)

    print(
        f"\nSaved stability plot:\n{OUTPUT_PLOT}"
    )


def main() -> None:
    """Run the lifetime stability scan."""

    decay_times_us = load_decay_times(
        INPUT_CSV
    )

    results = []

    for fit_min_us in FIT_MIN_VALUES_US:
        result = analyze_fit_range(
            decay_times_us=decay_times_us,
            fit_min_us=fit_min_us,
            fit_max_us=FIT_MAX_US,
        )

        results.append(
            result
        )

    results_dataframe = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print_results(
        results_dataframe
    )

    print(
        f"\nSaved fit-range table:\n{OUTPUT_CSV}"
    )

    plot_stability(
        results_dataframe
    )


if __name__ == "__main__":
    main()