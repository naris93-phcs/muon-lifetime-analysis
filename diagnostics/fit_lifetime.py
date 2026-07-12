import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.config import (
    DATA_DIR,
    FILE_PATTERN,
    FIT_BINS,
    FIT_MAX_US,
    LIFETIME_FIT_PATH,
    T_MIN_US,
)
from src.pipeline import calculate_lifetimes


def exponential(
    time_us: np.ndarray,
    amplitude: float,
    lifetime_us: float,
    background: float,
) -> np.ndarray:
    """Evaluate an exponential decay model with constant background."""

    return amplitude * np.exp(-time_us / lifetime_us) + background


def fit_lifetime_distribution(
    lifetimes_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an exponential model to the binned lifetime distribution.

    Returns
    -------
    tuple of numpy.ndarray
        Best-fit parameters and their standard errors.
    """

    counts, bin_edges = np.histogram(
        lifetimes_us,
        bins=FIT_BINS,
        range=(T_MIN_US, FIT_MAX_US),
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    nonempty_mask = counts > 0
    x_fit = bin_centers[nonempty_mask]
    y_fit = counts[nonempty_mask]

    if len(x_fit) < 3:
        raise ValueError("Not enough non-empty histogram bins for fitting.")

    initial_parameters = [
        float(np.max(y_fit)),
        2.2,
        0.0,
    ]

    fitted_parameters, covariance = curve_fit(
        exponential,
        x_fit,
        y_fit,
        p0=initial_parameters,
        maxfev=10_000,
    )

    parameter_errors = np.sqrt(np.diag(covariance))

    return fitted_parameters, parameter_errors


def plot_lifetime_fit(
    lifetimes_us: np.ndarray,
    fitted_parameters: np.ndarray,
    parameter_errors: np.ndarray,
) -> None:
    """Plot and save the lifetime histogram with its exponential fit."""

    amplitude, lifetime_us, background = fitted_parameters
    lifetime_error_us = parameter_errors[1]

    time_curve = np.linspace(T_MIN_US, FIT_MAX_US, 300)
    fitted_curve = exponential(
        time_curve,
        amplitude,
        lifetime_us,
        background,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        lifetimes_us,
        bins=FIT_BINS,
        range=(T_MIN_US, FIT_MAX_US),
        alpha=0.7,
        label="Data",
    )

    ax.plot(
        time_curve,
        fitted_curve,
        linewidth=2,
        label=(
            f"Fit: tau = {lifetime_us:.2f} "
            f"± {lifetime_error_us:.2f} μs"
        ),
    )

    ax.set_xlabel("Muon lifetime τ (μs)")
    ax.set_ylabel("Counts")
    ax.set_title("Muon Lifetime Histogram with Exponential Fit")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    LIFETIME_FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LIFETIME_FIT_PATH, dpi=150)

    plt.show()


def main() -> None:
    """Fit and plot the reconstructed muon lifetime distribution."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    lifetimes_s = calculate_lifetimes(files)
    lifetimes_us = np.asarray(lifetimes_s) * 1e6

    if len(lifetimes_us) == 0:
        raise ValueError("No valid muon lifetime events were reconstructed.")

    fitted_parameters, parameter_errors = fit_lifetime_distribution(
        lifetimes_us
    )

    lifetime_us = fitted_parameters[1]
    lifetime_error_us = parameter_errors[1]

    print(f"Events used: {len(lifetimes_us)}")
    print("========================")
    print("Exponential Fit")
    print("========================")
    print(
        f"Fitted lifetime tau = "
        f"{lifetime_us:.3f} ± {lifetime_error_us:.3f} μs"
    )
    print("========================")

    plot_lifetime_fit(
        lifetimes_us,
        fitted_parameters,
        parameter_errors,
    )

    print(f"Fit plot saved to: {LIFETIME_FIT_PATH}")


if __name__ == "__main__":
    main()