from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def extract_accepted_lifetimes(
    results: Iterable[dict],
) -> np.ndarray:
    """
    Extract decay times from accepted detector candidates.

    Parameters
    ----------
    results:
        Detector-result dictionaries.

    Returns
    -------
    np.ndarray
        Accepted decay times in microseconds.
    """

    lifetimes = [
        float(result["decay_time_us"])
        for result in results
        if result.get("accepted", False) and result.get("decay_time_us") is not None
    ]

    return np.asarray(lifetimes, dtype=float)


def select_fit_range(
    lifetimes_us: np.ndarray,
    t_min_us: float,
    t_max_us: float,
) -> np.ndarray:
    """
    Select lifetimes inside the requested fit interval.
    """

    lifetimes_us = np.asarray(
        lifetimes_us,
        dtype=float,
    )

    finite_mask = np.isfinite(lifetimes_us)

    fit_mask = finite_mask & (lifetimes_us >= t_min_us) & (lifetimes_us <= t_max_us)

    return lifetimes_us[fit_mask]


def _truncated_exponential_nll(
    tau_us: float,
    lifetimes_us: np.ndarray,
    t_min_us: float,
    t_max_us: float,
) -> float:
    """
    Negative log-likelihood for an exponential distribution
    truncated to [t_min_us, t_max_us].
    """

    if tau_us <= 0:
        return np.inf

    fit_width_us = t_max_us - t_min_us

    if fit_width_us <= 0:
        raise ValueError("t_max_us must be greater than t_min_us.")

    shifted_times = lifetimes_us - t_min_us

    normalization = 1.0 - np.exp(-fit_width_us / tau_us)

    if normalization <= 0:
        return np.inf

    event_count = lifetimes_us.size

    negative_log_likelihood = (
        event_count * np.log(tau_us)
        + np.sum(shifted_times) / tau_us
        + event_count * np.log(normalization)
    )

    return float(negative_log_likelihood)


def estimate_truncated_exponential_lifetime(
    lifetimes_us: np.ndarray,
    t_min_us: float,
    t_max_us: float,
) -> tuple[float, float]:
    """
    Estimate the exponential lifetime using a likelihood
    normalized inside a finite fit interval.

    Returns
    -------
    tuple[float, float]
        Estimated lifetime and approximate statistical
        uncertainty, both in microseconds.
    """

    selected_lifetimes = select_fit_range(
        lifetimes_us=lifetimes_us,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    if selected_lifetimes.size == 0:
        return np.nan, np.nan

    optimization_result = minimize_scalar(
        _truncated_exponential_nll,
        args=(
            selected_lifetimes,
            t_min_us,
            t_max_us,
        ),
        bounds=(1.0e-4, 100.0),
        method="bounded",
    )

    if not optimization_result.success:
        return np.nan, np.nan

    tau_mle_us = float(optimization_result.x)

    # Numerical curvature of the negative log-likelihood.
    step = max(
        tau_mle_us * 1.0e-4,
        1.0e-5,
    )

    nll_center = _truncated_exponential_nll(
        tau_us=tau_mle_us,
        lifetimes_us=selected_lifetimes,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    nll_left = _truncated_exponential_nll(
        tau_us=max(tau_mle_us - step, 1.0e-8),
        lifetimes_us=selected_lifetimes,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    nll_right = _truncated_exponential_nll(
        tau_us=tau_mle_us + step,
        lifetimes_us=selected_lifetimes,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    second_derivative = (nll_right - 2.0 * nll_center + nll_left) / (step**2)

    if second_derivative > 0:
        tau_error_us = float(np.sqrt(1.0 / second_derivative))
    else:
        tau_error_us = np.nan

    return tau_mle_us, tau_error_us


def calculate_summary(
    lifetimes_us: np.ndarray,
    t_min_us: float,
    t_max_us: float,
) -> dict:
    """
    Calculate descriptive statistics and a truncated
    exponential maximum-likelihood estimate.
    """

    lifetimes_us = np.asarray(
        lifetimes_us,
        dtype=float,
    )

    fit_lifetimes = select_fit_range(
        lifetimes_us=lifetimes_us,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    tau_mle_us, tau_error_us = estimate_truncated_exponential_lifetime(
        lifetimes_us=fit_lifetimes,
        t_min_us=t_min_us,
        t_max_us=t_max_us,
    )

    if fit_lifetimes.size == 0:
        return {
            "events_used": 0,
            "t_min_us": float(t_min_us),
            "t_max_us": float(t_max_us),
            "mean_us": np.nan,
            "median_us": np.nan,
            "std_us": np.nan,
            "minimum_us": np.nan,
            "maximum_us": np.nan,
            "tau_mle_us": np.nan,
            "tau_error_us": np.nan,
        }

    return {
        "events_used": int(fit_lifetimes.size),
        "t_min_us": float(t_min_us),
        "t_max_us": float(t_max_us),
        "mean_us": float(np.mean(fit_lifetimes)),
        "median_us": float(np.median(fit_lifetimes)),
        "std_us": (
            float(
                np.std(
                    fit_lifetimes,
                    ddof=1,
                )
            )
            if fit_lifetimes.size > 1
            else 0.0
        ),
        "minimum_us": float(np.min(fit_lifetimes)),
        "maximum_us": float(np.max(fit_lifetimes)),
        "tau_mle_us": float(tau_mle_us),
        "tau_error_us": float(tau_error_us),
    }


def calculate_fit_range_stability(
    lifetimes_us: np.ndarray,
    t_min_values_us: Iterable[float],
    t_max_us: float,
) -> list[dict]:
    """
    Calculate the fitted lifetime for several lower
    fit thresholds.
    """

    stability_results: list[dict] = []

    for t_min_us in t_min_values_us:
        summary = calculate_summary(
            lifetimes_us=lifetimes_us,
            t_min_us=float(t_min_us),
            t_max_us=t_max_us,
        )

        stability_results.append(
            {
                "t_min_us": summary["t_min_us"],
                "t_max_us": summary["t_max_us"],
                "events_used": summary["events_used"],
                "tau_mle_us": summary["tau_mle_us"],
                "tau_error_us": summary["tau_error_us"],
                "mean_us": summary["mean_us"],
                "median_us": summary["median_us"],
            }
        )

    return stability_results


def make_histogram(
    lifetimes_us: np.ndarray,
    savepath: str | Path,
    t_min_us: float | None = None,
    t_max_us: float | None = None,
    title: str = ("Cosmic Muon Decay-Time Distribution"),
    bins: int = 30,
) -> None:
    """
    Save a histogram of decay times.
    """

    lifetimes_us = np.asarray(
        lifetimes_us,
        dtype=float,
    )

    if t_min_us is not None and t_max_us is not None:
        plotted_lifetimes = select_fit_range(
            lifetimes_us=lifetimes_us,
            t_min_us=t_min_us,
            t_max_us=t_max_us,
        )
    else:
        plotted_lifetimes = lifetimes_us[np.isfinite(lifetimes_us)]

    if plotted_lifetimes.size == 0:
        return

    savepath = Path(savepath)
    savepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.hist(
        plotted_lifetimes,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    axis.set_title(title)
    axis.set_xlabel("Decay time (μs)")
    axis.set_ylabel("Counts")
    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        savepath,
        dpi=160,
    )

    plt.close(figure)


def make_fit_range_stability_plot(
    stability_results: list[dict],
    savepath: str | Path,
    accepted_lifetime_us: float | None = None,
) -> None:
    """
    Plot fitted lifetime as a function of the lower
    fit threshold.
    """

    valid_results = [
        result for result in stability_results if np.isfinite(result["tau_mle_us"])
    ]

    if not valid_results:
        return

    t_min_values = np.asarray(
        [result["t_min_us"] for result in valid_results],
        dtype=float,
    )

    tau_values = np.asarray(
        [result["tau_mle_us"] for result in valid_results],
        dtype=float,
    )

    tau_errors = np.asarray(
        [result["tau_error_us"] for result in valid_results],
        dtype=float,
    )

    savepath = Path(savepath)
    savepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    axis.errorbar(
        t_min_values,
        tau_values,
        yerr=tau_errors,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Measured lifetime",
    )

    if accepted_lifetime_us is not None:
        axis.axhline(
            accepted_lifetime_us,
            linestyle="--",
            label=("Reference free-muon lifetime " f"({accepted_lifetime_us:.3f} μs)"),
        )

    axis.set_xlabel("Lower fit threshold $t_{min}$ (μs)")
    axis.set_ylabel("Fitted lifetime τ (μs)")
    axis.set_title("Muon-Lifetime Fit-Range Stability")
    axis.grid(
        alpha=0.25,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        savepath,
        dpi=160,
    )

    plt.close(figure)


def format_summary(
    summary: dict,
) -> str:
    """
    Format an analysis summary as readable text.
    """

    return (
        "Muon Lifetime Analysis\n"
        "======================\n"
        f"Fit range        : "
        f"{summary['t_min_us']:.2f}–"
        f"{summary['t_max_us']:.2f} μs\n"
        f"Events used      : "
        f"{summary['events_used']}\n"
        f"Mean delay       : "
        f"{summary['mean_us']:.4f} μs\n"
        f"Median delay     : "
        f"{summary['median_us']:.4f} μs\n"
        f"Std deviation    : "
        f"{summary['std_us']:.4f} μs\n"
        f"Minimum delay    : "
        f"{summary['minimum_us']:.4f} μs\n"
        f"Maximum delay    : "
        f"{summary['maximum_us']:.4f} μs\n"
        f"Lifetime MLE     : "
        f"{summary['tau_mle_us']:.4f} "
        f"± {summary['tau_error_us']:.4f} μs"
    )


def save_summary(
    summary: dict,
    savepath: str | Path,
) -> None:
    """
    Save the formatted summary to a text file.
    """

    savepath = Path(savepath)

    savepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    savepath.write_text(
        format_summary(summary) + "\n",
        encoding="utf-8",
    )
