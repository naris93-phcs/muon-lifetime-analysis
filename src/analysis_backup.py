"""
Statistical analysis and visualization of accepted muon-decay candidates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_accepted_lifetimes(
    results: list[dict],
) -> np.ndarray:
    """
    Extract valid decay times from accepted detector results.

    Parameters
    ----------
    results:
        Detector-result dictionaries produced by the analysis pipeline.

    Returns
    -------
    numpy.ndarray
        Accepted decay times in microseconds.
    """

    lifetimes_us = [
        result["decay_time_us"]
        for result in results
        if result.get("accepted", False)
        and result.get("decay_time_us") is not None
        and np.isfinite(result["decay_time_us"])
    ]

    return np.asarray(lifetimes_us, dtype=float)


def calculate_summary(
    lifetimes_us: np.ndarray,
    t_min_us: float,
) -> dict:
    """
    Calculate descriptive statistics and the lower-truncated
    exponential maximum-likelihood estimate.

    For events selected above t_min, the MLE is

        tau = mean(t - t_min)
    """

    if lifetimes_us.size == 0:
        return {
            "events": 0,
            "mean_us": np.nan,
            "median_us": np.nan,
            "std_us": np.nan,
            "min_us": np.nan,
            "max_us": np.nan,
            "tau_mle_us": np.nan,
            "tau_uncertainty_us": np.nan,
        }

    shifted_lifetimes = lifetimes_us - t_min_us

    valid_shifted = shifted_lifetimes[shifted_lifetimes >= 0.0]

    if valid_shifted.size == 0:
        tau_mle_us = np.nan
        tau_uncertainty_us = np.nan
    else:
        tau_mle_us = float(np.mean(valid_shifted))
        tau_uncertainty_us = tau_mle_us / np.sqrt(valid_shifted.size)

    return {
        "events": int(lifetimes_us.size),
        "mean_us": float(np.mean(lifetimes_us)),
        "median_us": float(np.median(lifetimes_us)),
        "std_us": float(np.std(lifetimes_us)),
        "min_us": float(np.min(lifetimes_us)),
        "max_us": float(np.max(lifetimes_us)),
        "tau_mle_us": tau_mle_us,
        "tau_uncertainty_us": tau_uncertainty_us,
    }


def make_histogram(
    lifetimes_us: np.ndarray,
    savepath: Path | str,
    bins: int = 25,
) -> None:
    """
    Create and save the accepted decay-time histogram.
    """

    if lifetimes_us.size == 0:
        raise ValueError("Cannot create histogram: no accepted lifetimes.")

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(
        lifetimes_us,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
    )

    axis.set_xlabel("Decay time (μs)")
    axis.set_ylabel("Counts")
    axis.set_title("Cosmic Muon Decay-Time Distribution")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(savepath, dpi=150)
    plt.close(figure)


def format_summary(summary: dict) -> str:
    """
    Format analysis statistics as readable text.
    """

    if summary["events"] == 0:
        return (
            "Muon Lifetime Analysis\n"
            "======================\n"
            "No accepted decay candidates were found.\n"
        )

    return (
        "Muon Lifetime Analysis\n"
        "======================\n"
        f"Events used      : {summary['events']}\n"
        f"Mean delay       : {summary['mean_us']:.4f} μs\n"
        f"Median delay     : {summary['median_us']:.4f} μs\n"
        f"Std deviation    : {summary['std_us']:.4f} μs\n"
        f"Minimum delay    : {summary['min_us']:.4f} μs\n"
        f"Maximum delay    : {summary['max_us']:.4f} μs\n"
        f"Lifetime MLE     : "
        f"{summary['tau_mle_us']:.4f} "
        f"± {summary['tau_uncertainty_us']:.4f} μs\n"
    )


def save_summary(
    summary: dict,
    savepath: Path | str,
) -> None:
    """
    Save the formatted statistical summary to a text file.
    """

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    savepath.write_text(
        format_summary(summary),
        encoding="utf-8",
    )
