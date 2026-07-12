import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ACCEPTED_MUON_LIFETIME_US,
    DATA_DIR,
    FILE_PATTERN,
    FIT_MAX_US,
    PUBLICATION_HISTOGRAM_PATH,
    T_MIN_US,
)
from src.pipeline import calculate_lifetimes


HISTOGRAM_BINS = 16


def plot_publication_histogram(
    lifetimes_us: np.ndarray,
) -> None:
    """Create and save a publication-style muon lifetime histogram."""

    if len(lifetimes_us) == 0:
        raise ValueError("No valid muon lifetime events were reconstructed.")

    mean_lifetime_us = np.mean(lifetimes_us)
    std_lifetime_us = np.std(lifetimes_us)
    n_events = len(lifetimes_us)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        lifetimes_us,
        bins=HISTOGRAM_BINS,
        range=(T_MIN_US, FIT_MAX_US),
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        ACCEPTED_MUON_LIFETIME_US,
        linestyle="--",
        linewidth=2,
        label=(
            "Accepted muon lifetime "
            f"≈ {ACCEPTED_MUON_LIFETIME_US:.3f} μs"
        ),
    )

    ax.axvline(
        mean_lifetime_us,
        linestyle="-",
        linewidth=2,
        label=f"Mean reconstructed = {mean_lifetime_us:.3f} μs",
    )

    statistics_text = (
        f"Events: {n_events}\n"
        f"Mean: {mean_lifetime_us:.3f} μs\n"
        f"Std: {std_lifetime_us:.3f} μs"
    )

    ax.text(
        0.98,
        0.95,
        statistics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    ax.set_xlabel("Reconstructed lifetime (μs)")
    ax.set_ylabel("Counts")
    ax.set_title("Cosmic Muon Lifetime Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    PUBLICATION_HISTOGRAM_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        PUBLICATION_HISTOGRAM_PATH,
        dpi=200,
    )

    plt.show()


def main() -> None:
    """Generate the publication-style lifetime histogram."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    lifetimes_s = calculate_lifetimes(files)
    lifetimes_us = np.asarray(lifetimes_s) * 1e6

    plot_publication_histogram(lifetimes_us)

    print(
        "Publication histogram saved to: "
        f"{PUBLICATION_HISTOGRAM_PATH}"
    )


if __name__ == "__main__":
    main()