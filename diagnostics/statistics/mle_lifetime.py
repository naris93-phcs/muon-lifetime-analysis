import numpy as np
import pandas as pd

T_MIN_US = 0.80


def estimate_truncated_lifetime(
    lifetimes_us: np.ndarray,
    t_min_us: float,
) -> tuple[float, float, int]:
    """
    Estimate the lifetime of a truncated exponential distribution.

    Parameters
    ----------
    lifetimes_us : numpy.ndarray
        Reconstructed lifetimes in microseconds.
    t_min_us : float
        Lower detection threshold in microseconds.

    Returns
    -------
    tuple
        Estimated lifetime, statistical uncertainty, and number of events.
    """

    shifted_lifetimes = lifetimes_us - t_min_us
    shifted_lifetimes = shifted_lifetimes[shifted_lifetimes > 0]

    if len(shifted_lifetimes) == 0:
        raise ValueError("No lifetime events remain after the lower cutoff.")

    tau_mle = np.mean(shifted_lifetimes)
    tau_error = tau_mle / np.sqrt(len(shifted_lifetimes))

    return tau_mle, tau_error, len(shifted_lifetimes)


def main() -> None:
    """Run the truncated-exponential maximum-likelihood estimate."""

    accepted = pd.read_csv("results/root_full_dataset/accepted_candidates.csv")

    lifetimes_us = accepted["decay_time_us"].to_numpy()
    bins = np.arange(0.8, 9.2, 0.2)
    counts, edges = np.histogram(lifetimes_us, bins=bins)

    print("\nHistogram (0.2 μs bins)")
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        print(f"{left:4.1f} – {right:4.1f} μs : {count}")

    tau_mle, tau_error, events_used = estimate_truncated_lifetime(
        lifetimes_us,
        T_MIN_US,
    )

    print("========================")
    print("Muon Lifetime MLE")
    print("========================")
    print(f"Events used: {events_used}")
    print(f"t_min = {T_MIN_US:.2f} μs")
    print(f"tau_MLE = {tau_mle:.4f} ± {tau_error:.4f} μs")
    print("========================")


if __name__ == "__main__":
    main()
