import numpy as np

from src.config import DATA_DIR, FILE_PATTERN, T_MIN_US
from src.pipeline import calculate_lifetimes


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

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    lifetimes_s = calculate_lifetimes(files)
    lifetimes_us = np.asarray(lifetimes_s) * 1e6

    tau_mle, tau_error, events_used = estimate_truncated_lifetime(
        lifetimes_us,
        T_MIN_US,
    )

    print("========================")
    print("Muon Lifetime MLE")
    print("========================")
    print(f"Events used: {events_used}")
    print(f"t_min = {T_MIN_US:.2f} μs")
    print(f"tau_MLE = {tau_mle:.3f} ± {tau_error:.3f} μs")
    print("========================")


if __name__ == "__main__":
    main()
