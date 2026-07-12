import numpy as np

from src.config import (
    DATA_DIR,
    FILE_PATTERN,
    MAX_LIFETIME,
    MIN_DELAY,
    SUMMARY_PATH,
)
from src.pipeline import calculate_lifetimes


def create_summary() -> str:
    """Create a text summary of the muon lifetime analysis."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    lifetimes_s = calculate_lifetimes(files)
    lifetimes_us = np.asarray(lifetimes_s) * 1e6

    if len(files) == 0:
        raise ValueError(f"No input files found in {DATA_DIR}")

    if len(lifetimes_us) == 0:
        raise ValueError("No valid muon lifetime events were reconstructed.")

    detection_efficiency = 100 * len(lifetimes_us) / len(files)

    return f"""Muon Lifetime Analysis Summary
==============================

Input files
-----------
Total files analyzed      : {len(files)}
Events used               : {len(lifetimes_us)}
Detection efficiency      : {detection_efficiency:.1f} %

Lifetime statistics
-------------------
Mean lifetime             : {np.mean(lifetimes_us):.3f} μs
Standard deviation        : {np.std(lifetimes_us):.3f} μs
Median lifetime           : {np.median(lifetimes_us):.3f} μs
Minimum lifetime          : {np.min(lifetimes_us):.3f} μs
Maximum lifetime          : {np.max(lifetimes_us):.3f} μs

Detector configuration
----------------------
Trigger channel           : CH2
Decay search channel      : CH1
Early-time veto           : {MIN_DELAY * 1e6:.1f} μs
Maximum lifetime          : {MAX_LIFETIME * 1e6:.1f} μs
Pulse selection           : dual polarity
Selection metric          : peak prominence

Notes
-----
The reconstructed lifetime distribution is compared with the
free-muon lifetime of approximately 2.2 μs. No detector
acceptance or efficiency correction has been applied.
"""


def main() -> None:
    """Generate and save the analysis summary."""

    summary = create_summary()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()