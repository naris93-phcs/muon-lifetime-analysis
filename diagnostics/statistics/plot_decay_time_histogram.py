from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project root:
# diagnostics/plot_decay_time_histogram.py -> project root is one level above
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "diagnostics"

OUTPUT_FILE = OUTPUT_DIR / "accepted_decay_time_histogram.png"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Accepted-candidates file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Available columns:")
    print(df.columns.tolist())

    # Change this only if the printed CSV column has a different name.
    time_column = "decay_time_us"

    if time_column not in df.columns:
        raise KeyError(
            f"Column '{time_column}' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    decay_times = pd.to_numeric(
        df[time_column],
        errors="coerce",
    ).dropna()

    if decay_times.empty:
        raise ValueError("No valid decay times were found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.yscale("log")

    plt.hist(
        decay_times,
        bins=80,
        edgecolor="black",
        alpha=0.8,
    )

    plt.axvline(
        0.8,
        linestyle="--",
        linewidth=2,
        label="Search cutoff: 0.8 μs",
    )

    plt.axvline(
        1.5,
        linestyle="--",
        linewidth=2,
        label="Diagnostic reference: 1.5 μs",
    )

    plt.xlabel("Accepted decay time (μs)")
    plt.ylabel("Number of events")
    plt.title("Accepted Muon Decay-Time Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
    )

    plt.show()

    print()
    print(f"Events plotted: {len(decay_times)}")
    print(f"Minimum time: {decay_times.min():.4f} μs")
    print(f"Maximum time: {decay_times.max():.4f} μs")
    print(f"Mean time: {decay_times.mean():.4f} μs")
    print(f"Median time: {decay_times.median():.4f} μs")
    print(f"Histogram saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
