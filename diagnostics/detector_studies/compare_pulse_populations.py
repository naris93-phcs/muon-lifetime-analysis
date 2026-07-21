from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "pulse_diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_LIMIT_US = 1.5

FEATURES = [
    "channel2_amplitude_v",
    "channel2_prominence_v",
    "channel2_width_ns",
    "channel2_snr",
    "channel1_to_channel2_ratio",
]


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "decay_time_us",
        *FEATURES,
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError("Missing required columns:\n" + "\n".join(missing_columns))

    print()
    print("Data loaded successfully")
    print("------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def split_populations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    early = df[df["decay_time_us"] < EARLY_LIMIT_US].copy()

    tail = df[df["decay_time_us"] >= EARLY_LIMIT_US].copy()

    print()
    print("Population split")
    print("----------------")
    print(f"Early limit: {EARLY_LIMIT_US:.2f} μs")
    print(f"Early events: {len(early)}")
    print(f"Tail events:  {len(tail)}")
    print(f"Total events: {len(early) + len(tail)}")

    return early, tail


def compare_feature(
    feature: str,
    early: pd.DataFrame,
    tail: pd.DataFrame,
) -> dict:
    early_values = early[feature].dropna()
    tail_values = tail[feature].dropna()

    early_mean = early_values.mean()
    tail_mean = tail_values.mean()

    early_median = early_values.median()
    tail_median = tail_values.median()

    early_std = early_values.std()
    tail_std = tail_values.std()

    early_iqr = early_values.quantile(0.75) - early_values.quantile(0.25)

    tail_iqr = tail_values.quantile(0.75) - tail_values.quantile(0.25)

    median_difference_percent = (
        100.0 * (early_median - tail_median) / tail_median
        if tail_median != 0
        else np.nan
    )

    test_result = mannwhitneyu(
        early_values,
        tail_values,
        alternative="two-sided",
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        early_values,
        bins=50,
        alpha=0.6,
        label="Early",
    )

    plt.hist(
        tail_values,
        bins=50,
        alpha=0.6,
        label="Tail",
    )

    plt.xlabel(feature)
    plt.ylabel("Counts")
    plt.title(feature.replace("_", " "))
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_hist.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(6, 5))

    plt.boxplot(
        [early_values, tail_values],
        tick_labels=["Early", "Tail"],
        showfliers=False,
    )
    plt.ylabel(feature)

    plt.title(feature.replace("_", " "))

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_boxplot.png",
        dpi=300,
    )

    plt.close()

    print()
    print(f"Comparing: {feature}")
    print("-" * (11 + len(feature)))
    print(f"Early median: {early_median:.6g}")
    print(f"Tail median:  {tail_median:.6g}")
    print("Median difference: " f"{median_difference_percent:+.2f}%")
    print(f"Mann-Whitney p-value: {test_result.pvalue:.6e}")

    return {
        "feature": feature,
        "early_count": len(early_values),
        "tail_count": len(tail_values),
        "early_mean": early_mean,
        "tail_mean": tail_mean,
        "early_median": early_median,
        "tail_median": tail_median,
        "median_difference_percent": median_difference_percent,
        "early_std": early_std,
        "tail_std": tail_std,
        "early_iqr": early_iqr,
        "tail_iqr": tail_iqr,
        "mann_whitney_u": test_result.statistic,
        "mann_whitney_p_value": test_result.pvalue,
    }


def main() -> None:
    df = load_data()
    early, tail = split_populations(df)

    results = []

    for feature in FEATURES:
        results.append(
            compare_feature(
                feature,
                early,
                tail,
            )
        )

    results_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "pulse_population_statistics.csv"

    results_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Statistics saved to:\n{output_file}")


if __name__ == "__main__":
    main()
