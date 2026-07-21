from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "file_diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_LIMIT_US = 1.5


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Available CSV columns:")
    print(df.columns.tolist())

    filename_candidates = [
        "filename",
        "file_name",
        "source_file",
        "root_file",
        "file",
        "filepath",
        "file_path",
    ]

    filename_column = next(
        (column for column in filename_candidates if column in df.columns),
        None,
    )

    if filename_column is None:
        raise KeyError(
            "Could not identify the ROOT filename column.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if "decay_time_us" not in df.columns:
        raise KeyError(
            "Column 'decay_time_us' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Using file column: {filename_column}")

    summary_rows = []

    for filename, group in df.groupby(filename_column):
        accepted = len(group)

        early = (group["decay_time_us"] < EARLY_LIMIT_US).sum()

        tail = accepted - early

        summary_rows.append(
            {
                "filename": filename,
                "accepted_events": accepted,
                "early_events": early,
                "tail_events": tail,
                "early_fraction": early / accepted,
                "tail_fraction": tail / accepted,
                "mean_delay_us": (group["decay_time_us"].mean()),
                "median_delay_us": (group["decay_time_us"].median()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("filename").reset_index(drop=True)

    summary.to_csv(
        OUTPUT_DIR / "file_summary.csv",
        index=False,
    )

    print()
    print(summary)

    print()
    print("Overall statistics")
    print("------------------")
    print(
        summary[
            [
                "early_fraction",
                "accepted_events",
                "mean_delay_us",
            ]
        ].describe()
    )

    # ---------------------------------------
    # Plot 1
    # ---------------------------------------

    plt.figure(figsize=(12, 5))

    plt.bar(
        summary["filename"],
        summary["early_fraction"],
    )

    plt.xticks(rotation=90)

    plt.ylabel("Early fraction")

    plt.title("Early-event fraction by ROOT file")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "early_fraction_per_file.png",
        dpi=300,
    )

    # ---------------------------------------
    # Plot 2
    # ---------------------------------------

    plt.figure(figsize=(12, 5))

    plt.bar(
        summary["filename"],
        summary["accepted_events"],
    )

    plt.xticks(rotation=90)

    plt.ylabel("Accepted events")

    plt.title("Accepted candidates per ROOT file")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accepted_events_per_file.png",
        dpi=300,
    )

    # ---------------------------------------
    # Plot 3
    # ---------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        summary["early_fraction"],
        summary["accepted_events"],
    )

    for _, row in summary.iterrows():

        plt.text(
            row["early_fraction"],
            row["accepted_events"],
            row["filename"].replace(".root", ""),
            fontsize=7,
        )

    plt.xlabel("Early fraction")

    plt.ylabel("Accepted events")

    plt.title("Accepted events vs Early fraction")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accepted_vs_early_fraction.png",
        dpi=300,
    )

    print()
    print("Saved:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
