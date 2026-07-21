from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "root_full_dataset" / "accepted_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "root_full_dataset" / "pulse_diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print()
    print("Pulse-feature inventory")
    print("-----------------------")
    print(f"Input file: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Available columns")
    print("-----------------")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. " f"{column:<40} " f"{df[column].dtype}")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns were found in the input CSV.")

    summary = numeric_df.describe().T

    summary["missing_values"] = numeric_df.isna().sum()

    summary["missing_fraction"] = numeric_df.isna().mean()

    summary["unique_values"] = numeric_df.nunique(dropna=True)

    summary = summary[
        [
            "count",
            "missing_values",
            "missing_fraction",
            "unique_values",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]

    output_file = OUTPUT_DIR / "pulse_feature_inventory.csv"

    summary.to_csv(output_file)

    print()
    print("Numeric-feature summary")
    print("-----------------------")

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(summary)

    print()
    print(f"Saved summary to:\n{output_file}")


if __name__ == "__main__":
    main()
