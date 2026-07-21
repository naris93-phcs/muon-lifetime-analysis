from pathlib import Path

import pandas as pd

from src.analysis import (
    calculate_summary,
    extract_accepted_lifetimes,
    format_summary,
    make_histogram,
    save_summary,
)
from src.config import (
    ACCEPTED_CANDIDATES_PATH,
    DATA_DIR,
    LIFETIME_HISTOGRAM_PATH,
    RESULTS_DIR,
    ROOT_FILE_PATTERN,
    SUMMARY_PATH,
    T_MIN_US,
)
from src.io import iter_root_events
from src.pipeline import analyze_events


def main() -> None:
    """
    Run the complete ROOT muon-decay analysis pipeline.
    """

    root_files = sorted(Path(DATA_DIR).glob(ROOT_FILE_PATTERN))

    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in:\n{DATA_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("Muon Lifetime ROOT Analysis")
    print("=" * 70)
    print(f"ROOT files found: {len(root_files)}")
    print()

    all_results: list[dict] = []

    for file_number, file_path in enumerate(
        root_files,
        start=1,
    ):
        print(
            f"[{file_number:02d}/{len(root_files):02d}] " f"Processing {file_path.name}"
        )

        try:
            events = iter_root_events(file_path)
            file_results = analyze_events(events)
            all_results.extend(file_results)

        except Exception as error:
            print()
            print(f"WARNING: Skipping {file_path.name}")
            print(error)
            print()
            continue

        file_accepted = sum(result.get("accepted", False) for result in file_results)

        print(f"    Results: {len(file_results)} | " f"Accepted: {file_accepted}")

    accepted_results = [
        result for result in all_results if result.get("accepted", False)
    ]

    rejected_results = [
        result for result in all_results if not result.get("accepted", False)
    ]

    lifetimes_us = extract_accepted_lifetimes(all_results)

    summary = calculate_summary(
        lifetimes_us=lifetimes_us,
        t_min_us=T_MIN_US,
    )

    if all_results:
        results_dataframe = pd.DataFrame(all_results)

        results_dataframe.to_csv(
            ACCEPTED_CANDIDATES_PATH,
            index=False,
        )

    if lifetimes_us.size > 0:
        make_histogram(
            lifetimes_us=lifetimes_us,
            savepath=LIFETIME_HISTOGRAM_PATH,
        )

    save_summary(
        summary=summary,
        savepath=SUMMARY_PATH,
    )

    print()
    print("=" * 70)
    print("Analysis finished")
    print("=" * 70)
    print(f"Total detector results : {len(all_results)}")
    print(f"Accepted candidates    : {len(accepted_results)}")
    print(f"Rejected results       : {len(rejected_results)}")
    print()
    print(format_summary(summary))
    print(f"Results CSV : {ACCEPTED_CANDIDATES_PATH}")
    print(f"Histogram   : {LIFETIME_HISTOGRAM_PATH}")
    print(f"Summary     : {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
