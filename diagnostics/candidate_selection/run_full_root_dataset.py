from pathlib import Path
from time import perf_counter

import pandas as pd
import uproot

from diagnostics import select_root_decay_candidates as selector

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"

RESULTS_DIR = PROJECT_ROOT / "results" / "root_full_dataset"

SELECTION_RESULTS_CSV = RESULTS_DIR / "selection_results.csv"

ACCEPTED_CSV = RESULTS_DIR / "accepted_candidates.csv"

REJECTED_CSV = RESULTS_DIR / "rejected_candidates.csv"

FILE_SUMMARY_CSV = RESULTS_DIR / "file_summary.csv"

RUN_SUMMARY_TXT = RESULTS_DIR / "run_summary.txt"


def get_total_entries(file_path: Path) -> int:
    """Return the number of events in the ROOT TTree."""

    with uproot.open(file_path) as root_file:
        tree = root_file["t1"]
        return int(tree.num_entries)


def split_candidate_results(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate delayed-pulse candidates into accepted and rejected sets.
    """

    if results.empty or "decay_time_us" not in results.columns:
        empty = pd.DataFrame()
        return empty, empty, empty

    candidates = results[results["decay_time_us"].notna()].copy()

    if candidates.empty:
        empty = pd.DataFrame()
        return candidates, empty, empty

    accepted = candidates[candidates["accepted"] == True].copy()

    rejected = candidates[candidates["accepted"] == False].copy()

    return candidates, accepted, rejected


def concatenate_frames(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Safely concatenate non-empty DataFrames."""

    non_empty_frames = [frame for frame in frames if not frame.empty]

    if not non_empty_frames:
        return pd.DataFrame()

    return pd.concat(
        non_empty_frames,
        ignore_index=True,
    )


def save_current_results(
    selection_frames: list[pd.DataFrame],
    accepted_frames: list[pd.DataFrame],
    rejected_frames: list[pd.DataFrame],
    file_summaries: list[dict],
) -> None:
    """
    Save progress after every ROOT file.

    This prevents losing the completed analysis if a later file fails.
    """

    all_selection_results = concatenate_frames(selection_frames)

    all_accepted = concatenate_frames(accepted_frames)

    all_rejected = concatenate_frames(rejected_frames)

    if not all_selection_results.empty:
        all_selection_results.to_csv(
            SELECTION_RESULTS_CSV,
            index=False,
        )

    if not all_accepted.empty:
        all_accepted = all_accepted.sort_values(
            ["file", "event_index"],
        )

        all_accepted.to_csv(
            ACCEPTED_CSV,
            index=False,
        )

    if not all_rejected.empty:
        all_rejected = all_rejected.sort_values(
            ["file", "event_index"],
        )

        all_rejected.to_csv(
            REJECTED_CSV,
            index=False,
        )

    pd.DataFrame(file_summaries).to_csv(
        FILE_SUMMARY_CSV,
        index=False,
    )


def create_run_summary(
    root_files: list[Path],
    file_summaries: list[dict],
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    elapsed_seconds: float,
) -> str:
    """Create the final human-readable run summary."""

    summary_df = pd.DataFrame(file_summaries)

    successful = summary_df[summary_df["status"] == "completed"]

    failed = summary_df[summary_df["status"] == "failed"]

    total_events = int(successful["total_events"].sum())

    total_candidates = int(successful["delayed_candidates"].sum())

    total_accepted = len(accepted)
    total_rejected = len(rejected)

    acceptance_all_events = (
        100.0 * total_accepted / total_events if total_events > 0 else 0.0
    )

    acceptance_candidates = (
        100.0 * total_accepted / total_candidates if total_candidates > 0 else 0.0
    )

    lines = [
        "=" * 78,
        "MUON V3 — FULL ROOT DATASET SELECTION",
        "=" * 78,
        "",
        f"ROOT files discovered:       {len(root_files)}",
        f"Files completed:             {len(successful)}",
        f"Files failed:                {len(failed)}",
        "",
        f"Total acquisition events:    {total_events}",
        f"Delayed CH2 candidates:      {total_candidates}",
        f"Accepted decay candidates:   {total_accepted}",
        f"Rejected by CH1 veto:        {total_rejected}",
        "",
        ("Accepted / all events:      " f"{acceptance_all_events:.4f}%"),
        ("Accepted / candidates:      " f"{acceptance_candidates:.4f}%"),
        "",
    ]

    if not accepted.empty:
        decay_times = accepted["decay_time_us"]

        lines.extend(
            [
                "Accepted decay-time summary:",
                f"Mean:                       {decay_times.mean():.6f} μs",
                f"Median:                     {decay_times.median():.6f} μs",
                f"Standard deviation:         {decay_times.std():.6f} μs",
                f"Minimum:                    {decay_times.min():.6f} μs",
                f"Maximum:                    {decay_times.max():.6f} μs",
                "",
            ]
        )

    lines.append(f"Elapsed time:                {elapsed_seconds / 60:.2f} min")

    if not failed.empty:
        lines.extend(
            [
                "",
                "Failed files:",
            ]
        )

        for _, row in failed.iterrows():
            lines.append(f"- {row['file']}: {row['error']}")

    lines.extend(
        [
            "",
            "=" * 78,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run the validated selector over every ROOT file."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_files = selector.find_root_files(ROOT_DATA_DIR)

    if not root_files:
        raise FileNotFoundError(f"No ROOT files were found in:\n" f"{ROOT_DATA_DIR}")

    print()
    print("=" * 78)
    print("MUON V3 — FULL ROOT DATASET RUN")
    print("=" * 78)
    print(f"ROOT directory: {ROOT_DATA_DIR}")
    print(f"Files discovered: {len(root_files)}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 78)
    print()

    for index, file_path in enumerate(
        root_files,
        start=1,
    ):
        size_mb = file_path.stat().st_size / (1024**2)

        print(
            f"[{index:02d}/{len(root_files):02d}] "
            f"{file_path.name} "
            f"({size_mb:.2f} MB)"
        )

    print()
    input("Press ENTER to launch the full dataset analysis...")

    selection_frames = []
    accepted_frames = []
    rejected_frames = []
    file_summaries = []

    run_start = perf_counter()

    for file_number, file_path in enumerate(
        root_files,
        start=1,
    ):
        print()
        print("#" * 78)
        print(
            f"[{file_number:02d}/{len(root_files):02d}] " f"ANALYZING {file_path.name}"
        )
        print("#" * 78)

        file_start = perf_counter()

        try:
            total_events = get_total_entries(file_path)

            results = selector.scan_root_file(file_path)

            (
                candidates,
                accepted,
                rejected,
            ) = split_candidate_results(results)

            selection_frames.append(results)

            accepted_frames.append(accepted)

            rejected_frames.append(rejected)

            elapsed_file_seconds = perf_counter() - file_start

            acceptance_percent = (
                100.0 * len(accepted) / total_events if total_events > 0 else 0.0
            )

            file_summary = {
                "file": file_path.name,
                "status": "completed",
                "total_events": total_events,
                "delayed_candidates": len(candidates),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "acceptance_percent": acceptance_percent,
                "elapsed_seconds": elapsed_file_seconds,
                "error": "",
            }

            file_summaries.append(file_summary)

            print()
            print("-" * 78)
            print(f"Completed: {file_path.name}")
            print(f"Events: {total_events}")
            print(f"Delayed candidates: {len(candidates)}")
            print(f"Accepted: {len(accepted)}")
            print(f"Rejected: {len(rejected)}")
            print(f"Acceptance: {acceptance_percent:.4f}%")
            print(f"File time: " f"{elapsed_file_seconds / 60:.2f} min")
            print("-" * 78)

        except Exception as error:
            elapsed_file_seconds = perf_counter() - file_start

            print()
            print("!" * 78)
            print(f"FAILED: {file_path.name}")
            print(f"Reason: {error}")
            print("!" * 78)

            file_summaries.append(
                {
                    "file": file_path.name,
                    "status": "failed",
                    "total_events": 0,
                    "delayed_candidates": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "acceptance_percent": 0.0,
                    "elapsed_seconds": elapsed_file_seconds,
                    "error": str(error),
                }
            )

        save_current_results(
            selection_frames=selection_frames,
            accepted_frames=accepted_frames,
            rejected_frames=rejected_frames,
            file_summaries=file_summaries,
        )

        print("Progress saved successfully.")

    elapsed_seconds = perf_counter() - run_start

    all_accepted = concatenate_frames(accepted_frames)

    all_rejected = concatenate_frames(rejected_frames)

    summary_text = create_run_summary(
        root_files=root_files,
        file_summaries=file_summaries,
        accepted=all_accepted,
        rejected=all_rejected,
        elapsed_seconds=elapsed_seconds,
    )

    RUN_SUMMARY_TXT.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)

    print()
    print("FILES SAVED")
    print("-" * 78)
    print(SELECTION_RESULTS_CSV)
    print(ACCEPTED_CSV)
    print(REJECTED_CSV)
    print(FILE_SUMMARY_CSV)
    print(RUN_SUMMARY_TXT)
    print("-" * 78)


if __name__ == "__main__":
    main()
