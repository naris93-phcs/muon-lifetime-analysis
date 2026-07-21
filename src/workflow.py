from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis import (
    calculate_fit_range_stability,
    calculate_summary,
    extract_accepted_lifetimes,
    make_fit_range_stability_plot,
    make_histogram,
    save_summary,
)
from src.config import (
    ACCEPTED_CANDIDATES_PATH,
    ACCEPTED_MUON_LIFETIME_US,
    DATA_DIR,
    DETECTOR_RESULTS_PATH,
    FIT_MAX_US,
    FIT_RANGE_STABILITY_PATH,
    FIT_RANGE_STABILITY_PLOT_PATH,
    FIT_RANGE_T_MIN_VALUES_US,
    LIFETIME_HISTOGRAM_PATH,
    PER_FILE_HISTOGRAM_DIR,
    PER_FILE_SUMMARY_PATH,
    REJECTED_CANDIDATES_PATH,
    RESULTS_DIR,
    ROOT_FILE_PATTERN,
    SUMMARY_PATH,
    T_MIN_US,
)
from src.io import iter_root_events
from src.pipeline import analyze_events


@dataclass
class AnalysisRun:
    """
    Container holding the outputs of one full analysis run.
    """

    all_results: list[dict]
    accepted_results: list[dict]
    rejected_results: list[dict]
    per_file_summaries: list[dict]
    skipped_files: list[dict]
    overall_summary: dict
    fit_range_stability: list[dict]


def find_root_files() -> list[Path]:
    """
    Find all ROOT files configured for the analysis.
    """

    root_files = sorted(Path(DATA_DIR).glob(ROOT_FILE_PATTERN))

    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in:\n{DATA_DIR}")

    return root_files


def build_per_file_summary(
    file_path: Path,
    file_results: list[dict],
) -> dict:
    """
    Build the quality-control summary for one ROOT file.
    """

    accepted_results = [
        result for result in file_results if result.get("accepted", False)
    ]

    rejected_results = [
        result for result in file_results if not result.get("accepted", False)
    ]

    lifetimes_us = extract_accepted_lifetimes(file_results)

    summary = calculate_summary(
        lifetimes_us=lifetimes_us,
        t_min_us=T_MIN_US,
        t_max_us=FIT_MAX_US,
    )

    acceptance_fraction = (
        len(accepted_results) / len(file_results) if file_results else 0.0
    )

    return {
        "file": file_path.name,
        "detector_results": len(file_results),
        "accepted_candidates": len(accepted_results),
        "rejected_candidates": len(rejected_results),
        "acceptance_fraction": acceptance_fraction,
        "fit_events": summary["events_used"],
        "t_min_us": T_MIN_US,
        "t_max_us": FIT_MAX_US,
        "mean_us": summary["mean_us"],
        "median_us": summary["median_us"],
        "tau_mle_us": summary["tau_mle_us"],
        "tau_error_us": summary["tau_error_us"],
    }


def save_per_file_histogram(
    file_path: Path,
    file_results: list[dict],
) -> None:
    """
    Save the accepted decay-time histogram for one file.
    """

    lifetimes_us = extract_accepted_lifetimes(file_results)

    histogram_path = PER_FILE_HISTOGRAM_DIR / f"{file_path.stem}_decay_times.png"

    make_histogram(
        lifetimes_us=lifetimes_us,
        savepath=histogram_path,
        t_min_us=T_MIN_US,
        t_max_us=FIT_MAX_US,
        title=(f"Muon Decay Times — {file_path.name}"),
        bins=25,
    )


def process_root_file(
    file_path: Path,
) -> tuple[list[dict], dict]:
    """
    Process one ROOT file and return its detector results
    and quality-control summary.
    """

    events = iter_root_events(file_path)
    file_results = analyze_events(events)

    for result in file_results:
        result.setdefault(
            "source_file",
            file_path.name,
        )

    file_summary = build_per_file_summary(
        file_path=file_path,
        file_results=file_results,
    )

    save_per_file_histogram(
        file_path=file_path,
        file_results=file_results,
    )

    return file_results, file_summary


def save_result_tables(
    all_results: list[dict],
    accepted_results: list[dict],
    rejected_results: list[dict],
    per_file_summaries: list[dict],
    fit_range_stability: list[dict],
) -> None:
    """
    Save all CSV analysis outputs.
    """

    if all_results:
        pd.DataFrame(all_results).to_csv(
            DETECTOR_RESULTS_PATH,
            index=False,
        )

    if accepted_results:
        pd.DataFrame(accepted_results).to_csv(
            ACCEPTED_CANDIDATES_PATH,
            index=False,
        )

    if rejected_results:
        pd.DataFrame(rejected_results).to_csv(
            REJECTED_CANDIDATES_PATH,
            index=False,
        )

    pd.DataFrame(per_file_summaries).to_csv(
        PER_FILE_SUMMARY_PATH,
        index=False,
    )

    pd.DataFrame(fit_range_stability).to_csv(
        FIT_RANGE_STABILITY_PATH,
        index=False,
    )


def save_analysis_plots(
    lifetimes_us,
    fit_range_stability: list[dict],
) -> None:
    """
    Save the combined lifetime and stability plots.
    """

    make_histogram(
        lifetimes_us=lifetimes_us,
        savepath=LIFETIME_HISTOGRAM_PATH,
        t_min_us=T_MIN_US,
        t_max_us=FIT_MAX_US,
        title=(
            "Cosmic Muon Decay-Time Distribution "
            f"({T_MIN_US:.2f}–{FIT_MAX_US:.2f} μs)"
        ),
        bins=30,
    )

    make_fit_range_stability_plot(
        stability_results=fit_range_stability,
        savepath=FIT_RANGE_STABILITY_PLOT_PATH,
        accepted_lifetime_us=(ACCEPTED_MUON_LIFETIME_US),
    )


def run_analysis(
    root_files: list[Path],
) -> AnalysisRun:
    """
    Run the complete ROOT muon-lifetime workflow.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PER_FILE_HISTOGRAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: list[dict] = []
    per_file_summaries: list[dict] = []
    skipped_files: list[dict] = []

    for file_number, file_path in enumerate(
        root_files,
        start=1,
    ):
        print(
            f"[{file_number:02d}/{len(root_files):02d}] " f"Processing {file_path.name}"
        )

        try:
            (
                file_results,
                file_summary,
            ) = process_root_file(file_path)

        except ValueError as error:
            print()
            print(f"WARNING: Skipping {file_path.name}")
            print(f"Reason: {error}")
            print()

            skipped_files.append(
                {
                    "file": file_path.name,
                    "reason": str(error),
                }
            )

            continue

        all_results.extend(file_results)
        per_file_summaries.append(file_summary)

        print(
            f"    Results: "
            f"{file_summary['detector_results']} | "
            f"Accepted: "
            f"{file_summary['accepted_candidates']} | "
            f"Fit events: "
            f"{file_summary['fit_events']} | "
            f"Tau: "
            f"{file_summary['tau_mle_us']:.4f} μs"
        )

    accepted_results = [
        result for result in all_results if result.get("accepted", False)
    ]

    rejected_results = [
        result for result in all_results if not result.get("accepted", False)
    ]

    lifetimes_us = extract_accepted_lifetimes(all_results)

    overall_summary = calculate_summary(
        lifetimes_us=lifetimes_us,
        t_min_us=T_MIN_US,
        t_max_us=FIT_MAX_US,
    )

    fit_range_stability = calculate_fit_range_stability(
        lifetimes_us=lifetimes_us,
        t_min_values_us=(FIT_RANGE_T_MIN_VALUES_US),
        t_max_us=FIT_MAX_US,
    )

    save_result_tables(
        all_results=all_results,
        accepted_results=accepted_results,
        rejected_results=rejected_results,
        per_file_summaries=per_file_summaries,
        fit_range_stability=fit_range_stability,
    )

    save_analysis_plots(
        lifetimes_us=lifetimes_us,
        fit_range_stability=fit_range_stability,
    )

    save_summary(
        summary=overall_summary,
        savepath=SUMMARY_PATH,
    )

    return AnalysisRun(
        all_results=all_results,
        accepted_results=accepted_results,
        rejected_results=rejected_results,
        per_file_summaries=per_file_summaries,
        skipped_files=skipped_files,
        overall_summary=overall_summary,
        fit_range_stability=fit_range_stability,
    )
