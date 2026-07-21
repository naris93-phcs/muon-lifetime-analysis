import pandas as pd

from src.analysis import format_summary
from src.config import (
    ACCEPTED_CANDIDATES_PATH,
    DETECTOR_RESULTS_PATH,
    FIT_RANGE_STABILITY_PATH,
    FIT_RANGE_STABILITY_PLOT_PATH,
    LIFETIME_HISTOGRAM_PATH,
    PER_FILE_HISTOGRAM_DIR,
    PER_FILE_SUMMARY_PATH,
    REJECTED_CANDIDATES_PATH,
    SUMMARY_PATH,
)
from src.workflow import AnalysisRun


def print_analysis_header(
    root_file_count: int,
    t_min_us: float,
    t_max_us: float,
) -> None:
    """
    Print the analysis-start report.
    """

    print()
    print("=" * 70)
    print("Muon Lifetime ROOT Analysis")
    print("=" * 70)
    print(f"ROOT files found: {root_file_count}")
    print(f"Main fit range : " f"{t_min_us:.2f}–{t_max_us:.2f} μs")
    print()


def print_fit_range_stability(
    stability_results: list[dict],
) -> None:
    """
    Print the fit-range stability table.
    """

    dataframe = pd.DataFrame(stability_results)

    print()
    print("Fit-range stability")
    print("===================")

    print(
        dataframe.to_string(
            index=False,
            columns=[
                "t_min_us",
                "t_max_us",
                "events_used",
                "tau_mle_us",
                "tau_error_us",
            ],
            formatters={
                "t_min_us": "{:.2f}".format,
                "t_max_us": "{:.2f}".format,
                "tau_mle_us": "{:.4f}".format,
                "tau_error_us": "{:.4f}".format,
            },
        )
    )


def print_per_file_summary(
    per_file_summaries: list[dict],
) -> None:
    """
    Print the per-file quality-control table.
    """

    dataframe = pd.DataFrame(per_file_summaries)

    print()
    print("Per-file quality control")
    print("========================")

    print(
        dataframe.to_string(
            index=False,
            columns=[
                "file",
                "detector_results",
                "accepted_candidates",
                "acceptance_fraction",
                "fit_events",
                "tau_mle_us",
                "tau_error_us",
            ],
            formatters={
                "acceptance_fraction": ("{:.3f}".format),
                "tau_mle_us": "{:.4f}".format,
                "tau_error_us": "{:.4f}".format,
            },
        )
    )


def print_skipped_files(
    skipped_files: list[dict],
) -> None:
    """
    Print ROOT files that were skipped.
    """

    if not skipped_files:
        return

    print()
    print("Skipped files")
    print("=============")

    for skipped_file in skipped_files:
        print(f"- {skipped_file['file']}: " f"{skipped_file['reason']}")


def print_output_paths() -> None:
    """
    Print the generated output paths.
    """

    print()
    print(f"Detector CSV      : " f"{DETECTOR_RESULTS_PATH}")
    print(f"Accepted CSV      : " f"{ACCEPTED_CANDIDATES_PATH}")
    print(f"Rejected CSV      : " f"{REJECTED_CANDIDATES_PATH}")
    print(f"Per-file summary  : " f"{PER_FILE_SUMMARY_PATH}")
    print(f"Stability CSV     : " f"{FIT_RANGE_STABILITY_PATH}")
    print(f"Histogram         : " f"{LIFETIME_HISTOGRAM_PATH}")
    print(f"Stability plot    : " f"{FIT_RANGE_STABILITY_PLOT_PATH}")
    print(f"Summary           : " f"{SUMMARY_PATH}")
    print(f"Per-file plots    : " f"{PER_FILE_HISTOGRAM_DIR}")


def print_analysis_report(
    analysis_run: AnalysisRun,
) -> None:
    """
    Print the final terminal report.
    """

    print()
    print("=" * 70)
    print("Analysis finished")
    print("=" * 70)

    print(f"Total detector results : " f"{len(analysis_run.all_results)}")
    print(f"Accepted candidates    : " f"{len(analysis_run.accepted_results)}")
    print(f"Rejected results       : " f"{len(analysis_run.rejected_results)}")
    print(f"Skipped ROOT files     : " f"{len(analysis_run.skipped_files)}")

    print()
    print(format_summary(analysis_run.overall_summary))

    print_fit_range_stability(analysis_run.fit_range_stability)

    print_per_file_summary(analysis_run.per_file_summaries)

    print_skipped_files(analysis_run.skipped_files)

    print_output_paths()

    print("=" * 70)
