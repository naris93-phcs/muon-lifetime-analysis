from src.config import (
    FIT_MAX_US,
    T_MIN_US,
)
from src.reporting import (
    print_analysis_header,
    print_analysis_report,
)
from src.workflow import (
    find_root_files,
    run_analysis,
)


def main() -> None:
    """
    Run the complete muon-lifetime analysis.
    """

    root_files = find_root_files()

    print_analysis_header(
        root_file_count=len(root_files),
        t_min_us=T_MIN_US,
        t_max_us=FIT_MAX_US,
    )

    analysis_run = run_analysis(root_files=root_files)

    print_analysis_report(analysis_run)


if __name__ == "__main__":
    main()
