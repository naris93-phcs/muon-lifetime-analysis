from src.analysis import make_histogram
from src.pipeline import calculate_lifetimes
from src.config import (
    DATA_DIR,
    FILE_PATTERN,
    HISTOGRAM_PATH,
    RESULTS_DIR,
)


def main() -> None:
    """Run the complete muon lifetime analysis pipeline."""

    files = sorted(DATA_DIR.glob(FILE_PATTERN))

    print(f"Found {len(files)} files")

    lifetimes = calculate_lifetimes(files)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    make_histogram(
        lifetimes,
        savepath=HISTOGRAM_PATH,
    )


if __name__ == "__main__":
    main()
