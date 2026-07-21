"""
Central configuration for the ROOT muon-lifetime analysis.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "root"
RESULTS_DIR = PROJECT_ROOT / "results" / "root_analysis"

ROOT_FILE_PATTERN = "*.root"

ACCEPTED_CANDIDATES_PATH = RESULTS_DIR / "accepted_candidates.csv"

REJECTED_CANDIDATES_PATH = RESULTS_DIR / "rejected_candidates.csv"

LIFETIME_HISTOGRAM_PATH = RESULTS_DIR / "accepted_decay_times.png"

SUMMARY_PATH = RESULTS_DIR / "summary.txt"


# ---------------------------------------------------------------------
# Physics reference values
# ---------------------------------------------------------------------

T_MIN_US = 0.80
FIT_MAX_US = 6.00
ACCEPTED_MUON_LIFETIME_US = 2.197


# ---------------------------------------------------------------------
# ROOT detector configuration
# ---------------------------------------------------------------------

# Pre-trigger region used for baseline and noise estimation.
BASELINE_END_US = -0.10

# Initial coincidence-trigger window.
TRIGGER_START_US = -0.10
TRIGGER_END_US = 0.20
TRIGGER_MIN_SIGMA = 20.0

# Delayed-pulse search window.
DECAY_SEARCH_START_US = 0.80
DECAY_SEARCH_END_US = 9.00

# Delayed CH2 pulse requirements.
DELAYED_HEIGHT_SIGMA = 6.0
DELAYED_PROMINENCE_SIGMA = 6.0

MIN_PEAK_DISTANCE_NS = 40.0
MIN_PEAK_WIDTH_NS = 2.0

# CH1 coincidence veto.
COINCIDENCE_WINDOW_NS = 40.0
CH1_VETO_SIGMA = 8.0
CH1_TO_CH2_MAX_RATIO = 0.20

# Retain only the strongest delayed CH2 candidate in each event.
ONE_CANDIDATE_PER_EVENT = True


# ---------------------------------------------------------------------
# ROOT input configuration
# ---------------------------------------------------------------------

ROOT_TREE_NAME = "t1"

TIME_BRANCH = "time"
CHANNEL1_BRANCH = "channel1"
CHANNEL2_BRANCH = "channel2"

ROOT_CHUNK_SIZE = 100

# ---------------------------------------------------------------------------
# Physics-analysis configuration
# ---------------------------------------------------------------------------

FIT_RANGE_T_MIN_VALUES_US = (
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

ACCEPTED_CANDIDATES_PATH = RESULTS_DIR / "accepted_candidates.csv"

REJECTED_CANDIDATES_PATH = RESULTS_DIR / "rejected_candidates.csv"

DETECTOR_RESULTS_PATH = RESULTS_DIR / "detector_results.csv"

PER_FILE_SUMMARY_PATH = RESULTS_DIR / "per_file_summary.csv"

FIT_RANGE_STABILITY_PATH = RESULTS_DIR / "fit_range_stability.csv"

LIFETIME_HISTOGRAM_PATH = RESULTS_DIR / "accepted_decay_times.png"

FIT_RANGE_STABILITY_PLOT_PATH = RESULTS_DIR / "fit_range_stability.png"

SUMMARY_PATH = RESULTS_DIR / "summary.txt"

PER_FILE_HISTOGRAM_DIR = RESULTS_DIR / "per_file_histograms"

VALIDATION_DIR = RESULTS_DIR / "validation"
