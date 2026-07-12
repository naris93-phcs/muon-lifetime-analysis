from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

FILE_PATTERN = "TriggerAuto_*.csv"
HISTOGRAM_PATH = RESULTS_DIR / "lifetime_hist.png"


DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
SUMMARY_PATH = RESULTS_DIR / "summary.txt"
PULSE_STATISTICS_PATH = RESULTS_DIR / "pulse_statistics.csv"
LIFETIME_FIT_PATH = RESULTS_DIR / "lifetime_fit.png"
PUBLICATION_HISTOGRAM_PATH = RESULTS_DIR / "publication_lifetime_hist.png"

T_MIN_US = 0.8
ACCEPTED_MUON_LIFETIME_US = 2.197


# ---------------------------------------------------------------------
# Detector configuration
# ---------------------------------------------------------------------

MIN_DELAY = 0.8e-6          # seconds
MIN_HEIGHT = 0.012          # volts
MIN_PROMINENCE = 0.005      # volts
MIN_WIDTH = 2               # samples
MAX_LIFETIME = 10e-6        # seconds

FIT_MAX_US = 6.0
FIT_BINS = 12
DIAGNOSTIC_FILE_LIMIT = 30

DIAGNOSTIC_X_MIN_OFFSET = 0.6e-6
DIAGNOSTIC_X_MAX_OFFSET = 4.0e-6

DIAGNOSTIC_Y_MIN = -0.03
DIAGNOSTIC_Y_MAX = 0.08