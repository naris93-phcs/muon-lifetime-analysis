from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks

from src.io import load_csv


DATA_PATH = Path("data/raw")
OUT_PATH = Path("results/diagnostics")
OUT_PATH.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_PATH.glob("TriggerAuto_*.csv"))

N_FILES = 30

for i, f in enumerate(files[:N_FILES]):

    df = load_csv(f)

    time = df["TIME"].values
    ch1 = df["CH1"].values
    ch2 = df["CH2"].values

    t0_idx = np.argmax(ch2)
    t0 = time[t0_idx]

    min_delay = 0.8e-6
    search_start = np.searchsorted(time, time[t0_idx] + min_delay)

    if search_start >= len(ch1):
        continue

    ch1_after = ch1[search_start:]
    time_after = time[search_start:]

    peaks, properties = scipy_find_peaks(
        ch1_after,
        height=0.012,
        prominence=0.005,
        width=2
    )

    plt.figure(figsize=(10, 5))

    plt.plot(time * 1e6, ch1, label="CH1")
    

    plt.axvline(t0 * 1e6, linestyle="--", label="t0 trigger")
    plt.axvline(time[search_start] * 1e6, linestyle=":", label="search start")

    if len(peaks) > 0:
        peak_times = time_after[peaks]
        peak_values = ch1_after[peaks]

        plt.scatter(
            peak_times * 1e6,
            peak_values,
            marker="x",
            label="candidate peaks"
        )

        best_peak_rel = peaks[np.argmax(properties["prominences"])]
        best_time = time_after[best_peak_rel]
        best_value = ch1_after[best_peak_rel]

        plt.scatter(
            best_time * 1e6,
            best_value,
            s=100,
            marker="o",
            label="selected peak"
        )

        tau = best_time - t0
        title = f"{f.name} | tau = {tau * 1e6:.3f} μs"
    else:
        title = f"{f.name} | no decay candidate"

    plt.title(title)
    plt.xlabel("Time (μs)")
    plt.ylabel("Voltage")
    plt.xlim((t0 + 0.6e-6) * 1e6, (t0 + 4.0e-6) * 1e6)
    plt.ylim(-0.03, 0.08)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    savefile = OUT_PATH / f"diagnostic_{i:03d}.png"
    plt.savefig(savefile, dpi=150)
    plt.close()

print(f"Saved diagnostic plots in: {OUT_PATH}")