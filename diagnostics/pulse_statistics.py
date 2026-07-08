from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from pathlib import Path
import csv
import numpy as np
from scipy.signal import find_peaks

from src.io import load_csv

DATA_PATH = Path("data/raw")
OUTFILE = Path("results/pulse_statistics.csv")

files = sorted(DATA_PATH.glob("TriggerAuto_*.csv"))

rows = []

for f in files:

    df = load_csv(f)

    time = df["TIME"].values
    ch1 = df["CH1"].values
    ch2 = df["CH2"].values

    time = np.array(time)
    ch1 = np.array(ch1)
    ch2 = np.array(ch2)

    # Trigger
    t0_idx = np.argmax(ch2)
    t0 = time[t0_idx]

    # Start searching after veto
    min_delay = 0.8e-6
    search_start = np.searchsorted(time, t0 + min_delay)

    if search_start >= len(ch1):
        continue

    ch = ch1[search_start:]

    candidates = []

    # Positive pulses
    peaks, props = find_peaks(
        ch,
        height=0.012,
        prominence=0.005,
        width=2
    )

    for i, p in enumerate(peaks):
        candidates.append({
            "idx": p,
            "polarity": "positive",
            "prominence": props["prominences"][i],
            "width": props["widths"][i],
            "height": props["peak_heights"][i]
        })

    # Negative pulses
    peaks, props = find_peaks(
        -ch,
        height=0.012,
        prominence=0.005,
        width=2
    )

    for i, p in enumerate(peaks):
        candidates.append({
            "idx": p,
            "polarity": "negative",
            "prominence": props["prominences"][i],
            "width": props["widths"][i],
            "height": props["peak_heights"][i]
        })

    if len(candidates) == 0:
        continue

    best = max(candidates, key=lambda x: x["prominence"])

    t1 = time[search_start + best["idx"]]
    tau = (t1 - t0) * 1e6

    rows.append([
        f.name,
        tau,
        best["polarity"],
        best["height"],
        best["prominence"],
        best["width"]
    ])

with open(OUTFILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "file",
        "tau_us",
        "polarity",
        "height",
        "prominence",
        "width"
    ])

    writer.writerows(rows)

print(f"Saved {len(rows)} events to {OUTFILE}")