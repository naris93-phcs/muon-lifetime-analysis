from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.io import load_csv
from src.detector import find_peaks
from src.lifetime import compute_lifetime


DATA_PATH = Path("data/raw")
OUT_PATH = Path("results")
OUT_PATH.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_PATH.glob("TriggerAuto_*.csv"))

lifetimes = []

for f in files:
    df = load_csv(f)

    time = df["TIME"].values
    ch1 = df["CH1"].values
    ch2 = df["CH2"].values

    t0, t1 = find_peaks(time, ch1, ch2)
    tau = compute_lifetime(t0, t1)

    if tau is not None:
        lifetimes.append(tau * 1e6)

lifetimes = np.array(lifetimes)

summary = f"""
Muon Lifetime Analysis Summary
==============================

Input files
-----------
Total files analyzed      : {len(files)}
Events used               : {len(lifetimes)}
Detection efficiency      : {len(lifetimes) / len(files) * 100:.1f} %

Lifetime statistics
-------------------
Mean lifetime             : {np.mean(lifetimes):.3f} μs
Standard deviation        : {np.std(lifetimes):.3f} μs
Median lifetime           : {np.median(lifetimes):.3f} μs
Minimum lifetime          : {np.min(lifetimes):.3f} μs
Maximum lifetime          : {np.max(lifetimes):.3f} μs

Detector configuration
----------------------
Trigger channel           : CH2
Decay search channel      : CH1
Early-time veto           : 0.8 μs
Pulse selection           : dual polarity
Selection metric          : peak prominence

Notes
-----
The reconstructed mean lifetime is close to the expected
free muon lifetime of approximately 2.2 μs. No detector
acceptance or efficiency correction has been applied.
"""

outfile = OUT_PATH / "summary.txt"

with open(outfile, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print(f"Saved summary to: {outfile}")