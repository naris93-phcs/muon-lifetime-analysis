from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.io import load_csv
from src.detector import find_peaks
from src.lifetime import compute_lifetime


DATA_PATH = Path("data/raw")
OUT_PATH = Path("results")
OUT_PATH.mkdir(parents=True, exist_ok=True)

ACCEPTED_MUON_LIFETIME = 2.197  # microseconds

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

mean_tau = np.mean(lifetimes)
std_tau = np.std(lifetimes)
n_events = len(lifetimes)

plt.figure(figsize=(8, 5))

plt.hist(
    lifetimes,
    bins=16,
    range=(0.8, 6.0),
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    ACCEPTED_MUON_LIFETIME,
    linestyle="--",
    linewidth=2,
    label=f"Accepted μ lifetime ≈ {ACCEPTED_MUON_LIFETIME:.3f} μs"
)

plt.axvline(
    mean_tau,
    linestyle="-",
    linewidth=2,
    label=f"Mean reconstructed = {mean_tau:.3f} μs"
)

text = (
    f"Events: {n_events}\n"
    f"Mean: {mean_tau:.3f} μs\n"
    f"Std: {std_tau:.3f} μs"
)

plt.text(
    0.98,
    0.95,
    text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", alpha=0.15)
)

plt.xlabel("Reconstructed lifetime (μs)")
plt.ylabel("Counts")
plt.title("Cosmic Muon Lifetime Reconstruction")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

outfile = OUT_PATH / "publication_lifetime_hist.png"
plt.savefig(outfile, dpi=200)
plt.show()

print(f"Saved plot to: {outfile}")