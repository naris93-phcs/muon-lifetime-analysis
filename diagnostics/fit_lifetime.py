from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.io import load_csv
from src.detector import find_peaks
from src.lifetime import compute_lifetime


DATA_PATH = Path("data/raw")
OUT_PATH = Path("results")
OUT_PATH.mkdir(parents=True, exist_ok=True)


def exponential(t, A, tau, C):
    return A * np.exp(-t / tau) + C


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
        lifetimes.append(tau * 1e6)  # seconds -> μs


lifetimes = np.array(lifetimes)

print(f"Events used: {len(lifetimes)}")

# Histogram
counts, bin_edges = np.histogram(
    lifetimes,
    bins=12,
    range=(0.8, 6.0)
)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Fit μόνο στα bins που έχουν counts > 0
mask = counts > 0

x_fit = bin_centers[mask]
y_fit = counts[mask]

# Initial guesses
A0 = max(y_fit)
tau0 = 2.2
C0 = 0.0

popt, pcov = curve_fit(
    exponential,
    x_fit,
    y_fit,
    p0=[A0, tau0, C0],
    maxfev=10000
)

A_fit, tau_fit, C_fit = popt
tau_err = np.sqrt(np.diag(pcov))[1]

print("========================")
print("Exponential Fit")
print("========================")
print(f"Fitted lifetime tau = {tau_fit:.3f} ± {tau_err:.3f} μs")
print("========================")

# Smooth curve
t_curve = np.linspace(0.8, 6.0, 300)
y_curve = exponential(t_curve, A_fit, tau_fit, C_fit)

plt.figure(figsize=(8, 5))

plt.hist(
    lifetimes,
    bins=12,
    range=(0.8, 6.0),
    alpha=0.7,
    label="Data"
)

plt.plot(
    t_curve,
    y_curve,
    linewidth=2,
    label=f"Fit: tau = {tau_fit:.2f} ± {tau_err:.2f} μs"
)

plt.xlabel("Muon lifetime τ (μs)")
plt.ylabel("Counts")
plt.title("Muon Lifetime Histogram with Exponential Fit")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUT_PATH / "lifetime_fit.png", dpi=150)
plt.show()