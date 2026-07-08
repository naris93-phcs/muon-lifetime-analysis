from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.io import load_csv
from src.detector import find_peaks
from src.lifetime import compute_lifetime


DATA_PATH = Path("data/raw")

T_MIN_US = 0.8

files = sorted(DATA_PATH.glob("TriggerAuto_*.csv"))

lifetimes_us = []

for f in files:
    df = load_csv(f)

    time = df["TIME"].values
    ch1 = df["CH1"].values
    ch2 = df["CH2"].values

    t0, t1 = find_peaks(time, ch1, ch2)
    tau = compute_lifetime(t0, t1)

    if tau is not None:
        lifetimes_us.append(tau * 1e6)

lifetimes_us = np.array(lifetimes_us)

# MLE για truncated exponential με cutoff t_min:
# tau_hat = mean(t - t_min)
shifted = lifetimes_us - T_MIN_US
shifted = shifted[shifted > 0]

tau_mle = np.mean(shifted)
tau_err = tau_mle / np.sqrt(len(shifted))

print("========================")
print("Muon Lifetime MLE")
print("========================")
print(f"Events used: {len(shifted)}")
print(f"t_min = {T_MIN_US:.2f} μs")
print(f"tau_MLE = {tau_mle:.3f} ± {tau_err:.3f} μs")
print("========================")