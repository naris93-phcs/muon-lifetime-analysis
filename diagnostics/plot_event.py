from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import uproot
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"

file_path = ROOT_DATA_DIR / "CH_t12_150124.root"

with uproot.open(file_path) as root_file:
    tree = root_file["t1"]

    event = tree.arrays(
        ["time", "channel1", "channel2"],
        entry_start=1,
        entry_stop=2,
        library="ak",
    )

time = ak.to_numpy(event["time"][0])
ch1 = ak.to_numpy(event["channel1"][0])
ch2 = ak.to_numpy(event["channel2"][0])

print("len(time):", len(time))
print("len(ch1):", len(ch1))
print("len(ch2):", len(ch2))
print("CH1 min:", ch1.min())
print("CH1 max:", ch1.max())
print("CH1 unique values:", len(np.unique(ch1)))

print("CH2 min:", ch2.min())
print("CH2 max:", ch2.max())
print("CH2 unique values:", len(np.unique(ch2)))

# Αν τα μήκη δεν ταιριάζουν, σχεδίασε μόνο ό,τι κοινό έχουν.
n = min(len(time), len(ch1), len(ch2))

plt.figure(figsize=(12,5))
plt.plot(time[:n] * 1e6, ch1[:n], label="CH1")
plt.plot(time[:n] * 1e6, ch2[:n], label="CH2")

plt.xlabel("Time (μs)")
plt.ylabel("Voltage (V)")
plt.title("CH_t12_150124.root - Event 1")
plt.grid(True)
plt.legend()

plt.show()