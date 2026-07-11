from pathlib import Path
from src.io import load_csv
from src.detector import find_peaks
from src.lifetime import compute_lifetime
from src.analysis import make_histogram


DATA_PATH = Path("data/raw")

files = sorted(DATA_PATH.glob("TriggerAuto_*.csv"))

print(f"Found {len(files)} files")

lifetimes = []

for f in files:

    df = load_csv(f)

    time = df["TIME"].values
    ch1 = df["CH1"].values
    ch2 = df["CH2"].values

    t0, t1 = find_peaks(time, ch1, ch2)

    tau = compute_lifetime(t0, t1)


    if tau is not None:
        lifetimes.append(tau)


# Create output directory if it does not already exist
Path("results").mkdir(parents=True, exist_ok=True)

# Generate and save histogram
make_histogram(
    lifetimes,
    savepath="results/lifetime_hist.png"
    )