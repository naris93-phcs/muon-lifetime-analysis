import numpy as np
import matplotlib.pyplot as plt


def make_histogram(lifetimes, savepath=None):

    # remove invalid events
    lifetimes = [t for t in lifetimes if t is not None]

    # convert seconds -> microseconds
    lifetimes_us = np.array(lifetimes) * 1e6

    # -------------------------
    # Histogram
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.hist(lifetimes_us, bins=25, edgecolor="black", alpha=0.8)

    plt.xlabel("Muon lifetime (μs)")
    plt.ylabel("Counts")
    plt.title("Cosmic Muon Lifetime Distribution")

    plt.grid(alpha=0.3)

    if savepath:
        plt.savefig(savepath, dpi=150)

    plt.close()

    # -------------------------
    # Statistics
    # -------------------------
    mean = np.mean(lifetimes_us)
    std = np.std(lifetimes_us)

    print()
    print("========================")
    print("Muon Lifetime Analysis")
    print("========================")
    print(f"Events used: {len(lifetimes_us)}")
    print(f"Mean lifetime = {mean:.3f} μs")
    print(f"Std deviation = {std:.3f} μs")
    print("========================")

    return mean
