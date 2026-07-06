import numpy as np
import matplotlib.pyplot as plt


def make_histogram(lifetimes, savepath=None):

    lifetimes = [
        t for t in lifetimes
        if t is not None
    ]


    lifetimes_us = np.array(lifetimes) * 1e6


    plt.figure(figsize=(8,5))

    plt.hist(
        lifetimes_us,
        bins=25
    )


    plt.xlabel("Muon lifetime (μs)")
    plt.ylabel("Counts")
    plt.title("Muon Lifetime Distribution")


    if savepath:
        plt.savefig(savepath)


    plt.close()


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