import matplotlib.pyplot as plt


def plot_event(time, ch1, ch2, t0=None, t1=None, savepath=None):

    plt.figure(figsize=(10, 5))

    plt.plot(time, ch1, label="CH1 (Muon signal)")

    plt.plot(time, ch2, label="CH2 (Coincidence)")

    if t0 is not None:
        plt.axvline(t0, linestyle="--", label="t0")

    if t1 is not None:
        plt.axvline(t1, linestyle="--", label="t1")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)

    plt.close()
