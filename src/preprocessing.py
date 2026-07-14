import numpy as np


def normalize(signal):
    signal = np.array(signal)
    return signal - np.mean(signal)


def smooth(signal, window=5):
    signal = np.array(signal)
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")
