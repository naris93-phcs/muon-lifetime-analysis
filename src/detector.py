import numpy as np


def find_peaks(time, ch1, ch2):

    time = np.array(time)
    ch1 = np.array(ch1)
    ch2 = np.array(ch2)


    # trigger peak από CH2
    t0_idx = np.argmax(ch2)


    # decay peak μετά το trigger
    ch1_after = ch1[t0_idx:]


    if len(ch1_after) == 0:
        return None, None


    t1_idx_rel = np.argmax(ch1_after)

    t1_idx = t0_idx + t1_idx_rel


    return time[t0_idx], time[t1_idx]