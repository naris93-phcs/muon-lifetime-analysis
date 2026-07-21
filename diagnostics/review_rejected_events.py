from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


ROOT_FILE = Path("data/root/CH_t12_141223_1.root")
REJECTED_CSV = Path(
    "results/root_decay_selection/rejected_candidates.csv"
)
OUTPUT_DIR = Path(
    "results/root_decay_selection/rejected_event_review"
)

TREE_NAME = "t1"

# Time window around the rejected candidate.
WINDOW_BEFORE_US = 0.50
WINDOW_AFTER_US = 0.50


def baseline_subtract(
    waveform: np.ndarray,
    baseline_samples: int = 500,
) -> np.ndarray:
    """Subtract the median of the initial waveform samples."""
    baseline_samples = min(baseline_samples, len(waveform))
    baseline = np.median(waveform[:baseline_samples])
    return waveform - baseline


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ROOT_FILE.exists():
        raise FileNotFoundError(
            f"ROOT file not found: {ROOT_FILE}\n"
            "Update ROOT_FILE near the top of the script."
        )

    if not REJECTED_CSV.exists():
        raise FileNotFoundError(
            f"Rejected-candidate CSV not found: {REJECTED_CSV}"
        )

    rejected = pd.read_csv(REJECTED_CSV)

    required_columns = {
        "event_index",
        "decay_time_us",
        "rejection_reason",
        "channel2_amplitude_v",
        "channel1_veto_amplitude_v",
        "channel1_to_channel2_ratio",
    }

    missing_columns = required_columns - set(rejected.columns)

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            f"{sorted(missing_columns)}"
        )

    with uproot.open(ROOT_FILE) as root_file:
        tree = root_file[TREE_NAME]

        times = tree["time"].array(library="np")
        channel1 = tree["channel1"].array(library="np")
        channel2 = tree["channel2"].array(library="np")

        for _, row in rejected.iterrows():
            event_index = int(row["event_index"])
            decay_time_us = float(row["decay_time_us"])
            rejection_reason = str(row["rejection_reason"])

            time_s = np.asarray(times[event_index], dtype=float)
            ch1_raw = np.asarray(channel1[event_index], dtype=float)
            ch2_raw = np.asarray(channel2[event_index], dtype=float)

            # Convert time to microseconds.
            time_us = time_s * 1e6

            # Baseline subtraction and pulse inversion.
            # The detector pulses are negative in the raw waveform,
            # so inversion makes their amplitudes positive.
            ch1 = -baseline_subtract(ch1_raw)
            ch2 = -baseline_subtract(ch2_raw)

            candidate_index = int(
                np.argmin(np.abs(time_us - decay_time_us))
            )

            candidate_time_us = time_us[candidate_index]

            window_mask = (
                (time_us >= candidate_time_us - WINDOW_BEFORE_US)
                & (time_us <= candidate_time_us + WINDOW_AFTER_US)
            )

            if not np.any(window_mask):
                raise RuntimeError(
                    f"No samples found around event {event_index}, "
                    f"candidate time {candidate_time_us:.6f} us."
                )

            local_indices = np.flatnonzero(window_mask)

            ch1_local_index = local_indices[
                np.argmax(ch1[window_mask])
            ]
            ch2_local_index = local_indices[
                np.argmax(ch2[window_mask])
            ]

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
            )

            axes[0].plot(
                time_us[window_mask],
                ch1[window_mask],
                linewidth=1.0,
                label="CH1",
            )
            axes[0].axvline(
                candidate_time_us,
                linestyle="--",
                linewidth=1.2,
                label="Rejected candidate time",
            )
            axes[0].scatter(
                time_us[ch1_local_index],
                ch1[ch1_local_index],
                marker="x",
                s=90,
                linewidths=2,
                label="CH1 veto pulse",
                zorder=5,
            )

            axes[0].set_ylabel("Inverted amplitude (V)")
            axes[0].set_title(
                f"Event {event_index} — CH1 simultaneous-pulse veto"
            )
            axes[0].grid(alpha=0.25)
            axes[0].legend()

            axes[1].plot(
                time_us[window_mask],
                ch2[window_mask],
                linewidth=1.0,
                label="CH2",
            )
            axes[1].axvline(
                candidate_time_us,
                linestyle="--",
                linewidth=1.2,
                label="Rejected candidate time",
            )
            axes[1].scatter(
                time_us[ch2_local_index],
                ch2[ch2_local_index],
                marker="x",
                s=90,
                linewidths=2,
                label="CH2 delayed pulse",
                zorder=5,
            )

            axes[1].set_xlabel("Time (μs)")
            axes[1].set_ylabel("Inverted amplitude (V)")
            axes[1].grid(alpha=0.25)
            axes[1].legend()

            figure_text = (
                f"Reason: {rejection_reason}\n"
                f"Decay time: {decay_time_us:.4f} μs\n"
                f"CSV CH2 amplitude: "
                f"{row['channel2_amplitude_v']:.4f} V\n"
                f"CSV CH1 veto amplitude: "
                f"{row['channel1_veto_amplitude_v']:.4f} V\n"
                f"CH1 / CH2 ratio: "
                f"{row['channel1_to_channel2_ratio']:.2f}\n"
                f"CH2 SNR: {row['channel2_snr']:.1f}\n"
                f"CH1 veto SNR: {row['channel1_veto_snr']:.1f}"
            )

            fig.text(
                0.73,
                0.50,
                figure_text,
                va="center",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.1,
                },
            )

            fig.suptitle(
                "Rejected delayed-pulse candidate",
                fontsize=14,
            )

            fig.tight_layout(rect=(0, 0, 0.71, 0.95))

            output_path = (
                OUTPUT_DIR
                / f"rejected_event_{event_index:05d}.png"
            )

            fig.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(
                f"Saved event {event_index}: {output_path}"
            )

    print()
    print(f"Rejected candidates reviewed: {len(rejected)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()