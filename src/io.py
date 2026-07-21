from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

# ---------------------------------------------------------------------
# CSV configuration
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {"TIME", "CH1", "CH2"}


# ---------------------------------------------------------------------
# ROOT configuration
# ---------------------------------------------------------------------

ROOT_BRANCHES = {
    "time",
    "channel1",
    "channel2",
}


@dataclass(frozen=True, slots=True)
class WaveformEvent:
    """
    Single detector event loaded from a ROOT file.

    Attributes
    ----------
    file_path : pathlib.Path
        ROOT file containing the event.
    event_index : int
        Index of the event inside the selected TTree.
    time : numpy.ndarray
        Time samples.
    channel1 : numpy.ndarray
        Channel 1 waveform samples.
    channel2 : numpy.ndarray
        Channel 2 waveform samples.
    """

    file_path: Path
    event_index: int
    time: np.ndarray
    channel1: np.ndarray
    channel2: np.ndarray


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load an oscilloscope CSV file and standardize its columns.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the oscilloscope CSV file.

    Returns
    -------
    pandas.DataFrame
        Clean DataFrame containing TIME, CH1, and CH2 columns.
    """

    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    data_start = None

    for index, line in enumerate(lines):
        if line.strip().startswith("TIME") and "CH" in line:
            data_start = index
            break

    if data_start is None:
        raise ValueError(f"No data block found in {filepath}")

    df = pd.read_csv(filepath, skiprows=data_start)

    df.columns = df.columns.astype(str).str.strip().str.upper()

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))

        raise ValueError(
            f"Missing required columns in {filepath.name}: " f"{missing_columns}"
        )

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=list(REQUIRED_COLUMNS))


def load_waveforms(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TIME, CH1, and CH2 from an oscilloscope CSV file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the oscilloscope CSV file.

    Returns
    -------
    tuple of numpy.ndarray
        Time samples, CH1 waveform, and CH2 waveform.
    """

    df = load_csv(filepath)

    time = df["TIME"].to_numpy(dtype=float)
    ch1 = df["CH1"].to_numpy(dtype=float)
    ch2 = df["CH2"].to_numpy(dtype=float)

    return time, ch1, ch2


# ---------------------------------------------------------------------
# ROOT loading
# ---------------------------------------------------------------------


def iter_root_events(
    filepath: str | Path,
    tree_name: str = "t1",
    step_size: str = "100 MB",
) -> Iterator[WaveformEvent]:
    """
    Iterate over waveform events stored in a ROOT TTree.

    Events are loaded in chunks so that large ROOT files do not need to
    be loaded entirely into memory.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the ROOT file.
    tree_name : str, default="t1"
        Name of the TTree containing the waveform data. If several ROOT
        cycles exist, uproot automatically selects the latest cycle.
    step_size : str, default="100 MB"
        Approximate amount of data loaded into memory per chunk.

    Yields
    ------
    WaveformEvent
        Event metadata together with time, channel1, and channel2 arrays.

    Raises
    ------
    FileNotFoundError
        If the ROOT file does not exist.
    KeyError
        If the requested TTree does not exist.
    ValueError
        If required branches are missing or waveform lengths are invalid.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"ROOT file not found: {filepath}")

    with uproot.open(filepath) as root_file:
        if tree_name not in root_file:
            available_objects = [str(key).split(";")[0] for key in root_file.keys()]

            raise KeyError(
                f"TTree '{tree_name}' not found in {filepath.name}. "
                f"Available objects: {sorted(set(available_objects))}"
            )

        tree = root_file[tree_name]

        available_branches = set(tree.keys())
        missing_branches = ROOT_BRANCHES - available_branches

        if missing_branches:
            raise ValueError(
                f"Missing required ROOT branches in {filepath.name}: "
                f"{sorted(missing_branches)}"
            )

        event_index = 0

        for arrays in tree.iterate(
            expressions=sorted(ROOT_BRANCHES),
            step_size=step_size,
            library="ak",
        ):
            number_of_events = len(arrays["time"])

            for local_index in range(number_of_events):
                time = np.asarray(
                    ak.to_numpy(arrays["time"][local_index]),
                    dtype=float,
                )
                channel1 = np.asarray(
                    ak.to_numpy(arrays["channel1"][local_index]),
                    dtype=float,
                )
                channel2 = np.asarray(
                    ak.to_numpy(arrays["channel2"][local_index]),
                    dtype=float,
                )

                if time.size == 0:
                    raise ValueError(
                        f"Empty waveform in {filepath.name}, " f"event {event_index}"
                    )

                if not (time.size == channel1.size == channel2.size):
                    raise ValueError(
                        f"Waveform length mismatch in {filepath.name}, "
                        f"event {event_index}: "
                        f"time={time.size}, "
                        f"channel1={channel1.size}, "
                        f"channel2={channel2.size}"
                    )

                yield WaveformEvent(
                    file_path=filepath,
                    event_index=event_index,
                    time=time,
                    channel1=channel1,
                    channel2=channel2,
                )

                event_index += 1
