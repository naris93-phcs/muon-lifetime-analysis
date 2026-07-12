from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"TIME", "CH1", "CH2"}


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
        raise ValueError(
            f"Missing required columns in {filepath.name}: "
            f"{sorted(REQUIRED_COLUMNS - set(df.columns))}"
        )

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=list(REQUIRED_COLUMNS))


def load_waveforms(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TIME, CH1, and CH2 directly as NumPy arrays.

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

    time = df["TIME"].to_numpy()
    ch1 = df["CH1"].to_numpy()
    ch2 = df["CH2"].to_numpy()

    return time, ch1, ch2