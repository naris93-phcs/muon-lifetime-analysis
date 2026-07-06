import pandas as pd


def load_file(path):
    """
    Load muon CSV file and standardize format.
    """

    df = pd.read_csv(path, skiprows=15)

    # fallback if headers are broken
    if "CH1" not in df.columns:
        df = pd.read_csv(path, skiprows=15, header=None)
        df = df.iloc[:, :3]
        df.columns = ["TIME", "CH1", "CH2"]

    # clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # safety check
    required = {"TIME", "CH1", "CH2"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns: {df.columns}")

    # convert to numeric
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
    df["CH1"] = pd.to_numeric(df["CH1"], errors="coerce")
    df["CH2"] = pd.to_numeric(df["CH2"], errors="coerce")

    # drop NaNs
    df = df.dropna()

    # invert signal (physics convention)
    df["CH1"] = -df["CH1"]

    return df