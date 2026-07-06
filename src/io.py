import pandas as pd


def load_csv(filepath):

    with open(filepath, "r") as f:
        lines = f.readlines()


    data_start = None


    for i, line in enumerate(lines):

        if line.strip().startswith("TIME") and "CH" in line:
            data_start = i
            break


    if data_start is None:
        raise ValueError(f"No data block found in {filepath}")


    df = pd.read_csv(
        filepath,
        skiprows=data_start
    )


    df.columns = [
        c.strip().upper()
        for c in df.columns
    ]


    return df