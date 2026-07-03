import pandas as pd
import matplotlib.pyplot as plt
import glob

folder = r"C:\Users\naris\OneDrive\Desktop\muon_lifetime\muon_lifetime"
files = glob.glob(folder + r"\*.csv")

while True:
    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(i, f.split("\\")[-1])

    choice = input("\nChoose file index (or q to quit): ")

    if choice.lower() == "q":
        break

    choice = int(choice)

    df = pd.read_csv(files[choice], skiprows=15)

    if "CH1" not in df.columns:
        df = pd.read_csv(files[choice], skiprows=15, header=None)
        df = df.iloc[:, :3]
        df.columns = ["TIME", "CH1", "CH2"]

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "")
    )

    print("\nColumns detected:", df.columns)

    if "CH1" not in df.columns or "TIME" not in df.columns:
        raise ValueError(f"Missing required columns. Found: {df.columns}")

    df["CH1"] = -df["CH1"]

    plt.close('all')   # κλείνει ΟΛΑ τα προηγούμενα figures

    plt.figure()
    plt.scatter(df["TIME"], df["CH1"], s=1)
    plt.xlabel("Time")
    plt.ylabel("CH1")
    plt.title(f"Event {choice} — Muon Dataset")

    plt.show(block=False)
    plt.pause(0.1)