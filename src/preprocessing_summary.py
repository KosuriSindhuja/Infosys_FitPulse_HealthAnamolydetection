# src/preprocessing_summary.py

from pathlib import Path
import pandas as pd
import datetime as dt

ROOT = Path.cwd()
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(PROC / "preprocessed_data.csv")

    summary_path = OUT / "preprocessing_summary.md"

    with open(summary_path, "w") as f:
        f.write("# Preprocessing Summary\n\n")
        f.write(f"**Generated on:** {dt.datetime.now()}\n\n")
        f.write("## Steps Performed\n")
        f.write("1. Converted all timestamps to uniform format.\n")
        f.write("2. Removed invalid or negative values.\n")
        f.write("3. Cleaned heart rate outliers (<30 or >220 bpm).\n")
        f.write("4. Cleaned steps negative values.\n")
        f.write("5. Converted sleep minutes to hours.\n")
        f.write("6. Resampled everything into a 1-minute timeline.\n")
        f.write("7. Forward-filled missing values.\n")
        f.write("8. Merged heart rate, steps, and sleep into one dataset.\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Total rows: **{len(df)}**\n")
        f.write(f"- Columns: {list(df.columns)}\n\n")

    print(f"Preprocessing summary created at: {summary_path}")

if __name__ == "__main__":
    main()
