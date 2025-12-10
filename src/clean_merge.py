# src/clean_merge.py
"""
Fast, robust cleaning & merging for Fitbit CSVs.
This file parses the three CSVs (heart rate, steps, sleep) using the
formats discovered in your samples, converts timestamps to UTC,
resamples to 1-minute resolution, and writes processed outputs.

Run:
    py -3.11 src/clean_merge.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytz
import time
from typing import Optional

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

HR_FILE = RAW / "fitbit_heartrate_raw.csv"
STEPS_FILE = RAW / "fitbit_steps_raw.csv"
SLEEP_FILE = RAW / "fitbit_sleep_raw.csv"

# use '1min' instead of deprecated 'T'
RESAMPLE = "1min"     # 1 minute frequency
TEST_RATIO = 0.20     # 80/20 split
LOCAL_TZ = "Asia/Kolkata"
UTC = pytz.UTC

# Exact formats discovered from your sample files:
HR_FMT = "%m/%d/%y %H:%M:%S"
STEPS_FMT = "%Y-%m-%dT%H:%M:%S"
SLEEP_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_with_format(series: pd.Series, fmt: str, local_tz: str = LOCAL_TZ) -> pd.Series:
    """
    Parse a series of timestamp strings using a known format (vectorized), then
    localize naive times to local_tz and convert to UTC (vectorized).
    Returns tz-aware datetime64[ns, UTC] Series.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    n = len(series)
    if n == 0:
        return pd.Series(dtype="datetime64[ns, UTC]")

    t0 = time.time()

    # Vectorized parse with explicit format (fast)
    parsed = pd.to_datetime(series, format=fmt, errors="coerce", utc=False)

    # For any leftover NaT, try pandas general parser for those entries (rare)
    if parsed.isna().any():
        parsed.loc[parsed.isna()] = pd.to_datetime(series.loc[parsed.isna()], errors="coerce", utc=False)

    # Vectorized localization/conversion to UTC
    try:
        # If tz-naive series (common), tz_localize then convert
        if parsed.dt.tz is None:
            parsed_utc = parsed.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(UTC)
        else:
            parsed_utc = parsed.dt.tz_convert(UTC)
    except Exception:
        # If vectorized path fails for some reason, fall back to elementwise for the problematic entries only
        def _to_utc_elem(ts):
            if pd.isna(ts):
                return pd.NaT
            try:
                if getattr(ts, "tzinfo", None) is None:
                    return pytz.timezone(local_tz).localize(ts, is_dst=None).astimezone(UTC)
                else:
                    return ts.tz_convert(UTC)
            except Exception:
                return pd.NaT
        parsed_utc = parsed.apply(_to_utc_elem)

    parsed_utc = pd.to_datetime(parsed_utc, errors="coerce", utc=True)

    dt = time.time() - t0
    print(f"    Parsed {parsed_utc.notna().sum()}/{n} timestamps (format='{fmt}') in {dt:.2f}s")
    return parsed_utc


# -----------------------------
# Clean heart rate
# -----------------------------
def clean_heartrate():
    if not HR_FILE.exists():
        print(f"Warning: {HR_FILE} not found — returning empty HR frame.")
        return pd.DataFrame(columns=["heart_rate"]).astype({"heart_rate": "float64"})

    # read likely columns only to reduce memory overhead
    try:
        df = pd.read_csv(HR_FILE, usecols=lambda c: c.lower() in {"timestamp", "heart_rate", "value", "hr", "user_id", "source_type"})
    except Exception:
        df = pd.read_csv(HR_FILE)

    # Normalize timestamp column name
    if "timestamp" not in df.columns:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    # Parse timestamps to tz-aware UTC datetimes
    df["timestamp"] = parse_with_format(df["timestamp"], HR_FMT, LOCAL_TZ)
    parsed_count = df["timestamp"].notna().sum()
    print(f"  HR timestamps parsed: {parsed_count}/{len(df)}")

    # Drop rows without valid timestamp
    df = df.dropna(subset=["timestamp"]).copy()

    # Find heart_rate numeric column
    if "heart_rate" in df.columns:
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    elif "value" in df.columns:
        df["heart_rate"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df["heart_rate"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
        else:
            df["heart_rate"] = pd.Series(dtype="float")

    # Keep physiological values only
    df = df.dropna(subset=["heart_rate"]).copy()
    df = df[(df["heart_rate"] >= 30) & (df["heart_rate"] <= 220)].copy()

    # set datetime index (tz-aware)
    df = df.set_index("timestamp").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        # ensure tz is UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)

    # resample and interpolate small gaps
    df_numeric = df.select_dtypes(include=[np.number])
    if not df_numeric.empty:
        df_resampled = df_numeric.resample(RESAMPLE).mean()
        # interpolate time-based (index is guaranteed to be DatetimeIndex UTC)
        df_resampled = df_resampled.interpolate(method="time", limit=5)
        # ensure heart_rate column exists
        if "heart_rate" in df_resampled.columns:
            return df_resampled[["heart_rate"]]
        else:
            return df_resampled.iloc[:, [0]].rename(columns={df_resampled.columns[0]: "heart_rate"})
    return pd.DataFrame(columns=["heart_rate"])


# -----------------------------
# Clean steps
# -----------------------------
def clean_steps():
    if not STEPS_FILE.exists():
        print(f"Warning: {STEPS_FILE} not found — returning empty steps frame.")
        return pd.DataFrame(columns=["steps"]).astype({"steps": "float64"})

    try:
        df = pd.read_csv(STEPS_FILE, usecols=lambda c: c.lower() in {"timestamp", "steps", "value", "user_id", "source_type"})
    except Exception:
        df = pd.read_csv(STEPS_FILE)

    if "timestamp" not in df.columns:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    df["timestamp"] = parse_with_format(df["timestamp"], STEPS_FMT, LOCAL_TZ)
    parsed_count = df["timestamp"].notna().sum()
    print(f"  Steps timestamps parsed: {parsed_count}/{len(df)}")

    df = df.dropna(subset=["timestamp"]).copy()

    if "steps" in df.columns:
        df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
    elif "value" in df.columns:
        df["steps"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["steps"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0) if numeric_cols else 0

    df = df.set_index("timestamp").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)

    df_resampled = df.resample(RESAMPLE).sum().fillna(0)
    if "steps" in df_resampled.columns:
        return df_resampled[["steps"]]
    if not df_resampled.select_dtypes(include=[np.number]).empty:
        col = df_resampled.select_dtypes(include=[np.number]).columns[0]
        return df_resampled[[col]].rename(columns={col: "steps"})
    return pd.DataFrame(columns=["steps"])


# -----------------------------
# Clean sleep
# -----------------------------
def clean_sleep():
    if not SLEEP_FILE.exists():
        print(f"Warning: {SLEEP_FILE} not found — returning empty sleep frame.")
        return pd.DataFrame(columns=["sleep_hours"]).astype({"sleep_hours": "float64"})

    df = pd.read_csv(SLEEP_FILE)

    # detect start/end column names
    start_col = "startTime" if "startTime" in df.columns else ("start_time" if "start_time" in df.columns else None)
    end_col = "endTime" if "endTime" in df.columns else ("end_time" if "end_time" in df.columns else None)
    if start_col is None or end_col is None:
        datetime_like = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if len(datetime_like) >= 2:
            start_col, end_col = datetime_like[0], datetime_like[1]
        else:
            print("No start/end time columns found in sleep file.")
            return pd.DataFrame(columns=["sleep_hours"]).astype({"sleep_hours": "float64"})

    df["start_parsed"] = parse_with_format(df[start_col], SLEEP_FMT, LOCAL_TZ)
    df["end_parsed"] = parse_with_format(df[end_col], SLEEP_FMT, LOCAL_TZ)
    parsed_count_start = df["start_parsed"].notna().sum()
    parsed_count_end = df["end_parsed"].notna().sum()
    print(f"  Sleep timestamps parsed (start/end): {parsed_count_start}/{len(df)} {parsed_count_end}/{len(df)}")

    df = df.dropna(subset=["start_parsed", "end_parsed"]).copy()

    rows = []
    for _, r in df.iterrows():
        s = r["start_parsed"]
        e = r["end_parsed"]
        if pd.isna(s) or pd.isna(e):
            continue
        try:
            # both s and e are tz-aware (UTC)
            idx = pd.date_range(start=s, end=e, freq=RESAMPLE, closed="left", tz=UTC)
            if len(idx) == 0:
                idx = pd.DatetimeIndex([s])
        except Exception:
            try:
                idx = pd.DatetimeIndex([s])
            except Exception:
                continue
        rows.append(pd.DataFrame({"sleep_hours": 1.0 / 60.0}, index=idx))

    if rows:
        sleep_min = pd.concat(rows).sort_index()
        sleep_min.index.name = "timestamp"
        sleep_min = sleep_min.groupby(sleep_min.index).sum()
        # ensure tz-aware index
        if not isinstance(sleep_min.index, pd.DatetimeIndex):
            sleep_min.index = pd.to_datetime(sleep_min.index, utc=True)
        else:
            if sleep_min.index.tz is None:
                sleep_min.index = sleep_min.index.tz_localize(UTC)
            else:
                sleep_min.index = sleep_min.index.tz_convert(UTC)
        return sleep_min

    return pd.DataFrame(columns=["sleep_hours"])


# -----------------------------
# Main: merge and save
# -----------------------------
def main():
    print("Loading & cleaning heart rate...")
    hr = clean_heartrate()

    print("Loading & cleaning steps...")
    steps = clean_steps()

    print("Loading & cleaning sleep...")
    sleep = clean_sleep()

    # ensure indexes are datetime (may be empty frames)
    for df in (hr, steps, sleep):
        if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif not df.empty:
            # ensure tz-aware UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize(UTC)
            else:
                df.index = df.index.tz_convert(UTC)

    # union of indexes
    all_idx = pd.DatetimeIndex([])
    if not hr.empty:
        all_idx = all_idx.union(hr.index)
    if not steps.empty:
        all_idx = all_idx.union(steps.index)
    if not sleep.empty:
        all_idx = all_idx.union(sleep.index)

    # sort & force tz-aware DatetimeIndex in UTC
    all_idx = all_idx.sort_values()
    if len(all_idx) > 0:
        all_idx = pd.DatetimeIndex(all_idx)
        if all_idx.tz is None:
            all_idx = all_idx.tz_localize(UTC)
        else:
            all_idx = all_idx.tz_convert(UTC)

    if all_idx.empty:
        print("No data found in raw files. Exiting after creating empty outputs.")
        empty = pd.DataFrame(columns=["heart_rate", "steps", "sleep_hours"])
        empty.to_csv(PROC / "preprocessed_data.csv", index=False)
        empty.to_csv(PROC / "train.csv", index=False)
        empty.to_csv(PROC / "test.csv", index=False)
        (OUT / "data_dictionary.md").write_text("No data found.")
        (OUT / "preprocessing_summary.md").write_text("No data found.")
        return

    # Reindex each dataframe to the full timeline and ensure DatetimeIndex tz-aware before interpolation/fill
    if not hr.empty:
        hr = hr.reindex(all_idx)
        hr.index = pd.DatetimeIndex(hr.index, tz=UTC)
        # safe interpolation on time-indexed series
        hr = hr.interpolate(method="time", limit=5)
    else:
        hr = pd.DataFrame(index=all_idx, columns=["heart_rate"])

    if not steps.empty:
        steps = steps.reindex(all_idx).fillna(0)
        steps.index = pd.DatetimeIndex(steps.index, tz=UTC)
    else:
        steps = pd.DataFrame(index=all_idx, columns=["steps"]).fillna(0)

    if not sleep.empty:
        sleep = sleep.reindex(all_idx).fillna(0)
        sleep.index = pd.DatetimeIndex(sleep.index, tz=UTC)
    else:
        sleep = pd.DataFrame(index=all_idx, columns=["sleep_hours"]).fillna(0)

    # merge
    combined = pd.concat([hr, steps, sleep], axis=1)

    # fill missing columns if any
    if "heart_rate" not in combined.columns:
        combined["heart_rate"] = np.nan
    if "steps" not in combined.columns:
        combined["steps"] = 0
    if "sleep_hours" not in combined.columns:
        combined["sleep_hours"] = 0

    # ensure index is named and tz-aware
    combined.index.name = "timestamp"
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize(UTC)
    else:
        combined.index = combined.index.tz_convert(UTC)

    # save full dataset (timestamp index will be the first column when reading with index_col=0)
    combined.to_csv(PROC / "preprocessed_data.csv")
    print("Saved:", PROC / "preprocessed_data.csv")

    # train/test split
    n_test = int(len(combined) * TEST_RATIO)
    if n_test == 0:
        train = combined
        test = combined.iloc[0:0]
    else:
        train = combined.iloc[:-n_test]
        test = combined.iloc[-n_test:]

    train.to_csv(PROC / "train.csv")
    test.to_csv(PROC / "test.csv")
    print("Saved train/test split.")

    # documentation
    (OUT / "data_dictionary.md").write_text(
        "**heart_rate:** avg bpm per minute\n"
        "**steps:** steps in that minute\n"
        "**sleep_hours:** hours asleep in that minute (1/60 = 1 minute)\n"
    )

    (OUT / "preprocessing_summary.md").write_text(
        "Steps:\n"
        "- Cleaned HR, Steps, and Sleep\n"
        "- Converted timestamps to UTC\n"
        "- Resampled to 1-minute\n"
        "- Merged all signals\n"
        "- Created train/test split\n"
    )

    print("Documentation written in outputs/")


if __name__ == "__main__":
    main()
