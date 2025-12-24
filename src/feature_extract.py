"""
src/feature_extract.py

Module 2 - Feature Extraction & Modeling (complete, ready-to-run)

Features:
- Tries to use full cleaned dataframe from src/clean_merge.py if a helper exists.
- Otherwise concatenates data/processed/train.csv + test.csv.
- DEMO_MODE to run on recent subset (fast testing).
- TSFresh optional (disabled by default, enable with RUN_TSFRESH=True).
- Uses Prophet(mcmc_samples=0) for faster fitting.
- Headless plotting (no GUI blocking). Saves PNGs & numeric CSVs.
- Saves TFresh outputs and cluster CSVs if RUN_TSFRESH=True.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from datetime import timedelta

# Reduce noisy logs
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from prophet import Prophet

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------
# User-tunable flags (edit here)
# -------------------------
DEMO_MODE = True         # True -> use only last DEMO_DAYS days (faster)
DEMO_DAYS = 120

RUN_TSFRESH = True       # True -> run TSFresh feature extraction (can be slow)
TSFRESH_N_JOBS = 1       # parallel jobs for TSFresh (1 or 2 recommended on laptop)

RUN_TASK1 = True
RUN_TASK2 = True
RUN_TASK3 = True

PROPHET_MCMC_SAMPLES = 0  # 0 -> fast MAP optimization


# -------------------------
# Paths
# -------------------------
THIS_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "module2_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_PROCESSED, "train.csv")
TEST_CSV = os.path.join(DATA_PROCESSED, "test.csv")

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Processed CSV folder: {DATA_PROCESSED}")
print(f"Outputs will be saved to: {OUT_DIR}\n")

# Columns expected
TIMESTAMP_COL = "timestamp"
HR_COL = "heart_rate"
STEPS_COL = "steps"
SLEEP_COL = "sleep_hours"

# -------------------------
# Utility functions
# -------------------------
def try_load_full_from_clean_merge():
    """
    Try to import src.clean_merge and call a function that returns the full cleaned DataFrame.
    Supported function names: get_full_clean_df, load_full_df, build_full_df, get_full_df
    """
    sys.path.insert(0, SRC_DIR)
    candidates = ["get_full_clean_df", "load_full_df", "build_full_df", "get_full_df"]
    try:
        import importlib
        cm = importlib.import_module("clean_merge")
    except Exception:
        return None
    for fname in candidates:
        if hasattr(cm, fname) and callable(getattr(cm, fname)):
            try:
                df = getattr(cm, fname)()
                if isinstance(df, pd.DataFrame):
                    print(f"Obtained full DataFrame from clean_merge.{fname}()")
                    return df
            except Exception as e:
                print(f"clean_merge.{fname}() call failed: {e}")
                return None
    for const in ["FULL_DF_PATH", "FULL_CLEAN_PATH", "FULL_OUTPUT_PATH"]:
        if hasattr(cm, const):
            p = getattr(cm, const)
            if isinstance(p, str) and os.path.exists(p):
                print(f"Found full clean CSV path in clean_merge: {p}")
                return pd.read_csv(p)
    return None

def load_train_test_concat():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"train.csv not found at {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    if os.path.exists(TEST_CSV):
        test_df = pd.read_csv(TEST_CSV)
        full = pd.concat([train_df, test_df], ignore_index=True)
        print("Concatenated train.csv + test.csv -> full dataframe")
    else:
        full = train_df.copy()
        print("Using only train.csv as full dataframe (test.csv not found).")
    return full

def ensure_timestamp(df):
    if TIMESTAMP_COL in df.columns:
        return TIMESTAMP_COL
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
            print(f"Using '{c}' as timestamp column")
            return c
    raise ValueError("No timestamp-like column found in dataframe.")

def parse_and_normalize_timestamps_df(df, ts_col):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    nbad = int(df[ts_col].isna().sum())
    if nbad:
        raise ValueError(f"{nbad} timestamps failed to parse.")
    df[ts_col] = df[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def compute_daily_aggregates(full_df, ts_col):
    """
    Compute daily aggregates and save CSVs. Return dict with daily dfs for hr/steps/sleep where present.
    """
    cols = [ts_col]
    if HR_COL in full_df.columns:
        cols.append(HR_COL)
    if STEPS_COL in full_df.columns:
        cols.append(STEPS_COL)
    if SLEEP_COL in full_df.columns:
        cols.append(SLEEP_COL)
    df = full_df[cols].copy()
    df["date"] = df[ts_col].dt.floor("D")

    outputs = {}
    if HR_COL in df.columns:
        hr = df.dropna(subset=[HR_COL]).groupby("date")[HR_COL].mean().reset_index().rename(columns={"date":"ds", HR_COL:"y"})
        hr["ds"] = pd.to_datetime(hr["ds"])
        hr = hr.sort_values("ds").reset_index(drop=True)
        hr.to_csv(os.path.join(OUT_DIR, "daily_heart_rate.csv"), index=False)
        outputs["hr"] = hr
        print("Saved daily_heart_rate.csv")

    if STEPS_COL in df.columns:
        steps = df.dropna(subset=[STEPS_COL]).groupby("date")[STEPS_COL].sum().reset_index().rename(columns={"date":"ds", STEPS_COL:"y"})
        steps["ds"] = pd.to_datetime(steps["ds"])
        steps = steps.sort_values("ds").reset_index(drop=True)
        steps.to_csv(os.path.join(OUT_DIR, "daily_steps.csv"), index=False)
        outputs["steps"] = steps
        print("Saved daily_steps.csv")

    if SLEEP_COL in df.columns:
        sleep = df.dropna(subset=[SLEEP_COL]).groupby("date")[SLEEP_COL].sum().reset_index().rename(columns={"date":"ds", SLEEP_COL:"y"})
        sleep["ds"] = pd.to_datetime(sleep["ds"])
        sleep = sleep.sort_values("ds").reset_index(drop=True)
        sleep.to_csv(os.path.join(OUT_DIR, "daily_sleep.csv"), index=False)
        outputs["sleep"] = sleep
        print("Saved daily_sleep.csv")

    return outputs

def fast_prophet_fit(series_df, periods=7, weekly_seasonality=False, holidays=None):
    """
    Fit Prophet quickly with mcmc_samples=PROPHET_MCMC_SAMPLES (fast MAP if 0).
    Returns (model, forecast_df).
    """
    if holidays is not None:
        model = Prophet(mcmc_samples=PROPHET_MCMC_SAMPLES, holidays=holidays, weekly_seasonality=False)
    else:
        model = Prophet(mcmc_samples=PROPHET_MCMC_SAMPLES, weekly_seasonality=False)
    if weekly_seasonality:
        model.add_seasonality(name="weekly", period=7, fourier_order=3)
    model.fit(series_df)
    future = model.make_future_dataframe(periods=periods)
    fcst = model.predict(future)
    return model, fcst

def save_forecast_plot(series_df, fcst_df, title, filename):
    plt.figure(figsize=(10,5))
    plt.plot(series_df["ds"], series_df["y"], label="Actual")
    plt.plot(fcst_df["ds"], fcst_df["yhat"], label="Forecast")
    plt.fill_between(fcst_df["ds"].dt.to_pydatetime(), fcst_df["yhat_lower"], fcst_df["yhat_upper"], alpha=0.25, label="CI")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.tight_layout()
    outp = os.path.join(OUT_DIR, filename)
    plt.savefig(outp)
    plt.close()
    print(f"Saved plot: {outp}")

# -------------------------
# Tasks
# -------------------------
def task1_heart_rate(hr_df):
    print("\n=== Task 1: Heart Rate Forecasting ===")
    if len(hr_df) >= 60:
        series = hr_df.iloc[-60:].reset_index(drop=True)
    else:
        series = hr_df.copy()
        print(f"Warning: only {len(series)} days available (expected 60).")
    print(f"HR date range: {series['ds'].min().date()} -> {series['ds'].max().date()} ({len(series)} rows)")
    model, fcst = fast_prophet_fit(series, periods=14, weekly_seasonality=False)

    save_forecast_plot(series, fcst, "Task 1: Heart Rate Forecast", "task1_hr_forecast.png")

    if len(series) >= 7:
        holdout = series.iloc[-7:].copy()
        fcst_idx = fcst.set_index("ds")
        preds = fcst_idx.loc[holdout["ds"], "yhat"]
        mae = np.mean(np.abs(preds.values - holdout["y"].values))
        print(f"MAE on last 7-day holdout: {mae:.3f}")
    else:
        mae = None
        print("Not enough data to compute MAE on last 7 days.")

    start_date = series["ds"].min()
    day67_date = start_date + pd.Timedelta(days=66)
    row67 = fcst[fcst["ds"] == day67_date]
    if not row67.empty:
        print(f"Forecast for Day 67 ({day67_date.date()}): {float(row67['yhat'].values[0]):.3f}")
    else:
        print(f"Day 67 ({day67_date.date()}) not covered in forecast (last forecast date {fcst['ds'].max().date()})")
    return {"model": model, "forecast": fcst, "mae": mae}

def task2_sleep(sleep_df):
    print("\n=== Task 2: Sleep Pattern Forecasting ===")
    if len(sleep_df) >= 90:
        series = sleep_df.iloc[-90:].reset_index(drop=True)
    else:
        series = sleep_df.copy()
        print(f"Warning: only {len(series)} days available (expected 90).")
    print(f"Sleep date range: {series['ds'].min().date()} -> {series['ds'].max().date()} ({len(series)} rows)")
    model, fcst = fast_prophet_fit(series, periods=7, weekly_seasonality=True)

    save_forecast_plot(series, fcst, "Task 2: Sleep Forecast", "task2_sleep_forecast.png")

    fig = model.plot_components(fcst)
    comp_path = os.path.join(OUT_DIR, "task2_sleep_components.png")
    fig.savefig(comp_path)
    plt.close(fig)
    print(f"Saved components plot: {comp_path}")

    week_dates = pd.date_range(start="2020-01-06", periods=14, freq="D")
    week_df = pd.DataFrame({"ds": week_dates})
    week_pred = model.predict(week_df)
    week_pred["dow"] = week_pred["ds"].dt.day_name()
    if "weekly" in week_pred.columns:
        dow_scores = week_pred.groupby("dow")["weekly"].mean()
    else:
        dow_scores = week_pred.groupby("dow")["yhat"].mean()
    best = dow_scores.idxmax()
    worst = dow_scores.idxmin()

    trend_avgs = fcst[["ds", "trend"]].sort_values("ds")
    if len(trend_avgs) >= 28:
        first_avg = trend_avgs["trend"].iloc[:14].mean()
        last_avg = trend_avgs["trend"].iloc[-14:].mean()
        trend_dir = "increasing" if last_avg > first_avg else "decreasing"
    else:
        trend_dir = "no clear trend (insufficient data)"

    print(f"Best day to sleep (weekly pattern): {best}")
    print(f"Worst day to sleep (weekly pattern): {worst}")
    print(f"Sleep trend: {trend_dir}")
    return {"model": model, "forecast": fcst, "best_day": best, "worst_day": worst, "trend_dir": trend_dir}

def task3_steps(steps_df):
    print("\n=== Task 3: Steps Forecast with Holidays ===")
    if len(steps_df) >= 120:
        series = steps_df.iloc[-120:].reset_index(drop=True)
    else:
        series = steps_df.copy()
        print(f"Warning: only {len(series)} days available (expected 120).")
    print(f"Steps date range: {series['ds'].min().date()} -> {series['ds'].max().date()} ({len(series)} rows)")

    start = series["ds"].min()
    def date_for_day(one_indexed):
        return start + pd.Timedelta(days=(one_indexed - 1))

    holiday_rows = []
    for d in range(30, 38):
        holiday_rows.append({"holiday": "vacation", "ds": date_for_day(d)})
    for d in [60,61,62]:
        holiday_rows.append({"holiday": "sick", "ds": date_for_day(d)})
    holiday_rows.append({"holiday": "marathon", "ds": date_for_day(90)})
    holidays_df = pd.DataFrame(holiday_rows)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

    model_no, fcst_no = fast_prophet_fit(series, periods=30, weekly_seasonality=True)
    model_yes, fcst_yes = fast_prophet_fit(series, periods=30, weekly_seasonality=True, holidays=holidays_df)

    plt.figure(figsize=(12,5))
    plt.plot(series["ds"], series["y"], label="Actual")
    plt.plot(fcst_no["ds"], fcst_no["yhat"], label="Forecast w/o holidays")
    plt.plot(fcst_yes["ds"], fcst_yes["yhat"], label="Forecast with holidays")
    plt.legend()
    plt.title("Task 3: Steps Forecast - Without vs With Holidays")
    plt.tight_layout()
    comp_path = os.path.join(OUT_DIR, "task3_steps_forecast_comparison.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved plot: {comp_path}")

    comp_list = []
    for idx, row in holidays_df.iterrows():
        d = row["ds"]
        pred_no = float(fcst_no.loc[fcst_no["ds"] == d, "yhat"]) if not fcst_no.loc[fcst_no["ds"] == d].empty else np.nan
        pred_yes = float(fcst_yes.loc[fcst_yes["ds"] == d, "yhat"]) if not fcst_yes.loc[fcst_yes["ds"] == d].empty else np.nan
        comp_list.append({"ds": d.date(), "event": row["holiday"], "without_holiday": pred_no, "with_holiday": pred_yes, "diff": pred_yes - pred_no})
    comp_df = pd.DataFrame(comp_list)
    print("Event impact comparison:")
    print(comp_df)

    comp_df["absdiff"] = comp_df["diff"].abs()
    largest_idx = comp_df["absdiff"].idxmax()
    largest = comp_df.loc[largest_idx]
    print("Biggest effect event:")
    print(largest[["event","ds","diff"]].to_string(index=False))

    return {"model_no": model_no, "fcst_no": fcst_no, "model_yes": model_yes, "fcst_yes": fcst_yes, "comp_df": comp_df}

# -------------------------
# TSFresh + clustering (optional)
# -------------------------
def feature_extraction_and_clustering(daily_map):
    print("\n--- TSFresh feature extraction (may be slow) ---")
    def create_windows(daily_df, window=30):
        windows = []
        n = len(daily_df) // window
        for i in range(n):
            w = daily_df.iloc[i*window:(i+1)*window].copy().reset_index(drop=True)
            if len(w) == window:
                windows.append(w)
        return windows

    parts = []
    idc = 0
    for key in ["hr","steps","sleep"]:
        if key in daily_map:
            wins = create_windows(daily_map[key], 30)
            for w in wins:
                tmp = pd.DataFrame({"id": idc, "time": np.arange(len(w)), "value": w["y"].values, "source": key})
                parts.append(tmp)
                idc += 1
    if len(parts) == 0:
        print("Not enough windows for TSFresh. Skipping.")
        return None

    long_df = pd.concat(parts, ignore_index=True)
    print(f"Extracting features from {long_df['id'].nunique()} windows (TSFresh). This can take time...")
    extracted = extract_features(long_df, column_id="id", column_sort="time", column_value="value", n_jobs=TSFRESH_N_JOBS)
    impute(extracted)
    id_source = long_df[["id","source"]].drop_duplicates().set_index("id")["source"].to_dict()
    extracted["source"] = extracted.index.map(id_source)

    features_path = os.path.join(OUT_DIR, "features_tsfresh.csv")
    extracted.to_csv(features_path)
    print(f"Saved TSFresh features: {features_path}")

    X = extracted.drop(columns=["source"]).replace([np.inf,-np.inf], np.nan).dropna(axis=1, how="all")
    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = min(4, Xs.shape[0]) if Xs.shape[0] >= 2 else 1
    if k >= 2:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(Xs)
        klabels = kmeans.labels_
    else:
        klabels = np.zeros(Xs.shape[0], dtype=int)
    db = DBSCAN(eps=1.0, min_samples=2).fit(Xs)
    dlabels = db.labels_

    extracted["kmeans"] = klabels
    extracted["dbscan"] = dlabels
    clusters_path = os.path.join(OUT_DIR, "clusters.csv")
    extracted.to_csv(clusters_path)
    print(f"Saved clusters: {clusters_path}")
    return {"features": extracted, "kmeans": klabels, "dbscan": dlabels}

# -------------------------
# Main
# -------------------------
def main():
    # 1) Try to load full df from clean_merge (if available)
    full_df = None
    try:
        full_df = try_load_full_from_clean_merge()
    except Exception as e:
        print("clean_merge import error:", e)

    # 2) fallback
    if full_df is None:
        print("Falling back to concatenating train.csv + test.csv (if present).")
        full_df = load_train_test_concat()

    # 3) timestamp
    ts_col = ensure_timestamp(full_df)
    full_df = parse_and_normalize_timestamps_df(full_df, ts_col)

    # 4) demo mode (keep last DEMO_DAYS)
    if DEMO_MODE:
        last_date = full_df[ts_col].dt.floor("D").max()
        cutoff = last_date - pd.Timedelta(days=(DEMO_DAYS - 1))
        full_df = full_df[full_df[ts_col] >= cutoff].reset_index(drop=True)
        print(f"DEMO_MODE ON: using last {DEMO_DAYS} days ({len(full_df)} rows)")

    # 5) compute daily aggregates
    daily = compute_daily_aggregates(full_df, ts_col)

    # -------------------
    # Run Tasks and assign returns to t1,t2,t3
    # -------------------
    t1 = t2 = t3 = None

    if RUN_TASK1 and "hr" in daily:
        try:
            t1 = task1_heart_rate(daily["hr"])
        except Exception as e:
            print("Task1 failed:", repr(e))
    else:
        print("Skipping Task1 (no hr or disabled).")

    if RUN_TASK2 and "sleep" in daily:
        try:
            t2 = task2_sleep(daily["sleep"])
        except Exception as e:
            print("Task2 failed:", repr(e))
    else:
        print("Skipping Task2 (no sleep or disabled).")

    if RUN_TASK3 and "steps" in daily:
        try:
            t3 = task3_steps(daily["steps"])
        except Exception as e:
            print("Task3 failed:", repr(e))
    else:
        print("Skipping Task3 (no steps or disabled).")

    # Save numeric outputs for easy inspection
    try:
        if t1 is not None and isinstance(t1, dict) and 'forecast' in t1:
            t1['forecast'].to_csv(os.path.join(OUT_DIR, 'task1_hr_forecast_table.csv'), index=False)
            print("Saved numeric forecast: task1_hr_forecast_table.csv")
    except Exception as e:
        print("Could not save Task1 forecast:", e)

    try:
        if t2 is not None and isinstance(t2, dict) and 'forecast' in t2:
            t2['forecast'].to_csv(os.path.join(OUT_DIR, 'task2_sleep_forecast_table.csv'), index=False)
            print("Saved numeric forecast: task2_sleep_forecast_table.csv")
    except Exception as e:
        print("Could not save Task2 forecast:", e)

    try:
        if t3 is not None and isinstance(t3, dict):
            if 'comp_df' in t3 and t3['comp_df'] is not None:
                t3['comp_df'].to_csv(os.path.join(OUT_DIR, 'task3_events_impact.csv'), index=False)
                print("Saved events comparison: task3_events_impact.csv")
            if 'fcst_no' in t3 and t3['fcst_no'] is not None:
                t3['fcst_no'].to_csv(os.path.join(OUT_DIR, 'task3_forecast_no_holidays.csv'), index=False)
            if 'fcst_yes' in t3 and t3['fcst_yes'] is not None:
                t3['fcst_yes'].to_csv(os.path.join(OUT_DIR, 'task3_forecast_with_holidays.csv'), index=False)
    except Exception as e:
        print("Could not save Task3 outputs:", e)

    # TSFresh optional
    if RUN_TSFRESH:
        try:
            feature_extraction_and_clustering(daily)
        except Exception as e:
            print("TSFresh step failed:", repr(e))
    else:
        print("TSFresh skipped (RUN_TSFRESH=False).")

    print("\nModule 2 complete. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()