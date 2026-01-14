import os
import base64
from datetime import datetime, timedelta, date
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
from prophet import Prophet

st.set_page_config(
    page_title="FitPulse Health Insights",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.75rem; margin:0; padding:0;">
        <h1 style="margin:0; padding:0;">FitPulse Health Insights</h1>
        <div style="display:flex; align-items:center; justify-content:center;">
            <svg viewBox="0 0 64 64" width="60" height="60" xmlns="http://www.w3.org/2000/svg">
                <polyline points="4,34 16,34 22,22 30,44 38,18 46,34 60,34" 
                          fill="none" stroke="#ffffff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
        </div>
    </div>
    <p style="color:#9fb4c8;">Visualize your fitness patterns, analyze anomalies, and forecast health trends.</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="padding:0.35rem 0.9rem; border-radius:999px; background:linear-gradient(135deg, #ff8c00, #ffb347); color:#ffffff; font-weight:800; font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; letter-spacing:0.12em; font-size:0.85rem; text-align:center; margin-bottom:0.5rem;">
        FITPULSE
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <p style="font-size:0.8rem; color:#d0d8e5; margin-bottom:0.2rem;">
        An anomaly is a health reading that looks different from your normal pattern and may need a closer look.
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")

# Add tab descriptions
st.sidebar.markdown("""
### Tab Guide

**🏠 Home**  
Daily health summary with today's metrics and 7-day overview

**📊 Analysis**  
Explore health data with custom date ranges and detailed statistics

**🔍 Anomalies**  
Advanced multi-method anomaly detection with severity classification

**📄 Reports**  
Generate professional PDF reports with theory-based health insights
""")

st.sidebar.markdown("---")

# Path to Module 2 outputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # go up → project folder
MODULE2_DIR = os.path.join(PROJECT_ROOT, "module2_outputs")

# Path to processed data for holistic health status
PROCESSED_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "preprocessed_data.csv",
)


def metric_unit(metric_key: str) -> str:
    if "Heart" in metric_key:
        return "bpm"
    if "Step" in metric_key:
        return "steps"
    if "Sleep" in metric_key:
        return "hours"
    return "units"


@st.cache_data(show_spinner=False)
def load_processed_data(path: str):
        """Load preprocessed data used to derive a simple daily health status."""
        if not os.path.exists(path):
                return None
        try:
                df = pd.read_csv(path)
                if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        df = df.dropna(subset=["timestamp"])
                return df
        except Exception:
                return None

def derive_daily_health_status(df):
    """Return a short 2-3 word label and mood from the latest day.

    Output: (status_label, mood_label)
    mood_label in {"happy", "calm", "sleepy", "tired", "neutral"}
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return "Status Unknown", "neutral"

    work = df.copy()
    work["date"] = work["timestamp"].dt.date
    latest_date = work["date"].max()
    if pd.isna(latest_date):
        return "Status Unknown", "neutral"

    day = work[work["date"] == latest_date]
    if day.empty:
        return "Status Unknown", "neutral"

    avg_hr = day["heart_rate"].mean() if "heart_rate" in day.columns else None
    total_steps = day["steps"].sum() if "steps" in day.columns else None
    total_sleep = day["sleep_hours"].sum() if "sleep_hours" in day.columns else None

    # If any core metric is missing, fall back to a generic label
    if avg_hr is None or np.isnan(avg_hr) or total_steps is None or np.isnan(total_steps):
        return "Syncing Data", "neutral"

    # Basic rules of thumb combining movement, sleep, and heart rate
    if (total_sleep is not None and not np.isnan(total_sleep) and total_sleep < 5.5) or avg_hr > 95:
        return "Needs Rest", "tired"
    if total_steps < 3000:
        return "Low Activity", "sleepy"
    if (
        total_steps >= 10000
        and total_sleep is not None
        and not np.isnan(total_sleep)
        and 7 <= total_sleep <= 9
        and 55 <= avg_hr <= 85
    ):
        return "Great Balance", "happy"

    return "On Track", "calm"


def extract_metric_time_series(df: pd.DataFrame, filename: str = None):
    """Extract daily time series for Heart Rate, Steps, Sleep from a generic CSV.

    Returns a dict mapping metric name ("Heart Rate", "Steps", "Sleep") to
    a DataFrame with columns ['ds', 'y'] (daily aggregated).

    This is used by both the Analysis and Anomalies tabs so that users can
    upload *any* reasonable CSV (raw or processed, train/test, etc.).
    """

    if df is None or df.empty:
        return {}

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    # Detect date/time column
    date_col = None
    date_patterns = [
        "ds",
        "timestamp",
        "date",
        "datetime",
        "activitydate",
        "logdate",
        "sleepdate",
        "time",
        "starttime",
        "endtime",
        "startdate",
        "enddate",
    ]
    for col in work.columns:
        cl = col.lower()
        for patt in date_patterns:
            if cl == patt or cl.endswith(patt):
                date_col = col
                break
        if date_col:
            break

    if not date_col:
        return {}

    ds_series = pd.to_datetime(work[date_col], errors="coerce")
    # Remove timezone information if present (Prophet requires tz-naive)
    try:
        if getattr(ds_series.dt, "tz", None) is not None:
            try:
                ds_series = ds_series.dt.tz_convert(None)
            except TypeError:
                ds_series = ds_series.dt.tz_localize(None)
    except Exception:
        # If anything goes wrong, keep the original parsed dates
        pass

    work["ds"] = ds_series
    work = work.dropna(subset=["ds"])
    if work.empty:
        return {}

    # Metric-specific column patterns
    metric_patterns = {
        "Heart Rate": ["heart_rate", "heartrate", "hr", "bpm"],
        "Steps": ["steps", "step_count", "step"],
        "Sleep": [
            "sleep_hours",
            "minutesasleep",
            "minutes_asleep",
            "sleepduration",
            "sleep_duration",
            "totalsleep",
        ],
    }

    series = {}

    for metric_name, patterns in metric_patterns.items():
        # Find the first matching column for this metric
        value_col = None
        for col in work.columns:
            cl = col.lower()
            if col == date_col or col == "ds":
                continue
            if any(patt in cl for patt in patterns):
                value_col = col
                break

        if not value_col:
            continue

        mdf = pd.DataFrame()
        mdf["ds"] = work["ds"].dt.floor("D")
        mdf["y"] = pd.to_numeric(work[value_col], errors="coerce")
        mdf = mdf.dropna(subset=["y"])
        if mdf.empty:
            continue

        # Aggregate to daily level: heart rate → mean, steps/sleep → sum
        if metric_name == "Heart Rate":
            agg = "mean"
        else:
            agg = "sum"

        mdf = (
            mdf.groupby("ds", as_index=False)["y"].agg(agg).sort_values("ds").reset_index(drop=True)
        )
        series[metric_name] = mdf

    # Fallback: ds + generic 'y' column, infer metric from filename
    if not series and "y" in work.columns:
        inferred_metric = None
        if filename:
            fname = filename.lower()
            if "heart" in fname or "hr" in fname or "heartrate" in fname:
                inferred_metric = "Heart Rate"
            elif "step" in fname:
                inferred_metric = "Steps"
            elif "sleep" in fname:
                inferred_metric = "Sleep"

        if inferred_metric:
            mdf = pd.DataFrame()
            mdf["ds"] = work["ds"].dt.floor("D")
            mdf["y"] = pd.to_numeric(work["y"], errors="coerce")
            mdf = mdf.dropna(subset=["y"])
            if not mdf.empty:
                agg = "mean" if inferred_metric == "Heart Rate" else "sum"
                mdf = (
                    mdf.groupby("ds", as_index=False)["y"].agg(agg).sort_values("ds").reset_index(drop=True)
                )
                series[inferred_metric] = mdf

    return series


def build_and_show_pdf_report(report_text: str, metric_type: str):
    """Create a PDF from the given report text and show download/preview controls in Streamlit."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    text_width = pdf.w - pdf.l_margin - pdf.r_margin

    def sanitize_for_pdf(text):
        """Remove non-ASCII characters and Unicode symbols, replacing with safe ASCII equivalents."""
        if not isinstance(text, str):
            text = str(text)

        replacements = {
            'σ': 'std_dev', 'Σ': 'SIGMA', 'μ': 'mean',
            '↑': '[UP]', '↓': '[DOWN]', '→': '[RIGHT]', '←': '[LEFT]',
            '⚠️': '[ALERT]', '⚠': '[ALERT]', '✓': '[OK]', '✗': '[FAIL]',
            '•': '-', '◆': '*', '○': 'o', '●': '*', '★': '*', '☆': '*',
            '™': '(TM)', '©': '(C)', '®': '(R)',
            '€': 'EUR', '£': 'GBP', '¥': 'YEN',
            '°': ' deg', '±': ' +/- ', '×': 'x', '÷': '/',
            '≈': ' approx ', '≠': ' != ', '≤': ' <= ', '≥': ' >= ',
            '√': 'sqrt', '∞': 'inf', '∑': 'sum', '∫': 'integral',
            '∆': 'DELTA', 'λ': 'lambda', 'π': 'pi', 'θ': 'theta',
            'Δ': 'DELTA', 'Ω': 'OMEGA',
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'κ': 'kappa',
            'ν': 'nu', 'ρ': 'rho', 'τ': 'tau', 'υ': 'upsilon',
            'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            '–': '-', '—': '-', '‐': '-', '‑': '-', '−': '-',
            '…': '...', '·': '.', '‰': '/1000',
            '′': "'", '″': '"', '‴': "'''",
            '‵': "'", '‶': '"',
            'ʹ': "'", 'ʺ': '"', 'ʻ': "'", 'ʼ': "'", 'ʽ': "'", 'ʾ': "'", 'ʿ': "'",
        }

        result = text
        for unicode_char, replacement in replacements.items():
            result = result.replace(unicode_char, replacement)

        try:
            result = result.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            result = ''.join(char if ord(char) < 128 else '?' for char in result)

        return result

    # Add text to PDF with proper encoding and width handling
    for line in report_text.split('\n'):
        if line.strip():
            clean_line = sanitize_for_pdf(line)
            if clean_line.strip():
                pdf.multi_cell(text_width, 5, clean_line, align='L')
        else:
            pdf.ln(2)

    # Save PDF to disk
    pdf_filename = f"HealthReport_{metric_type.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_filename)

    # Read PDF bytes once
    with open(pdf_filename, "rb") as f:
        pdf_bytes = f.read()

    # Display success and download button
    st.success("✓ Report generated successfully!")
    st.markdown("---")

    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
    )

    # Inline PDF preview (centered, like a printed page)
    st.markdown("### Report Preview")
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'''
    <div style="display:flex; justify-content:center;">
      <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        style="width:100%; max-width:900px; height:800px; border:1px solid #ccc; box-shadow:0 0 8px rgba(0,0,0,0.1);"
        type="application/pdf">
      </iframe>
    </div>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)


def append_recommendations_and_disclaimer(report_text: str, context_line: str) -> str:
     return report_text + f"""
{'='*80}

HEALTH RECOMMENDATIONS & ACTION PLAN
-------------------------------------

{context_line}

1. IMMEDIATE ACTIONS:
    - Continue monitoring your health metrics regularly
    - Note any lifestyle changes that correlate with anomalies
    - Maintain current positive patterns

2. SHORT-TERM IMPROVEMENTS (1-2 weeks):
    - Identify and document triggers for anomalies
    - Adjust routines based on patterns identified
    - Increase consistency in health habits

3. LONG-TERM STRATEGY (1-3 months):
    - Work toward optimal health targets
    - Build sustainable habits
    - Track improvement trends

4. WHEN TO SEEK PROFESSIONAL HELP:
    - If anomalies become frequent or severe
    - If trends are consistently negative
    - If you experience symptoms or feel unwell
    - After emergency alerts or concerning readings

PROFESSIONAL CONSULTATION:
If you have concerns about your health metrics or the anomalies detected, 
please consult with your healthcare provider for personalized medical advice.

{'='*80}

Report Disclaimer:
This report is generated for informational and educational purposes only.
It is not a substitute for professional medical diagnosis or treatment.
Please consult with a qualified healthcare provider for medical concerns.

Generated by: FitPlus Health Insights Dashboard
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


def format_metric_analysis_block(m_key: str, m_mean: float, m_median: float, m_std: float,
                                                                 m_min: float, m_max: float, m_recent_trend: float,
                                                                 m_unit: str) -> str:
        """Return the repeated analysis text block for a single metric (stats + narrative)."""

        block = f"""
Metric: {m_key}
~~~~~~~~~~~~~~~

Key Statistics:
    - Average: {m_mean:.2f} {m_unit}
    - Median: {m_median:.2f} {m_unit}
    - Standard Deviation: {m_std:.2f}
    - Range: {m_min:.2f} - {m_max:.2f} {m_unit}
    - 7-Day Trend: {'Increasing' if m_recent_trend > 0 else 'Decreasing'} ({abs(m_recent_trend):.2f} {m_unit})

"""

        # Heart-rate specific narrative
        if "Heart" in m_key:
                health_status = (
                        "Normal" if 60 <= m_mean <= 100 else "Elevated" if m_mean > 100 else "Low"
                )
                emergency = m_mean > 120 or m_mean < 40

                block += f"""
HEART RATE ANALYSIS
-------------------

Health Status: {health_status}

Your average resting heart rate of {m_mean:.1f} bpm indicates a {health_status.lower()} 
cardiovascular state. The medical standard for healthy resting heart rate in adults is 
60-100 bpm.

Pattern Analysis:
    - Consistency Score: {100 - (m_std / max(m_mean, 1) * 100):.1f}%
    - Your heart rate shows {'stable' if m_std < m_mean * 0.15 else 'variable'} patterns
    - Recent trend: {'Improving cardiovascular fitness' if m_recent_trend < 0 else 'Increased stress or activity'}

"""

                if emergency:
                        block += f"""
[EMERGENCY ALERT]
Your heart rate readings show concerning values ({m_min:.0f}-{m_max:.0f} bpm).
ACTION REQUIRED: Consult a healthcare provider immediately.

"""

        # Steps-specific narrative
        elif "Step" in m_key:
                daily_avg = m_mean
                activity_level = (
                        "Sedentary"
                        if daily_avg < 5000
                        else "Low Active"
                        if daily_avg < 7500
                        else "Somewhat Active"
                        if daily_avg < 10000
                        else "Active"
                        if daily_avg < 12500
                        else "Very Active"
                )

                block += f"""
DAILY ACTIVITY ANALYSIS
-----------------------

Activity Level: {activity_level}

Your average daily step count of {daily_avg:.0f} steps indicates a {activity_level.lower()} 
lifestyle. The World Health Organization recommends 10,000 steps daily for optimal 
health benefits.

Pattern Analysis:
    - Consistency Score: {100 - (m_std / max(m_mean, 1) * 100):.1f}%
    - Activity patterns: {'Regular and consistent' if m_std < m_mean * 0.25 else 'Highly variable'}
    - Recent trend: {'Increasing physical activity' if m_recent_trend > 0 else 'Declining activity levels'}

"""

        # Sleep-specific narrative
        elif "Sleep" in m_key:
                sleep_quality = (
                        "Excellent"
                        if 7 <= m_mean <= 9
                        else "Good"
                        if 6 <= m_mean < 7
                        else "Poor"
                        if m_mean < 6
                        else "Too Much"
                )

                block += f"""
SLEEP DURATION ANALYSIS
-----------------------

Sleep Quality Rating: {sleep_quality}

Your average nightly sleep of {m_mean:.1f} hours is rated as {sleep_quality.lower()}. 
The medical standard recommends 7-9 hours of quality sleep per night for adults.

Pattern Analysis:
    - Consistency Score: {100 - (m_std / max(m_mean, 1) * 100):.1f}%
    - Sleep patterns: {'Very stable sleep schedule' if m_std < m_mean * 0.15 else 'Irregular sleep patterns'}
    - Recent trend: {'Improving sleep duration' if m_recent_trend > 0 else 'Declining sleep hours'}

"""

        return block

files = {
    "Heart Rate": os.path.join(MODULE2_DIR, "daily_heart_rate.csv"),
    "Steps": os.path.join(MODULE2_DIR, "daily_steps.csv"),
    "Sleep": os.path.join(MODULE2_DIR, "daily_sleep.csv")
}


def normalize_metric_df(metric_df: pd.DataFrame) -> pd.DataFrame:
    df_m = metric_df.rename(columns={"ds": "date", "y": "value"}).copy()
    df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
    df_m["value"] = pd.to_numeric(df_m["value"], errors="coerce")
    return df_m.dropna(subset=["date", "value"]).sort_values("date")


@st.cache_data(show_spinner=False)
def load_steps_forecast_data():
    """Load precomputed forecast data for steps with and without holidays.

    Returns a dict with keys: actual, with_holidays, without_holidays, events.
    Any missing piece will be set to None.
    """
    data = {"actual": None, "with_holidays": None, "without_holidays": None, "events": None}

    # Actual daily steps
    steps_path = os.path.join(MODULE2_DIR, "daily_steps.csv")
    if os.path.exists(steps_path):
        try:
            df_steps = pd.read_csv(steps_path)
            if "ds" in df_steps.columns and "y" in df_steps.columns:
                df_steps["ds"] = pd.to_datetime(df_steps["ds"], errors="coerce")
                df_steps = df_steps.dropna(subset=["ds"]).sort_values("ds")
                data["actual"] = df_steps
        except Exception:
            pass

    # Forecasts without holidays
    no_holidays_path = os.path.join(MODULE2_DIR, "task3_forecast_no_holidays.csv")
    if os.path.exists(no_holidays_path):
        try:
            df_no = pd.read_csv(no_holidays_path)
            if "ds" in df_no.columns and "yhat" in df_no.columns:
                df_no["ds"] = pd.to_datetime(df_no["ds"], errors="coerce")
                df_no = df_no.dropna(subset=["ds"]).sort_values("ds")
                data["without_holidays"] = df_no
        except Exception:
            pass

    # Forecasts with holidays
    with_holidays_path = os.path.join(MODULE2_DIR, "task3_forecast_with_holidays.csv")
    if os.path.exists(with_holidays_path):
        try:
            df_with = pd.read_csv(with_holidays_path)
            if "ds" in df_with.columns and "yhat" in df_with.columns:
                df_with["ds"] = pd.to_datetime(df_with["ds"], errors="coerce")
                df_with = df_with.dropna(subset=["ds"]).sort_values("ds")
                data["with_holidays"] = df_with
        except Exception:
            pass

    # Events impact (used to highlight holiday-related anomalies)
    events_path = os.path.join(MODULE2_DIR, "task3_events_impact.csv")
    if os.path.exists(events_path):
        try:
            df_events = pd.read_csv(events_path)
            if "ds" in df_events.columns:
                df_events["ds"] = pd.to_datetime(df_events["ds"], errors="coerce")
                df_events = df_events.dropna(subset=["ds"]).sort_values("ds")
                data["events"] = df_events
        except Exception:
            pass

    return data


def render_steps_forecast_section(
    steps_df: pd.DataFrame,
    anom_mask: Optional[pd.Series] = None,
    anom_text: Optional[pd.Series] = None,
    show_anomalies: bool = True,
    show_events: bool = True,
):
    """Render two graphs for steps: forecasts with and without holidays.

    This version always uses the provided ``steps_df`` (uploaded / selected
    data) instead of precomputed Module 2 CSVs. When ``anom_mask`` is
    provided, the same anomaly flags used in the Anomalies tab are shown
    on both graphs, so changing severity or sensitivity updates all
    three views consistently.
    """
    if steps_df is None or steps_df.empty:
        st.info("No steps data available for forecast comparison.")
        return

    work = steps_df.copy()
    if "ds" not in work.columns or "y" not in work.columns:
        st.warning("Steps data must have 'ds' and 'y' columns for forecasting.")
        return

    work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["ds", "y"]).sort_values("ds")
    if work.empty:
        st.info("No valid steps data available after cleaning.")
        return

    # Prepare base daily series
    base = work[["ds", "y"]].rename(columns={"ds": "ds", "y": "y"})

    # Fit Prophet model without holidays
    try:
        model_no = Prophet(
            interval_width=0.95,
            yearly_seasonality=len(base) > 365,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model_no.fit(base)
        fcst_no = model_no.predict(base[["ds"]])
    except Exception as e:
        st.warning(f"Could not fit steps forecast model: {e}")
        return

    # Construct a simple synthetic holiday calendar similar to Module 2
    unique_days = base["ds"].dt.floor("D").sort_values().unique()
    holidays_df = None
    if len(unique_days) > 0:
        start_day = pd.to_datetime(unique_days[0])

        def date_for_day(one_indexed: int) -> pd.Timestamp:
            return start_day + pd.Timedelta(days=one_indexed - 1)

        holiday_rows = []
        # Vacation block around day 30
        for d in range(30, 38):
            holiday_rows.append({"holiday": "vacation", "ds": date_for_day(d)})
        # Sick days around day 60
        for d in [60, 61, 62]:
            holiday_rows.append({"holiday": "sick", "ds": date_for_day(d)})
        # Marathon day at day 90
        holiday_rows.append({"holiday": "marathon", "ds": date_for_day(90)})

        holidays_df = pd.DataFrame(holiday_rows)
        holidays_df["ds"] = pd.to_datetime(holidays_df["ds"], errors="coerce")

    # Fit Prophet model with holidays (if we could build a calendar)
    if holidays_df is not None and not holidays_df.empty:
        try:
            model_yes = Prophet(
                interval_width=0.95,
                yearly_seasonality=len(base) > 365,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=holidays_df,
            )
            model_yes.fit(base)
            fcst_yes = model_yes.predict(base[["ds"]])
        except Exception:
            fcst_yes = fcst_no.copy()
    else:
        fcst_yes = fcst_no.copy()

    # Keep forecasts aligned to actual days only
    merged_no = pd.merge(
        base,
        fcst_no[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )
    merged_with = pd.merge(
        base,
        fcst_yes[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )

    if merged_no.empty or merged_with.empty:
        st.warning("Aligned steps forecast data could not be prepared.")
        return

    # Use ensemble anomaly flags from the caller when provided
    if anom_mask is not None and len(anom_mask) == len(work):
        merged_no["is_anom"] = anom_mask.values
        merged_with["is_anom"] = anom_mask.values
        if anom_text is not None and len(anom_text) == len(work):
            merged_no["anom_text"] = anom_text.values
            merged_with["anom_text"] = anom_text.values
        else:
            merged_no["anom_text"] = "Flagged anomaly"
            merged_with["anom_text"] = "Flagged anomaly"
    else:
        merged_no["is_anom"] = False
        merged_no["anom_text"] = ""
        merged_with["is_anom"] = False
        merged_with["anom_text"] = ""

    col_a, col_b = st.columns(2)

    # --- Without holidays ---
    with col_a:
        st.markdown("#### Steps Forecast (Without Holidays)")
        fig_no = go.Figure()
        fig_no.add_trace(
            go.Scatter(
                x=merged_no["ds"],
                y=merged_no["y"],
                mode="lines+markers",
                name="Actual Steps",
                line=dict(color="#2e86de", width=1.8),
                marker=dict(size=4),
            )
        )
        fig_no.add_trace(
            go.Scatter(
                x=merged_no["ds"],
                y=merged_no["yhat"],
                mode="lines",
                name="Forecast (No Holidays)",
                line=dict(color="#16a085", width=2, dash="dash"),
            )
        )
        # Confidence band
        fig_no.add_trace(
            go.Scatter(
                x=pd.concat([merged_no["ds"], merged_no["ds"][::-1]]),
                y=pd.concat([merged_no["yhat_upper"], merged_no["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Anomaly markers (optional, driven by ensemble flags when available)
        if show_anomalies:
            anom_no = merged_no[merged_no["is_anom"]]
            if not anom_no.empty:
                fig_no.add_trace(
                    go.Scatter(
                        x=anom_no["ds"],
                        y=anom_no["y"],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(size=8, color="red", symbol="x"),
                        text=anom_no["anom_text"],
                        hovertemplate="%{x|%Y-%m-%d}<br>Steps: %{y:.0f}<br>%{text}<extra></extra>",
                    )
                )

        fig_no.update_layout(
            xaxis_title="Date",
            yaxis_title="Steps",
            height=320,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_no, use_container_width=True)

    # --- With holidays ---
    with col_b:
        st.markdown("#### Steps Forecast (With Holidays)")
        fig_with = go.Figure()
        fig_with.add_trace(
            go.Scatter(
                x=merged_with["ds"],
                y=merged_with["y"],
                mode="lines+markers",
                name="Actual Steps",
                line=dict(color="#2e86de", width=1.8),
                marker=dict(size=4),
            )
        )
        fig_with.add_trace(
            go.Scatter(
                x=merged_with["ds"],
                y=merged_with["yhat"],
                mode="lines",
                name="Forecast (With Holidays)",
                line=dict(color="#8e44ad", width=2, dash="dash"),
            )
        )
        # Confidence band
        fig_with.add_trace(
            go.Scatter(
                x=pd.concat([merged_with["ds"], merged_with["ds"][::-1]]),
                y=pd.concat([merged_with["yhat_upper"], merged_with["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(155, 89, 182, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Anomaly markers based on ensemble flags (optional)
        if show_anomalies:
            anom_with = merged_with[merged_with["is_anom"]]
            if not anom_with.empty:
                fig_with.add_trace(
                    go.Scatter(
                        x=anom_with["ds"],
                        y=anom_with["y"],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(size=8, color="red", symbol="x"),
                        text=anom_with["anom_text"],
                        hovertemplate="%{x|%Y-%m-%d}<br>Steps: %{y:.0f}<br>%{text}<extra></extra>",
                    )
                )

        # Highlight specific holiday / event days if available (optional)
        if show_events:
            events_path = os.path.join(MODULE2_DIR, "task3_events_impact.csv")
            if os.path.exists(events_path):
                try:
                    df_events = pd.read_csv(events_path)
                    if "ds" in df_events.columns and "event" in df_events.columns:
                        df_events["ds"] = pd.to_datetime(df_events["ds"], errors="coerce")
                        df_events = df_events.dropna(subset=["ds"]).sort_values("ds")
                        events_merged = pd.merge(
                            df_events[["ds", "event"]],
                            fcst_yes[["ds", "yhat"]],
                            on="ds",
                            how="inner",
                        )
                        if not events_merged.empty:
                            fig_with.add_trace(
                                go.Scatter(
                                    x=events_merged["ds"],
                                    y=events_merged["yhat"],
                                    mode="markers",
                                    name="Holiday / Event Days",
                                    marker=dict(size=9, color="orange", symbol="diamond"),
                                    text=events_merged["event"],
                                    hovertemplate="%{x|%Y-%m-%d}<br>Event: %{text}<br>Forecast: %{y:.0f} steps<extra></extra>",
                                )
                            )
                except Exception:
                    pass

        fig_with.update_layout(
            xaxis_title="Date",
            yaxis_title="Steps",
            height=320,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_with, use_container_width=True)

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["Home", "Analysis", "Anomalies", "Reports"])

# -----------------------------
# HOME TAB
# -----------------------------
with tabs[0]:
    # Rotating greeting
    greetings = [
        "Welcome back, Champion!",
        "Ready to crush your goals today?",
        "Stay strong, stay healthy!",
        "Your health journey continues!",
        "Every step counts. Let's go!",
    ]
    import random
    greeting = random.choice(greetings)
    st.markdown(
        f"<h2 style='font-size:2.5em; color:#2e86de; margin-bottom:0.2em;'>{greeting}</h2>",
        unsafe_allow_html=True,
    )

    # Simple human-friendly summary for the Home tab
    st.caption("This page gives you a quick view of today and the last week so you can spot any unusual heart, steps, or sleep days at a glance.")

    # Derive a compact health status
    uploaded_series = st.session_state.get("analysis_user_data", {})

    if uploaded_series:
        # Use latest day from user-uploaded daily series
        latest_date = None
        for mdf in uploaded_series.values():
            if not mdf.empty:
                d = mdf["ds"].max().date()
                if latest_date is None or d > latest_date:
                    latest_date = d

        avg_hr = None
        total_steps = None
        total_sleep = None

        if latest_date is not None:
            if "Heart Rate" in uploaded_series:
                df_hr = uploaded_series["Heart Rate"]
                row = df_hr[df_hr["ds"].dt.date == latest_date]
                if not row.empty:
                    avg_hr = row["y"].iloc[-1]

            if "Steps" in uploaded_series:
                df_steps = uploaded_series["Steps"]
                row = df_steps[df_steps["ds"].dt.date == latest_date]
                if not row.empty:
                    total_steps = row["y"].iloc[-1]

            if "Sleep" in uploaded_series:
                df_sleep = uploaded_series["Sleep"]
                row = df_sleep[df_sleep["ds"].dt.date == latest_date]
                if not row.empty:
                    total_sleep = row["y"].iloc[-1]

        # Basic rules matching derive_daily_health_status
        if latest_date is None or avg_hr is None or np.isnan(avg_hr) or total_steps is None or np.isnan(total_steps):
            health_status_label, health_mood = "Status Unknown", "neutral"
        elif (total_sleep is not None and not np.isnan(total_sleep) and total_sleep < 5.5) or avg_hr > 95:
            health_status_label, health_mood = "Needs Rest", "tired"
        elif total_steps < 3000:
            health_status_label, health_mood = "Low Activity", "sleepy"
        elif (
            total_steps >= 10000
            and total_sleep is not None
            and not np.isnan(total_sleep)
            and 7 <= total_sleep <= 9
            and 55 <= avg_hr <= 85
        ):
            health_status_label, health_mood = "Great Balance", "happy"
        else:
            health_status_label, health_mood = "On Track", "calm"
    else:
        # Fallback to project sample data if no uploads are available
        processed_df = load_processed_data(PROCESSED_DATA_PATH)
        health_status_label, health_mood = derive_daily_health_status(processed_df)

    # Show status badge as a short 2-3 word summary with an emoji
    status_container = st.container()
    badge_colors = {
        "happy": "#27ae60",
        "calm": "#2980b9",
        "sleepy": "#f1c40f",
        "tired": "#e74c3c",
        "neutral": "#7f8c8d",
    }
    mood_emojis = {
        "happy": "😊",
        "calm": "😌",
        "sleepy": "😴",
        "tired": "😓",
        "neutral": "😐",
    }
    badge_color = badge_colors.get(health_mood, "#7f8c8d")
    mood_emoji = mood_emojis.get(health_mood, "😐")
    with status_container:
        st.markdown(
            f"""
                        <div style="margin:0.25rem 0 0.25rem 0;">
                            <span style="font-weight:600; color:#2c3e50; margin-right:0.35rem;">Health Status:</span>
                            <span style="display:inline-flex; align-items:center; gap:0.3rem; padding:0.15rem 0.6rem; border-radius:999px; background:{badge_color}; color:white; font-weight:600; font-size:0.9rem;">
                                <span>{mood_emoji}</span>
                                <span>{health_status_label}</span>
                            </span>
                        </div>
            """,
            unsafe_allow_html=True,
        )

        # If no user-uploaded data is driving the status, show a small note
        if not uploaded_series:
            st.caption("No current user-uploaded data detected. Showing sample project data.")

    # Load today's stats
    today = pd.Timestamp.now().normalize()

    if uploaded_series:
        # Use uploaded daily series for today's metrics
        def get_today_from_series(metric_name):
            mdf = uploaded_series.get(metric_name)
            if mdf is None or mdf.empty:
                return None
            row = mdf[mdf["ds"].dt.normalize() == today]
            if not row.empty:
                return row["y"].iloc[-1]
            return None

        hr = get_today_from_series("Heart Rate")
        steps = get_today_from_series("Steps")
        sleep = get_today_from_series("Sleep")
    else:
        def get_today_stat(path, value_col):
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "ds" in df.columns:
                    df["ds"] = pd.to_datetime(df["ds"])
                    # Find rows where ds is today (any time)
                    row = df[df["ds"].dt.normalize() == today]
                    if not row.empty:
                        return row.iloc[0][value_col]
            return None

        hr = get_today_stat(files["Heart Rate"], "y")
        steps = get_today_stat(files["Steps"], "y")
        sleep = get_today_stat(files["Sleep"], "y")

    col1, col2, col3 = st.columns(3)
    col1.metric("Today's Avg Heart Rate", f"{hr:.0f} bpm" if hr is not None else "-", delta=None)
    col2.metric("Today's Steps", f"{steps:.0f}" if steps is not None else "-", delta=None)
    col3.metric("Last Night's Sleep", f"{sleep:.1f} hrs" if sleep is not None else "-", delta=None)

    # Anomaly summary (today)
    anomaly_summary = "No anomalies detected today."
    # Try to load anomaly data if available
    anomaly_path = os.path.join(MODULE2_DIR, "task3_events_impact.csv")
    if os.path.exists(anomaly_path):
        adf = pd.read_csv(anomaly_path)
        if "date" in adf.columns:
            adf["date"] = pd.to_datetime(adf["date"]).dt.normalize()
            today_anoms = adf[adf["date"] == today]
            if not today_anoms.empty:
                anomaly_summary = f"{len(today_anoms)} anomaly(s) detected today."
    st.info(anomaly_summary)

    # 7-day sparklines
    st.markdown("#### Last 7 Days Overview")
    spark_col1, spark_col2, spark_col3 = st.columns(3)

    def sparkline_series(mdf, label, col):
        if mdf is None or mdf.empty:
            return
        mdf = mdf.copy()
        mdf["date"] = mdf["ds"].dt.normalize()
        end_date = today
        start_date = today - pd.Timedelta(days=6)
        date_index = pd.date_range(start_date, end_date, freq="D")
        daily = (
            mdf.sort_values("date")
               .drop_duplicates(subset="date", keep="last")
               .set_index("date")["y"]
        )
        series = daily.reindex(date_index, fill_value=0)
        col.markdown(f"**{label}**")
        col.line_chart(series, height=60)

    if uploaded_series:
        sparkline_series(uploaded_series.get("Heart Rate"), "Heart Rate", spark_col1)
        sparkline_series(uploaded_series.get("Steps"), "Steps", spark_col2)
        sparkline_series(uploaded_series.get("Sleep"), "Sleep", spark_col3)
    else:
        def sparkline(path, label, col):
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "ds" in df.columns:
                    df["ds"] = pd.to_datetime(df["ds"])
                    df["date"] = df["ds"].dt.normalize()
                    # Build a fixed 7-day window ending today
                    end_date = today
                    start_date = today - pd.Timedelta(days=6)
                    date_index = pd.date_range(start_date, end_date, freq="D")

                    # Aggregate to one value per day (last value if multiple)
                    daily = (
                        df.sort_values("date")
                          .drop_duplicates(subset="date", keep="last")
                          .set_index("date")["y"]
                    )
                    series = daily.reindex(date_index, fill_value=0)

                    col.markdown(f"**{label}**")
                    col.line_chart(series, height=60)

        sparkline(files["Heart Rate"], "Heart Rate", spark_col1)
        sparkline(files["Steps"], "Steps", spark_col2)
        sparkline(files["Sleep"], "Sleep", spark_col3)


# -----------------------------
# ANALYSIS TAB (Optimized)
# -----------------------------
with tabs[1]:
    st.subheader("📊 Health Data Analysis")

    # Simple human-friendly summary for the Analysis tab
    st.caption("Use this page to explore how your heart, steps, and sleep change over time for any date range you choose.")

    # User-provided metric files (required for a user-centric deployment)
    st.markdown("**📁 Upload Health CSV Files**")
    uploaded_analysis_files = st.file_uploader(
        "Upload up to 6 CSV files (raw or processed)",
        type=["csv"],
        accept_multiple_files=True,
        key="analysis_upload",
        help=(
            "Upload any Fitbit health CSVs (raw or processed, train/test/etc.). "
            "The dashboard will auto-detect date and metric columns and extract "
            "daily series for Heart Rate, Steps, and Sleep based on column names."
        ),
    )

    # Map uploaded files to metric names used in the checkboxes
    if uploaded_analysis_files:
        if len(uploaded_analysis_files) > 6:
            st.warning("⚠️ Only the first 6 files will be used for analysis.")
            uploaded_analysis_files = uploaded_analysis_files[:6]

        user_data = {}
        metric_sources = {}
        raw_files = []
        for f in uploaded_analysis_files:
            try:
                df_u = pd.read_csv(f, low_memory=False)
                raw_files.append({"name": f.name, "df": df_u})

                metric_series = extract_metric_time_series(df_u, f.name)
                if not metric_series:
                    st.warning(
                        f"{f.name}: could not detect Heart Rate, Steps, or Sleep columns. "
                        "Check that it has a date column and metric columns such as heart_rate, steps, sleep_hours, etc.")
                    continue

                # Merge each detected metric into the session mapping
                for metric_key, mdf in metric_series.items():
                    if metric_key in user_data:
                        combined = (
                            pd.concat([user_data[metric_key], mdf], ignore_index=True)
                              .drop_duplicates(subset=["ds"])
                              .sort_values("ds")
                        )
                        user_data[metric_key] = combined
                    else:
                        user_data[metric_key] = mdf

                    # Track which files contributed to each metric
                    metric_sources.setdefault(metric_key, set()).add(f.name)

                detected = ", ".join(metric_series.keys())
                st.success(f"✅ Loaded {f.name}: detected {detected} for Analysis tab.")
            except Exception as e:
                st.error(f"Error reading {f.name}: {str(e)}")

        if user_data:
            st.session_state["analysis_user_data"] = user_data
            # Store human-readable source list per metric for display under headings
            st.session_state["analysis_metric_sources"] = {
                k: sorted(list(v)) for k, v in metric_sources.items()
            }
            # Also keep raw uploaded dataframes so other tabs (Reports, Home) can reuse them
            st.session_state["analysis_raw_files"] = raw_files
    else:
        # If nothing uploaded in this run, keep any existing session data as-is
        pass
    
    # Date Range Selection with Calendar
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**📅 Select Date Range**")
        min_date = date(2000, 1, 1)
        max_date = date.today()
        date_range = st.date_input(
            "Date Range:",
            value=(date.today() - timedelta(days=30), date.today()),
            min_value=min_date,
            max_value=max_date,
            key="analysis_dates"
        )
    
    with col2:
        st.markdown("**📊 Data Options**")
        use_all_data = st.checkbox("Analyze All Data", value=False)
        if use_all_data:
            st.info("Will analyze all available data")
        # Persist the "analyze all data" flag so the Anomalies tab can
        # use the exact same date-range logic as the Analysis tab.
        st.session_state["analysis_use_all_data"] = use_all_data
    
    # Metric Selection
    st.markdown("**🔍 Select Metrics to Analyze**")
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_hr = st.checkbox("Heart Rate", value=True)
    with col2:
        analyze_steps = st.checkbox("Steps", value=False)
    with col3:
        analyze_sleep = st.checkbox("Sleep", value=False)
    
    selected_metrics = []
    if analyze_hr:
        selected_metrics.append("Heart Rate")
    if analyze_steps:
        selected_metrics.append("Steps")
    if analyze_sleep:
        selected_metrics.append("Sleep")
    
    # Run Analysis Button
    if st.button("🔍 Run Analysis", use_container_width=True):
        if "analysis_user_data" not in st.session_state or not st.session_state["analysis_user_data"]:
            st.error("Please upload at least one valid CSV file above before running analysis.")
        elif not selected_metrics:
            st.error("Please select at least one metric to analyze.")
        else:
            # Remember which metrics were selected for use in the Reports tab
            st.session_state["analysis_selected_metrics"] = selected_metrics.copy()

            # Determine date range
            if use_all_data:
                start_date = min_date
                end_date = max_date
                date_label = "All Available Data"
            else:
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range
                date_label = f"{start_date} to {end_date}"
            
            # Analyze each selected metric
            forecast = None  # will hold Prophet forecast if model fits successfully
            for metric in selected_metrics:
                st.markdown(f"---")
                st.markdown(f"## {metric} Analysis ({date_label})")

                # Show which uploaded files contributed to this metric (if known)
                src_map = st.session_state.get("analysis_metric_sources", {})
                if metric in src_map:
                    sources_str = ", ".join(src_map[metric])
                    st.caption(f"Data source(s): {sources_str}")

                # Require user-uploaded data for analysis (no backend fallback)
                if "analysis_user_data" not in st.session_state or metric not in st.session_state["analysis_user_data"]:
                    st.error(
                        f"No uploaded data found for {metric}. "
                        "Please upload a CSV file with this metric above.")
                    continue

                df = st.session_state["analysis_user_data"][metric].copy()

                try:
                    if 'ds' not in df.columns or 'y' not in df.columns:
                        st.error(f"Invalid format for {metric} (missing 'ds' or 'y').")
                        continue
                    
                    ds_series = pd.to_datetime(df['ds'], errors='coerce')
                    # Ensure ds is tz-naive for Prophet compatibility
                    try:
                        if getattr(ds_series.dt, "tz", None) is not None:
                            try:
                                ds_series = ds_series.dt.tz_convert(None)
                            except TypeError:
                                ds_series = ds_series.dt.tz_localize(None)
                    except Exception:
                        pass

                    df['ds'] = ds_series
                    df = df.sort_values('ds')

                    # Choose a friendly y-axis label based on metric type
                    if metric == "Heart Rate":
                        y_label = "Heart Rate (bpm)"
                    elif metric == "Steps":
                        y_label = "Steps"
                    elif metric == "Sleep":
                        y_label = "Minutes Asleep"
                    else:
                        y_label = "Value"
                    
                    # Filter by date range
                    df_range = df[(df['ds'].dt.date >= start_date) & (df['ds'].dt.date <= end_date)].copy()
                    
                    if df_range.empty:
                        st.warning(f"No data for {metric} in selected range.")
                        continue
                    
                    # Create 3-column layout for graphs
                    col1, col2, col3 = st.columns(3)
                    
                    # Graph 1: Time Series Line Plot
                    with col1:
                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(
                            x=df_range['ds'], y=df_range['y'],
                            mode='lines+markers',
                            name=metric,
                            line=dict(color='#2e86de', width=2),
                            marker=dict(size=4)
                        ))
                        fig_ts.update_layout(
                            title="Time Series",
                            xaxis_title="Date",
                            yaxis_title=y_label,
                            height=400,
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Graph 2: Scatter Plot
                    with col2:
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=df_range['ds'], y=df_range['y'],
                            mode='markers',
                            name=metric,
                            marker=dict(size=8, color='#ff6348', opacity=0.7)
                        ))
                        fig_scatter.update_layout(
                            title="Scatter Plot",
                            xaxis_title="Date",
                            yaxis_title=y_label,
                            height=400,
                            hovermode='closest',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Graph 3: Prophet Decomposition
                    with col3:
                        if len(df_range) >= 14:
                            try:
                                df_prophet = df_range[['ds', 'y']].copy()
                                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, 
                                              daily_seasonality=False, interval_width=0.95)
                                with st.spinner("Fitting Prophet model..."):
                                    model.fit(df_prophet)
                                
                                future = model.make_future_dataframe(periods=0)
                                forecast = model.predict(future)
                                
                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=df_prophet['ds'], y=df_prophet['y'],
                                    mode='lines', name='Actual',
                                    line=dict(color='#2e86de', width=2)
                                ))
                                fig_trend.add_trace(go.Scatter(
                                    x=forecast['ds'], y=forecast['trend'],
                                    mode='lines', name='Trend',
                                    line=dict(color='#ff6348', width=2, dash='dash')
                                ))
                                fig_trend.update_layout(
                                    title="Trend Component",
                                    xaxis_title="Date",
                                    yaxis_title=y_label,
                                    height=400,
                                    hovermode='x unified',
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Prophet error: {str(e)}")
                        else:
                            st.info(f"Need ≥14 data points for Prophet. Current: {len(df_range)}")
                    
                    # Weekly Seasonality (only if Prophet succeeded and forecast is available)
                    if len(df_range) >= 14 and forecast is not None:
                        try:
                            col_w, col_y = st.columns(2)
                            
                            with col_w:
                                if 'weekly' in forecast.columns:
                                    fig_weekly = go.Figure()
                                    fig_weekly.add_trace(go.Scatter(
                                        x=forecast['ds'], y=forecast['weekly'] if 'weekly' in forecast.columns else forecast.get('weekly_0', [0]*len(forecast)),
                                        mode='lines', name='Weekly',
                                        line=dict(color='#1dd1a1', width=2)
                                    ))
                                    fig_weekly.update_layout(
                                        title="Weekly Seasonality",
                                        xaxis_title="Date",
                                        yaxis_title="Seasonality Component",
                                        height=350,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_weekly, use_container_width=True)
                            
                            with col_y:
                                if 'yearly' in forecast.columns:
                                    fig_yearly = go.Figure()
                                    fig_yearly.add_trace(go.Scatter(
                                        x=forecast['ds'], y=forecast['yearly'] if 'yearly' in forecast.columns else forecast.get('yearly_0', [0]*len(forecast)),
                                        mode='lines', name='Yearly',
                                        line=dict(color='#f368e0', width=2)
                                    ))
                                    fig_yearly.update_layout(
                                        title="Yearly Seasonality",
                                        xaxis_title="Date",
                                        yaxis_title="Seasonality Component",
                                        height=350,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_yearly, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Seasonality visualization error: {str(e)}")
                    
                    # Additional forecast comparison specifically for Steps
                    if metric == "Steps":
                        st.markdown("### 📈 Steps Forecast: With vs Without Holidays")
                        # In Analysis we show clean forecasts (no anomaly markers),
                        # but always based on the same uploaded steps series
                        render_steps_forecast_section(
                            steps_df=df_range,
                            anom_mask=None,
                            anom_text=None,
                            show_anomalies=False,
                            show_events=False,
                        )

                    # Summary Statistics Table
                    st.markdown("### 📊 Summary Statistics")
                    
                    summary_stats = {
                        "Metric": ["Count", "Mean", "Median", "Std Dev", "Min", "Max", "Range"],
                        "Value": [
                            f"{len(df_range)}",
                            f"{df_range['y'].mean():.2f}",
                            f"{df_range['y'].median():.2f}",
                            f"{df_range['y'].std():.2f}",
                            f"{df_range['y'].min():.2f}",
                            f"{df_range['y'].max():.2f}",
                            f"{df_range['y'].max() - df_range['y'].min():.2f}"
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_stats)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error analyzing {metric}: {str(e)}")

## ===== ANOMALY DETECTION TAB (Advanced Multi-Method Ensemble) =====
with tabs[2]:
    st.subheader("🔍 Advanced Anomaly Detection")

    # Simple human-friendly summary for the Anomalies tab
    st.caption("This page highlights days that look very different from your usual pattern so you know when to pay closer attention.")

    # Use the same date-range logic as the Analysis tab, if available.
    # If the user checked "Analyze All Data" there, we mirror that here.
    start_date = None
    end_date = None
    analysis_dates = st.session_state.get("analysis_dates")
    analysis_use_all = st.session_state.get("analysis_use_all_data", False)

    if analysis_use_all:
        # Match Analysis tab behaviour: all available dates
        start_date = date(2000, 1, 1)
        end_date = date.today()
    elif analysis_dates is not None:
        # Mirror the currently selected date range from the Analysis tab
        if isinstance(analysis_dates, tuple) and len(analysis_dates) == 2:
            start_date, end_date = analysis_dates
        else:
            start_date = end_date = analysis_dates

    # Metric selection and detection sensitivity (data comes from Analysis tab uploads)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**🔍 Select Metrics to Detect Anomalies**")
        anom_hr = st.checkbox("Heart Rate", value=True, key="anom_hr")
        anom_steps = st.checkbox("Steps", value=True, key="anom_steps")
        anom_sleep = st.checkbox("Sleep", value=True, key="anom_sleep")

        selected_anom_metrics = []
        if anom_hr:
            selected_anom_metrics.append("Heart Rate")
        if anom_steps:
            selected_anom_metrics.append("Steps")
        if anom_sleep:
            selected_anom_metrics.append("Sleep")

    with col2:
        st.markdown("**⚙️ Detection Sensitivity**")
        sensitivity_choice = st.selectbox(
            "How sensitive should anomaly detection be?",
            [
                "Balanced (recommended)",
                "catch more anomalies",
                "only strong anomalies",
            ],
            index=0,
            key="sensitivity_mode",
            help=(
                "Balanced works well for most cases. "
                "Use 'catch more anomalies' to flag more subtle changes, or 'only strong anomalies' to only show larger deviations."
            ),
        )

        # Backend rule: for Steps, always treat any Prophet
        # forecast band breach as an anomaly (no user toggle).
        steps_all_prophet = True

        # Map sensitivity choice to internal numeric thresholds for Low / Medium / High severity
        if sensitivity_choice == "catch more anomalies":
            # More sensitive: flag more points as anomalies
            low_threshold, medium_threshold, high_threshold = 0.25, 0.55, 0.8
        elif sensitivity_choice == "only strong anomalies":
            # Less sensitive: only stronger deviations are treated as anomalies
            low_threshold, medium_threshold, high_threshold = 0.5, 0.7, 0.9
        else:  # Balanced (recommended)
            low_threshold, medium_threshold, high_threshold = 0.3, 0.7, 0.85

    # SINGLE RUN BUTTON - NO RERUNS (data is taken from Analysis tab uploads)
    if st.button("🚀 Analyze", use_container_width=True, key="analyze_btn"):
        if not selected_anom_metrics:
            st.error("Please select at least one metric for anomaly detection.")
        elif "analysis_user_data" not in st.session_state or not st.session_state["analysis_user_data"]:
            st.error("Please upload and process data in the Analysis tab first.")
        else:
            with st.spinner("Processing anomalies..."):
                try:
                    processed_data = {}

                    for metric_type in selected_anom_metrics:
                        if metric_type not in st.session_state["analysis_user_data"]:
                            st.warning(
                                f"No data found for {metric_type} in the uploaded files. "
                                "Please upload a CSV containing this metric in the Analysis tab. It will be skipped for now."
                            )
                            continue

                        df_clean = st.session_state["analysis_user_data"][metric_type].copy()
                        if 'ds' not in df_clean.columns or 'y' not in df_clean.columns:
                            st.error(f"{metric_type}: expected columns 'ds' and 'y' in uploaded data.")
                            continue

                        df_clean['ds'] = pd.to_datetime(df_clean['ds'], errors='coerce')
                        df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')
                        df_clean = df_clean.dropna(subset=['ds', 'y']).sort_values('ds').reset_index(drop=True)

                        # Apply the same date filter used in the Analysis tab,
                        # so both tabs operate on an identical window.
                        if start_date is not None and end_date is not None:
                            mask = (df_clean['ds'].dt.date >= start_date) & (df_clean['ds'].dt.date <= end_date)
                            df_clean = df_clean.loc[mask].reset_index(drop=True)

                        if len(df_clean) < 5:
                            st.error(
                                f"{metric_type}: Need at least 5 valid data points "
                                f"(found {len(df_clean)}) in the selected date range"
                            )
                            continue

                        processed_data[metric_type] = df_clean

                    if processed_data:
                        st.session_state['anom_data'] = processed_data
                        st.session_state['anom_config'] = {
                            'low_threshold': low_threshold,
                            'medium_threshold': medium_threshold,
                            'high_threshold': high_threshold,
                            'sensitivity_mode': sensitivity_choice,
                            # This behaviour is now compulsory in the backend:
                            # any Prophet band breach for Steps is always treated
                            # as an anomaly.
                            'steps_all_prophet': True,
                            # Persist the effective date window for use when
                            # rendering forecasts (e.g., steps with/without holidays).
                            'start_date': start_date,
                            'end_date': end_date,
                        }
                        # Remember which metrics the user asked to analyze so we can
                        # show explicit "no data" messages for missing ones later.
                        st.session_state['anom_selected_metrics'] = list(selected_anom_metrics)
                        st.success("✅ Ready to display results")
                        st.rerun()
                    else:
                        st.error("No usable data found for the selected metrics.")
                except Exception as e:
                    st.error(f"Error: {str(e)[:150]}")

    # ===== DISPLAY CACHED RESULTS (OPTIMIZED) =====
    if 'anom_data' in st.session_state:
        config = st.session_state['anom_config']

        # Use the last selected metrics if available, otherwise just the ones we have data for
        selected_for_display = st.session_state.get(
            'anom_selected_metrics', list(st.session_state['anom_data'].keys())
        )

        for metric_name in selected_for_display:
            if metric_name not in st.session_state['anom_data']:
                # User selected this metric but there was no data in uploaded files
                st.warning(
                    f"{metric_name}: No data available from uploaded files. "
                    "Upload a CSV for this metric in the Analysis tab to see anomalies here."
                )
                continue

            df = st.session_state['anom_data'][metric_name].copy()

            # Handle edge cases
            if len(df) < 3 or df['y'].std() == 0:
                st.warning(f"⚠️ {metric_name}: Insufficient variance for analysis")
                continue

            mean_val = df['y'].mean()
            std_val = df['y'].std()

            # Friendly label for the value axis based on metric
            if "heart" in metric_name.lower():
                value_label = "Heart Rate (bpm)"
            elif "step" in metric_name.lower():
                value_label = "Steps"
            elif "sleep" in metric_name.lower():
                value_label = "Minutes Asleep"
            else:
                value_label = "Value"

            # FAST: Vectorized anomaly detection
            anomaly_scores = np.zeros(len(df))
            anomaly_types = np.array(['Normal'] * len(df))
            # Track how many methods vote each point as anomalous
            method_votes = np.zeros(len(df), dtype=int)
            # Track Prophet band breaches explicitly so we can
            # always include them for Steps if the user chooses.
            prophet_band_hits = np.zeros(len(df), dtype=bool)
            # Collect textual reasons for each potential anomaly
            reason_flags = [[] for _ in range(len(df))]

            # Point anomalies (fast, vectorized)
            outliers = (df['y'] < mean_val - 2.5 * std_val) | (df['y'] > mean_val + 2.5 * std_val)
            anomaly_scores[outliers] += 0.4
            anomaly_types[outliers] = 'Point'
            method_votes[outliers] += 1
            outlier_idx = np.where(outliers)[0]
            for i in outlier_idx:
                reason_flags[i].append("Unusually high or low compared with your typical values.")

            # Contextual anomalies (fast, vectorized)
            changes = np.abs(df['y'].diff()).fillna(0)
            spikes = changes > std_val * 2
            anomaly_scores[spikes] += 0.3
            anomaly_types[spikes] = 'Contextual'
            method_votes[spikes] += 1
            spike_idx = np.where(spikes)[0]
            for i in spike_idx:
                if i == 0 or pd.isna(df['y'].iloc[i - 1]):
                    desc = "Sudden change compared with earlier days."
                else:
                    if df['y'].iloc[i] > df['y'].iloc[i - 1]:
                        desc = "Sudden jump up compared with the previous day."
                    else:
                        desc = "Sudden drop compared with the previous day."
                reason_flags[i].append(desc)

            # Prophet (once, cached via session) - OPTIMIZED for large data
            if len(df) >= 20:
                # Initialize cache dict if needed
                if 'prophet_cache' not in st.session_state:
                    st.session_state['prophet_cache'] = {}

                try:
                    cache = st.session_state['prophet_cache']
                    forecast = cache.get(metric_name)

                    # Refit Prophet if no forecast cached or length mismatch
                    if forecast is None or len(forecast) != len(df):
                        df_p = df[['ds', 'y']].copy()
                        df_p.columns = ['ds', 'y']

                        # Suppress Prophet warnings
                        import logging
                        logging.getLogger('prophet').setLevel(logging.WARNING)

                        model = Prophet(
                            interval_width=0.95,
                            yearly_seasonality=len(df) > 365,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                        )

                        # Fit with reduced verbosity
                        with st.spinner(f"Fitting Prophet for {metric_name}..."):
                            model.fit(df_p)

                        forecast = model.predict(df_p[['ds']])
                        cache[metric_name] = forecast

                    # Now forecast length should match df length
                    model_anom = (
                        df['y'].values < forecast['yhat_lower'].values
                    ) | (
                        df['y'].values > forecast['yhat_upper'].values
                    )
                    anomaly_scores[model_anom] += 0.4
                    method_votes[model_anom] += 1
                    prophet_band_hits = model_anom
                    prophet_idx = np.where(model_anom)[0]
                    for i in prophet_idx:
                        reason_flags[i].append("Outside the forecast range predicted from your past data.")
                except Exception:
                    # If Prophet fails for any reason, just skip model-based anomalies
                    pass
                
                # Cluster-based (OPTIMIZED for large data)
                try:
                    from sklearn.cluster import KMeans, DBSCAN
                    X = np.column_stack([(df['y'] - mean_val) / (std_val + 1e-6), np.arange(len(df)) / len(df)])
                    
                    # Limit clustering to first 5000 points for performance
                    sample_size = min(len(df), 5000)
                    sample_indices = np.random.choice(len(df), sample_size, replace=False)
                    X_sample = X[sample_indices]
                    
                    km = KMeans(n_clusters=min(4, len(df) // 2), random_state=42, n_init=5)
                    labels_sample = km.fit_predict(X_sample)
                    labels = km.predict(X)
                    
                    _, counts = np.unique(labels, return_counts=True)
                    small = np.unique(labels)[counts < len(df) * 0.05]
                    kmeans_anom = np.isin(labels, small)
                    anomaly_scores[kmeans_anom] += 0.3
                    anomaly_types[kmeans_anom] = 'Collective'
                    method_votes[kmeans_anom] += 1
                    kmeans_idx = np.where(kmeans_anom)[0]
                    for i in kmeans_idx:
                        reason_flags[i].append("Unusual pattern compared with your typical days.")
                    
                    db = DBSCAN(eps=0.5, min_samples=3)
                    dbscan_anom = db.fit_predict(X) == -1
                    anomaly_scores[dbscan_anom] += 0.4
                    anomaly_types[dbscan_anom] = 'Collective'
                    method_votes[dbscan_anom] += 1
                    dbscan_idx = np.where(dbscan_anom)[0]
                    for i in dbscan_idx:
                        reason_flags[i].append("Isolated behaviour that does not fit nearby days.")
                except:
                    pass

                # Normalize scores into [0, 1] so that the
                # sensitivity thresholds (low/medium/high) are
                # applied consistently across metrics.
                if anomaly_scores.max() > 0:
                    anomaly_scores = anomaly_scores / anomaly_scores.max()

                df['score'] = anomaly_scores
                df['type'] = anomaly_types
                df['votes'] = method_votes

                # Use both the selected sensitivity mode and the
                # ensemble score to control how many points are
                # treated as anomalies.
                sensitivity_mode = config.get('sensitivity_mode', 'Balanced (recommended)')
                # Force Steps to always treat Prophet band breaches
                # as anomalies (no longer user-configurable).
                steps_all_prophet = True

                if sensitivity_mode == 'catch more anomalies':
                    # Any point flagged by at least one method is an anomaly
                    base_is_anom = method_votes >= 1
                elif sensitivity_mode == 'only strong anomalies':
                    # Require agreement from multiple methods and a higher score
                    base_is_anom = (method_votes >= 2) & (anomaly_scores >= config['medium_threshold'])
                else:  # Balanced
                    # At least one method plus a moderate score
                    base_is_anom = (method_votes >= 1) & (anomaly_scores >= config['low_threshold'])

                # Optionally: for Steps, always include any Prophet
                # band breach as an anomaly so counts match the
                # with/without-holidays views more closely.
                if steps_all_prophet and "step" in metric_name.lower():
                    df['is_anom'] = base_is_anom | prophet_band_hits
                else:
                    df['is_anom'] = base_is_anom

                # Assign severity ONLY to flagged anomalies; others stay "Normal".
                severity = np.full(len(df), 'Normal', dtype=object)
                # Low severity: between low and medium thresholds
                low_mask = df['is_anom'] & (anomaly_scores < config['medium_threshold'])
                severity[low_mask] = 'Low'
                # Medium severity: between medium and high thresholds
                med_mask = df['is_anom'] & (anomaly_scores >= config['medium_threshold']) & (anomaly_scores < config['high_threshold'])
                severity[med_mask] = 'Medium'
                # High severity: above high threshold
                high_mask = df['is_anom'] & (anomaly_scores >= config['high_threshold'])
                severity[high_mask] = 'High'

                df['severity'] = severity

                # Build a human-readable explanation per point for hover tooltips
                explanations = []
                for i in range(len(df)):
                    if not df['is_anom'].iloc[i]:
                        explanations.append("Not flagged as an anomaly.")
                    else:
                        parts = reason_flags[i] if reason_flags[i] else ["Flagged by the anomaly detection ensemble."]
                        # Remove duplicates while preserving order
                        seen = set()
                        clean_parts = []
                        for p in parts:
                            if p not in seen:
                                clean_parts.append(p)
                                seen.add(p)
                        reason_text = "; ".join(clean_parts)
                        sev = df['severity'].iloc[i]
                        explanations.append(f"{reason_text} (Severity: {sev}).")

                df['anomaly_explanation'] = explanations
                
                # DISPLAY (OPTIMIZED - fewer but informative charts)
                st.markdown(f"---\n## {metric_name}")

                # Count only flagged anomalies per severity, and make
                # Total Anomalies the sum of Low + Medium + High.
                n_high = ((df['severity'] == 'High') & df['is_anom']).sum()
                n_med = ((df['severity'] == 'Medium') & df['is_anom']).sum()
                n_low = ((df['severity'] == 'Low') & df['is_anom']).sum()
                n_anom = n_high + n_med + n_low

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Anomalies", n_anom)
                col2.metric("🔴 High", n_high)
                col3.metric("🟠 Medium", n_med)
                col4.metric("🟡 Low", n_low)

                # Compact summary flashcard for this metric type
                if n_anom > 0:
                    if "heart" in metric_name.lower():
                        reason_text = (
                            "Heart rate anomalies can come from intense exercise, stress, "
                            "caffeine, or possible cardiovascular issues when they repeat."
                        )
                    elif "step" in metric_name.lower():
                        reason_text = (
                            "Step anomalies usually reflect very active days, sick/rest days, "
                            "travel, or major routine changes."
                        )
                    elif "sleep" in metric_name.lower():
                        reason_text = (
                            "Sleep anomalies are often caused by stress, late work, travel, "
                            "illness, or changes in sleep schedule."
                        )
                    else:
                        reason_text = (
                            "Anomalies indicate values that behave very differently from your "
                            "usual pattern and should be checked against your recent activities."
                        )

                    # Build richer summary content depending on metric type
                    extra_lines = []

                    # Steps: highlight highest / lowest step days in recent week window
                    if "step" in metric_name.lower():
                        try:
                            temp = df.copy()
                            temp["date"] = temp["ds"].dt.normalize()
                            daily_steps = temp.groupby("date")["y"].sum().sort_index()
                            if len(daily_steps) > 0:
                                window = daily_steps.tail(7) if len(daily_steps) >= 7 else daily_steps
                                max_date = window.idxmax()
                                min_date = window.idxmin()
                                max_val = window.max()
                                min_val = window.min()
                                extra_lines.append(
                                    f"Steps weekly pattern (last {len(window)} days):"
                                )
                                extra_lines.append(
                                    f"- Highest steps on {max_date.strftime('%A')} ({max_val:,.0f} steps)."
                                )
                                extra_lines.append(
                                    f"- Lowest steps on {min_date.strftime('%A')} ({min_val:,.0f} steps)."
                                )
                        except Exception:
                            pass

                    # Sleep: which weekday has most / least average sleep and trend
                    if "sleep" in metric_name.lower():
                        try:
                            temp = df.copy()
                            temp["date"] = temp["ds"].dt.normalize()
                            daily_sleep = temp.groupby("date")["y"].sum().sort_index()
                            if len(daily_sleep) > 0:
                                # Average duration by weekday name
                                sleep_df = daily_sleep.to_frame("duration")
                                sleep_df["weekday"] = sleep_df.index.day_name()
                                by_weekday = sleep_df.groupby("weekday")["duration"].mean()
                                if not by_weekday.empty:
                                    max_wd = by_weekday.idxmax()
                                    min_wd = by_weekday.idxmin()
                                    max_val = by_weekday.max()
                                    min_val = by_weekday.min()

                                    # Heuristic: if values are large (> 24), treat as minutes and
                                    # convert to hours; otherwise assume they are already hours.
                                    if max_val > 24:
                                        max_hours = max_val / 60.0
                                        min_hours = min_val / 60.0
                                    else:
                                        max_hours = max_val
                                        min_hours = min_val

                                    extra_lines.append("Sleep weekday pattern (average):")

                                    # If weekdays differ only slightly, avoid confusing
                                    # "most" vs "least" lines with the same rounded value.
                                    if abs(max_hours - min_hours) < 0.25:  # < 15 minutes difference
                                        avg_hours = (max_hours + min_hours) / 2.0
                                        extra_lines.append(
                                            f"- Sleep duration is similar across weekdays (~{avg_hours:.1f} hours per night)."
                                        )
                                    else:
                                        extra_lines.append(
                                            f"- Most sleep on {max_wd} (~{max_hours:.1f} hours)."
                                        )
                                        extra_lines.append(
                                            f"- Least sleep on {min_wd} (~{min_hours:.1f} hours)."
                                        )

                                # Simple trend: compare last 7 vs previous 7 days
                                if len(daily_sleep) >= 14:
                                    last7 = daily_sleep.tail(7).mean()
                                    prev7 = daily_sleep.tail(14).head(7).mean()
                                    diff = last7 - prev7
                                    if abs(diff) < 0.1:
                                        trend_text = "Sleep duration is relatively stable over recent weeks."
                                    elif diff > 0:
                                        trend_text = "Sleep duration is increasing in the most recent week."
                                    else:
                                        trend_text = "Sleep duration is decreasing in the most recent week."
                                    extra_lines.append(trend_text)
                        except Exception:
                            pass

                    # Use Streamlit's default blue info box style
                    # Show severity breakdown, then reasons, then any extra insights
                    summary_lines = [
                        f"Summary: {n_anom} anomalies detected:",
                        f"- High: {n_high}",
                        f"- Medium: {n_med}",
                        f"- Low: {n_low}",
                        "",
                        f"Possible reasons: {reason_text}",
                    ]

                    text = "\n".join(summary_lines + extra_lines)
                    st.info(text)
                
                # TWO GRAPHS ONLY (FAST)
                col_left, col_right = st.columns(2)
                
                # Scatter Plot
                with col_left:
                    st.markdown("#### 📊 Anomalies Timeline")
                    fig = go.Figure()
                    norm = df[~df['is_anom']]
                    if len(norm) > 0:
                        fig.add_trace(go.Scatter(x=norm['ds'], y=norm['y'], mode='markers', name='Normal',
                            marker=dict(size=4, color='blue', opacity=0.25)))
                    for sev, col in [('High','red'), ('Medium','orange'), ('Low','yellow')]:
                        anom = df[(df['is_anom']) & (df['severity']==sev)]
                        if len(anom) > 0:
                            fig.add_trace(go.Scatter(
                                x=anom['ds'],
                                y=anom['y'],
                                mode='markers',
                                name=sev,
                                marker=dict(size=6, color=col, symbol='diamond'),
                                text=anom['anomaly_explanation'],
                                hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.2f}<br>%{text}<extra></extra>",
                            ))
                    q_lower = df['y'].quantile(0.1)
                    q_upper = df['y'].quantile(0.9)
                    fig.add_hline(y=q_lower, line_dash="dash", line_color="green", opacity=0.4)
                    fig.add_hline(y=q_upper, line_dash="dash", line_color="red", opacity=0.4)
                    fig.update_layout(
                        title="Anomalies Timeline",
                        xaxis_title="Date",
                        yaxis_title=value_label,
                        height=280,
                        template='plotly_white',
                        margin=dict(l=30, r=30, t=20, b=20),
                        hovermode='closest',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Density
                with col_right:
                    st.markdown("#### 🔥 Anomaly Frequency")
                    df['date'] = df['ds'].dt.normalize()
                    density = df.groupby('date')['is_anom'].sum()
                    fig = go.Figure(data=[go.Bar(x=density.index, y=density.values,
                        marker=dict(color=density.values, colorscale='RdYlGn_r'))])
                    fig.update_layout(height=280, template='plotly_white', margin=dict(l=30, r=30, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Forecast (if ≥20 points)
                if len(df) >= 20:
                    st.markdown("#### 📈 Forecast with Anomalies")
                    try:
                        forecast = st.session_state['prophet_cache'][metric_name]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='blue', width=1.5)))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='green', width=1.5, dash='dash')))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(0,200,0,0.15)', fill='tonexty', name='CI'))
                        anom = df[df['is_anom']]
                        if len(anom) > 0:
                            fig.add_trace(go.Scatter(
                                x=anom['ds'],
                                y=anom['y'],
                                mode='markers',
                                name='Flagged',
                                marker=dict(size=5, color='red', symbol='x'),
                                text=anom['anomaly_explanation'],
                                hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.2f}<br>%{text}<extra></extra>",
                            ))
                        fig.update_layout(
                            title="Forecast with Anomaly Bands",
                            xaxis_title="Date",
                            yaxis_title=value_label,
                            height=280,
                            template='plotly_white',
                            margin=dict(l=30, r=30, t=20, b=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

                # For steps data, always show precomputed forecasts with and without holidays
                if "step" in metric_name.lower():
                    st.markdown("#### 📈 Steps Forecast: With vs Without Holidays")
                    render_steps_forecast_section(
                        steps_df=df[['ds', 'y']],
                        anom_mask=df['is_anom'],
                        anom_text=df.get('anomaly_explanation'),
                        show_anomalies=True,
                        show_events=True,
                    )
                
                # Results Table
                if n_anom > 0:
                    st.markdown(f"**📋 Results ({n_anom} anomalies)**")
                    anom_df = df[df['is_anom']][['ds', 'y', 'score', 'type', 'severity']].copy()
                    anom_df.columns = ['Date', 'Value', 'Score', 'Type', 'Severity']
                    anom_df['Score'] = (anom_df['Score'] * 100).round(1).astype(str) + '%'
                    anom_df = anom_df.sort_values('Date', ascending=False).head(15)
                    st.dataframe(anom_df, use_container_width=True, hide_index=True)
                    st.download_button("📥 Download", anom_df.to_csv(index=False), file_name=f"anom_{metric_name.lower()}.csv", key=f"dl_{metric_name}")
                else:
                    st.info("✅ No anomalies detected")
# ===== COMPREHENSIVE REPORTS TAB =====
with tabs[3]:
    st.subheader("Comprehensive Health Report Generator")

    # Simple human-friendly summary for the Reports tab
    st.caption("Use this page to generate a clear health summary report that explains your recent patterns and anomalies in plain language.")
    
    # File Upload Section
    col1, col2 = st.columns([2, 1])
    with col1:
        # 1) Choose whether to report on existing Analysis/Anomalies data
        #    or on a newly uploaded file.
        report_source_mode = st.radio(
            "Select report source",
            ["Report on analyzed data", "Report on uploaded file"],
            index=0,
            help=(
                "Use the metrics you've already processed in the Analysis/Anomalies tabs, "
                "or upload/select a CSV and let the report run its own analysis in the background."
            ),
        )

        st.markdown("---")

        # 2) Optional upload section (only used when reporting on an uploaded file)
        st.markdown("**Upload file for report (optional)**")
        uploaded_file = st.file_uploader(
            "Choose CSV file (preprocessed or raw)",
            type=["csv"],
            key="report_upload",
            help="Only used when 'Report on uploaded file' is selected above.",
        )

        # 3) Metric selection that applies **only** to the uploaded-file path
        upload_metric_selection = []
        if report_source_mode == "Report on uploaded file":
            upload_metric_selection = st.multiselect(
                "Select metric(s) for uploaded file",
                ["Heart Rate", "Steps", "Sleep"],
                default=["Heart Rate"],
                help="These metric choices apply only when generating a report from an uploaded file.",
            )
    
    with col2:
        st.markdown("**Report Settings**")

        # Common settings: apply to both analyzed-data and uploaded-file reports
        include_analysis = st.checkbox("Include Analysis section", value=True)
        include_anomalies = st.checkbox("Include Anomalies section", value=True)

    # Global action button to generate the report based on the
    # selections above (source + settings).
    st.markdown("---")
    generate_report_clicked = st.button(
        "Generate Report", use_container_width=True, key="gen_comprehensive_report"
    )
    
    df = None
    filename = None
    source_mode = None  # 'csv' or 'dashboard'
    dashboard_metric_key = None
    dashboard_metrics_for_report = []
    analysis_data = st.session_state.get("analysis_user_data", {})
    anom_data = st.session_state.get("anom_data", {})

    # Build the base dataframe depending on the chosen report source
    if report_source_mode == "Report on uploaded file":
        source_mode = "csv"

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                filename = uploaded_file.name
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                df = None

    else:  # Report on analyzed data from Analysis/Anomalies tabs
        source_mode = "dashboard"
        if not analysis_data:
            st.error(
                "No processed data found from the Analysis tab. "
                "Please upload and run Analysis first, or switch to 'Report on uploaded file'."
            )
        else:
            # Metrics selected in Analysis tab (for Analysis Overview section)
            analysis_selected = st.session_state.get("analysis_selected_metrics", [])
            if analysis_selected:
                analysis_metrics_for_report = [m for m in analysis_selected if m in analysis_data]
            else:
                analysis_metrics_for_report = list(analysis_data.keys())

            # Metrics selected in Anomalies tab (for Anomalies section)
            if anom_data:
                anom_selected = st.session_state.get("anom_selected_metrics", list(anom_data.keys()))
                anomaly_metrics_for_report = [m for m in anom_selected if m in analysis_data]
            else:
                anomaly_metrics_for_report = []

            # Union of all metrics we might talk about
            dashboard_metrics_for_report = list({*analysis_metrics_for_report, *anomaly_metrics_for_report})

            if not dashboard_metrics_for_report:
                st.error(
                    "No metrics found from previous tabs. "
                    "Please run Analysis (and optionally Anomalies) first."
                )
            else:
                # Deterministic choice for column detection / filename
                dashboard_metric_key = dashboard_metrics_for_report[0]

            if dashboard_metric_key and dashboard_metric_key in analysis_data:
                df_metric = analysis_data[dashboard_metric_key].copy()
                if "ds" in df_metric.columns and "y" in df_metric.columns:
                    df = df_metric.rename(columns={"ds": "date", "y": "value"})
                    filename = f"{dashboard_metric_key.replace(' ', '')}_from_dashboard"
                else:
                    st.error(
                        f"{dashboard_metric_key}: expected columns 'ds' and 'y' in Analysis data. "
                        "Please re-run Analysis for this metric."
                    )
            elif dashboard_metric_key:
                st.error(
                    f"No data available for {dashboard_metric_key} from Analysis tab. "
                    "Please enable this metric in Analysis or upload a CSV."
                )

    if df is not None:
        try:
            if source_mode == "csv":
                # ----- Multi-metric path for uploaded CSV -----
                if not upload_metric_selection:
                    st.error("Please select at least one metric for the uploaded file.")
                else:
                    series_dict = extract_metric_time_series(df, filename=filename)
                    if not series_dict:
                        st.error(
                            "Could not detect Heart Rate, Steps, or Sleep columns in the uploaded file. "
                            "Please ensure the CSV contains recognizable metric columns."
                        )
                    elif generate_report_clicked:
                        with st.spinner("Generating comprehensive report from uploaded file(s)..."):
                            report_text = ""

                            # ANALYSIS OVERVIEW
                            if include_analysis:
                                report_text += f"""
FitPlus Health Report - Uploaded Data
{'='*80}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source File: {filename}

{'='*80}

ANALYSIS OVERVIEW
------------------

"""

                                for m_key in upload_metric_selection:
                                    metric_df = series_dict.get(m_key)
                                    if metric_df is None or not isinstance(metric_df, pd.DataFrame):
                                        continue

                                    if "ds" not in metric_df.columns or "y" not in metric_df.columns:
                                        continue

                                    df_m = metric_df.rename(columns={"ds": "date", "y": "value"}).copy()
                                    df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
                                    df_m["value"] = pd.to_numeric(df_m["value"], errors="coerce")
                                    df_m = df_m.dropna(subset=["date", "value"]).sort_values("date")

                                    if df_m.empty:
                                        continue

                                    m_mean = df_m["value"].mean()
                                    m_median = df_m["value"].median()
                                    m_std = df_m["value"].std()
                                    m_min = df_m["value"].min()
                                    m_max = df_m["value"].max()

                                    m_recent_trend = (
                                        df_m["value"].iloc[-7:].mean()
                                        - df_m["value"].iloc[-14:-7].mean()
                                        if len(df_m) >= 14
                                        else 0
                                    )

                                    m_unit = metric_unit(m_key)

                                    report_text += format_metric_analysis_block(
                                        m_key,
                                        m_mean,
                                        m_median,
                                        m_std,
                                        m_min,
                                        m_max,
                                        m_recent_trend,
                                        m_unit,
                                    )

                            # ANOMALIES SECTION for uploaded data
                            if include_anomalies:
                                report_text += f"""
{'='*80}

ANOMALIES OVERVIEW (Uploaded Data)
----------------------------------

"""

                                for m_key in upload_metric_selection:
                                    metric_df = series_dict.get(m_key)
                                    if metric_df is None or not isinstance(metric_df, pd.DataFrame):
                                        continue

                                    if "ds" not in metric_df.columns or "y" not in metric_df.columns:
                                        continue

                                    df_m = normalize_metric_df(metric_df)

                                    if df_m.empty:
                                        continue

                                    m_mean = df_m["value"].mean()
                                    m_std = df_m["value"].std()

                                    m_outliers = (
                                        (df_m["value"] < m_mean - 2.5 * m_std)
                                        | (df_m["value"] > m_mean + 2.5 * m_std)
                                    )
                                    m_anom_idx = np.where(m_outliers)[0]
                                    m_anom_count = len(m_anom_idx)

                                    m_unit = metric_unit(m_key)

                                    report_text += f"""
Metric: {m_key}
~~~~~~~~~~~~~~~

Anomalies Detected: {m_anom_count}
Percentage of Data: {(m_anom_count/len(df_m)*100):.2f}%

"""
                                    if m_anom_count > 0:
                                        report_text += """
Top Recent Anomalies:
"""
                                        for i, idx in enumerate(reversed(m_anom_idx[-5:]), 1):
                                            anom_date = df_m["date"].iloc[idx]
                                            anom_val = df_m["value"].iloc[idx]
                                            deviation = (
                                                (anom_val - m_mean) / m_std if m_std else 0
                                            )

                                            report_text += f"""
{i}. Date: {anom_date.strftime('%Y-%m-%d')}
   Value: {anom_val:.2f} {m_unit}
   Deviation: {deviation:.2f} std_dev from mean
"""
                                    else:
                                        report_text += """
No significant anomalies detected. Patterns appear stable.
"""

                            # Recommendations & disclaimer (shared)
                            report_text = append_recommendations_and_disclaimer(
                                report_text,
                                "Based on your uploaded metrics analysis and anomalies:",
                            )

                            # For uploaded multi-metric mode, use a generic filename label
                            metric_type = "UploadedMetrics"

                            # Build and show the PDF
                            build_and_show_pdf_report(report_text, metric_type)

            else:
                # ----- Dashboard-based multi-metric report (Analysis/Anomalies tabs) -----
                if not dashboard_metrics_for_report:
                    st.error(
                        "No metrics available from Analysis/Anomalies to include in the report."
                    )
                elif generate_report_clicked:
                    with st.spinner("Generating comprehensive report from analyzed data..."):
                        report_text = ""

                        # ANALYSIS OVERVIEW: metrics selected in Analysis tab
                        analysis_selected = st.session_state.get("analysis_selected_metrics", [])
                        if analysis_selected:
                            analysis_metrics_for_report = [
                                m for m in analysis_selected if m in analysis_data
                            ]
                        else:
                            analysis_metrics_for_report = list(analysis_data.keys())

                        # ANOMALIES: metrics selected in Anomalies tab
                        if anom_data:
                            anom_selected = st.session_state.get(
                                "anom_selected_metrics", list(anom_data.keys())
                            )
                            anomaly_metrics_for_report = [
                                m for m in anom_selected if m in analysis_data
                            ]
                        else:
                            anomaly_metrics_for_report = []

                        if include_analysis and analysis_metrics_for_report:
                            report_text += f"""
FitPlus Health Report - Analyzed Data
{'='*80}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}

ANALYSIS OVERVIEW (From Analysis Tab)
-------------------------------------

"""

                            for m_key in analysis_metrics_for_report:
                                metric_df = analysis_data.get(m_key)
                                if metric_df is None or not isinstance(metric_df, pd.DataFrame):
                                    continue

                                if "ds" not in metric_df.columns or "y" not in metric_df.columns:
                                    continue

                                df_m = normalize_metric_df(metric_df)

                                if df_m.empty:
                                    continue

                                m_mean = df_m["value"].mean()
                                m_median = df_m["value"].median()
                                m_std = df_m["value"].std()
                                m_min = df_m["value"].min()
                                m_max = df_m["value"].max()

                                m_recent_trend = (
                                    df_m["value"].iloc[-7:].mean()
                                    - df_m["value"].iloc[-14:-7].mean()
                                    if len(df_m) >= 14
                                    else 0
                                )

                                m_unit = metric_unit(m_key)

                                report_text += format_metric_analysis_block(
                                    m_key,
                                    m_mean,
                                    m_median,
                                    m_std,
                                    m_min,
                                    m_max,
                                    m_recent_trend,
                                    m_unit,
                                )

                        # ANOMALIES SECTION (from Anomalies tab selections)
                        if include_anomalies and anomaly_metrics_for_report:
                            report_text += f"""
{'='*80}

ANOMALIES OVERVIEW (From Anomalies Tab)
---------------------------------------

"""

                            for m_key in anomaly_metrics_for_report:
                                metric_df = analysis_data.get(m_key)
                                if metric_df is None or not isinstance(metric_df, pd.DataFrame):
                                    continue

                                if "ds" not in metric_df.columns or "y" not in metric_df.columns:
                                    continue

                                df_m = normalize_metric_df(metric_df)

                                if df_m.empty:
                                    continue

                                m_mean = df_m["value"].mean()
                                m_std = df_m["value"].std()

                                m_outliers = (
                                    (df_m["value"] < m_mean - 2.5 * m_std)
                                    | (df_m["value"] > m_mean + 2.5 * m_std)
                                )
                                m_anom_idx = np.where(m_outliers)[0]
                                m_anom_count = len(m_anom_idx)

                                m_unit = metric_unit(m_key)

                                report_text += f"""
Metric: {m_key}
~~~~~~~~~~~~~~~

Anomalies Detected: {m_anom_count}
Percentage of Data: {(m_anom_count/len(df_m)*100):.2f}%

"""
                                if m_anom_count > 0:
                                    report_text += """
Top Recent Anomalies:
"""
                                    for i, idx in enumerate(reversed(m_anom_idx[-5:]), 1):
                                        anom_date = df_m["date"].iloc[idx]
                                        anom_val = df_m["value"].iloc[idx]
                                        deviation = (
                                            (anom_val - m_mean) / m_std if m_std else 0
                                        )

                                        report_text += f"""
{i}. Date: {anom_date.strftime('%Y-%m-%d')}
   Value: {anom_val:.2f} {m_unit}
   Deviation: {deviation:.2f} std_dev from mean
"""
                                else:
                                    report_text += """
No significant anomalies detected. Patterns appear stable.
"""

                        # Shared recommendations & disclaimer for dashboard mode
                        report_text = append_recommendations_and_disclaimer(
                            report_text,
                            "Based on your analyzed metrics and anomalies:",
                        )

                        # For dashboard multi-metric mode, use a generic label
                        metric_type = "AnalyzedMetrics"

                        # Build and show the PDF
                        build_and_show_pdf_report(report_text, metric_type)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
                