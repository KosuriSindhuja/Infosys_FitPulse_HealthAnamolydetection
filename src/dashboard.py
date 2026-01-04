import os
import io
import base64
from datetime import datetime, timedelta, date
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
from prophet import Prophet

st.set_page_config(
    page_title="FitPlus Health Insights Dashboard",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown(
    """
    <h1 style="margin:0; padding:0;">FitPlus Health Insights Dashboard</h1>
    <p style="color:#9fb4c8;">Visualize your fitness patterns, analyze anomalies, and forecast health trends.</p>
    """,
    unsafe_allow_html=True
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

files = {
    "Heart Rate": os.path.join(MODULE2_DIR, "daily_heart_rate.csv"),
    "Steps": os.path.join(MODULE2_DIR, "daily_steps.csv"),
    "Sleep": os.path.join(MODULE2_DIR, "daily_sleep.csv")
}


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


def render_steps_forecast_section(start_date=None, end_date=None, show_anomalies=True, show_events=True):
    """Render two graphs for steps: forecasts with and without holidays.

    Uses precomputed module2_outputs forecasts and the daily_steps actuals.
    If ``show_anomalies`` is False, the graphs will only show actuals,
    forecasts and confidence bands (no red anomaly markers or event points).
    """
    data = load_steps_forecast_data()

    steps_df = data.get("actual")
    df_no = data.get("without_holidays")
    df_with = data.get("with_holidays")
    df_events = data.get("events")

    if steps_df is None or df_no is None or df_with is None:
        st.info("Steps forecast data (with/without holidays) is not available.")
        return

    # Optional date filtering
    if start_date is not None and end_date is not None:
        mask_actual = (steps_df["ds"].dt.date >= start_date) & (steps_df["ds"].dt.date <= end_date)
        mask_no = (df_no["ds"].dt.date >= start_date) & (df_no["ds"].dt.date <= end_date)
        mask_with = (df_with["ds"].dt.date >= start_date) & (df_with["ds"].dt.date <= end_date)
        steps_df = steps_df.loc[mask_actual]
        df_no = df_no.loc[mask_no]
        df_with = df_with.loc[mask_with]

        if steps_df.empty or df_no.empty or df_with.empty:
            st.warning("No steps forecast data in the selected date range.")
            return

    # Merge actuals with forecasts (inner join on ds to keep aligned days)
    merged_no = pd.merge(
        steps_df[["ds", "y"]],
        df_no[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )
    merged_with = pd.merge(
        steps_df[["ds", "y"]],
        df_with[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )

    if merged_no.empty or merged_with.empty:
        st.warning("Aligned steps forecast data could not be prepared.")
        return

    # Identify anomalies based on forecast confidence intervals
    merged_no["is_anom"] = (merged_no["y"] > merged_no["yhat_upper"]) | (merged_no["y"] < merged_no["yhat_lower"])
    merged_with["is_anom"] = (merged_with["y"] > merged_with["yhat_upper"]) | (merged_with["y"] < merged_with["yhat_lower"])

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

        # Anomaly markers (optional)
        if show_anomalies:
            anom_no = merged_no[merged_no["is_anom"]]
            if not anom_no.empty:
                fig_no.add_trace(
                    go.Scatter(
                        x=anom_no["ds"],
                        y=anom_no["y"],
                        mode="markers",
                        name="Anomalies (No Holidays)",
                        marker=dict(size=8, color="red", symbol="x"),
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

        # Anomaly markers based on with-holiday forecast (optional)
        if show_anomalies:
            anom_with = merged_with[merged_with["is_anom"]]
            if not anom_with.empty:
                fig_with.add_trace(
                    go.Scatter(
                        x=anom_with["ds"],
                        y=anom_with["y"],
                        mode="markers",
                        name="Anomalies (With Holidays)",
                        marker=dict(size=8, color="red", symbol="x"),
                    )
                )

        # Highlight specific holiday / event days if available (optional)
        if show_events and df_events is not None and not df_events.empty:
            events_merged = pd.merge(df_events[["ds", "event"]], merged_with[["ds", "yhat"]], on="ds", how="inner")
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

    # Derive a compact health status from processed data
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
                        <div style="margin:0.25rem 0 0.75rem 0;">
                            <span style="font-weight:600; color:#2c3e50; margin-right:0.35rem;">Health Status:</span>
                            <span style="display:inline-flex; align-items:center; gap:0.3rem; padding:0.15rem 0.6rem; border-radius:999px; background:{badge_color}; color:white; font-weight:600; font-size:0.9rem;">
                                <span>{mood_emoji}</span>
                                <span>{health_status_label}</span>
                            </span>
                        </div>
            """,
            unsafe_allow_html=True,
        )

    # Load today's stats
    today = pd.Timestamp.now().normalize()

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
    col1.metric("Today's Avg Heart Rate", f"{hr:.0f} bpm" if hr else "-", delta=None)
    col2.metric("Today's Steps", f"{steps:.0f}" if steps else "-", delta=None)
    col3.metric("Last Night's Sleep", f"{sleep:.1f} hrs" if sleep else "-", delta=None)

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
        if not selected_metrics:
            st.error("Please select at least one metric to analyze.")
        else:
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
            for metric in selected_metrics:
                st.markdown(f"---")
                st.markdown(f"## {metric} Analysis ({date_label})")
                
                path = files[metric]
                if not os.path.exists(path):
                    st.error(f"{metric} data not found.")
                    continue
                
                try:
                    df = pd.read_csv(path)
                    if 'ds' not in df.columns or 'y' not in df.columns:
                        st.error(f"Invalid format for {metric} (missing 'ds' or 'y').")
                        continue
                    
                    df['ds'] = pd.to_datetime(df['ds'])
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
                    
                    # Weekly Seasonality
                    if len(df_range) >= 14:
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
                        # In Analysis we show clean forecasts (no anomaly markers)
                        render_steps_forecast_section(
                            start_date=start_date,
                            end_date=end_date,
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
    
    # ===== ADVANCED COLUMN DETECTION FUNCTION =====
    def smart_detect_columns(df):
        """Smart column detection for both raw and processed files."""
        df.columns = df.columns.str.lower().str.strip()
        
        # Detect DATE column (enhanced for raw files including sleep)
        date_col = None
        date_patterns = ['ds', 'timestamp', 'date', 'datetime', 'activitydate', 'logdate', 'sleepdate', 'time', 'starttime', 'endtime', 'startdate', 'enddate']
        for col in df.columns:
            col_lower = col.lower()
            for pattern in date_patterns:
                if col_lower == pattern or col_lower.endswith(pattern):
                    date_col = col
                    break
            if date_col:
                break
        
        # Detect VALUE column (enhanced for raw files including sleep)
        value_col = None
        # Order matters - check specific patterns first
        value_patterns = ['y', 'value', 'steps', 'step_count', 'heart_rate', 'heartrate', 'duration', 'duration_minutes', 'metric', 'val', 'sleep_hours', 'hr', 'bpm', 'count', 'minutesasleep', 'minutes_asleep']
        
        for pattern in value_patterns:
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == pattern:  # Exact match first
                    value_col = col
                    break
            if value_col:
                break
        
        # If no exact match, try partial matching
        if not value_col:
            for col in df.columns:
                col_lower = col.lower()
                for pattern in value_patterns:
                    if pattern in col_lower and col_lower != 'timestamp' and col_lower != 'date' and 'date' not in col_lower and 'time' not in col_lower:
                        value_col = col
                        break
                if value_col:
                    break
        
        return date_col, value_col
    
    # ===== FILE UPLOAD & CONFIG (MINIMAL) =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📁 Upload CSV Files (Raw or Processed)**")
        uploaded_files = st.file_uploader("Choose CSV files (up to 5)", type=["csv"], accept_multiple_files=True, key="anom_upload")
    
    with col2:
        st.markdown("**⚙️ Severity Levels**")
        low_threshold = st.slider(
            "Low",
            0.0,
            1.0,
            0.3,
            step=0.05,
            key="low_sev_global",
            help="Use this to control how many points are flagged as unusual. Slide left to see more, right to see only bigger changes.",
        )
        medium_threshold = st.slider(
            "Medium",
            low_threshold,
            1.0,
            0.7,
            step=0.05,
            key="med_sev_global",
            help="Use this to separate minor anomalies from ones you usually want to review.",
        )
        high_threshold = st.slider(
            "High",
            medium_threshold,
            1.0,
            0.85,
            step=0.05,
            key="high_sev_global",
            help="Use this to decide which anomalies you treat as urgent and act on first.",
        )
    
    # Internal settings (not visible to user)
    max_rows = 10000
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("⚠️ Maximum 5 files allowed")
            uploaded_files = uploaded_files[:5]
        
        # SINGLE RUN BUTTON - NO RERUNS
        if st.button("🚀 Analyze", use_container_width=True, key="analyze_btn"):
            with st.spinner("Processing anomalies..."):
                try:
                    processed_data = {}
                    
                    for idx, file in enumerate(uploaded_files):
                        progress_text = f"Processing {file.name}..."
                        st.info(progress_text)
                        
                        # Read CSV with optimization
                        df = pd.read_csv(file, low_memory=False)
                        file_size = len(df)
                        
                        # Sample if too large (for performance)
                        if file_size > max_rows:
                            st.warning(f"⚠️ {file.name}: {file_size:,} rows. Sampling {max_rows:,} for analysis")
                            df = df.sample(n=max_rows, random_state=42).sort_values(df.columns[0])
                        
                        # Smart column detection
                        date_col, value_col = smart_detect_columns(df)
                        
                        if date_col is None:
                            st.error(f"❌ {file.name}: Could not find date column. Columns: {', '.join(df.columns[:5])}")
                            continue
                        
                        if value_col is None:
                            st.error(f"❌ {file.name}: Could not find value column. Columns: {', '.join(df.columns[:5])}")
                            continue
                        
                        # Clean data with optimization
                        df_clean = pd.DataFrame()
                        
                        # Parse dates efficiently
                        try:
                            df_clean['ds'] = pd.to_datetime(df[date_col], errors='coerce')
                        except:
                            st.error(f"❌ {file.name}: Could not parse date column '{date_col}'")
                            continue
                        
                        # Parse values
                        try:
                            df_clean['y'] = pd.to_numeric(df[value_col], errors='coerce')
                        except:
                            st.error(f"❌ {file.name}: Could not parse value column '{value_col}'")
                            continue
                        
                        # Remove completely invalid rows
                        df_clean = df_clean.dropna(subset=['ds', 'y'])
                        
                        if len(df_clean) == 0:
                            st.error(f"❌ {file.name}: No valid data after cleaning")
                            continue
                        
                        # Fill remaining NaN values efficiently
                        df_clean['y'] = df_clean['y'].fillna(df_clean['y'].median())
                        
                        # Sort and reset
                        df_clean = df_clean.sort_values('ds').reset_index(drop=True)
                        
                        if len(df_clean) < 5:
                            st.error(f"❌ {file.name}: Need at least 5 valid data points (found {len(df_clean)})")
                            continue
                        
                        # Determine metric type from filename
                        fname = file.name.lower()
                        if 'heart' in fname or 'hr' in fname or 'heartrate' in fname:
                            metric_type = 'Heart Rate'
                        elif 'step' in fname:
                            metric_type = 'Steps'
                        elif 'sleep' in fname:
                            metric_type = 'Sleep'
                        else:
                            metric_type = f'Metric_{idx}'
                        
                        processed_data[metric_type] = df_clean
                        st.success(f"✅ {file.name} → {metric_type} ({len(df_clean)} records)")
                    
                    if processed_data:
                        st.session_state['anom_data'] = processed_data
                        st.session_state['anom_config'] = {
                            'low_threshold': low_threshold,
                            'medium_threshold': medium_threshold,
                            'high_threshold': high_threshold
                        }
                        st.success("✅ Ready to display results")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)[:150]}")
        
        # ===== DISPLAY CACHED RESULTS (OPTIMIZED) =====
        if 'anom_data' in st.session_state:
            config = st.session_state['anom_config']
            
            for metric_name, df in st.session_state['anom_data'].items():
                df = df.copy()
                
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
                
                # Point anomalies (fast, vectorized)
                outliers = (df['y'] < mean_val - 2.5 * std_val) | (df['y'] > mean_val + 2.5 * std_val)
                anomaly_scores[outliers] += 0.4
                anomaly_types[outliers] = 'Point'
                
                # Contextual anomalies (fast, vectorized)
                changes = np.abs(df['y'].diff()).fillna(0)
                spikes = changes > std_val * 2
                anomaly_scores[spikes] += 0.3
                anomaly_types[spikes] = 'Contextual'
                
                # Prophet (once, cached via session) - OPTIMIZED for large data
                if len(df) >= 20 and 'prophet_cache' not in st.session_state:
                    st.session_state['prophet_cache'] = {}
                
                if len(df) >= 20 and metric_name not in st.session_state.get('prophet_cache', {}):
                    try:
                        df_p = df[['ds', 'y']].copy()
                        df_p.columns = ['ds', 'y']
                        
                        # Suppress Prophet warnings
                        import logging
                        logging.getLogger('prophet').setLevel(logging.WARNING)
                        
                        model = Prophet(interval_width=0.95, yearly_seasonality=len(df)>365, 
                                      weekly_seasonality=True, daily_seasonality=False)
                        
                        # Fit with reduced verbosity
                        with st.spinner(f"Fitting Prophet for {metric_name}..."):
                            model.fit(df_p)
                        
                        forecast = model.predict(df_p[['ds']])
                        st.session_state['prophet_cache'][metric_name] = forecast
                        
                        model_anom = (df['y'].values < forecast['yhat_lower'].values) | (df['y'].values > forecast['yhat_upper'].values)
                        anomaly_scores[model_anom] += 0.4
                    except:
                        pass
                elif len(df) >= 20 and metric_name in st.session_state.get('prophet_cache', {}):
                    forecast = st.session_state['prophet_cache'][metric_name]
                    model_anom = (df['y'].values < forecast['yhat_lower'].values) | (df['y'].values > forecast['yhat_upper'].values)
                    anomaly_scores[model_anom] += 0.4
                
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
                    
                    db = DBSCAN(eps=0.5, min_samples=3)
                    dbscan_anom = db.fit_predict(X) == -1
                    anomaly_scores[dbscan_anom] += 0.4
                    anomaly_types[dbscan_anom] = 'Collective'
                except:
                    pass
                
                # Normalize
                if anomaly_scores.max() > 0:
                    anomaly_scores = anomaly_scores / anomaly_scores.max()
                
                severity = np.where(anomaly_scores < config['low_threshold'], 'Low',
                                   np.where(anomaly_scores < config['medium_threshold'], 'Medium', 'High'))
                
                df['score'] = anomaly_scores
                df['type'] = anomaly_types
                df['severity'] = severity
                df['is_anom'] = anomaly_scores >= config['low_threshold']
                
                # DISPLAY (OPTIMIZED - fewer but informative charts)
                st.markdown(f"---\n## {metric_name}")

                n_anom = df['is_anom'].sum()
                n_high = (df['severity'] == 'High').sum()
                n_med = (df['severity'] == 'Medium').sum()
                n_low = (df['severity'] == 'Low').sum()

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

                                    # Convert minutes to hours when values look like minutes
                                    max_hours = max_val / 60.0
                                    min_hours = min_val / 60.0

                                    extra_lines.append("Sleep weekday pattern (average):")
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
                            fig.add_trace(go.Scatter(x=anom['ds'], y=anom['y'], mode='markers', name=sev,
                                marker=dict(size=6, color=col, symbol='diamond')))
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
                            fig.add_trace(go.Scatter(x=anom['ds'], y=anom['y'], mode='markers', name='Flagged', marker=dict(size=5, color='red', symbol='x')))
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
                    # Use full available range from module outputs (independent of upload window)
                    render_steps_forecast_section(show_anomalies=True, show_events=True)
                
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
    
    # File Upload Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Upload Health Data**")
        uploaded_file = st.file_uploader("Choose CSV file (preprocessed or raw)", type=["csv"], key="report_upload")
    
    with col2:
        st.markdown("**Report Settings**")
        include_analysis = st.checkbox("Include Analysis", value=True)
        include_anomalies = st.checkbox("Include Anomalies", value=True)
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.lower().str.strip()
            
            # Auto-detect columns
            date_col = None
            value_col = None
            
            for col in ['ds', 'timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    date_col = col
                    break
            
            for col in ['y', 'value', 'metric', 'val', 'heart_rate', 'steps', 'sleep_hours', 'hr']:
                if col in df.columns:
                    value_col = col
                    break
            
            if date_col and value_col:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                df['value'] = pd.to_numeric(df[value_col], errors='coerce')
                df = df.dropna(subset=['date', 'value']).sort_values('date')
                
                # Detect metric type
                filename = uploaded_file.name.lower()
                if 'heart' in filename or 'hr' in filename:
                    metric_type = "Heart Rate"
                    unit = "bpm"
                elif 'step' in filename:
                    metric_type = "Steps"
                    unit = "steps"
                elif 'sleep' in filename:
                    metric_type = "Sleep"
                    unit = "hours"
                else:
                    metric_type = "Health Metric"
                    unit = "units"
                
                # Calculate statistics
                mean_val = df['value'].mean()
                median_val = df['value'].median()
                std_val = df['value'].std()
                min_val = df['value'].min()
                max_val = df['value'].max()
                
                # Anomaly detection
                outliers = (df['value'] < mean_val - 2.5 * std_val) | (df['value'] > mean_val + 2.5 * std_val)
                anomaly_indices = np.where(outliers)[0]
                anomaly_count = len(anomaly_indices)
                
                recent_trend = df['value'].iloc[-7:].mean() - df['value'].iloc[-14:-7].mean() if len(df) >= 14 else 0
                
                if st.button("Generate Report", use_container_width=True, key="gen_comprehensive_report"):
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            # Generate comprehensive report text
                            report_text = f"""
FitPlus Health Report - {metric_type}
{'='*80}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Points: {len(df)}
Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}

{'='*80}

ANALYSIS OVERVIEW
-----------------

Your {metric_type.lower()} data shows the following health patterns:

Key Statistics:
  - Average: {mean_val:.2f} {unit}
  - Median: {median_val:.2f} {unit}
  - Standard Deviation: {std_val:.2f}
  - Range: {min_val:.2f} - {max_val:.2f} {unit}
  - 7-Day Trend: {'Increasing' if recent_trend > 0 else 'Decreasing'} ({abs(recent_trend):.2f} {unit})

"""
                            
                            # Metric-specific analysis
                            if "Heart" in metric_type:
                                health_status = "Normal" if 60 <= mean_val <= 100 else "Elevated" if mean_val > 100 else "Low"
                                emergency = mean_val > 120 or mean_val < 40
                                
                                report_text += f"""
HEART RATE ANALYSIS
-------------------

Health Status: {health_status}

Your average resting heart rate of {mean_val:.1f} bpm indicates a {health_status.lower()} 
cardiovascular state. The medical standard for healthy resting heart rate in adults is 
60-100 bpm.

Pattern Analysis:
  - Consistency Score: {100 - (std_val / mean_val * 100):.1f}%
  - Your heart rate shows {'stable' if std_val < mean_val * 0.15 else 'variable'} patterns
  - Recent trend: {'Improving cardiovascular fitness' if recent_trend < 0 else 'Increased stress or activity'}

CARDIOVASCULAR HEALTH REASONING:
- Resting heart rate reflects cardiovascular fitness
- Lower resting heart rate generally indicates better heart efficiency
- Consistent patterns suggest stable health
- Variability may indicate stress, activity changes, or health fluctuations

"""
                                if emergency:
                                    report_text += f"""
[EMERGENCY ALERT]
Your heart rate readings show concerning values ({min_val:.0f}-{max_val:.0f} bpm).
ACTION REQUIRED: Consult a healthcare provider immediately.

"""
                            
                            elif "Step" in metric_type:
                                daily_avg = mean_val
                                activity_level = "Sedentary" if daily_avg < 5000 else "Low Active" if daily_avg < 7500 else "Somewhat Active" if daily_avg < 10000 else "Active" if daily_avg < 12500 else "Very Active"
                                
                                report_text += f"""
DAILY ACTIVITY ANALYSIS
-----------------------

Activity Level: {activity_level}

Your average daily step count of {daily_avg:.0f} steps indicates a {activity_level.lower()} 
lifestyle. The World Health Organization recommends 10,000 steps daily for optimal 
health benefits.

Pattern Analysis:
  - Consistency Score: {100 - (std_val / max(mean_val, 1) * 100):.1f}%
  - Activity patterns: {'Regular and consistent' if std_val < mean_val * 0.25 else 'Highly variable'}
  - Recent trend: {'Increasing physical activity' if recent_trend > 0 else 'Declining activity levels'}

ACTIVITY PATTERN REASONING:
- Daily steps measure physical activity and lifestyle
- Regular movement reduces disease risk and improves fitness
- Consistency indicates sustainable habits
- Variations may reflect work schedule, weather, or exercise routines

"""
                            
                            elif "Sleep" in metric_type:
                                sleep_quality = "Excellent" if 7 <= mean_val <= 9 else "Good" if 6 <= mean_val < 7 else "Poor" if mean_val < 6 else "Too Much"
                                
                                report_text += f"""
SLEEP DURATION ANALYSIS
-----------------------

Sleep Quality Rating: {sleep_quality}

Your average nightly sleep of {mean_val:.1f} hours is rated as {sleep_quality.lower()}. 
The medical standard recommends 7-9 hours of quality sleep per night for adults.

Pattern Analysis:
  - Consistency Score: {100 - (std_val / mean_val * 100):.1f}%
  - Sleep patterns: {'Very stable sleep schedule' if std_val < mean_val * 0.15 else 'Irregular sleep patterns'}
  - Recent trend: {'Improving sleep duration' if recent_trend > 0 else 'Declining sleep hours'}

SLEEP HEALTH REASONING:
- Sleep duration directly affects immune function and cognitive performance
- Consistent sleep schedule supports circadian rhythm
- Irregular sleep may indicate stress, lifestyle changes, or sleep disorders
- Quality and consistency matter as much as duration

"""
                            
                            # Anomalies section
                            if include_anomalies:
                                report_text += f"""
{'='*80}

ANOMALY DETECTION & ANALYSIS
-----------------------------

Anomalies Detected: {anomaly_count}
Percentage of Data: {(anomaly_count/len(df)*100):.2f}%

"""
                                if anomaly_count > 0:
                                    report_text += f"""
Top Recent Anomalies:
"""
                                    for i, idx in enumerate(reversed(anomaly_indices[-5:]), 1):
                                        anom_date = df['date'].iloc[idx]
                                        anom_val = df['value'].iloc[idx]
                                        deviation = (anom_val - mean_val) / std_val
                                        
                                        report_text += f"""
{i}. Date: {anom_date.strftime('%Y-%m-%d')}
   Value: {anom_val:.2f} {unit}
   Deviation: {deviation:.2f} std_dev from mean
"""
                                    
                                    # Why anomalies occurred
                                    report_text += f"""

WHY THESE ANOMALIES OCCURRED:

Anomalies represent unusual values that deviate significantly from normal patterns.
For {metric_type.lower()}, these could be caused by:

For Heart Rate Anomalies:
  - Stress or anxiety during measurement
  - Physical exercise or recovery period
  - Caffeine or medication effects
  - Sleep quality issues
  - Potential health concerns if persistent

For Steps Anomalies:
  - Rest days or illness
  - Intense exercise days
  - Travel or schedule changes
  - Weather conditions
  - Work-related changes in routine

For Sleep Anomalies:
  - Work stress or deadline pressure
  - Lifestyle changes or travel
  - Health issues or illness
  - Environmental changes
  - Schedule disruptions

WHAT CAN BE DONE:
  1) Monitor Trends: Track if anomalies form a pattern or are isolated incidents
  2) Investigate Causes: Correlate anomalies with activities, stress, diet
  3) Preventive Actions: Address root causes (stress management, routine changes)
  4) Seek Help: If anomalies persist, consult healthcare provider
  5) Lifestyle Adjustments: Implement targeted improvements based on findings

"""
                                else:
                                    report_text += f"""
No significant anomalies detected in your data. Your {metric_type.lower()} patterns 
are stable and consistent within normal ranges.
"""
                            
                            # Recommendations section
                            report_text += f"""
{'='*80}

HEALTH RECOMMENDATIONS & ACTION PLAN
-------------------------------------

Based on your {metric_type.lower()} analysis:

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

                            # Create PDF with embedded graphs
                            pdf = FPDF()
                            pdf.set_auto_page_break(auto=True, margin=15)
                            pdf.set_left_margin(20)
                            pdf.set_right_margin(20)
                            pdf.add_page()
                            pdf.set_font("Arial", size=10)
                            text_width = pdf.w - pdf.l_margin - pdf.r_margin

                            # Comprehensive character sanitization function for FPDF compatibility
                            def sanitize_for_pdf(text):
                                """Remove all non-ASCII characters and Unicode symbols, replace with safe ASCII equivalents."""
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
                        
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                
                # Show data preview
                if st.checkbox("Show data preview", value=False):
                    st.dataframe(df[['date', 'value']].head(20), use_container_width=True)
            
            else:
                st.error("Could not auto-detect date and value columns. Please ensure CSV has standard column names like 'ds', 'date', 'y', 'value'.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
                