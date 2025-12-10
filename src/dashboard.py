# -----------------------------
# FitPlus Health Insights Dashboard
# Streamlit Dashboard (Module 4)
# -----------------------------

import os
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# -----------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------
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

# Path to Module 2 outputs
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
OUT_DIR = os.path.join(BASE_DIR, "module2_outputs")

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["Home", "Analysis", "Anomalies", "Forecast & Reports"])

# -----------------------------
# HOME TAB
# -----------------------------
with tabs[0]:
    st.subheader("Welcome")
    st.write(
        """
        This dashboard allows you to explore your Fitbit health data:

        - 📉 **Time-Series Visualization** (Heart Rate, Steps, Sleep)  
        - 🔍 **Anomaly Detection** (Isolation Forest, Z-score)  
        - 📈 **Forecasting using Facebook Prophet**  
        - 🧾 **Generate PDF Reports**

        Use the sidebar and tabs to explore each module.
        """
    )

# -----------------------------
# ANALYSIS TAB (Module 2 visualizations)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # go up → project folder
MODULE2_DIR = os.path.join(PROJECT_ROOT, "module2_outputs")

files = {
    "Heart Rate": os.path.join(MODULE2_DIR, "daily_heart_rate.csv"),
    "Steps": os.path.join(MODULE2_DIR, "daily_steps.csv"),
    "Sleep": os.path.join(MODULE2_DIR, "daily_sleep.csv")
}

for label, path in files.items():
    st.markdown(f"### {label} Trend")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.line_chart(df['y'])  # simple trend display
    else:
        st.warning(f"{label} data not found at: {path}")

# -----------------------------
# ANOMALY DETECTION TAB
# -----------------------------
with tabs[2]:

    st.subheader("🔍 Anomaly Detection")

    uploaded = st.file_uploader("Upload CSV (timestamp + value)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.lower()

        st.write("Preview:")
        st.dataframe(df.head())

        ts_col = st.selectbox("Timestamp column", df.columns)
        val_col = st.selectbox("Value column", df.columns)

        df["timestamp"] = pd.to_datetime(df[ts_col])
        df["value"] = pd.to_numeric(df[val_col], errors="coerce")

        method = st.radio("Method", ["IsolationForest", "Z-score"])

        if st.button("Run Detection"):
            data = df.copy()

            if method == "IsolationForest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                iso.fit(data[["value"]])
                data["anomaly"] = iso.predict(data[["value"]]) == -1

            else:
                z = (data["value"] - data["value"].mean()) / data["value"].std()
                data["anomaly"] = abs(z) > 3

            st.success(f"Found {data['anomaly'].sum()} anomalies")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["timestamp"], y=data["value"], mode="lines", name="Value"))
            fig.add_trace(go.Scatter(
                x=data.loc[data["anomaly"], "timestamp"],
                y=data.loc[data["anomaly"], "value"],
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=8)
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(data.loc[data["anomaly"], ["timestamp", "value"]])

# -----------------------------
# FORECAST & REPORTS TAB
# -----------------------------
with tabs[3]:
    st.subheader("📈 Forecasting")

    if not PROPHET_AVAILABLE:
        st.warning("Prophet is not installed. Run: pip install prophet")
    else:
        file = st.file_uploader("Upload CSV for forecasting", type=["csv"], key="forecast")

        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.lower()

            ts_col = st.selectbox("Timestamp column", df.columns, key="ts2")
            val_col = st.selectbox("Value column", df.columns, key="val2")

            df["ds"] = pd.to_datetime(df[ts_col])
            df["y"] = pd.to_numeric(df[val_col], errors="coerce")

            periods = st.slider("Forecast days", 7, 90, 30)

            if st.button("Run Forecast"):
                model = Prophet()
                model.fit(df[["ds", "y"]])

                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
                fig.update_layout(title="Forecast", height=400)

                st.plotly_chart(fig, use_container_width=True)

                st.success("Forecast complete!")

    st.markdown("---")
    st.info("Report generation coming soon in next milestone.")

