# -----------------------------
# FitPlus Health Insights Dashboard
# Streamlit Dashboard (Module 4)
# -----------------------------

import os
import io
from datetime import datetime, timedelta, date
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
from prophet import Prophet

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

files = {
    "Heart Rate": os.path.join(MODULE2_DIR, "daily_heart_rate.csv"),
    "Steps": os.path.join(MODULE2_DIR, "daily_steps.csv"),
    "Sleep": os.path.join(MODULE2_DIR, "daily_sleep.csv")
}

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
        "Every step counts. Let's go!"
    ]
    import random
    greeting = random.choice(greetings)
    st.markdown(f"<h2 style='font-size:2.5em; color:#2e86de; margin-bottom:0.2em;'>{greeting}</h2>", unsafe_allow_html=True)

    # Load today's stats
    today = pd.Timestamp.now().normalize()
    def get_today_stat(path, value_col):
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
                # Find rows where ds is today (any time)
                row = df[df['ds'].dt.normalize() == today]
                if not row.empty:
                    return row.iloc[0][value_col]
        return None

    hr = get_today_stat(files["Heart Rate"], 'y')
    steps = get_today_stat(files["Steps"], 'y')
    sleep = get_today_stat(files["Sleep"], 'y')

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
        if 'date' in adf.columns:
            adf['date'] = pd.to_datetime(adf['date']).dt.normalize()
            today_anoms = adf[adf['date'] == today]
            if not today_anoms.empty:
                anomaly_summary = f"{len(today_anoms)} anomaly(s) detected today."
    st.info(anomaly_summary)

    # 7-day sparklines
    st.markdown("#### Last 7 Days Overview")
    spark_col1, spark_col2, spark_col3 = st.columns(3)
    def sparkline(path, label, col):
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
                last7 = df.sort_values('ds').tail(7)
                col.markdown(f"**{label}**")
                col.line_chart(last7['y'].reset_index(drop=True), height=60)
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
                            yaxis_title="Value",
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
                            yaxis_title="Value",
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
                                    yaxis_title="Value",
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
                                        yaxis_title="Component",
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
                                        yaxis_title="Component",
                                        height=350,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_yearly, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Seasonality visualization error: {str(e)}")
                    
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
    
    # ===== FILE UPLOAD & CONFIG (MINIMAL) =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader("📁 Upload CSV Files (up to 5)", type=["csv"], accept_multiple_files=True, key="anom_upload")
    
    with col2:
        st.markdown("**Severity Thresholds**")
        low_threshold = st.slider("Low", 0.0, 1.0, 0.3, step=0.05, key="low_sev_global")
        medium_threshold = st.slider("Medium", low_threshold, 1.0, 0.7, step=0.05, key="med_sev_global")
        high_threshold = st.slider("High", medium_threshold, 1.0, 0.85, step=0.05, key="high_sev_global")
    
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
                        df = pd.read_csv(file)
                        df.columns = df.columns.str.lower().str.strip()
                        
                        # Find date column
                        date_col = None
                        for col_name in ['ds', 'timestamp', 'date', 'time', 'datetime']:
                            if col_name in df.columns:
                                date_col = col_name
                                break
                        if date_col is None:
                            st.error(f"❌ {file.name}: Missing date column (expected: ds, timestamp, date)")
                            continue
                        
                        # Find value column
                        value_col = None
                        for col_name in ['y', 'value', 'metric', 'val', 'data', 'heart_rate', 'steps', 'sleep_hours', 'hr']:
                            if col_name in df.columns:
                                value_col = col_name
                                break
                        if value_col is None:
                            st.error(f"❌ {file.name}: Missing value column (expected: y, value, heart_rate, steps, sleep_hours)")
                            continue
                        
                        # Clean data
                        df_clean = pd.DataFrame()
                        df_clean['ds'] = pd.to_datetime(df[date_col], errors='coerce')
                        df_clean['y'] = pd.to_numeric(df[value_col], errors='coerce')
                        
                        if df_clean['ds'].isna().all() or df_clean['y'].isna().all():
                            st.error(f"❌ {file.name}: Invalid data")
                            continue
                        
                        df_clean['y'] = df_clean['y'].fillna(df_clean['y'].mean())
                        df_clean = df_clean.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
                        
                        if len(df_clean) < 5:
                            st.error(f"❌ {file.name}: Need at least 5 points")
                            continue
                        
                        # Determine metric
                        fname = file.name.lower()
                        if 'heart' in fname or 'hr' in fname:
                            metric_type = 'Heart Rate'
                        elif 'step' in fname:
                            metric_type = 'Steps'
                        elif 'sleep' in fname:
                            metric_type = 'Sleep'
                        else:
                            metric_type = f'Metric_{idx}'
                        
                        processed_data[metric_type] = df_clean
                    
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
                    st.error(f"Error: {str(e)[:100]}")
        
        # ===== DISPLAY CACHED RESULTS =====
        if 'anom_data' in st.session_state:
            config = st.session_state['anom_config']
            
            for metric_name, df in st.session_state['anom_data'].items():
                df = df.copy()
                mean_val = df['y'].mean()
                std_val = df['y'].std()
                
                # FAST: Vectorized anomaly detection
                anomaly_scores = np.zeros(len(df))
                anomaly_types = np.array(['Normal'] * len(df))
                
                # Point anomalies
                outliers = (df['y'] < mean_val - 2.5 * std_val) | (df['y'] > mean_val + 2.5 * std_val)
                anomaly_scores[outliers] += 0.4
                anomaly_types[outliers] = 'Point'
                
                # Contextual anomalies
                changes = np.abs(df['y'].diff()).fillna(0)
                spikes = changes > std_val * 2
                anomaly_scores[spikes] += 0.3
                anomaly_types[spikes] = 'Contextual'
                
                # Prophet (once, cached via session)
                if len(df) >= 20 and 'prophet_cache' not in st.session_state:
                    st.session_state['prophet_cache'] = {}
                
                if len(df) >= 20 and metric_name not in st.session_state.get('prophet_cache', {}):
                    try:
                        df_p = df[['ds', 'y']].copy()
                        df_p.columns = ['ds', 'y']
                        model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
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
                
                # Cluster-based
                try:
                    from sklearn.cluster import KMeans, DBSCAN
                    X = np.column_stack([(df['y'] - mean_val) / (std_val + 1e-6), np.arange(len(df)) / len(df)])
                    
                    km = KMeans(n_clusters=min(4, len(df) // 2), random_state=42, n_init=10)
                    labels = km.fit_predict(X)
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
                
                # DISPLAY
                st.markdown(f"---\n## {metric_name}")
                
                n_anom = df['is_anom'].sum()
                n_high = (df['severity'] == 'High').sum()
                n_med = (df['severity'] == 'Medium').sum()
                n_low = (df['severity'] == 'Low').sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", n_anom)
                col2.metric("🔴 High", n_high)
                col3.metric("🟠 Medium", n_med)
                col4.metric("🟡 Low", n_low)
                
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
                    fig.update_layout(height=280, template='plotly_white', margin=dict(l=30, r=30, t=20, b=20), hovermode='closest')
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
                        fig.update_layout(height=280, template='plotly_white', margin=dict(l=30, r=30, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                
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
                            # Create visualizations
                            fig_analysis = go.Figure()
                            fig_analysis.add_trace(go.Scatter(
                                x=df['date'], y=df['value'],
                                mode='lines+markers',
                                name=metric_type,
                                line=dict(color='#2e86de', width=2),
                                marker=dict(size=4)
                            ))
                            fig_analysis.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
                            
                            # Mark anomalies
                            if len(anomaly_indices) > 0:
                                anom_dates = df['date'].iloc[anomaly_indices]
                                anom_vals = df['value'].iloc[anomaly_indices]
                                fig_analysis.add_trace(go.Scatter(
                                    x=anom_dates, y=anom_vals,
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(size=10, color='red', symbol='x')
                                ))
                            
                            fig_analysis.update_layout(
                                title=f"{metric_type} Analysis",
                                xaxis_title="Date",
                                yaxis_title=f"Value ({unit})",
                                height=400,
                                template='plotly_white'
                            )
                            
                            # Convert to image
                            img_analysis = io.BytesIO()
                            fig_analysis.write_image(img_analysis, format="png", width=700, height=400)
                            img_analysis.seek(0)
                            
                            # Distribution plot
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Histogram(
                                x=df['value'],
                                nbinsx=30,
                                marker=dict(color='#ff6348'),
                                name=metric_type
                            ))
                            fig_dist.update_layout(
                                title="Value Distribution",
                                xaxis_title=f"Value ({unit})",
                                yaxis_title="Frequency",
                                height=400,
                                template='plotly_white'
                            )
                            
                            img_dist = io.BytesIO()
                            fig_dist.write_image(img_dist, format="png", width=700, height=400)
                            img_dist.seek(0)
                            
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
                            pdf.add_page()
                            pdf.set_font("Arial", size=10)
                            
                            # Comprehensive character sanitization function for FPDF compatibility
                            def sanitize_for_pdf(text):
                                """Remove all non-ASCII characters and Unicode symbols, replace with safe ASCII equivalents."""
                                if not isinstance(text, str):
                                    text = str(text)
                                
                                # Unicode to ASCII replacements (comprehensive)
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
                                
                                # Apply direct replacements
                                result = text
                                for unicode_char, replacement in replacements.items():
                                    result = result.replace(unicode_char, replacement)
                                
                                # Remove any remaining non-ASCII characters beyond 127
                                try:
                                    result = result.encode('ascii', 'ignore').decode('ascii')
                                except:
                                    result = ''.join(char if ord(char) < 128 else '?' for char in result)
                                
                                return result
                            
                            # Add text to PDF with proper encoding and width handling
                            for line in report_text.split('\n'):
                                if line.strip():
                                    # Sanitize line for FPDF (remove all non-ASCII characters)
                                    clean_line = sanitize_for_pdf(line)
                                    # Use multi_cell with proper width (185 = full page with margins)
                                    if clean_line.strip():
                                        pdf.multi_cell(185, 5, clean_line, align='L')
                                else:
                                    pdf.ln(2)
                            
                            # Add page break and graphs
                            pdf.add_page()
                            pdf.set_font("Arial", size=14, style="B")
                            pdf.cell(0, 10, "Data Visualizations", ln=True)
                            pdf.ln(5)
                            
                            # Add analysis graph
                            pdf.image(img_analysis, x=10, w=190)
                            pdf.ln(2)
                            pdf.set_font("Arial", size=10)
                            pdf.multi_cell(0, 4, "Analysis Graph: Time-series visualization of your health metrics with identified anomalies marked in red.")
                            
                            pdf.ln(5)
                            
                            # Add distribution graph
                            pdf.image(img_dist, x=10, w=190)
                            pdf.ln(2)
                            pdf.multi_cell(0, 4, "Distribution Graph: Frequency distribution showing how your values are spread across the range.")
                            
                            # Save PDF
                            pdf_filename = f"HealthReport_{metric_type.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            pdf.output(pdf_filename)
                            
                            # Display success and download button
                            st.success(f"✓ Report generated successfully!")
                            st.markdown("---")
                            
                            with open(pdf_filename, "rb") as f:
                                st.download_button(
                                    "📥 Download PDF Report",
                                    f,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            
                            # Show preview
                            st.markdown("### Report Preview")
                            st.text(report_text)
                            
                            # Display graphs
                            st.markdown("### Visualizations")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_analysis, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_dist, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                
                # Show data preview
                if st.checkbox("Show data preview", value=False):
                    st.dataframe(df[['date', 'value']].head(20), use_container_width=True)
            
            else:
                st.error("Could not auto-detect date and value columns. Please ensure CSV has standard column names like 'ds', 'date', 'y', 'value'.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")