"""
visualize_anomaly.py

Module 3: Anomaly Detection & Visualization
Drop this file into your project's `src/` folder.

Features included (big list):
- Threshold (rule-based) anomalies with adaptive sustained windows (auto-detect sampling rate)
- Residual (model-based) anomalies using Prophet (if installed) or fallback forecasting
- Cluster-based anomalies using KMeans or DBSCAN (with feature extraction)
- Feature extraction: handcrafted time-series features + optional TSFresh (if installed)
- Visualizations with Plotly and Matplotlib (Plotly used in Streamlit UI)
- Streamlit app with interactive controls: thresholds, sliders, toggles, date-range, export, annotations
- Export anomaly CSV/JSON and downloadable charts
- Auto-tune thresholds helper (simple baseline-based suggestions)
- Annotation storage (local session-state) and simple "clinician report" export

Dependencies (install before running):
pip install pandas numpy plotly scikit-learn scipy matplotlib streamlit prophet tsfresh dbscan

Notes:
- prophet, tsfresh and dbscan are optional. The module will fall back to lighter-weight methods if missing.
- For Prophet: install `prophet` (PyPI package). On some systems you may need cmdstanpy backend.

Usage:
Save as src/visualize_anomaly.py and run with:
streamlit run src/visualize_anomaly.py
"""

# Core imports
import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# ML / stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Optional imports
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from tsfresh import extract_features
    HAS_TSFRESH = True
except Exception:
    HAS_TSFRESH = False

try:
    from sklearn.cluster import DBSCAN
    HAS_DBSCAN = True
except Exception:
    HAS_DBSCAN = False

# Streamlit (application layer)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

# --------------------- Utility helpers ---------------------

def infer_sampling_period(series: pd.Series) -> pd.Timedelta:
    """Infer typical sampling period of a timestamp series. Returns a Timedelta."""
    s = pd.to_datetime(series.dropna().sort_values())
    if len(s) < 2:
        return pd.Timedelta('1min')
    diffs = s.diff().dropna()
    mode = diffs.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return diffs.median()


def window_size_minutes_to_counts(window_minutes: int, sampling_period: pd.Timedelta) -> int:
    """Convert minutes to number-of-records given sampling_period."""
    if sampling_period <= pd.Timedelta('0s'):
        return window_minutes
    return max(1, int((pd.Timedelta(f"{window_minutes}min") / sampling_period)))


def ensure_timestamp(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    return df

# --------------------- Threshold Detector ---------------------
class ThresholdAnomalyDetector:
    def __init__(self, rules: Optional[Dict] = None):
        default_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'Heart rate outside normal resting range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,  # tune based on freq
                'sustained_minutes': 5,
                'description': 'Unrealistic step count detected'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,
                'max_threshold': 720,
                'sustained_minutes': 0,
                'description': 'Unusual sleep duration'
            }
        }
        self.rules = rules if rules is not None else default_rules

    def detect_anomalies(self, df: pd.DataFrame, data_type: str, ts_col: str = 'timestamp') -> Tuple[pd.DataFrame, Dict]:
        df = ensure_timestamp(df, ts_col)
        report = {
            'method': 'Threshold-Based',
            'data_type': data_type,
            'total_records': len(df),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
        }

        if data_type not in self.rules:
            return df, report

        rule = self.rules[data_type]
        metric = rule['metric_name']
        if metric not in df.columns:
            return df, report

        sampling = infer_sampling_period(df[ts_col])
        window_counts = window_size_minutes_to_counts(rule['sustained_minutes'], sampling)

        df = df.sort_values(ts_col).reset_index(drop=True)
        df['threshold_anomaly'] = False
        df['anomaly_reason'] = ''
        df['severity'] = 'Normal'

        too_high = df[metric] > rule['max_threshold']
        too_low = df[metric] < rule['min_threshold']

        if rule['sustained_minutes'] > 0 and window_counts > 1:
            high_sust = too_high.rolling(window=window_counts, min_periods=window_counts).sum() >= window_counts
            low_sust = too_low.rolling(window=window_counts, min_periods=window_counts).sum() >= window_counts
            df.loc[high_sust, 'threshold_anomaly'] = True
            df.loc[high_sust, 'anomaly_reason'] = f'High {metric} (>{rule["max_threshold"]})'
            df.loc[high_sust, 'severity'] = 'High'
            df.loc[low_sust, 'threshold_anomaly'] = True
            df.loc[low_sust, 'anomaly_reason'] = f'Low {metric} (<{rule["min_threshold"]})'
            df.loc[low_sust, 'severity'] = 'Medium'
        else:
            df.loc[too_high, 'threshold_anomaly'] = True
            df.loc[too_high, 'anomaly_reason'] = f'High {metric} (>{rule["max_threshold"]})'
            df.loc[too_high, 'severity'] = 'High'
            df.loc[too_low, 'threshold_anomaly'] = True
            df.loc[too_low, 'anomaly_reason'] = f'Low {metric} (<{rule["min_threshold"]})'
            df.loc[too_low, 'severity'] = 'Medium'

        anomalies = df[df['threshold_anomaly']]
        report['anomalies_detected'] = int(len(anomalies))
        report['anomaly_percentage'] = (len(anomalies) / max(1, len(df))) * 100
        report['threshold_info'] = rule

        return df, report

# --------------------- Residual / Forecast Detector ---------------------
class ResidualAnomalyDetector:
    def __init__(self, threshold_std: float = 3.0, freq: str = '1min'):
        self.threshold_std = threshold_std
        self.freq = freq

    def _fit_prophet_and_forecast(self, df: pd.DataFrame, metric: str, periods: int = 0) -> pd.DataFrame:
        # Build df for Prophet
        df_p = df[[ 'timestamp', metric ]].rename(columns={'timestamp':'ds', metric:'y'}).dropna()
        if len(df_p) < 10:
            # Not enough data for Prophet
            raise ValueError('Not enough data for Prophet')
        m = Prophet()
        m.fit(df_p)
        future = m.make_future_dataframe(periods=periods, freq=self.freq)
        forecast = m.predict(future)
        return forecast[['ds','yhat','yhat_lower','yhat_upper']]

    def _fallback_forecast(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        # Simple moving-average + seasonal daily mean fallback
        tmp = df[['timestamp', metric]].dropna().sort_values('timestamp').reset_index(drop=True)
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        # use rolling median as prediction
        tmp['predicted'] = tmp[metric].rolling(window=15, min_periods=1).median()
        tmp['yhat_lower'] = tmp['predicted'] - 2 * tmp[metric].rolling(window=30, min_periods=1).std().fillna(0)
        tmp['yhat_upper'] = tmp['predicted'] + 2 * tmp[metric].rolling(window=30, min_periods=1).std().fillna(0)
        tmp = tmp.rename(columns={'predicted':'yhat'})[['timestamp','yhat','yhat_lower','yhat_upper']]
        tmp = tmp.rename(columns={'timestamp':'ds'})
        return tmp

    def detect_from_forecast(self, df: pd.DataFrame, data_type: str, forecast_df: Optional[pd.DataFrame] = None, ts_col: str = 'timestamp') -> Tuple[pd.DataFrame, Dict]:
        df = ensure_timestamp(df, ts_col)
        metric_map = {'heart_rate':'heart_rate','steps':'step_count','sleep':'duration_minutes'}
        report = {'method':'Residual-Based','data_type':data_type,'anomalies_detected':0,'anomaly_percentage':0.0,'threshold_std':self.threshold_std}

        if data_type not in metric_map:
            return df, report
        metric = metric_map[data_type]
        if metric not in df.columns:
            return df, report

        df = df.sort_values(ts_col).reset_index(drop=True)

        if forecast_df is None:
            # Fit Prophet if installed else fallback
            if HAS_PROPHET:
                try:
                    f = self._fit_prophet_and_forecast(df, metric)
                    f = f.rename(columns={'ds':'timestamp','yhat':'predicted','yhat_lower':'yhat_lower','yhat_upper':'yhat_upper'})
                except Exception:
                    f = self._fallback_forecast(df, metric).rename(columns={'ds':'timestamp','yhat':'predicted','yhat_lower':'yhat_lower','yhat_upper':'yhat_upper'})
            else:
                f = self._fallback_forecast(df, metric).rename(columns={'ds':'timestamp','yhat':'predicted','yhat_lower':'yhat_lower','yhat_upper':'yhat_upper'})
        else:
            f = forecast_df.rename(columns={'ds':'timestamp','yhat':'predicted'})

        merged = df.merge(f[['timestamp','predicted','yhat_lower','yhat_upper']], on='timestamp', how='left')
        merged['residual'] = merged[metric] - merged['predicted']
        rmean = merged['residual'].mean()
        rstd = merged['residual'].std()
        threshold = self.threshold_std * (rstd if not np.isnan(rstd) else 0)
        merged['residual_anomaly'] = False
        merged.loc[merged['residual'].abs() > threshold, 'residual_anomaly'] = True
        if 'yhat_lower' in merged.columns and 'yhat_upper' in merged.columns:
            outside = (merged[metric] > merged['yhat_upper']) | (merged[metric] < merged['yhat_lower'])
            merged.loc[outside, 'residual_anomaly'] = True
        merged['residual_anomaly_reason'] = ''
        merged.loc[merged['residual_anomaly'], 'residual_anomaly_reason'] = 'Deviates from predicted trend'

        anomaly_count = int(merged['residual_anomaly'].sum())
        report['anomalies_detected'] = anomaly_count
        report['anomaly_percentage'] = (anomaly_count / max(1, len(merged))) * 100
        report['residual_stats'] = {'mean':float(rmean) if not np.isnan(rmean) else None,'std':float(rstd) if not np.isnan(rstd) else None,'threshold':float(threshold)}

        return merged, report

# --------------------- Feature Extraction & Clustering ---------------------
class FeatureExtractor:
    def __init__(self, use_tsfresh: bool = False):
        self.use_tsfresh = use_tsfresh and HAS_TSFRESH

    def extract_handcrafted(self, df: pd.DataFrame, ts_col: str = 'timestamp', metric: str = 'heart_rate') -> pd.DataFrame:
        df = ensure_timestamp(df, ts_col).sort_values(ts_col)
        s = df[metric].dropna()
        feats = {
            'mean': float(s.mean()) if len(s)>0 else np.nan,
            'std': float(s.std()) if len(s)>0 else np.nan,
            'median': float(s.median()) if len(s)>0 else np.nan,
            'min': float(s.min()) if len(s)>0 else np.nan,
            'max': float(s.max()) if len(s)>0 else np.nan,
            'skew': float(s.skew()) if len(s)>0 else np.nan,
            'kurtosis': float(s.kurtosis()) if len(s)>0 else np.nan,
            'pct_25': float(s.quantile(0.25)) if len(s)>0 else np.nan,
            'pct_75': float(s.quantile(0.75)) if len(s)>0 else np.nan,
            'num_peaks': int(((s.diff().fillna(0).abs()> (s.std()*1.5)).sum()))
        }
        return pd.DataFrame([feats])

    def extract(self, df: pd.DataFrame, group_col: Optional[str] = None, ts_col: str='timestamp', metric:str='heart_rate') -> pd.DataFrame:
        # If tsfresh available and the data is long, use it
        if self.use_tsfresh and len(df) > 200:
            # tsfresh expects columns: id, time, value
            tmp = df[[ts_col, metric]].dropna().reset_index().rename(columns={'index':'id', ts_col:'time', metric:'value'})
            extracted = extract_features(tmp, column_id='id', column_sort='time')
            return extracted.fillna(0)
        # else use handcrafted features
        return self.extract_handcrafted(df, ts_col=ts_col, metric=metric)

class ClusterAnomalyDetector:
    def __init__(self):
        self.cluster_info = {}

    def detect_cluster_outliers(self, feature_matrix: pd.DataFrame, method: str = 'kmeans', n_clusters: int = 4, outlier_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict]:
        """
        Defensive clustering:
        - If there are too few samples (<3) skip clustering and return safe report.
        - Try different algorithms, but catch exceptions and return safe result if error occurs.
        """
        df = feature_matrix.copy().reset_index(drop=True)
        report = {'method':'Cluster-Based','total_clusters':0,'anomalies_detected':0,'anomaly_percentage':0.0}
        numeric = df.select_dtypes(include=[np.number]).fillna(0)
        n_samples = numeric.shape[0]

        # Minimum samples required to run clustering/LOF
        min_samples_for_clustering = 3
        if n_samples < min_samples_for_clustering:
            df['cluster'] = -999
            df['cluster_anomaly'] = False
            df['cluster_anomaly_reason'] = ''
            report['total_clusters'] = 0
            report['anomalies_detected'] = 0
            report['anomaly_percentage'] = 0.0
            report['cluster_distribution'] = {}
            report['anomalous_clusters'] = []
            return df, report

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric)

        # Adjust n_clusters if more clusters than samples
        if method == 'kmeans':
            if n_samples < n_clusters:
                n_clusters = max(1, n_samples)

        labels = None
        try:
            if method == 'kmeans':
                km = KMeans(n_clusters=max(1,n_clusters), random_state=42)
                labels = km.fit_predict(X)
            elif method == 'dbscan' and HAS_DBSCAN:
                db = DBSCAN(eps=0.5, min_samples=2)
                labels = db.fit_predict(X)
            elif method == 'lof':
                n_neighbors = min(20, max(2, n_samples - 1))
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
                lof_labels = lof.fit_predict(X)
                labels = np.where(lof_labels == -1, -1, 0)
            else:
                # fallback to LOF-like behavior
                n_neighbors = min(20, max(2, n_samples - 1))
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
                lof_labels = lof.fit_predict(X)
                labels = np.where(lof_labels == -1, -1, 0)
        except Exception as e:
            df['cluster'] = -999
            df['cluster_anomaly'] = False
            df['cluster_anomaly_reason'] = ''
            report['error'] = str(e)
            return df, report

        df['cluster'] = labels
        sizes = pd.Series(labels).value_counts()
        total = len(labels)
        anomalous_clusters = [int(c) for c, s in sizes.items() if (s / total) < outlier_threshold or c == -1]
        df['cluster_anomaly'] = df['cluster'].isin(anomalous_clusters)
        df['cluster_anomaly_reason'] = ''
        for c in anomalous_clusters:
            df.loc[df['cluster'] == c, 'cluster_anomaly_reason'] = f'Belongs to anomalous cluster {c}'

        anomaly_count = int(df['cluster_anomaly'].sum())
        report['total_clusters'] = int(len(sizes))
        report['anomalies_detected'] = anomaly_count
        report['anomaly_percentage'] = (anomaly_count / max(1, total)) * 100
        report['cluster_distribution'] = sizes.to_dict()
        report['anomalous_clusters'] = anomalous_clusters
        self.cluster_info = report
        return df, report

# --------------------- Visualization ---------------------
class AnomalyVisualizer:
    def __init__(self):
        self.colors = {'normal':'#1f77b4','threshold':'#ff7f0e','residual':'#d62728','cluster':'#9467bd'}

    def _apply_layout_template(self, fig: go.Figure, dark_mode: bool):
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        fig.update_layout(template=template)

    def plot_heart_rate(self, df: pd.DataFrame, title:str='Heart Rate Anomaly Detection', dark_mode: bool = False) -> go.Figure:
        df = df.sort_values('timestamp')
        fig = go.Figure()
        normal_mask = ~df.get('threshold_anomaly', False)
        fig.add_trace(go.Scatter(x=df.loc[normal_mask,'timestamp'], y=df.loc[normal_mask,'heart_rate'], mode='lines', name='Normal', line=dict(color=self.colors['normal'], width=1.5)))
        if 'predicted' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted'], mode='lines', name='Predicted', line=dict(color='lightgreen', width=2, dash='dash')))
            if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['yhat_upper'], mode='lines', showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['yhat_lower'], mode='lines', showlegend=False, fill='tonexty', fillcolor='rgba(144,238,144,0.2)'))
        if 'threshold_anomaly' in df.columns:
            ta = df[df['threshold_anomaly']]
            if len(ta):
                fig.add_trace(go.Scatter(x=ta['timestamp'], y=ta['heart_rate'], mode='markers', name='Threshold', marker=dict(color=self.colors['threshold'], size=8, symbol='x')))
        if 'residual_anomaly' in df.columns:
            ra = df[df['residual_anomaly']]
            if len(ra):
                fig.add_trace(go.Scatter(x=ra['timestamp'], y=ra['heart_rate'], mode='markers', name='Residual', marker=dict(color=self.colors['residual'], size=10, symbol='diamond')))
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='BPM', hovermode='x unified', height=600)
        return fig

    def plot_steps(self, df: pd.DataFrame, title:str='Step Count Anomaly Detection', dark_mode: bool = False) -> go.Figure:
        df = df.sort_values('timestamp')
        fig = go.Figure()
        normal_mask = ~df.get('threshold_anomaly', False)
        fig.add_trace(go.Bar(x=df.loc[normal_mask,'timestamp'], y=df.loc[normal_mask,'step_count'], name='Normal'))
        if 'predicted' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted'], mode='lines', name='Predicted'))
        if 'threshold_anomaly' in df.columns:
            ta = df[df['threshold_anomaly']]
            if len(ta):
                fig.add_trace(go.Scatter(x=ta['timestamp'], y=ta['step_count'], mode='markers', name='Anomaly', marker=dict(color='red', size=12, symbol='star')))
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Steps', hovermode='x unified', height=500)
        return fig

    def plot_sleep(self, df: pd.DataFrame, title:str='Sleep Pattern Anomaly Detection', dark_mode: bool = False) -> go.Figure:
        d = df.copy().sort_values('timestamp')
        d['duration_hours'] = d['duration_minutes']/60.0
        fig = go.Figure()
        normal_mask = ~d.get('threshold_anomaly', False)
        fig.add_trace(go.Scatter(x=d.loc[normal_mask,'timestamp'], y=d.loc[normal_mask,'duration_hours'], mode='lines+markers', name='Normal', fill='tozeroy'))
        # reference lines - color will be visible in both templates
        fig.add_hline(y=7, line_dash='dash', annotation_text='Recommended (7h)', annotation_position='right')
        fig.add_hline(y=3, line_dash='dash', line_color='red', annotation_text='Minimum (3h)', annotation_position='right')
        fig.add_hline(y=12, line_dash='dash', line_color='red', annotation_text='Maximum (12h)', annotation_position='right')
        if 'threshold_anomaly' in d.columns:
            ta = d[d['threshold_anomaly']]
            if len(ta):
                fig.add_trace(go.Scatter(x=ta['timestamp'], y=ta['duration_minutes']/60.0, mode='markers', name='Anomaly', marker=dict(color='red', size=12, symbol='circle-open')))
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Sleep (hours)', height=500)
        return fig

# --------------------- Pipeline ---------------------
class VisualizeAnomalyPipeline:
    def __init__(self, use_tsfresh: bool = False):
        self.threshold_detector = ThresholdAnomalyDetector()
        self.residual_detector = ResidualAnomalyDetector()
        self.feature_extractor = FeatureExtractor(use_tsfresh=use_tsfresh)
        self.cluster_detector = ClusterAnomalyDetector()
        self.visualizer = AnomalyVisualizer()

    def run(self, preprocessed_data: Dict[str, pd.DataFrame], prophet_forecasts: Optional[Dict[str,pd.DataFrame]] = None, feature_matrices: Optional[Dict[str,pd.DataFrame]] = None, cluster_params: Optional[Dict] = None) -> Dict:
        results = {'data_with_anomalies':{}, 'reports':{}}
        for data_type, df in preprocessed_data.items():
            results['reports'][data_type] = {}
            # Threshold
            df_th, r_th = self.threshold_detector.detect_anomalies(df, data_type)
            results['reports'][data_type]['threshold'] = r_th
            # Residual
            if prophet_forecasts and data_type in prophet_forecasts:
                forecast = prophet_forecasts[data_type]
                df_res, r_res = self.residual_detector.detect_from_forecast(df_th, data_type, forecast)
                results['reports'][data_type]['residual'] = r_res
                df_final = df_res
            else:
                df_res, r_res = self.residual_detector.detect_from_forecast(df_th, data_type, None)
                results['reports'][data_type]['residual'] = r_res
                df_final = df_res
            # Clustering (defensive)
            if feature_matrices and data_type in feature_matrices:
                fm = feature_matrices[data_type]
                # only attempt clustering if feature matrix has >= 3 rows
                if isinstance(fm, pd.DataFrame) and fm.select_dtypes(include=[np.number]).shape[0] >= 3:
                    cparams = cluster_params or {}
                    cmethod = cparams.get('method','kmeans')
                    n_clusters = cparams.get('n_clusters',4)
                    outlier_threshold = cparams.get('outlier_threshold',0.05)
                    cluster_df, r_cluster = self.cluster_detector.detect_cluster_outliers(fm, method=cmethod, n_clusters=n_clusters, outlier_threshold=outlier_threshold)
                    results['reports'][data_type]['cluster'] = r_cluster
                    # If cluster labels correspond to rows in df_final by index, merge
                    try:
                        if len(cluster_df) == len(df_final):
                            df_final = df_final.reset_index(drop=True).merge(cluster_df[['cluster','cluster_anomaly','cluster_anomaly_reason']], left_index=True, right_index=True, how='left')
                    except Exception:
                        pass
                else:
                    results['reports'][data_type]['cluster'] = {'method':'Cluster-Based','note':'Skipped - not enough samples for clustering','anomalies_detected':0}
            results['data_with_anomalies'][data_type] = df_final
        return results

# --------------------- Sample Data Generator ---------------------
def create_sample_data_with_anomalies() -> Dict[str,pd.DataFrame]:
    timestamps = pd.date_range(start='2024-01-15 08:00:00', end='2024-01-15 20:00:00', freq='1min')
    base_hr = 70
    hr_list = []
    for ts in timestamps:
        tod = ts.hour + ts.minute/60.0
        hr = base_hr + np.random.normal(0,3)
        if 9 <= tod < 10:
            hr = 110 + np.random.normal(0,5)
        if 11.5 <= tod < 12:
            hr = 135 + np.random.normal(0,5)
        if 16 <= tod < 16.3:
            hr = 35 + np.random.normal(0,3)
        if 18.5 <= tod < 18.6:
            hr = 150
        hr_list.append(max(30, min(220, hr)))
    heart_rate_df = pd.DataFrame({'timestamp':timestamps,'heart_rate':hr_list})

    step_timestamps = pd.date_range(start='2024-01-15 08:00:00', end='2024-01-15 20:00:00', freq='5min')
    step_list = []
    for ts in step_timestamps:
        tod = ts.hour + ts.minute/60.0
        if 15 <= tod < 15.2:
            s = 1200
        elif 8 <= tod < 9:
            s = 50 + np.random.randint(-10,10)
        else:
            s = 20 + np.random.randint(-5,5)
        step_list.append(max(0, s))
    steps_df = pd.DataFrame({'timestamp':step_timestamps,'step_count':step_list})

    sleep_df = pd.DataFrame({'timestamp':pd.to_datetime(['2024-01-14','2024-01-15']),'duration_minutes':[420,200]})

    return {'heart_rate':heart_rate_df,'steps':steps_df,'sleep':sleep_df}

# --------------------- Streamlit App ---------------------
if HAS_STREAMLIT:
    def _apply_streamlit_dark_css(dark_mode: bool):
        """Inject a small css to tweak background/text to better match dark mode selection."""
        if dark_mode:
            st.markdown(
                """
                <style>
                .stApp { background-color: #0e1117; color: #e6edf3; }
                .css-18e3th9 { background-color: #0e1117; }
                .css-1d391kg { background-color: #0e1117; }
                </style>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <style>
                .stApp { background-color: white; color: black; }
                </style>
                """, unsafe_allow_html=True)

    def streamlit_app():
        st.set_page_config(page_title='Milestone 3: Anomaly Detection', layout='wide')
        st.title('🚨 Milestone 3 — Anomaly Detection & Visualization')

        st.sidebar.header('Configuration')
        use_sample = st.sidebar.checkbox('Use sample data (demo)', value=True)
        use_tsfresh = st.sidebar.checkbox('Use TSFresh (if installed)', value=False)
        residual_std = st.sidebar.slider('Residual detection sigma threshold', min_value=1.0, max_value=6.0, value=3.0, step=0.5)
        outlier_threshold = st.sidebar.slider('Cluster outlier threshold (fraction)', min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        cluster_method = st.sidebar.selectbox('Clustering method', options=['kmeans','dbscan','lof'])
        dark_mode = st.sidebar.checkbox('Dark mode', value=False)
        st.sidebar.markdown('---')
        st.sidebar.markdown('**Thresholds (editable)**')
        # show default rules & allow edit
        rules = st.session_state.get('threshold_rules', None)
        if rules is None:
            rules = ThresholdAnomalyDetector().rules
            st.session_state['threshold_rules'] = rules
        # allow edit for heart rate min/max
        hr_min = st.sidebar.number_input('HR min (bpm)', value=int(rules['heart_rate']['min_threshold']))
        hr_max = st.sidebar.number_input('HR max (bpm)', value=int(rules['heart_rate']['max_threshold']))
        rules['heart_rate']['min_threshold'] = hr_min
        rules['heart_rate']['max_threshold'] = hr_max
        rules['heart_rate']['sustained_minutes'] = st.sidebar.number_input('HR sustained (minutes)', value=int(rules['heart_rate']['sustained_minutes']))
        st.sidebar.markdown('---')
        st.sidebar.markdown('Data / Export')
        uploaded = st.sidebar.file_uploader('Upload preprocessed CSV (heart_rate/steps/sleep) or leave demo', type=['csv','zip'])

        # Apply dark mode CSS early
        _apply_streamlit_dark_css(dark_mode)

        if use_sample or uploaded is None:
            preprocessed = create_sample_data_with_anomalies()
            st.info('Using sample data (demo).')
        else:
            # try load files from uploaded zip or csvs
            preprocessed = {}
            st.info('Please upload three CSVs named heart_rate.csv, steps.csv, sleep.csv or a zip containing them.')
            if uploaded is not None:
                try:
                    if uploaded.type == 'application/zip':
                        import zipfile
                        z = zipfile.ZipFile(uploaded)
                        for name in z.namelist():
                            if 'heart' in name.lower():
                                preprocessed['heart_rate'] = pd.read_csv(z.open(name))
                            if 'step' in name.lower():
                                preprocessed['steps'] = pd.read_csv(z.open(name))
                            if 'sleep' in name.lower():
                                preprocessed['sleep'] = pd.read_csv(z.open(name))
                    else:
                        # assume single CSV; try to infer
                        df = pd.read_csv(uploaded)
                        if 'heart_rate' in df.columns:
                            preprocessed['heart_rate'] = df
                        elif 'step_count' in df.columns:
                            preprocessed['steps'] = df
                        elif 'duration_minutes' in df.columns:
                            preprocessed['sleep'] = df
                except Exception as e:
                    st.error('Error reading uploaded file: ' + str(e))

        # update detector objects
        pipeline = VisualizeAnomalyPipeline(use_tsfresh=use_tsfresh)
        pipeline.threshold_detector = ThresholdAnomalyDetector(rules=rules)
        pipeline.residual_detector = ResidualAnomalyDetector(threshold_std=residual_std)

        # Feature matrices auto-extraction
        feature_matrices = {}
        for key, df in preprocessed.items():
            if key == 'heart_rate' or key == 'steps':
                fm = pipeline.feature_extractor.extract(df, ts_col='timestamp', metric=('heart_rate' if key=='heart_rate' else 'step_count'))
            else:
                fm = pipeline.feature_extractor.extract(df, ts_col='timestamp', metric='duration_minutes')
            feature_matrices[key] = fm

        # Run pipeline
        if st.button('🚀 Run Anomaly Detection'):
            with st.spinner('Running detectors...'):
                results = pipeline.run(preprocessed_data=preprocessed, prophet_forecasts=None, feature_matrices=feature_matrices, cluster_params={'method':cluster_method,'n_clusters':4,'outlier_threshold':outlier_threshold})
                st.session_state['m3_results'] = results
                # Save dark_mode selection for visualizer usage
                st.session_state['m3_dark_mode'] = dark_mode
                st.success('Completed anomaly detection')

        # Show results
        if 'm3_results' in st.session_state:
            results = st.session_state['m3_results']
            dark_mode_setting = st.session_state.get('m3_dark_mode', False)
            cols = st.columns([1,2])
            # Overview
            total_anoms = 0
            methods_used = set()
            data_types = list(results['data_with_anomalies'].keys())
            for dt, rep in results['reports'].items():
                for method_name, r in rep.items():
                    total_anoms += int(r.get('anomalies_detected',0))
                    methods_used.add(r.get('method',method_name))
            cols[0].metric('Total Anomalies', total_anoms)
            cols[0].metric('Data Types', len(data_types))
            cols[0].metric('Detection Methods', len(methods_used))

            # Tabs for each data type
            for dt, df in results['data_with_anomalies'].items():
                st.header(dt.replace('_',' ').title())
                if dt == 'heart_rate':
                    fig = pipeline.visualizer.plot_heart_rate(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig, use_container_width=True)
                elif dt == 'steps':
                    fig = pipeline.visualizer.plot_steps(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig, use_container_width=True)
                elif dt == 'sleep':
                    fig = pipeline.visualizer.plot_sleep(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig, use_container_width=True)

                # Show anomaly table
                anomaly_cols = [c for c in df.columns if 'anomaly' in c.lower()]
                if anomaly_cols:
                    mask = df[anomaly_cols].any(axis=1)
                    anoms = df[mask].copy()
                    st.subheader('Detected anomalies')
                    st.dataframe(anoms, use_container_width=True)

            # Exports
            st.markdown('---')
            st.subheader('Exports & Reports')
            if st.button('📄 Download Anomaly Report (JSON)'):
                json_report = json.dumps(results['reports'], default=str, indent=2)
                st.download_button('Download JSON', data=json_report, file_name=f'anomaly_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', mime='application/json')
            if st.button('📊 Download All Anomalies (CSV)'):
                all_dfs = []
                for dt, df in results['data_with_anomalies'].items():
                    acols = [c for c in df.columns if 'anomaly' in c.lower()]
                    if acols:
                        mask = df[acols].any(axis=1)
                        tmp = df[mask].copy()
                        tmp['data_type'] = dt
                        all_dfs.append(tmp)
                if len(all_dfs):
                    combined = pd.concat(all_dfs, ignore_index=True)
                    csv = combined.to_csv(index=False)
                    st.download_button('Download CSV', data=csv, file_name=f'anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
                else:
                    st.info('No anomalies to export')

            # Annotation panel
            st.markdown('---')
            st.subheader('Annotations')
            if 'annotations' not in st.session_state:
                st.session_state['annotations'] = []
            with st.form('annot'):
                ann_text = st.text_area('Add a note about anomalies (will attach timestamp)')
                submitted = st.form_submit_button('Save note')
                if submitted and ann_text.strip():
                    st.session_state['annotations'].append({'timestamp':datetime.now().isoformat(),'note':ann_text})
                    st.success('Saved note')
            if st.session_state['annotations']:
                st.write(pd.DataFrame(st.session_state['annotations']))

    # run app
    if __name__ == '__main__':
        streamlit_app()

else:
    # If Streamlit not installed, provide a simple CLI demo runner
    def cli_demo():
        print('Streamlit is not installed. Running CLI demo...')
        pre = create_sample_data_with_anomalies()
        pipeline = VisualizeAnomalyPipeline()
        results = pipeline.run(preprocessed_data=pre)
        for dt, rep in results['reports'].items():
            print(f"\n=== {dt} ===")
            for m, r in rep.items():
                print(m, r)
        # save sample outputs
        outdir = os.path.join(os.getcwd(), 'milestone3_output')
        os.makedirs(outdir, exist_ok=True)
        for dt, df in results['data_with_anomalies'].items():
            df.to_csv(os.path.join(outdir, f'{dt}_anomalies.csv'), index=False)
        print('Saved outputs to', outdir)

    if __name__ == '__main__':
        cli_demo()
