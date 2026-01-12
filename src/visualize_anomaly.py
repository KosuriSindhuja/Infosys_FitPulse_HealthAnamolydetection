
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

# --------------------- ENSEMBLE ANOMALY SCORING SYSTEM ---------------------
# MILESTONE 3: Multi-method consensus-based anomaly detection
# Combines rule-based, model-based, and cluster-based methods
class EnsembleAnomalyScorer:
    """
    Ensemble scorer that combines flags from multiple detection methods.
    
    Strategy:
    - Rule-based, Model-based, Cluster-based each produce a boolean flag
    - Confidence score = (# methods voting anomaly) / 3
    - Severity = combination of confidence + temporal context
    
    Anomaly Types Handled:
    - Point: Single value outside thresholds
    - Contextual: Value normal generally but abnormal in specific context
    - Collective: Pattern of consecutive anomalies
    """
    
    def __init__(self):
        self.confidence_threshold = 0.67  # At least 2/3 methods must agree
        
    def score_ensemble(self, df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create ensemble scoring from existing anomaly columns.
        
        Expects columns: threshold_anomaly, residual_anomaly, cluster_anomaly
        Produces: ensemble_anomaly, confidence_score, severity, anomaly_type
        """
        df = df.copy()
        n = len(df)
        
        # Initialize ensemble columns
        df['ensemble_anomaly'] = False
        df['confidence_score'] = 0.0
        df['severity'] = 'Normal'
        df['anomaly_type'] = 'None'
        df['ensemble_reason'] = ''
        
        # Count votes from each method
        votes = np.zeros(n)
        method_flags = []
        
        if 'threshold_anomaly' in df.columns:
            method_flags.append(df['threshold_anomaly'].fillna(False).astype(int))
        else:
            method_flags.append(np.zeros(n))
            
        if 'residual_anomaly' in df.columns:
            method_flags.append(df['residual_anomaly'].fillna(False).astype(int))
        else:
            method_flags.append(np.zeros(n))
            
        if 'cluster_anomaly' in df.columns:
            method_flags.append(df['cluster_anomaly'].fillna(False).astype(int))
        else:
            method_flags.append(np.zeros(n))
        
        # Sum votes
        for flags in method_flags:
            votes += flags.values if hasattr(flags, 'values') else flags
        
        # Calculate confidence scores
        confidence_scores = votes / len(method_flags)
        df['confidence_score'] = confidence_scores
        
        # Final ensemble: at least 2 out of 3 methods agree
        df['ensemble_anomaly'] = confidence_scores >= self.confidence_threshold
        
        # Assign severity based on confidence + temporal patterns
        severity = self._assign_severity(df, confidence_scores)
        df['severity'] = severity
        
        # Classify anomaly type
        anomaly_types = self._classify_anomaly_type(df)
        df['anomaly_type'] = anomaly_types
        
        # Reason: which methods triggered
        reasons = self._build_reason_string(df, method_flags)
        df['ensemble_reason'] = reasons
        
        return df
    
    def _assign_severity(self, df: pd.DataFrame, confidence_scores: np.ndarray) -> np.ndarray:
        """
        Assign severity levels (1=Low, 2=Medium, 3=High).
        
        Criteria:
        - Confidence score
        - Temporal pattern (isolated vs sustained)
        - Multiple methods agreeing
        """
        severity = np.full(len(df), 'Normal', dtype=object)
        
        # Check for temporal patterns
        is_anomaly = df['ensemble_anomaly'].fillna(False).values
        
        for i in range(len(df)):
            if not is_anomaly[i]:
                continue
            
            conf = confidence_scores[i]
            
            # Base severity from confidence
            if conf >= 0.95:
                base_sev = 'High'
            elif conf >= 0.67:
                base_sev = 'Medium'
            else:
                base_sev = 'Low'
            
            # Check temporal context (sustained vs isolated)
            window_start = max(0, i - 2)
            window_end = min(len(is_anomaly), i + 3)
            nearby_anomalies = is_anomaly[window_start:window_end].sum()
            
            # If part of pattern (multiple nearby anomalies), escalate
            if nearby_anomalies > 1 and base_sev != 'High':
                base_sev = 'High' if base_sev == 'Medium' else 'Medium'
            
            severity[i] = base_sev
        
        return severity
    
    def _classify_anomaly_type(self, df: pd.DataFrame) -> np.ndarray:
        """
        Classify anomaly into: Point, Contextual, or Collective.
        
        - Point: Single spike (isolated anomaly)
        - Contextual: Detectable by multiple methods, time/activity dependent
        - Collective: Sustained pattern (consecutive anomalies)
        """
        anomaly_types = np.full(len(df), 'None', dtype=object)
        is_anomaly = df['ensemble_anomaly'].fillna(False).values
        
        for i in range(len(df)):
            if not is_anomaly[i]:
                continue
            
            # Check surroundings
            window_start = max(0, i - 3)
            window_end = min(len(is_anomaly), i + 4)
            nearby_count = is_anomaly[window_start:window_end].sum()
            
            # Collective: 3+ consecutive anomalies
            if nearby_count >= 3:
                anomaly_types[i] = 'Collective'
            # Point: isolated spike (1 anomaly, surrounded by normal)
            elif nearby_count == 1 and is_anomaly[i]:
                anomaly_types[i] = 'Point'
            # Contextual: detected by multiple methods, not extreme
            else:
                anomaly_types[i] = 'Contextual'
        
        return anomaly_types
    
    def _build_reason_string(self, df: pd.DataFrame, method_flags: List[np.ndarray]) -> np.ndarray:
        """Build human-readable reason for anomaly."""
        reasons = np.full(len(df), '', dtype=object)
        is_anomaly = df['ensemble_anomaly'].fillna(False).values
        method_names = ['Rule-Based', 'Model-Based', 'Cluster-Based']
        
        for i in range(len(df)):
            if not is_anomaly[i]:
                continue
            
            triggered = []
            for j, method_name in enumerate(method_names):
                if j < len(method_flags) and method_flags[j][i] if hasattr(method_flags[j], '__getitem__') else False:
                    triggered.append(method_name)
            
            reasons[i] = 'Detected by: ' + ', '.join(triggered if triggered else ['Unknown'])
        
        return reasons

# --------------------- Visualization ---------------------
class AnomalyVisualizer:
    def __init__(self):
        self.colors = {'normal':'#1f77b4','threshold':'#ff7f0e','residual':'#d62728','cluster':'#9467bd','ensemble':'#e74c3c'}

    def _apply_layout_template(self, fig: go.Figure, dark_mode: bool):
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        fig.update_layout(template=template)
    
    def plot_ensemble_heart_rate(self, df: pd.DataFrame, title: str = 'Heart Rate - Ensemble Anomaly Detection', dark_mode: bool = False) -> go.Figure:
        """
        Plot heart rate with ensemble anomaly detection results.
        Shows: normal values, confidence intervals, ensemble anomalies with severity coloring.
        """
        df = df.sort_values('timestamp')
        fig = go.Figure()
        
        # Normal values
        normal_mask = ~df.get('ensemble_anomaly', False)
        fig.add_trace(go.Scatter(
            x=df.loc[normal_mask, 'timestamp'],
            y=df.loc[normal_mask, 'heart_rate'],
            mode='lines',
            name='Normal',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Confidence interval (if Prophet data available)
        if 'yhat_lower' in df.columns and 'yhat_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['yhat_upper'],
                mode='lines',
                showlegend=False,
                line=dict(width=0)
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['yhat_lower'],
                mode='lines',
                showlegend=False,
                fill='tonexty',
                fillcolor='rgba(100,150,200,0.15)',
                name='95% Confidence'
            ))
            
            # Predicted trend
            if 'predicted' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['predicted'],
                    mode='lines',
                    name='Predicted Trend',
                    line=dict(color='#3498db', dash='dash', width=2)
                ))
        
        # Ensemble anomalies colored by severity
        if 'ensemble_anomaly' in df.columns and 'severity' in df.columns:
            for severity_level, color in [('High', '#e74c3c'), ('Medium', '#f39c12'), ('Low', '#f1c40f')]:
                mask = (df['ensemble_anomaly']) & (df['severity'] == severity_level)
                if mask.any():
                    anom_df = df[mask]
                    fig.add_trace(go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['heart_rate'],
                        mode='markers',
                        name=f'Anomaly ({severity_level})',
                        marker=dict(color=color, size=10, symbol='X', line=dict(width=2, color='white'))
                    ))
        
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Heart Rate (bpm)',
            hovermode='x unified',
            height=600,
            legend=dict(x=0.01, y=0.99)
        )
        return fig

    def plot_method_comparison(self, df: pd.DataFrame, title: str = 'Detection Method Comparison', dark_mode: bool = False) -> go.Figure:
        """
        Compare anomaly detection results from all three methods.
        Shows side-by-side traces: Rule-Based, Model-Based, Cluster-Based, and Ensemble.
        """
        from plotly.subplots import make_subplots
        
        df = df.sort_values('timestamp')
        
        # Create subplots (4 rows: original + 3 methods + ensemble)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Original Time-Series', 'Rule-Based Detection', 'Model-Based Detection', 'Ensemble Result'),
            vertical_spacing=0.08
        )
        
        # Row 1: Original time-series
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df.get('heart_rate', df.get('step_count', df.get('duration_minutes', df.iloc[:, -1]))),
                      mode='lines', name='Data', line=dict(color='#3498db', width=1.5)),
            row=1, col=1
        )
        
        # Row 2: Threshold anomalies
        if 'threshold_anomaly' in df.columns:
            ta = df[df['threshold_anomaly']]
            if len(ta) > 0:
                fig.add_trace(
                    go.Scatter(x=ta['timestamp'], y=ta.get('heart_rate', ta.get('step_count', ta.get('duration_minutes', ta.iloc[:, -1]))),
                              mode='markers', name='Threshold', marker=dict(color='#f39c12', size=8)),
                    row=2, col=1
                )
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="Threshold", row=2, col=1)
        
        # Row 3: Residual anomalies
        if 'residual_anomaly' in df.columns:
            ra = df[df['residual_anomaly']]
            if len(ra) > 0:
                fig.add_trace(
                    go.Scatter(x=ra['timestamp'], y=ra.get('heart_rate', ra.get('step_count', ra.get('duration_minutes', ra.iloc[:, -1]))),
                              mode='markers', name='Model-Based', marker=dict(color='#e67e22', size=8)),
                    row=3, col=1
                )
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="Model-Based", row=3, col=1)
        
        # Row 4: Ensemble result with severity
        if 'ensemble_anomaly' in df.columns:
            ensemble_anoms = df[df['ensemble_anomaly']]
            if len(ensemble_anoms) > 0:
                for severity_level, color in [('High', '#e74c3c'), ('Medium', '#f39c12'), ('Low', '#f1c40f')]:
                    mask = ensemble_anoms['severity'] == severity_level
                    if mask.any():
                        subset = ensemble_anoms[mask]
                        fig.add_trace(
                            go.Scatter(x=subset['timestamp'], y=subset.get('heart_rate', subset.get('step_count', subset.get('duration_minutes', subset.iloc[:, -1]))),
                                      mode='markers', name=f'Ensemble {severity_level}', marker=dict(color=color, size=10, symbol='X')),
                            row=4, col=1
                        )
        
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(title=title, height=900, hovermode='x unified')
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=1)
        return fig

    def plot_confidence_heatmap(self, df: pd.DataFrame, title: str = 'Anomaly Confidence Heatmap', dark_mode: bool = False) -> go.Figure:
        """
        Heatmap showing confidence scores over time, aggregated by week.
        Darker colors = higher confidence in anomaly detection.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by week and aggregate confidence
        df['week'] = df['timestamp'].dt.to_period('W')
        weekly_stats = df.groupby('week').agg({
            'confidence_score': 'mean',
            'ensemble_anomaly': 'sum'
        }).reset_index()
        
        weekly_stats['week_str'] = weekly_stats['week'].astype(str)
        
        fig = go.Figure(data=go.Heatmap(
            z=[[weekly_stats['confidence_score'].values]],
            x=weekly_stats['week_str'],
            y=['Weekly Avg Confidence'],
            colorscale='RdYlGn_r',
            text=[[f"{x:.2%}" for x in weekly_stats['confidence_score'].values]],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Confidence')
        ))
        
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(
            title=title,
            xaxis_title='Week',
            yaxis_title='Metric',
            height=300
        )
        return fig

    def plot_severity_breakdown(self, df: pd.DataFrame, title: str = 'Severity Breakdown', dark_mode: bool = False) -> go.Figure:
        """
        Bar chart showing distribution of anomalies by severity level.
        """
        if 'severity' not in df.columns:
            return go.Figure()
        
        severity_counts = df[df['ensemble_anomaly'] if 'ensemble_anomaly' in df.columns else True]['severity'].value_counts()
        
        colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#f1c40f', 'Normal': '#2ecc71'}
        trace_colors = [colors.get(sev, '#95a5a6') for sev in severity_counts.index]
        
        fig = go.Figure(data=go.Bar(
            x=severity_counts.index,
            y=severity_counts.values,
            marker=dict(color=trace_colors),
            text=severity_counts.values,
            textposition='auto'
        ))
        
        self._apply_layout_template(fig, dark_mode)
        fig.update_layout(
            title=title,
            xaxis_title='Severity Level',
            yaxis_title='Count',
            height=400,
            showlegend=False
        )
        return fig

    # ===== EXISTING VISUALIZATION METHODS (MAINTAINED) =====
    
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
        self.ensemble_scorer = EnsembleAnomalyScorer()  # MILESTONE 3: Ensemble scoring
        self.visualizer = AnomalyVisualizer()

    def run(self, preprocessed_data: Dict[str, pd.DataFrame], prophet_forecasts: Optional[Dict[str,pd.DataFrame]] = None, feature_matrices: Optional[Dict[str,pd.DataFrame]] = None, cluster_params: Optional[Dict] = None, use_ensemble: bool = True) -> Dict:
        """
        MILESTONE 3 Enhanced Pipeline:
        Runs all three anomaly detection methods, then optionally applies ensemble scoring.
        
        Args:
            preprocessed_data: Dict of metric_name -> dataframe
            prophet_forecasts: Optional pre-computed Prophet forecasts
            feature_matrices: Optional pre-computed feature matrices for clustering
            cluster_params: Clustering method parameters
            use_ensemble: If True, apply ensemble scoring and severity assignment
        
        Returns:
            Dict with results, reports, and ensemble scores
        """
        results = {'data_with_anomalies':{}, 'reports':{}, 'ensemble_summary':{}}
        
        for data_type, df in preprocessed_data.items():
            results['reports'][data_type] = {}
            
            # Threshold detection
            df_th, r_th = self.threshold_detector.detect_anomalies(df, data_type)
            results['reports'][data_type]['threshold'] = r_th
            
            # Residual (Model-based) detection
            if prophet_forecasts and data_type in prophet_forecasts:
                forecast = prophet_forecasts[data_type]
                df_res, r_res = self.residual_detector.detect_from_forecast(df_th, data_type, forecast)
                results['reports'][data_type]['residual'] = r_res
                df_final = df_res
            else:
                df_res, r_res = self.residual_detector.detect_from_forecast(df_th, data_type, None)
                results['reports'][data_type]['residual'] = r_res
                df_final = df_res
            
            # Cluster-based detection
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
            
            # MILESTONE 3: Ensemble scoring
            if use_ensemble:
                df_final = self.ensemble_scorer.score_ensemble(df_final)
                
                # Ensemble summary
                ensemble_summary = {
                    'total_anomalies': int(df_final['ensemble_anomaly'].sum()) if 'ensemble_anomaly' in df_final.columns else 0,
                    'avg_confidence': float(df_final['confidence_score'].mean()) if 'confidence_score' in df_final.columns else 0,
                    'severity_counts': {}
                }
                
                if 'severity' in df_final.columns:
                    for sev in ['Low', 'Medium', 'High']:
                        ensemble_summary['severity_counts'][sev] = int((df_final[df_final['ensemble_anomaly']]['severity'] == sev).sum())
                
                results['ensemble_summary'][data_type] = ensemble_summary
                results['reports'][data_type]['ensemble'] = ensemble_summary
            
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

        # ===== SMART COLUMN DETECTION FUNCTION =====
        def smart_detect_columns(df):
            """Smart column detection for both raw and processed files."""
            df.columns = df.columns.str.lower().str.strip()
            
            # Detect DATE column
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
            
            # Detect VALUE column
            value_col = None
            value_patterns = ['y', 'value', 'steps', 'step_count', 'heart_rate', 'heartrate', 'duration', 'duration_minutes', 
                             'metric', 'val', 'sleep_hours', 'hr', 'bpm', 'count', 'minutesasleep', 'minutes_asleep']
            
            for pattern in value_patterns:
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower == pattern:
                        value_col = col
                        break
                if value_col:
                    break
            
            if not value_col:
                for col in df.columns:
                    col_lower = col.lower()
                    for pattern in value_patterns:
                        if pattern in col_lower and col_lower != 'timestamp' and col_lower != 'date' and 'date' not in col_lower:
                            value_col = col
                            break
                    if value_col:
                        break
            
            return date_col, value_col

        st.sidebar.header('Configuration')
        use_sample = st.sidebar.checkbox('Use sample data (demo)', value=True)
        use_tsfresh = st.sidebar.checkbox('Use TSFresh (if installed)', value=False)
        residual_std = st.sidebar.slider('Residual detection sigma threshold', min_value=1.0, max_value=6.0, value=3.0, step=0.5)
        outlier_threshold = st.sidebar.slider('Cluster outlier threshold (fraction)', min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        cluster_method = st.sidebar.selectbox('Clustering method', options=['kmeans','dbscan','lof'])
        dark_mode = st.sidebar.checkbox('Dark mode', value=False)
        
        # ===== NEW: File size optimization settings =====
        st.sidebar.markdown('---')
        st.sidebar.markdown('**Performance Settings**')
        max_rows = st.sidebar.number_input("Max rows per file (for large datasets)", 5000, 50000, 10000, step=1000)
        
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
        uploaded = st.sidebar.file_uploader('Upload CSV file (processed or raw FitBit)', type=['csv','zip'])

        # Apply dark mode CSS early
        _apply_streamlit_dark_css(dark_mode)

        if use_sample or uploaded is None:
            preprocessed = create_sample_data_with_anomalies()
            st.info('Using sample data (demo).')
        else:
            # ===== OPTIMIZED: Smart file loading with sampling =====
            preprocessed = {}
            try:
                if uploaded.type == 'application/zip':
                    import zipfile
                    z = zipfile.ZipFile(uploaded)
                    for name in z.namelist():
                        if name.endswith('.csv'):
                            df = pd.read_csv(z.open(name), low_memory=False)
                            file_size = len(df)
                            
                            # Auto-sample for performance
                            if file_size > max_rows:
                                st.warning(f"⚠️ {name}: {file_size:,} rows. Sampling to {max_rows:,} rows")
                                df = df.sample(n=max_rows, random_state=42).sort_values(df.columns[0])
                            
                            # Smart column detection
                            date_col, value_col = smart_detect_columns(df)
                            
                            if date_col and value_col:
                                df_clean = pd.DataFrame()
                                try:
                                    df_clean['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
                                    df_clean['value'] = pd.to_numeric(df[value_col], errors='coerce')
                                    df_clean = df_clean.dropna()
                                    
                                    if len(df_clean) > 0:
                                        # Infer type from filename
                                        fname = name.lower()
                                        if 'heart' in fname or 'hr' in fname:
                                            preprocessed['heart_rate'] = df_clean.rename(columns={'value': 'heart_rate'})
                                        elif 'step' in fname:
                                            preprocessed['steps'] = df_clean.rename(columns={'value': 'step_count'})
                                        elif 'sleep' in fname:
                                            preprocessed['sleep'] = df_clean.rename(columns={'value': 'duration_minutes'})
                                        st.success(f"✅ Loaded {name} ({len(df_clean)} records)")
                                except Exception as e:
                                    st.error(f"❌ {name}: Could not parse - {str(e)[:100]}")
                            else:
                                st.error(f"❌ {name}: Could not detect date/value columns. Found: {list(df.columns[:5])}")
                else:
                    # Single CSV file
                    df = pd.read_csv(uploaded, low_memory=False)
                    file_size = len(df)
                    
                    # Auto-sample for performance
                    if file_size > max_rows:
                        st.warning(f"⚠️ {uploaded.name}: {file_size:,} rows. Sampling to {max_rows:,} rows")
                        df = df.sample(n=max_rows, random_state=42).sort_values(df.columns[0])
                    
                    # Smart column detection
                    date_col, value_col = smart_detect_columns(df)
                    
                    if date_col and value_col:
                        df_clean = pd.DataFrame()
                        try:
                            df_clean['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
                            df_clean['value'] = pd.to_numeric(df[value_col], errors='coerce')
                            df_clean = df_clean.dropna()
                            
                            if len(df_clean) > 0:
                                fname = uploaded.name.lower()
                                if 'heart' in fname or 'hr' in fname:
                                    preprocessed['heart_rate'] = df_clean.rename(columns={'value': 'heart_rate'})
                                elif 'step' in fname:
                                    preprocessed['steps'] = df_clean.rename(columns={'value': 'step_count'})
                                elif 'sleep' in fname:
                                    preprocessed['sleep'] = df_clean.rename(columns={'value': 'duration_minutes'})
                                st.success(f"✅ Loaded {uploaded.name} ({len(df_clean)} records)")
                        except Exception as e:
                            st.error(f"❌ Could not parse - {str(e)[:100]}")
                    else:
                        st.error(f"❌ Could not detect date/value columns. Found: {list(df.columns[:5])}")
                        st.info("Expected columns: date (ds, timestamp, activityDate, sleepDate) + value (y, value, heart_rate, steps, duration)")
                        
            except Exception as e:
                st.error('Error reading uploaded file: ' + str(e)[:150])

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
                # MILESTONE 3: Use ensemble scoring
                results = pipeline.run(preprocessed_data=preprocessed, prophet_forecasts=None, feature_matrices=feature_matrices, cluster_params={'method':cluster_method,'n_clusters':4,'outlier_threshold':outlier_threshold}, use_ensemble=True)
                st.session_state['m3_results'] = results
                # Save dark_mode selection for visualizer usage
                st.session_state['m3_dark_mode'] = dark_mode
                st.success('Milestone 3 Ensemble Anomaly Detection Complete!')

        # Show results
        if 'm3_results' in st.session_state:
            results = st.session_state['m3_results']
            dark_mode_setting = st.session_state.get('m3_dark_mode', False)
            
            # MILESTONE 3: Enhanced Summary with Ensemble Stats
            st.markdown('### Ensemble Detection Summary')
            cols = st.columns([1,1,1,1])
            
            total_anoms = sum(results['ensemble_summary'].get(dt, {}).get('total_anomalies', 0) for dt in results['ensemble_summary'])
            avg_confidence = np.mean([results['ensemble_summary'].get(dt, {}).get('avg_confidence', 0) for dt in results['ensemble_summary']])
            
            cols[0].metric('Total Anomalies (Ensemble)', int(total_anoms))
            cols[1].metric('Avg Confidence', f'{avg_confidence:.2%}')
            cols[2].metric('Data Types', len(results['data_with_anomalies']))
            
            # Severity breakdown
            total_high = sum(results['ensemble_summary'].get(dt, {}).get('severity_counts', {}).get('High', 0) for dt in results['ensemble_summary'])
            cols[3].metric('HIGH Severity', int(total_high), delta=None, delta_color='inverse')

            # MILESTONE 3: Tabs for different views
            view_tabs = st.tabs(['Ensemble Results', 'Method Comparison', 'Heatmaps', 'Details'])
            
            with view_tabs[0]:
                st.markdown('#### Ensemble Detection Results')
                for dt, df in results['data_with_anomalies'].items():
                    st.subheader(dt.replace('_',' ').title())
                    
                    # Show ensemble visualization
                    fig = pipeline.visualizer.plot_ensemble_heart_rate(df, dark_mode=dark_mode_setting) if dt == 'heart_rate' else pipeline.visualizer.plot_ensemble_heart_rate(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Severity breakdown chart
                    if 'severity' in df.columns:
                        fig_sev = pipeline.visualizer.plot_severity_breakdown(df, dark_mode=dark_mode_setting)
                        st.plotly_chart(fig_sev, use_container_width=True)
                    
                    # Ensemble statistics
                    ensemble_stats = results['ensemble_summary'].get(dt, {})
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Ensemble Anomalies:** {ensemble_stats.get('total_anomalies', 0)}")
                    col2.write(f"**Avg Confidence:** {ensemble_stats.get('avg_confidence', 0):.3f}")
                    sev_counts = ensemble_stats.get('severity_counts', {})
                    col3.write(f"**High/Med/Low:** {sev_counts.get('High', 0)}/{sev_counts.get('Medium', 0)}/{sev_counts.get('Low', 0)}")
                    
                    # Top anomalies table
                    if 'ensemble_anomaly' in df.columns:
                        anoms = df[df['ensemble_anomaly']].copy()
                        if len(anoms) > 0:
                            anoms_display = anoms[['timestamp', 'ensemble_anomaly', 'confidence_score', 'severity', 'anomaly_type', 'ensemble_reason']].head(10)
                            st.write('**Top Anomalies (by confidence):**')
                            st.dataframe(anoms_display, use_container_width=True)
            
            with view_tabs[1]:
                st.markdown('#### Detection Method Comparison')
                for dt, df in results['data_with_anomalies'].items():
                    st.subheader(f'{dt.replace("_"," ").title()} - Methods Comparison')
                    fig_comp = pipeline.visualizer.plot_method_comparison(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Method breakdown
                    col1, col2, col3, col4 = st.columns(4)
                    reports = results['reports'].get(dt, {})
                    col1.write(f"**Threshold:** {reports.get('threshold', {}).get('anomalies_detected', 0)}")
                    col2.write(f"**Model-Based:** {reports.get('residual', {}).get('anomalies_detected', 0)}")
                    col3.write(f"**Cluster-Based:** {reports.get('cluster', {}).get('anomalies_detected', 0)}")
                    col4.write(f"**Ensemble (2+ votes):** {results['ensemble_summary'].get(dt, {}).get('total_anomalies', 0)}")
            
            with view_tabs[2]:
                st.markdown('#### Anomaly Confidence Heatmaps')
                for dt, df in results['data_with_anomalies'].items():
                    st.subheader(dt.replace('_',' ').title())
                    fig_heat = pipeline.visualizer.plot_confidence_heatmap(df, dark_mode=dark_mode_setting)
                    st.plotly_chart(fig_heat, use_container_width=True)
            
            with view_tabs[3]:
                st.markdown('#### Anomaly Details & Interpretations')
                for dt, df in results['data_with_anomalies'].items():
                    st.subheader(dt.replace('_',' ').title())
                    
                    if 'ensemble_anomaly' in df.columns:
                        anoms = df[df['ensemble_anomaly']].copy()
                        if len(anoms) > 0:
                            st.write(f'**{len(anoms)} anomalies detected**')
                            
                            # Show detailed table
                            display_cols = ['timestamp', 'confidence_score', 'severity', 'anomaly_type', 'ensemble_reason']
                            if dt == 'heart_rate' and 'heart_rate' in df.columns:
                                display_cols.insert(1, 'heart_rate')
                            elif dt == 'steps' and 'step_count' in df.columns:
                                display_cols.insert(1, 'step_count')
                            elif dt == 'sleep' and 'duration_minutes' in df.columns:
                                display_cols.insert(1, 'duration_minutes')
                            
                            available_cols = [c for c in display_cols if c in anoms.columns]
                            st.dataframe(anoms[available_cols], use_container_width=True)
                        else:
                            st.info('No anomalies detected with ensemble method')

            # Exports
            st.markdown('---')
            st.subheader('Exports & Reports')
            
            # Ensemble report
            if st.button('📋 Generate Ensemble Report'):
                report_text = "# Milestone 3: Ensemble Anomaly Detection Report\n\n"
                report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report_text += "## Summary\n"
                report_text += f"- Total Anomalies Detected (Ensemble): {total_anoms}\n"
                report_text += f"- Average Confidence: {avg_confidence:.2%}\n"
                report_text += f"- HIGH Severity: {total_high}\n\n"
                
                report_text += "## Detection Methods Used\n"
                report_text += "1. **Rule-Based**: Medical thresholds + percentile-based\n"
                report_text += "2. **Model-Based**: Prophet forecasting + residual analysis\n"
                report_text += "3. **Cluster-Based**: DBSCAN pattern detection\n"
                report_text += "**Ensemble**: Votes from all 3 methods, confidence ≥ 2/3\n\n"
                
                report_text += "## Severity Levels\n"
                report_text += "- **HIGH**: All 3 methods agree or part of sustained pattern\n"
                report_text += "- **MEDIUM**: 2 methods agree with moderate-high confidence\n"
                report_text += "- **LOW**: Single method triggers with moderate confidence\n\n"
                
                report_text += "## Anomaly Types\n"
                report_text += "- **Point**: Isolated single spike\n"
                report_text += "- **Contextual**: Detectable by multiple methods in specific context\n"
                report_text += "- **Collective**: Sustained pattern of consecutive anomalies\n\n"
                
                report_text += "## Details by Metric\n"
                for dt in results['ensemble_summary']:
                    stats = results['ensemble_summary'][dt]
                    report_text += f"\n### {dt.replace('_', ' ').title()}\n"
                    report_text += f"- Total Anomalies: {stats.get('total_anomalies', 0)}\n"
                    report_text += f"- Avg Confidence: {stats.get('avg_confidence', 0):.3f}\n"
                    sev = stats.get('severity_counts', {})
                    report_text += f"- Severity: HIGH={sev.get('High', 0)} | MEDIUM={sev.get('Medium', 0)} | LOW={sev.get('Low', 0)}\n"
                
                st.download_button(
                    'Download Report (TXT)',
                    data=report_text,
                    file_name=f'ensemble_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                    mime='text/plain'
                )
            
            if st.button('📄 Download JSON Report'):
                json_report = json.dumps(results['reports'], default=str, indent=2)
                st.download_button('Download JSON', data=json_report, file_name=f'ensemble_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', mime='application/json')
            
            if st.button('📊 Download All Anomalies (CSV)'):
                all_dfs = []
                for dt, df in results['data_with_anomalies'].items():
                    if 'ensemble_anomaly' in df.columns:
                        anoms = df[df['ensemble_anomaly']].copy()
                        if len(anoms) > 0:
                            anoms['data_type'] = dt
                            all_dfs.append(anoms)
                if len(all_dfs):
                    combined = pd.concat(all_dfs, ignore_index=True)
                    csv = combined.to_csv(index=False)
                    st.download_button('Download CSV', data=csv, file_name=f'ensemble_anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
                else:
                    st.info('No anomalies to export')

            # Annotation panel
            st.markdown('---')
            st.subheader('Clinical Notes & Annotations')
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