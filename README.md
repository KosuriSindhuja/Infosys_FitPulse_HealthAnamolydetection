# Infosys FitPulse Health Anomaly Detection

## Overview
FitPlus is a comprehensive health analytics platform that processes Fitbit data to detect anomalies, analyze health patterns, and generate professional health insights reports. The project includes data processing pipelines, statistical analysis, interactive dashboards, and theory-based health report generation.

---

## Directory Structure
```
├── data/
│   ├── raw/                    # Raw Fitbit data files
│   │   ├── fitbit_heartrate_raw.csv
│   │   ├── fitbit_sleep_raw.csv
│   │   └── fitbit_steps_raw.csv
│   └── processed/              # Preprocessed datasets
│       ├── preprocessed_data.csv
│       ├── train.csv
│       └── test.csv
│
├── module2_outputs/            # Analysis results
│   ├── daily_heart_rate.csv
│   ├── daily_sleep.csv
│   ├── daily_steps.csv
│   ├── clusters.csv
│   ├── features_tsfresh.csv
│   ├── task1_hr_forecast_table.csv
│   ├── task2_sleep_forecast_table.csv
│   ├── task3_events_impact.csv
│   └── task3_forecast_*.csv
│
├── outputs/                    # Documentation
│   ├── data_dictionary.md
│   └── preprocessing_summary.md
│
├── src/                        # Source code
│   ├── dashboard.py           # Main Streamlit dashboard (4 tabs)
│   ├── clean_merge.py         # Data cleaning & merging
│   ├── feature_extract.py     # Feature extraction
│   ├── preprocessing_summary.py
│   ├── visualize_anomaly.py   # Anomaly visualization
│   ├── extract_bson.py        # BSON data extraction
│   ├── check_bson.py          # BSON validation
│   └── preview_data.py        # Data preview utilities
│
└── README.md                   # This file
```

---

## Getting Started

### Installation
1. Clone the repository
2. Install Python 3.8+
3. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn tsfresh streamlit plotly prophet fpdf kaleido
```

### Quick Start
```bash
# Launch dashboard
streamlit run src/dashboard.py

# Open in browser
http://localhost:8501
```

---

## Requirements
- **Python**: 3.8+
- **Core Libraries**:
  - `pandas` - Data processing
  - `numpy` - Numerical calculations
  - `scikit-learn` - ML algorithms (Isolation Forest, KMeans, DBSCAN)
  - `streamlit` - Web dashboard
  - `plotly` - Interactive graphs
  - `prophet` - Time-series forecasting
  - `fpdf` - PDF generation
  - `kaleido` - Graph to image conversion
  - `matplotlib` - Visualization support
  - `tsfresh` - Time series feature extraction

---

## Dashboard Tabs

### Tab 1: Home
Interactive daily health summary:
- Rotating motivational health messages
- Compact 2-3 word health status tag with emoji derived from processed data
- Today's metrics (Heart Rate, Steps, Sleep)
- Daily anomaly summary
- 7-day sparkline charts

### Tab 2: Analysis
Interactive health data exploration driven entirely by **user-uploaded CSVs**:
- **Flexible Uploads**: Accepts up to 6 raw or processed CSVs (including train/test, preprocessed_data, raw Fitbit exports)
- **Automatic Metric Detection**:
  - Detects date/time columns (ds, timestamp, date, datetime, activityDate, logDate, sleepDate, etc.)
  - Extracts daily series for **Heart Rate**, **Steps**, and **Sleep** based on column names (heart_rate, heartrate, hr, bpm, steps, step_count, sleep_hours, minutesAsleep, etc.)
- **Metric Selection**: User chooses which metric(s) to analyze; if a selected metric is **not present** in the uploaded data, the dashboard clearly reports that it is missing instead of failing
- **Visualizations** (per selected metric):
  - Time Series: Line chart with markers
  - Scatter Plot: Value distribution
  - Prophet Trend: Trend decomposition
  - Seasonality: Weekly/yearly patterns
- **Summary Statistics**: Mean, median, std dev, min, max, range

### Tab 3: Anomalies
Advanced multi-method anomaly detection supporting **both raw and processed CSV files**, including files that contain
multiple metrics (heart rate, steps, sleep) together and reusing data already uploaded in the Analysis tab:

#### Features:
- **Shared Data Source with Analysis Tab**
  - Uses the same uploaded CSV files and extracted daily series as the Analysis tab
  - User simply selects which metrics (Heart Rate, Steps, Sleep) to run anomaly detection on; if a selected metric has no data,
    the dashboard clearly reports that instead of failing
- **Smart Multi-Metric Detection** 
  - Auto-detects date columns: ds, timestamp, date, datetime, activityDate, logDate, sleepDate, etc.
  - From each uploaded CSV, automatically extracts separate daily series for **Heart Rate**, **Steps**, and **Sleep** when
    those columns are present (e.g., heart_rate, steps, sleep_hours, minutesAsleep), so a single uploaded file can yield multiple
    anomaly traces
  - Also supports legacy processed files in ds/y format with metric inferred from filename
- **Performance Optimization**
  - Automatic sampling for large files (>10,000 rows) to prevent hanging
  - Efficient vectorized anomaly detection
  - Prophet caching to avoid recomputation
  - Clustering limited to 5,000 samples for speed
- **5 Detection Methods**:
  1. Statistical (±2.5σ threshold)
  2. Contextual (sudden spikes)
  3. Prophet prediction intervals
  4. KMeans clustering
  5. DBSCAN clustering
- **Severity Classification**: Low/Medium/High
- **Visualizations**:
  - Timeline scatter plot with color severity
  - Anomaly frequency bar chart by date
  - Prophet forecast with anomalies
  - Results table with export to CSV
- **Data Formats Supported**:
  - ✅ Processed CSV (ds, y columns)
  - ✅ Raw FitBit Heart Rate (activityDate, value)
  - ✅ Raw FitBit Steps (activityDate, steps)
  - ✅ Raw FitBit Sleep (sleepDate, duration_minutes)
  - ✅ Any CSV with date + value columns

#### Known Issues Fixed (v2.1):
| Issue | Cause | Solution |
|-------|-------|----------|
| Laptop hanging on large files | Loading full dataset at once | Auto-sampling to max 10,000 rows |
| Missing date column error (sleep_raw) | Column name not recognized (sleepDate) | Enhanced pattern matching for raw files |
| Heart rate raw not loading | activityDate column not detected | Added flexible column detection patterns |
| Slow visualization with >50K rows | Prophet fitting on all data | Reduced sample size for clustering, caching |

#### Optimization Details:
```python
# 1. Smart Column Detection (Multiple Patterns)
date_patterns = ['ds', 'timestamp', 'date', 'time', 'datetime', 
                 'activitydate', 'logdate', 'sleepdate']
value_patterns = ['y', 'value', 'heart_rate', 'steps', 'sleep_hours', 
                  'duration', 'duration_minutes', 'hr', 'bpm']

# 2. Auto-Sampling for Performance
if file_size > max_rows:  # default: 10,000
    df = df.sample(n=max_rows, random_state=42).sort_values(date_col)

# 3. Vectorized Anomaly Detection (Fast)
anomaly_scores = np.zeros(len(df))
outliers = (df['y'] < mean - 2.5*std) | (df['y'] > mean + 2.5*std)
anomaly_scores[outliers] += 0.4  # vectorized assignment

# 4. Prophet Caching (Avoid Recomputation)
if metric_name not in st.session_state.get('prophet_cache', {}):
    model = Prophet(...)  # fit once
    forecast = model.predict(...)
    st.session_state['prophet_cache'][metric_name] = forecast

# 5. Clustering Optimization (Limited Sample)
sample_size = min(len(df), 5000)  # max 5000 for clustering
sample_indices = np.random.choice(len(df), sample_size, replace=False)
km = KMeans(n_clusters=..., n_init=5)  # reduced from 10 for speed
```

#### How to Use with Raw Files:
1. Upload raw FitBit CSV (or processed CSV)
2. System auto-detects columns automatically
3. Choose max rows setting for large files (default 10,000)
4. Click "Analyze" → Results display instantly
5. Download anomaly results as CSV

✅ **Fast, smooth visualization even with large datasets!**

### Tab 4: Reports - Comprehensive Health Report Generator

**Professional theory-based PDF reports from CSV data**

#### Features:
✅ CSV upload (preprocessed or raw format), or reuse of files already uploaded in the Analysis tab  
✅ Auto-detect columns and metric type  
✅ Comprehensive statistical analysis  
✅ Multi-method anomaly detection  
✅ Professional PDF export with graphs  
✅ Theory-based explanations  

#### Workflow:
```
1. Upload CSV Data (in the Reports tab or in the Analysis tab)
   ↓
2. Auto-detect Columns & Metric Type
   ↓
3. Configure Report Settings
   ↓
4. Generate Comprehensive Analysis
   ↓
5. Download Professional PDF Report
```

#### Supported Data Formats:

**Auto-Detected Column Names:**
- Date: 'ds', 'timestamp', 'date', 'time', 'datetime'
- Value: 'y', 'value', 'metric', 'val', 'heart_rate', 'steps', 'sleep_hours', 'hr'

**Metric Detection (from filename):**
- "heart" or "hr" → Heart Rate (bpm)
- "step" → Steps (steps/day)
- "sleep" → Sleep Duration (hours)

#### Report Structure - Page 1 (Text):

**1. Report Header**
- Metric type title
- Generation timestamp
- Total data points analyzed
- Date range covered

**2. Analysis Overview**
- Key Statistics: Mean, Median, Std Dev, Min, Max, Range
- 7-Day Trend: Increasing/Decreasing direction and magnitude
- Current Status: Comparison with health standards

**3. Metric-Specific Analysis**

**Heart Rate Analysis:**
- Health Status: Normal (60-100) / Elevated (>100) / Low (<60)
- Cardiovascular Fitness Assessment
- Consistency Score: Based on data variability
- Recent Trend Analysis
- **Theory**: Why resting heart rate reflects cardiovascular health
- **Medical Context**: Standard health ranges and implications
- **Emergency Alerts**: >120 or <40 bpm warnings

**Steps/Activity Analysis:**
- Activity Level: Sedentary → Low Active → Somewhat Active → Active → Very Active
- WHO Standard: 10,000 steps daily recommendation
- Consistency Score: Activity pattern stability
- Recent Trend: Activity level changes
- **Theory**: Role of daily movement in disease prevention
- **Health Impact**: Physical fitness and longevity benefits

**Sleep Analysis:**
- Sleep Quality Rating: Poor / Good / Excellent
- Duration: Assessed vs 7-9 hour medical standard
- Consistency Score: Sleep schedule regularity
- Circadian Rhythm: Sleep pattern stability
- **Theory**: Sleep's role in immune function and cognition
- **Health Impact**: Why consistent sleep matters for wellness

**4. Anomaly Detection & Analysis**

**Detected Anomalies:**
- Total count and percentage of flagged data
- Top 5 most recent anomalies:
  - Date of occurrence
  - Value with deviation magnitude (std devs from mean)
  - Severity level classification

**WHY Anomalies Occurred - Context-Specific Reasoning:**

*Heart Rate:*
- Stress or anxiety during measurement
- Physical exercise or recovery period
- Caffeine or medication effects
- Sleep quality issues
- Potential health concerns if persistent

*Steps:*
- Rest days or illness recovery
- Intense exercise sessions
- Travel or schedule changes
- Weather conditions
- Work-related activity shifts

*Sleep:*
- Work stress or deadline pressure
- Travel or schedule disruption
- Acute health issues or illness
- Environmental changes
- Lifestyle modifications

**What Can Be Done:**
1. **Monitor & Track**: Identify pattern formation
2. **Investigate**: Correlate with activities, stress, diet
3. **Prevent**: Address root causes systematically
4. **Consult**: Seek professional help if persistent
5. **Adjust**: Implement targeted lifestyle improvements

**5. Health Recommendations & Action Plan**

**Immediate Actions** (Same Day):
- Continue regular monitoring
- Note lifestyle correlations
- Maintain positive patterns

**Short-Term Improvements** (1-2 Weeks):
- Identify anomaly triggers
- Adjust daily routines
- Build consistency in habits

**Long-Term Strategy** (1-3 Months):
- Work toward optimal health targets
- Establish sustainable practices
- Track improvement trends

**Professional Consultation Guidance:**
- When to seek healthcare provider
- Red flag symptoms requiring attention
- Documentation for medical visits

**6. Disclaimer**
Clear statement that report is informational only and not substitute for professional medical advice.

#### Report Structure - Page 2 (Visualizations):

**Analysis Graph:**
- Time-series line chart of all values
- Mean reference line (green dashed)
- Anomalies highlighted in red (X markers)
- Full date range displayed
- Clear axes with value units

**Distribution Graph:**
- Histogram with 30-bin frequency distribution
- Shows data spread and natural clusters
- Identifies normal vs. outlier ranges
- X-axis: Values | Y-axis: Frequency

#### PDF Output Details:
- **Format**: Professional multi-page PDF
- **Filename**: `HealthReport_[MetricType]_[YYYYMMDD_HHMMSS].pdf`
- **Pages**: 2 (Text report + Visualizations)
- **Features**: Embedded graphs, professional formatting, timestamps

---

## Technical Details

### Character Encoding & PDF Compatibility

Comprehensive Unicode-to-ASCII character sanitization ensures all PDF content renders correctly:

**Handled Replacements:**
- Greek letters: σ→std_dev, μ→mean, π→pi, θ→theta, α→alpha, etc.
- Arrows: ↑→[UP], ↓→[DOWN], →→[RIGHT], ←→[LEFT]
- Mathematical: ±→+/-, ×→x, ÷→/, √→sqrt, ∞→inf, ≈→approx
- Symbols: •→-, ©→(C), ®→(R), ™→(TM)
- Currencies: €→EUR, £→GBP, ¥→YEN
- Quotes: Smart quotes→Regular quotes (', ")
- Dashes: Various dashes→Standard dash (-)
- Alerts: ⚠️→[ALERT], ✓→[OK]

**FPDF Best Practices Implemented:**
- Font: Arial (ASCII-compatible)
- Font Size: 9pt body, 12pt titles
- Line Height: 4.5 for proper spacing
- Text Width: 190mm (A4 with 10mm margins)
- Margins: 10mm all sides
- Automatic page breaks when content exceeds page height

### Anomaly Detection Ensemble

**5 Complementary Methods:**
1. **Statistical**: Outliers beyond ±2.5 standard deviations
2. **Contextual**: Sudden spikes (change > 2σ)
3. **Prophet**: Values outside prediction confidence intervals
4. **KMeans**: Points in small clusters identified as anomalies
5. **DBSCAN**: Points not in dense regions flagged as anomalies

**Severity Scoring:**
- Low: 0.3-0.7 (potential anomaly)
- Medium: 0.7-0.85 (likely anomaly)
- High: 0.85-1.0 (strong anomaly)

### Data Processing Pipeline

```
Raw Data (CSV) → Cleaning → Merging → Validation →
  ↓
Feature Extraction → Normalization → Statistical Analysis →
  ↓
Anomaly Detection → Visualization Generation → Report Creation →
  ↓
PDF Assembly → User Download
```

---

## Outputs & Results

### Module 2 Outputs
- `daily_heart_rate.csv`: Daily average heart rate metrics
- `daily_steps.csv`: Daily step counts and activity
- `daily_sleep.csv`: Daily sleep duration data
- `clusters.csv`: KMeans clustering analysis results
- `features_tsfresh.csv`: Time-series features extracted
- `task1_hr_forecast_table.csv`: Heart rate forecast data
- `task2_sleep_forecast_table.csv`: Sleep forecast data
- `task3_events_impact.csv`: Holiday/event impact analysis
- `task3_forecast_*.csv`: Forecast comparisons

### Generated Documentation
- `outputs/data_dictionary.md`: Column name descriptions and data types
- `outputs/preprocessing_summary.md`: Data cleaning and transformation steps
- Generated PDF Reports: Professional health analysis with recommendations

---

## Troubleshooting

### Dashboard Won't Start
```bash
# Verify Python
python --version

# Check Streamlit installed
pip list | grep streamlit

# Run with debug output
streamlit run src/dashboard.py --logger.level=debug
```

### File Upload Issues
- Ensure CSV format (comma-separated values)
- Use standard column names (ds, date, timestamp, value, y)
- Verify numeric values in data column
- Check date format (YYYY-MM-DD recommended)

### PDF Generation Errors
- Check for Unicode characters in source data
- Ensure file write permissions in working directory
- Verify sufficient RAM for graph generation
- Update FPDF library: `pip install --upgrade fpdf2`

### Performance Issues
- Use smaller datasets for testing
- Close other applications to free RAM
- Disable Prophet seasonality for very large datasets
- Clear browser cache

---

## Code Quality & Testing

✅ **Syntax Validation**: `python -m py_compile src/dashboard.py`  
✅ **No Unhandled Exceptions**: Robust error handling throughout  
✅ **User-Friendly Feedback**: Clear error messages and guidance  
✅ **Modular Design**: Clean function separation and reusability  
✅ **Well-Documented**: Comprehensive code comments and docstrings  
✅ **PDF Character Safety**: All Unicode handled properly  

---

## Recent Improvements

1. **Reports Tab Redesign**: Removed forecasting, added report generation
2. **Character Encoding Fix**: Implemented comprehensive Unicode-to-ASCII sanitization
3. **PDF Optimization**: Added automatic page breaks and margin handling
4. **Documentation**: Consolidated all .md files into comprehensive README
5. **Error Handling**: Robust validation with user-friendly feedback
6. **Layout Optimization**: Ensured all text stays within page boundaries

---

## Support & Contributing

For issues or questions:
1. Review error messages and troubleshooting section
2. Verify data format and column names match documentation
3. Ensure all dependencies are installed correctly
4. Check file permissions in working directory
5. Consult generated error logs for detailed information

---

## Legal & Disclaimer

**FitPlus Health Insights Dashboard is provided for informational and educational purposes ONLY.**

### Important Notice:
This dashboard is **NOT a substitute for professional medical diagnosis or treatment.**

**Always consult with a qualified healthcare provider** for:
- Medical concerns or symptoms
- Persistent anomalies in health metrics
- Emergency alerts or critical readings
- Pre-existing health conditions
- Personalized medical advice

### Liability:
- Generated reports are informational tools only
- Users assume all responsibility for health-related decisions
- Healthcare provider consultation is strongly recommended
- Project developers assume no liability for health-related outcomes

### Data Privacy:
- Uploaded data is processed locally
- No data is stored on external servers
- Users maintain full data ownership
- PDF reports are generated on user's machine

---

## License & Attribution

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full terms.

This project is part of the Infosys FitPulse Health Anomaly Detection initiative.

### Technologies Used:
- **Streamlit**: Open-source app framework
- **Plotly**: Interactive graphing library
- **Prophet**: Time series forecasting
- **scikit-learn**: Machine learning algorithms
- **pandas/numpy**: Data processing and analysis
- **FPDF**: PDF document generation

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Production Ready
