# Infosys FitPulse Health Anomaly Detection

## Overview
This repository contains code and data for detecting health anomalies using Fitbit data. The project processes raw health data, extracts features, performs clustering, and forecasts health metrics. It is structured for modular analysis and reporting.

## Directory Structure
- `data/`
  - `raw/`: Contains raw Fitbit data files (`fitbit_sleep_raw.csv`, `fitbit_steps_raw.csv`).
  - `processed/`: Preprocessed datasets for training and testing (`preprocessed_data.csv`, `train.csv`, `test.csv`).
- `module2_outputs/`: Outputs from module 2, including clustering results, daily metrics, feature extraction, and forecasting tables/plots.
- `outputs/`: Documentation and summaries (`data_dictionary.md`, `preprocessing_summary.md`).
- `src/`: Source code for data processing, feature extraction, visualization, and dashboarding.
  - `check_bson.py`: Checks BSON data integrity.
  - `clean_merge.py`: Cleans and merges datasets.
  - `dashboard.py`: Dashboard visualization.
  - `extract_bson.py`: Extracts data from BSON files.
  - `feature_extract.py`: Feature extraction logic.
  - `preprocessing_summary.py`: Summarizes preprocessing steps.
  - `preview_data.py`: Previews data samples.
  - `visualize_anomaly.py`: Visualizes detected anomalies.

## Getting Started
1. Clone the repository.
2. Install required Python packages (see below).
3. Run scripts in `src/` for data processing and analysis.

## Requirements
- Python 3.8+
- Common packages: pandas, numpy, matplotlib, scikit-learn, tsfresh

Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn tsfresh
```

## Usage
- Place raw Fitbit data in `data/raw/`.
- Run preprocessing and feature extraction scripts from `src/`.
- Review outputs in `module2_outputs/` and documentation in `outputs/`.

## Outputs
- Clustering results, daily metrics, feature tables, and forecast visualizations.
- Data dictionary and preprocessing summary for reference.
