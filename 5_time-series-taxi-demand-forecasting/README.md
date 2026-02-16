# Taxi Demand Forecasting – Time Series Analysis

## Overview
Time series forecasting project predicting hourly taxi demand at airport locations. This project applies statistical modeling and machine learning techniques to analyze temporal patterns, seasonal trends, and build predictive models that optimize fleet allocation and operational planning.

## Dataset
- **Source:** TripleTen Bootcamp – Historical Taxi Order Data
- **Records:** Several months of hourly taxi orders
- **Features:** Timestamp, number of orders
- **Target:** Taxi order volume for the next hour

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_Learn-Medium-2F3D40?style=flat&logo=scikitlearn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Medium-182625?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Resampled data to hourly intervals.
- Created **lag features** and **rolling statistics** (moving averages, rolling std).
- Chronological data splitting for train/validation/test sets.

### Exploratory Data Analysis
- **Decomposition**: Trend analysis and seasonal pattern identification (daily, weekly).
- **Stationarity**: Augmented Dickey-Fuller tests.
- **Autocorrelation**: ACF and PACF plots for feature identification.

### Machine Learning Pipeline
- **Models**: ARIMA, Linear Regression with time features, Random Forest, XGBoost/LightGBM.
- **Feature Engineering**: Lag features (1, 2, 3, 24h), rolling mean/std windows, time-based features (hour, day of week).
- **Optimization**: Time series cross-validation and hyperparameter tuning.

## Key Findings
- Clear daily and weekly seasonality patterns identified.
- Peak hours: Morning (7-9 AM) and Evening (5-8 PM).
- Gradient Boosting models outperformed traditional ARIMA for complex patterns.
- Model enables 1-hour ahead forecasting within target RMSE thresholds.

## Project Structure
```
5_time-series-taxi-demand-forecasting/
├───dataset
│   └───taxi.csv
└───notebook
    └───time_series_analysis.ipynb
```

## Skills Demonstrated
- Time Series Analysis
- Seasonal Decomposition
- ARIMA & ML Regression
- Lag Feature Engineering
- Temporal Pattern Recognition
- Demand Forecasting
