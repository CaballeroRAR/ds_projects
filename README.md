# 1. Employee Salary Analysis – Small Company in India

## Overview
Comprehensive exploratory data analysis and unsupervised learning project examining compensation patterns across a 50-employee dataset from a small Indian company. The analysis investigates salary determinants through statistical analysis, encoding techniques, feature scaling, and K-Means clustering to identify distinct employee segments based on compensation and demographic characteristics.

## Dataset
- **Source:** Kaggle – Indian Employee Salaries Dataset
- **Records:** 50 employees
- **Features:** EmployeeID, Name, Age, Gender, City (Bangalore, Chennai, Delhi, Hyderabad, Mumbai), Years of Experience, Department (Finance, HR, IT, Marketing, Operations), Education Level (High School, Bachelor, Master, PhD), Monthly Salary (INR)
- **Data Quality:** Complete dataset with no missing values

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Medium-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Loaded and inspected dataset using Pandas with comprehensive data profiling (shape, info, describe, duplicates check)
- Verified data completeness: zero missing values across all features
- Generated frequency distributions and cross-tabulations for categorical variables (Department, Education, Gender, City)
- Created derived features: salary department ratios and aggregated statistics by multiple groupings

### Feature Engineering & Encoding
- Applied **Ordinal Encoding** to Experience_Years and Education_Level to preserve hierarchical relationships
- Implemented **One-Hot Encoding** for nominal categorical variables (Department, City, Gender) to prepare data for machine learning
- Performed **Min-Max Scaling** (0-1 normalization) on all numerical and encoded features for clustering analysis
- Dropped non-predictive identifiers (EmployeeID, Name) to focus on analytical features

### Analysis
**Univariate Analysis**
- Examined salary distributions by Age using groupby aggregations and visualizations
- Analyzed departmental salary budgets and employee counts across all departments
- Investigated gender distribution within departments and cities
- Evaluated education level distribution and average salaries by degree type

**Bivariate & Multivariate Analysis**
- Explored Department vs. Education Level cross-tabulations to understand workforce composition
- Analyzed salary patterns across Department, Education, and Gender simultaneously
- Investigated average experience years by Education Level and Gender to identify career trajectory patterns
- Examined geographic salary variations across five major Indian cities

**Unsupervised Learning: K-Means Clustering**
- Applied K-Means clustering algorithm on scaled features to segment employees into distinct groups
- Generated employee clusters based on combined characteristics: experience, education, demographics, and salary
- Enabled pattern discovery for workforce segmentation and compensation structure identification

**Visualizations**
- Histograms and distribution plots for salary patterns by age groups
- Bar charts comparing departmental salary allocations and employee counts
- Cross-tabulation heatmaps for categorical variable relationships
- Grouped comparisons visualizing salary differences across multiple dimensions

## Key Findings
- **Department Analysis:** Marketing department accounts for 30.5% of total salary budget with 13 employees; IT follows with 18.6% allocation for 10 employees
- **Education Impact:** Master's degree holders earn highest average monthly salary (₹86,258), contrary to expectations, PhD holders earn less (₹72,944)
- **Gender Patterns:** Female employees with Bachelor's degrees average 10.8 years of experience vs. 6.7 years for males; distribution relatively balanced across departments
- **Geographic Distribution:** Delhi houses 30% of workforce (15 employees), followed by Hyderabad (24%) and Bangalore (22%)
- **Experience Distribution:** Relatively uniform experience levels across education tiers (9-11 years average), with slight variations by gender
- **Clustering Results:** K-Means identified distinct employee segments enabling targeted HR strategies for compensation and retention

## Project Structure
```
1_employee_salary_analysis-kaggle_salary/
├─ dataset/
│  ├─ employee_salary_dataset
├─ graphs/
├─ notebook/
│  ├─ salary_analysis.ipynb

```

## Deliverables
Statistical insights, cross-tabulated workforce profiles, clustering-based employee segmentation models, and actionable visualizations supporting HR compensation benchmarking, departmental budget optimization, talent acquisition strategy, and evidence-based workforce planning for small-to-medium enterprises.

**Skills Demonstrated:** Exploratory Data Analysis (EDA), Feature Engineering, Data Encoding (Ordinal & One-Hot), Feature Scaling, Unsupervised Machine Learning (K-Means Clustering), Statistical Analysis, Data Aggregation & Grouping, Business Analytics

# 2. Titanic Survival Prediction – Machine Learning Classification

## Overview
End-to-end machine learning pipeline for predicting passenger survival on the Titanic. This project applies multiple classification algorithms with hyperparameter tuning and cross-validation to achieve optimal performance on the Kaggle Titanic dataset.

## Dataset
- **Source:** Kaggle – Titanic: Machine Learning from Disaster
- **Records:** 891 training samples, 418 test samples
- **Features:** PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Target:** Survived (0 = No, 1 = Yes)

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_Learn-Medium-2F3D40?style=flat&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Handled missing values in Age, Cabin, and Embarked columns
- Feature engineering: family size, title extraction from names, fare binning
- Encoded categorical variables (Sex, Embarked, Pclass)
- Outlier detection and treatment in Age and Fare

### Exploratory Data Analysis
**Univariate Analysis**
- Distribution analysis of Age, Fare, and Pclass using histograms and box plots

**Bivariate Analysis**
- Survival rates by Sex, Pclass, and Embarked with count plots
- Age distribution by survival status
- Correlation heatmap of numerical features

**Visualizations**
- Count plots showing survival patterns by gender and class
- Box plots comparing fare distribution across passenger classes
- Heatmaps for feature correlation and missing data patterns

### Machine Learning Pipeline
**Models Implemented**
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting (XGBoost, LightGBM)

**Model Optimization**
- Cross-validation (K-Fold) for robust performance evaluation
- Hyperparameter tuning with RandomizedSearchCV
- Feature importance analysis to identify key predictors

**Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves for model comparison
- Confusion matrix analysis

## Key Findings
- Gender was the strongest predictor of survival (females had higher survival rates)
- Passenger class (Pclass) significantly impacted survival probability
- Age and family size showed moderate correlation with survival
- Ensemble methods (Random Forest, Gradient Boosting) outperformed linear models
- Achieved competitive Kaggle leaderboard score through feature engineering and model tuning

## Project Structure
```
2_titanic_machine_learning-kaggle_titanic/
├── train.csv
├── test.csv
├── titanic_analysis.ipynb
├── submission.csv
├── README.md
└── images/
```

## Deliverables
Complete machine learning workflow from data exploration to model deployment, demonstrating classification techniques, hyperparameter optimization, and performance evaluation suitable for production environments.

---
**Skills Demonstrated:** ETL, Feature Engineering, Classification Models, Hyperparameter Tuning, Cross-Validation, Model Evaluation, Kaggle Competition Submission

# 3. Gold Recovery Prediction – Industrial Process Optimization

## Overview
Regression modeling project for predicting gold recovery rates in an industrial mining operation. This project optimizes the flotation process by building predictive models that estimate gold concentration at different purification stages, enabling data-driven process improvements and cost reduction.

## Dataset
- **Source:** TripleTen Bootcamp – Gold Recovery Industrial Dataset
- **Records:** ~16,000 process measurements
- **Features:** Flotation reagent feeds, air amounts, fluid levels, metal concentrations (Au, Ag, Pb)
- **Target:** Gold recovery percentage at rougher and final stages

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_Learn-Medium-2F3D40?style=flat&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Cleaned multi-stage process data with Pandas: handled missing values and anomalies
- Feature engineering: calculated recovery rates using metallurgical formulas
- Removed outliers in concentration measurements
- Split data by process stages (rougher, primary cleaner, final)

### Exploratory Data Analysis
**Univariate Analysis**
- Distribution analysis of metal concentrations and recovery rates
- Time series visualization of process parameters

**Bivariate Analysis**
- Correlation between feed characteristics and recovery efficiency
- Impact of reagent dosages on gold concentration
- Stage-by-stage recovery analysis with scatter plots

**Visualizations**
- Line plots tracking concentration changes across purification stages
- Heatmaps for feature correlation matrices
- Box plots comparing recovery distributions by operational parameters

### Machine Learning Pipeline
**Models Implemented**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Model Optimization**
- Cross-validation with custom sMAPE (symmetric Mean Absolute Percentage Error) metric
- Hyperparameter tuning for ensemble methods
- Feature importance analysis to identify key process variables

**Evaluation Metrics**
- sMAPE for rougher and final recovery stages
- MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
- R² score for model fit assessment

## Key Findings
- Feed particle size and reagent concentrations were strongest predictors of recovery
- Rougher stage performance significantly influenced final gold output
- Ensemble methods (Random Forest, Gradient Boosting) achieved lowest sMAPE scores
- Identified optimal process parameter ranges for maximizing recovery efficiency
- Model predictions enable proactive adjustments to flotation conditions

## Project Structure
```
3_gold_recovery_machine_learning-tripleten_gold recovery/
├── gold_recovery_full.csv
├── gold_recovery_train.csv
├── gold_recovery_test.csv
├── gold_recovery_analysis.ipynb
├── README.md
└── images/
```
## Deliverables
End-to-end regression pipeline demonstrating industrial process optimization through machine learning. Models provide actionable insights for metallurgical engineers to improve gold extraction efficiency and reduce operational waste.

---
**Skills Demonstrated:** Regression Modeling, Feature Engineering, Industrial Process Analysis, Custom Metrics (sMAPE), Cross-Validation, Ensemble Methods, Domain-Specific Problem Solving

# 4. Time Series Analysis – Taxi Demand Forecasting

## Overview
Time series forecasting project predicting hourly taxi demand at airport locations. This project applies statistical modeling and machine learning techniques to analyze temporal patterns, seasonal trends, and build predictive models that optimize fleet allocation and operational planning.

## Dataset
- **Source:** TripleTen Bootcamp – Historical Taxi Order Data
- **Records:** Several months of hourly taxi orders
- **Features:** Timestamp, number of orders
- **Target:** Taxi order volume for next hour

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
- Resampled time series data to hourly intervals with Pandas
- Handled missing timestamps and interpolated gaps
- Created lag features and rolling statistics (moving averages, rolling std)
- Split data chronologically for train/validation/test sets

### Exploratory Data Analysis
**Time Series Decomposition**
- Trend analysis using moving averages
- Seasonal pattern identification (daily, weekly cycles)
- Stationarity tests (Augmented Dickey-Fuller test)

**Temporal Analysis**
- Hourly demand patterns across days of week
- Peak hour identification and capacity planning insights
- Correlation between time-based features and order volume

**Visualizations**
- Line plots showing demand trends over time
- Seasonal decomposition plots (trend, seasonality, residuals)
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots
- Heatmaps for hourly/daily demand patterns

### Machine Learning Pipeline
**Models Implemented**
- ARIMA (AutoRegressive Integrated Moving Average)
- Linear Regression with time features
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)

**Feature Engineering**
- Lag features (previous 1, 2, 3, 24 hours)
- Rolling mean and standard deviation windows
- Time-based features: hour, day of week, month
- Holiday and weekend indicators

**Model Optimization**
- Cross-validation with time series split
- Hyperparameter tuning for tree-based models
- Feature importance analysis for interpretability

**Evaluation Metrics**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Key Findings
- Clear daily and weekly seasonality patterns in taxi demand
- Peak hours identified: morning (7-9 AM) and evening (5-8 PM)
- Weekend demand differs significantly from weekday patterns
- Gradient Boosting models outperformed traditional ARIMA for complex patterns
- Lag features and rolling statistics significantly improved prediction accuracy
- Model enables 1-hour ahead forecasting with <10% RMSE threshold

## Project Structure
```
4_time_series_analysis-tripleten_bootcamp/
├── taxi_orders.csv
├── time_series_analysis.ipynb
├── README.md
└── images/
```

## Deliverables
Complete time series forecasting pipeline demonstrating statistical analysis, feature engineering, and predictive modeling for demand forecasting. Results support operational decisions for fleet management, driver scheduling, and resource optimization.

---
**Skills Demonstrated:** Time Series Analysis, Seasonal Decomposition, ARIMA Modeling, Lag Feature Engineering, Temporal Pattern Recognition, Demand Forecasting, Cross-Validation for Time Series

