# 1. Employee Salary Analysis – Small Company in India

## Overview
Exploratory data analysis of employee compensation patterns in a small Indian company. This project examines how role, experience, department, and education level influence salary structures, delivering actionable insights for HR and management decision-making.

## Dataset
- **Source:** Kaggle – Indian Employee Salaries
- **Records:** 50 employees
- **Features:** Age, Years of Experience, Department, Job Role, Education Level, Salary

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Cleaned dataset with Pandas: handled missing values and outlier treatment
- Standardized categorical variables for consistency
- Validated data types and statistical distributions

### Analysis
**Univariate Analysis**
- Distribution analysis of Salary, Age, and Experience using histograms and box plots

**Bivariate Analysis**
- Experience vs. Salary correlation with scatter plots
- Department and Role salary comparisons via bar charts
- Education Level impact on compensation brackets

**Visualizations**
- Box plots for salary distribution by department and role
- Scatter plots showing experience-salary relationship
- Heatmaps for feature correlations

## Key Findings
- Strong positive correlation between years of experience and salary (quantified)
- Identified highest/lowest-paying departments and roles
- Education level significantly impacts starting salary and growth trajectory
- Clear salary bands emerged across job roles and seniority levels

## Project Structure
```
1_employee_salary_analysis-kaggle_salary/
├── employee_salary_dataset.csv
├── salary_analysis.ipynb
├── README.md
└── images/
```

## Deliverables
Visual summaries and statistical insights supporting HR compensation strategy, role benchmarking, and talent acquisition planning.

**Skills Demonstrated:** Data Cleaning, EDA, Statistical Analysis, Data Visualization, Business Analytics

# 2. Titanic: Machine Learning from Disaster
---

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue) ![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project tackles the famous **Kaggle Titanic - Machine Learning from Disaster** competition. The goal is to build a predictive model that answers the question: *"What sorts of people were more likely to survive?"* using passenger data (e.g., name, age, gender, socio-economic class). This serves as an introductory project to applied machine learning and data science workflows.

## Project Structure
kaggle_titanic/
├── titanic.ipynb # Main Jupyter notebook with full analysis
├── data/
│ ├── train.csv # Training dataset (provided by Kaggle)
│ ├── test.csv # Test dataset (provided by Kaggle)
│ └── gender_submission.csv # Example submission file
├── submission.csv # Generated predictions for Kaggle
├── README.md # This file
└── requirements.txt # Python dependencies

## Goals

- Perform **exploratory data analysis (EDA)** to understand relationships and patterns.
- Apply **data preprocessing** including handling missing values, feature engineering, and encoding.
- Build, compare, and evaluate multiple **machine learning models**.
- Optimize model performance through **hyperparameter tuning**.
- Generate a valid **submission file** for the Kaggle competition.

## Key Steps

### 1. Data Exploration & Visualization
- Examine survival rates by features: `Sex`, `Pclass`, `Age`, `Fare`, `Embarked`.
- Identify relationships and correlations using statistical analysis and visualizations.
- Create informative plots (bar charts, histograms, heatmaps, etc.) to uncover insights.

### 2. Feature Engineering & Preprocessing
- Extract titles from passenger names (`Mr`, `Mrs`, `Miss`, `Master`, etc.).
- Create family size and alone status features from `SibSp` and `Parch`.
- Fill/impute missing values in `Age`, `Embarked`, and `Fare` columns.
- Convert categorical features into numerical representations via encoding.

### 3. Modeling
Multiple classification algorithms are implemented and evaluated:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

### 4. Evaluation & Submission
- Models are evaluated using **accuracy** and **cross-validation** scores.
- The best-performing model is selected for final predictions.
- Predictions on the test set are saved as `submission.csv` for Kaggle submission.

### Main dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
