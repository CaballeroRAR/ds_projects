# 1. Employee Salary Analysis – Small Company in India 

## Project Overview
This project conducts an exploratory data analysis (EDA) of employee salary data from a small company in India. The objective is to uncover compensation trends based on role, experience, department, and education level, delivering data-driven insights into salary structures and key influencing factors.

## Dataset
- **File:** `employee_salary_dataset.csv`
- **Source:** Kaggle – Public dataset on Indian employee salaries
- **Records:** Approximately 500 employee entries
- **Key Features:**
  - Demographic: Age, Years of Experience
  - Employment: Department, Job Role, Education Level
  - Target: Salary

## Tools & Libraries
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- Jupyter Notebook

## Analysis Steps

### 1. Data Loading and Initial Inspection
- Loaded the dataset using Pandas.
- Inspected structure, data types, and summary statistics.
- Identified missing values and potential outliers.

### 2. Data Cleaning and Preparation
- Addressed missing data appropriately.
- Treated outliers in numerical columns, specifically Salary.
- Standardized categorical variables for consistency.

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Examined distributions of Salary, Age, and Experience.
- **Bivariate Analysis:**
  - Analyzed correlation between Salary and Years of Experience.
  - Compared Salary distributions across Departments and Job Roles.
  - Evaluated the impact of Education Level on compensation brackets.
- **Visualizations:**
  - Histograms and box plots for salary distributions.
  - Scatter plots illustrating the experience-salary relationship.
  - Bar charts comparing average salaries by role and department.

## Key Insights
- Identified the highest and lowest-paying roles and departments within the company.
- Quantified a strong positive correlation between years of experience and salary.
- Highlighted how education level influences starting salary and growth trajectory.
- Provided clear, visual summaries of salary trends to support HR and managerial decision-making.

## Project Structure
salary_dataset/
|-- employee_salary_dataset.csv # Raw dataset
|-- salary_analysis.ipynb # Main analysis notebook
|-- README.md # Project documentation
|-- images/ # Folder containing visualizations and charts

# Conclusion
This project demonstrates a structured approach to analyzing real-world salary data, 
from initial data wrangling to insightful visualization. It serves as a practical example of 
using Python for business analytics and can be extended with predictive modeling or deeper statistical tests.

# 2. Titanic: Machine Learning from Disaster

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
