# Employee Salary Analysis – EDA & Clustering

## Overview
Comprehensive exploratory data analysis and unsupervised learning project examining compensation patterns across a 50-employee dataset from a small Indian company. The analysis investigates salary determinants through statistical analysis, encoding techniques, feature scaling, and K-Means clustering to identify distinct employee segments based on compensation and demographic characteristics.

## Dataset
- **Source:** Kaggle – Indian Employee Salaries Dataset
- **Records:** 50 employees
- **Features:** EmployeeID, Name, Age, Gender, City, Years of Experience, Department, Education Level, Monthly Salary (INR)
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
- Loaded and inspected dataset using Pandas with comprehensive data profiling.
- Verified data completeness: zero missing values across all features.
- Generated frequency distributions and cross-tabulations for categorical variables.

### Feature Engineering & Encoding
- Applied **Ordinal Encoding** to Experience_Years and Education_Level.
- Implemented **One-Hot Encoding** for nominal categorical variables (Department, City, Gender).
- Performed **Min-Max Scaling** (0-1 normalization) on all features for clustering.

### Analysis
- **Univariate Analysis**: Examined salary distributions, departmental budgets, and demographic breakdowns.
- **Bivariate & Multivariate Analysis**: Explored correlations between Education, Experience, and Salary.
- **Unsupervised Learning (K-Means)**: Segmented employees into distinct groups based on compensation and demographic characteristics.

## Key Findings
- **Department Analysis:** Marketing accounts for 30.5% of total salary budget; IT follows with 18.6%.
- **Education Impact:** Master's degree holders earn the highest average salary (₹86,258).
- **Geographic Distribution:** Delhi houses 30% of the workforce, followed by Hyderabad (24%).
- **Clustering**: Identified distinct segments enabling targeted HR strategies for compensation.

## Project Structure
```
3_employee-salary-eda-clustering/
├───dataset
│   └───employee_salary_dataset.csv
├───graphs
└───notebook
    └───salary_analysis.ipynb
```

## Skills Demonstrated
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Encoding (Ordinal & One-Hot)
- Feature Scaling
- Unsupervised Machine Learning (K-Means Clustering)
- Statistical Analysis
