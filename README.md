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
