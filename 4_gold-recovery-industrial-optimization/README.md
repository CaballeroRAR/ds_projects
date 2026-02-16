# Gold Recovery Prediction – Industrial Process Optimization

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
- Cleaned multi-stage process data: handled missing values and anomalies.
- Feature engineering: calculated recovery rates using metallurgical formulas.
- Stage-by-stage data splitting (rougher, primary cleaner, final).

### Exploratory Data Analysis
- **Metal Concentrations**: TRACKED concentration changes across purification stages.
- **Correlation Analysis**: Impact of reagent dosages and feed characteristics on efficiency.

### Machine Learning Pipeline
- **Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
- **Evaluation**: Cross-validation with custom **sMAPE** (symmetric Mean Absolute Percentage Error) metric.
- **Optimization**: Hyperparameter tuning for ensemble methods.

## Key Findings
- Feed particle size and reagent concentrations were the strongest predictors of recovery.
- Rougher stage performance significantly influenced final gold output.
- Ensemble methods achieved the lowest sMAPE scores.
- Identified optimal process parameters for maximizing extraction efficiency.

## Project Structure
```
4_gold-recovery-industrial-optimization/
├───datasets
│   ├───gold_recovery_full.csv
│   ├───gold_recovery_test.csv
│   └───gold_recovery_train.csv
└───notebook
    └───gold_recovery_notebook.ipynb
```

## Skills Demonstrated
- Regression Modeling
- Industrial Process Analysis
- Custom Metrics (sMAPE)
- Cross-Validation
- Ensemble Methods
- Domain-Specific Problem Solving
