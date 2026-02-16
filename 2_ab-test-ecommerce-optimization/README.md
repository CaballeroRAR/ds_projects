# A/B Test - E-commerce Website Optimization

## Overview
This project evaluates the performance of a new e-commerce webpage against an old design through A/B testing. By analyzing user click-through rates and conversion rates, it determines whether the new page provides a statistically significant improvement. The analysis employs hypothesis testing to provide a data-driven recommendation on whether to launch the new design.

## Dataset
- **Source**: Internal A/B Test Data
- **Records**: User session data with page exposure and conversion events.
- **Features**: `user_id`, `timestamp`, `group` (con/exp), `click` (0 or 1).

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Medium-28403D?style=flat&logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Medium-182625?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Cleaning
- Aligned treatment and control groups to ensure accurate test exposure.
- Removed duplicate user entries to avoid data skew.
- Calculated observed conversion rates for both groups.

### Hypothesis Testing
- **Null Hypothesis (H₀)**: The conversion rate of the new page is less than or equal to the old page.
- **Alternative Hypothesis (H₁)**: The conversion rate of the new page is greater than the old page.
- Simulated conversion rates under the null hypothesis to create a sampling distribution.
- Calculated the p-value by comparing the observed difference in conversion rates to the simulated distribution.

### Statistical Analysis
- Used z-tests and computed z-scores and p-values to validate findings from the simulation.
- Established a significance level (alpha) of 0.10.

## Key Findings
- **Statistically Significant**: The A/B test yielded a p-value of approximately 0 (Z-score: 59.44), far below the 0.10 alpha threshold.
- **Conversion Impact**: Experimental variant (`exp`) achieved a **61.16% conversion rate** vs. 19.89% in control (`con`)—a **41.27% absolute increase**.
- **Practical Significance**: This effect size vastly exceeds the 1.07% MDE threshold.
- **Recommendation**: Immediate implementation of the experimental variant as it demonstrates substantial business value.

## Project Structure
```
2_ab-test-ecommerce-optimization/
├─ dataset/
│  ├─ ab_test_click_data.csv
├─ notebook/
│  ├─ a-b_test.ipynb
```

## Skills Demonstrated
- A/B Testing
- Hypothesis Testing
- Statistical Significance
- P-value Calculation
- Z-test
- Data-driven Decision Making
