# AIML Lab 2 - Exploratory Data Analysis of Healthcare Data

EDA on the **Pima Indians Diabetes** dataset (768 records) to understand
patient health trends, check data quality, and study relationships between
health parameters before modelling.

## Contents

| File | Description |
|------|-------------|
| `healthcare_eda.ipynb` | The EDA notebook (with all outputs and graphs) |
| `diabetes.csv` | Dataset |

## Steps covered

Load -> inspect structure -> missing values -> duplicates -> summary statistics ->
univariate -> bivariate -> multivariate -> outlier detection -> correlation ->
target analysis -> key findings.

## Key findings

- **Hidden missing data:** Insulin (374) and SkinThickness (227) had many `0`s
  encoding missing readings; imputed with the median.
- **Class imbalance:** 65.1% non-diabetic vs 34.9% diabetic (~1.87 : 1).
- **Strongest predictors of the outcome:** Glucose (0.49), then BMI (0.31) and
  Age (0.24).

## Run

```bash
pip install pandas numpy matplotlib seaborn
jupyter notebook healthcare_eda.ipynb
```
