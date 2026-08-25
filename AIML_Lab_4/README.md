# AIML Lab 4 — AI-Based Medical Prognosis using Logistic Regression

Predicts a **future** clinical outcome — whether a heart failure patient
survives the follow-up period — from their clinical records, using Logistic
Regression.

Where Lab 3 answered *what condition does this patient have now?*, this lab
answers *what is likely to happen to them next?*

## Dataset

**Heart Failure Clinical Records** (UCI, 299 patients) — committed as
`heart_failure_clinical_records_dataset.csv`, no download needed.

| | |
|---|---|
| Patients | 299 |
| Features | 12 |
| Survived (`DEATH_EVENT` = 0) | 203 (67.9%) |
| Died (`DEATH_EVENT` = 1) | 96 (32.1%) |
| Missing values | 0 |

## Files

| File | Description |
|------|-------------|
| `heart_failure_prognosis.ipynb` | The lab notebook, Parts A–D (with all outputs and graphs) |
| `heart_failure_clinical_records_dataset.csv` | Dataset |

## Features used

Seven clinical features: `age`, `ejection_fraction`, `serum_creatinine`,
`serum_sodium`, `diabetes`, `high_blood_pressure`, `smoking`.

**`time` is deliberately excluded.** It records the length of the follow-up
period, which is only known *after* the outcome — a patient who died early has a
short follow-up time. Training on it leaks the answer into the model and inflates
accuracy to ~85%, and it is not available for a new patient at admission.

## What each part does

| Part | Contents |
|------|----------|
| **A** | Load the data, record and feature counts, first five patients, class distribution, missing-value check |
| **B** | Select the 7 clinical features, split into X and y, standard-scale (fitted on the training set only), 80/20 stratified train-test split |
| **C** | Train Logistic Regression, predict, report train/test accuracy, plot the coefficients, run one example patient |
| **D** | Test accuracy, confusion matrix, classification report, actual vs predicted per patient, and prognosis for two new patients |

## Results

| Metric | Value |
|--------|-------|
| Training samples | 239 |
| Testing samples | 60 |
| Training accuracy | 76.57 % |
| **Test accuracy** | **68.33 %** (41 / 60 correct) |

Confusion matrix (rows = actual, columns = predicted):

```
                  Predicted
                  Survived  Death
Actual Survived         35      6
       Death            13      6
```

The strongest predictors, from the model coefficients: a **low ejection
fraction** (−0.82) and **high serum creatinine** (+0.80) push the prediction
towards death, followed by age (+0.47).

### The false-negative problem

13 of the 19 patients who died were predicted to survive — the error a clinician
would least want. Because only 32% of the dataset died, the model scores well by
leaning towards "Survived". Re-fitting with `class_weight="balanced"` fixes most
of it:

| | Plain model | Balanced |
|---|---|---|
| Test accuracy | 68.33 % | **73.33 %** |
| False negatives | 13 | **7** |
| Recall on death events | 0.32 | **0.63** |

The notebook keeps the plain model for the rest of the experiment, since Part C
specifies a standard Logistic Regression model.

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook heart_failure_prognosis.ipynb
```
