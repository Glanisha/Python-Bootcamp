# AIML Lab 1 — Disease Dataset Preprocessing Pipeline

End-to-end data preprocessing pipeline for a **Diabetes prediction** task,
built on the real **Pima Indians Diabetes** dataset (768 patient records).

## Pipeline stages

`Define → Collect → Clean → Integrate → Transform → Validate → Store → Predict`

| Stage | What happens |
|-------|--------------|
| **Define**    | Disease = Diabetes, goal = Prediction, binary target |
| **Collect**   | Reads two real sources: `clinical_records.csv` (EHR) + `lab_results.csv` (lab) |
| **Clean**     | Converts impossible `0`s to missing, median-imputes, drops duplicates, IQR-caps outliers |
| **Integrate** | Merges the two sources on `patient_id` |
| **Transform** | Encodes the target, engineers a `risk_score`, standard-scales numeric features |
| **Validate**  | Asserts no missing values, no duplicate patients, binary target |
| **Store**     | Writes the analysis-ready `diabetes_clean.csv` |
| **Predict**   | Trains a Logistic Regression classifier (~71% test accuracy) |

## Multiple data sources

The raw file `pima_raw.csv` is split **once** into two source files so the
pipeline genuinely reads from and integrates multiple sources — mirroring data
arriving from separate hospital systems that share a `patient_id`.

## Files

| File | Description |
|------|-------------|
| `disease_data_preprocessing.py` | The full pipeline |
| `pima_raw.csv` | Raw Pima Indians Diabetes dataset (input) |
| `clinical_records.csv` | Source A — EHR (generated) |
| `lab_results.csv` | Source B — lab results (generated) |
| `diabetes_clean.csv` | Final cleaned, analysis-ready output (generated) |

The three generated CSVs are recreated automatically on each run.

## Run

```bash
pip install pandas numpy scikit-learn
python disease_data_preprocessing.py
```
