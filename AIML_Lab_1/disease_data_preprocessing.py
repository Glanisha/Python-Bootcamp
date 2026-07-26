"""
Diabetes dataset preprocessing pipeline.

Steps: define -> collect -> clean -> integrate -> transform -> validate -> store -> predict
Dataset: Pima Indians Diabetes (768 records).

Run: python disease_data_preprocessing.py
Needs pima_raw.csv in the same folder. Install: pip install pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

DISEASE = "Diabetes"
GOAL = "Prediction"
TARGET_COLUMN = "diabetic"

RAW_FILE = "pima_raw.csv"
CLINICAL_FILE = "clinical_records.csv"
LAB_FILE = "lab_results.csv"
OUTPUT_FILE = "diabetes_clean.csv"

# In this dataset 0 is used where the reading is actually missing.
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness",
                   "insulin", "bmi"]


def build_source_files():
    # Split the raw file into a clinical source and a lab source so we
    # have two datasets to integrate later.
    if os.path.exists(CLINICAL_FILE) and os.path.exists(LAB_FILE):
        print("[Setup] Source files already exist - skipping split")
        return

    raw = pd.read_csv(RAW_FILE)
    raw.columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
    ]
    raw.insert(0, "patient_id", range(1, len(raw) + 1))

    clinical = raw[["patient_id", "pregnancies", "age",
                    "blood_pressure", "outcome"]].copy()
    clinical["diagnosis"] = clinical["outcome"].map(
        {0: "Not Diabetic", 1: "Diabetic"})
    clinical = clinical.drop(columns=["outcome"])
    clinical.to_csv(CLINICAL_FILE, index=False)

    lab = raw[["patient_id", "glucose", "skin_thickness",
               "insulin", "bmi", "diabetes_pedigree"]].copy()
    lab.to_csv(LAB_FILE, index=False)

    print(f"[Setup] Split {len(raw)} records into "
          f"{CLINICAL_FILE} + {LAB_FILE}")


def collect_data():
    clinical = pd.read_csv(CLINICAL_FILE)
    lab = pd.read_csv(LAB_FILE)
    print(f"[Collect] Clinical: {clinical.shape[0]} rows | "
          f"Lab: {lab.shape[0]} rows")
    return clinical, lab


def clean_data(clinical, lab):
    clinical = clinical.copy()
    lab = lab.copy()

    b_c, b_l = clinical.shape[0], lab.shape[0]
    clinical = clinical.drop_duplicates(subset="patient_id")
    lab = lab.drop_duplicates(subset="patient_id")
    print(f"[Clean] Removed duplicates -> clinical: {b_c - clinical.shape[0]}, "
          f"lab: {b_l - lab.shape[0]}")

    for col in ZERO_AS_MISSING:
        if col in clinical.columns:
            clinical[col] = clinical[col].replace(0, np.nan)
        if col in lab.columns:
            lab[col] = lab[col].replace(0, np.nan)

    # Fill missing values with the median.
    for col in ["age", "blood_pressure"]:
        clinical[col] = clinical[col].fillna(clinical[col].median())
    for col in ["glucose", "skin_thickness", "insulin",
                "bmi", "diabetes_pedigree"]:
        lab[col] = lab[col].fillna(lab[col].median())

    for col in ["insulin", "skin_thickness", "bmi"]:
        lab[col] = _cap_outliers_iqr(lab[col])

    total_missing = int(clinical.isna().sum().sum() + lab.isna().sum().sum())
    print(f"[Clean] Missing values remaining after imputation: {total_missing}")
    return clinical, lab


def _cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - k * iqr, upper=q3 + k * iqr)


def integrate_data(clinical, lab):
    merged = pd.merge(clinical, lab, on="patient_id", how="inner")
    print(f"[Integrate] Merged dataset: {merged.shape[0]} rows, "
          f"{merged.shape[1]} columns")
    return merged


def transform_data(df):
    df = df.copy()

    df[TARGET_COLUMN] = df["diagnosis"].map(
        {"Not Diabetic": 0, "Diabetic": 1})

    df["risk_score"] = (
        (df["glucose"] > 125).astype(int)
        + (df["bmi"] >= 30).astype(int)
        + (df["age"] > 50).astype(int)
    )

    numeric_features = ["pregnancies", "age", "blood_pressure", "glucose",
                        "skin_thickness", "insulin", "bmi", "diabetes_pedigree"]
    scaler = StandardScaler()
    scaled_cols = [f"{c}_scaled" for c in numeric_features]
    df[scaled_cols] = scaler.fit_transform(df[numeric_features])

    final_cols = ["patient_id"] + scaled_cols + ["risk_score", TARGET_COLUMN]
    print(f"[Transform] Engineered risk_score, encoded target, "
          f"scaled {len(numeric_features)} numeric features")
    return df[final_cols]


def validate_data(df):
    checks = {
        "no_missing_values": int(df.isna().sum().sum()) == 0,
        "no_duplicate_patients": df["patient_id"].is_unique,
        "target_is_binary": set(df[TARGET_COLUMN].unique()).issubset({0, 1}),
        "has_rows": len(df) > 0,
    }
    for name, passed in checks.items():
        print(f"[Validate] {name:<22} -> {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise ValueError("Validation failed - dataset is not analysis-ready.")
    print("[Validate] All checks passed. Dataset is analysis-ready.")
    return df


def store_data(df, path=OUTPUT_FILE):
    df.to_csv(path, index=False)
    print(f"[Store] Saved {df.shape[0]} rows x {df.shape[1]} cols -> {path}")
    return path


def predict_diabetes(df, test_size=0.2):
    feature_cols = [c for c in df.columns
                    if c not in ("patient_id", TARGET_COLUMN)]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[Predict] Trained Logistic Regression on {len(X_train)} rows, "
          f"tested on {len(X_test)}")
    print(f"[Predict] Test accuracy: {acc:.3f}\n")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual:0", "actual:1"],
        columns=["pred:0", "pred:1"]).to_string())
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))
    return model


def main():
    print(f"=== {DISEASE} preprocessing pipeline (goal: {GOAL}) ===\n")

    build_source_files()
    clinical, lab = collect_data()
    clinical, lab = clean_data(clinical, lab)
    merged = integrate_data(clinical, lab)
    transformed = transform_data(merged)
    validated = validate_data(transformed)
    store_data(validated)

    print("\n=== Final analysis-ready dataset (first 8 rows) ===")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(validated.head(8).round(3).to_string(index=False))
    print(f"\nClass balance -> {validated[TARGET_COLUMN].value_counts().to_dict()}")

    print("\n=== Prediction: Logistic Regression ===")
    predict_diabetes(validated)


if __name__ == "__main__":
    main()
