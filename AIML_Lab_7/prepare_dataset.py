"""Builds medical_reviews.csv from the Drugs.com patient review dataset.

Each review carries a 1 to 10 rating, which is mapped to a sentiment label:

    rating >= 8   Positive
    rating <= 4   Negative
    otherwise     Neutral

A balanced sample of 1200 reviews per class is written out so the file stays
small enough to keep in the repo.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/lewtun/drug-reviews/resolve/main/test.jsonl"
RAW = Path(__file__).parent / "drug_reviews_raw.jsonl"
OUT = Path(__file__).parent / "medical_reviews.csv"

PER_CLASS = 1200
SEED = 42


def label(rating):
    if rating >= 8:
        return "Positive"
    if rating <= 4:
        return "Negative"
    return "Neutral"


def download():
    if RAW.exists():
        print("Raw file already downloaded.")
        return
    print("Downloading drug review dataset (~32 MB) ...")
    urllib.request.urlretrieve(URL, RAW)


def build():
    records = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "Review": row["review"].strip('"'),
                "Condition": row.get("condition"),
                "Rating": row["rating"],
            })

    df = pd.DataFrame(records)
    df["Sentiment"] = df["Rating"].apply(label)
    print("Available reviews:", len(df))
    print(df["Sentiment"].value_counts().to_string())

    balanced = [g.sample(PER_CLASS, random_state=SEED) for _, g in df.groupby("Sentiment")]
    sample = (pd.concat(balanced)
                .sample(frac=1, random_state=SEED)
                .reset_index(drop=True))

    sample[["Review", "Condition", "Rating", "Sentiment"]].to_csv(OUT, index=False)
    print("\nWrote", OUT.name, "with", len(sample), "reviews")
    print(sample["Sentiment"].value_counts().to_string())


if __name__ == "__main__":
    download()
    build()
