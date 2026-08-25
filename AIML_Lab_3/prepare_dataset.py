"""
AIML Lab 3 - Dataset preparation.

Downloads the PneumoniaMNIST chest X-ray dataset (part of MedMNIST v2, which is
derived from the Kaggle Chest X-Ray Pneumonia dataset) and writes it out as
plain PNG images in one folder per class:

    dataset/
        NORMAL/
        PNEUMONIA/

That is the folder layout the lab expects. The notebook runs this automatically
if the dataset folder is missing, so it normally does not need to be run by hand.

    python prepare_dataset.py
"""

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist_128.npz?download=1"
ARCHIVE = Path(__file__).parent / "pneumoniamnist_128.npz"
DATASET_DIR = Path(__file__).parent / "dataset"

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
IMAGES_PER_CLASS = 1000  # balanced subset, keeps CPU training time reasonable
SEED = 42


def download_archive():
    """Fetch the .npz once and cache it next to this script."""
    if ARCHIVE.exists():
        print(f"Archive already downloaded: {ARCHIVE.name}")
        return
    print("Downloading PneumoniaMNIST (128x128), ~72 MB ...")
    urllib.request.urlretrieve(URL, ARCHIVE)
    print("Download complete.")


def build_folders():
    """Write a balanced subset of the X-rays as PNGs, one folder per class."""
    data = np.load(ARCHIVE)

    # The archive ships pre-split; merge the splits back together so we can make
    # our own 80/20 split in the notebook.
    images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]])
    labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]]).ravel()
    print(f"Total images available: {len(images)}")

    rng = np.random.default_rng(SEED)

    for label, name in CLASS_NAMES.items():
        out_dir = DATASET_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = np.flatnonzero(labels == label)
        if len(idx) > IMAGES_PER_CLASS:
            idx = rng.choice(idx, size=IMAGES_PER_CLASS, replace=False)

        for n, i in enumerate(sorted(idx)):
            Image.fromarray(images[i], mode="L").save(out_dir / f"{name.lower()}_{n:04d}.png")

        print(f"{name}: wrote {len(idx)} images to {out_dir}")


def main():
    download_archive()
    build_folders()
    print(f"\nDataset ready at: {DATASET_DIR}")


if __name__ == "__main__":
    main()
