# AIML Lab 3 — AI-Based Medical Diagnosis using a CNN

A simple AI-based medical diagnosis system that classifies chest X-ray images as
**Normal** or **Pneumonia** using a Convolutional Neural Network.

## Dataset

**PneumoniaMNIST** (MedMNIST v2), derived from the Kaggle Chest X-Ray Pneumonia
dataset — 5,856 real chest X-rays already provided at 128 × 128. The lab expects
a separate folder per class, so `prepare_dataset.py` writes a balanced subset of
**1000 images per class** as PNGs:

```
dataset/
    NORMAL/       normal_0000.png ...
    PNEUMONIA/    pneumonia_0000.png ...
```

The download (~72 MB) and the `dataset/` folder are git-ignored — they are
regenerated on demand, so nothing large is committed.

## Files

| File | Description |
|------|-------------|
| `medical_image_cnn.ipynb` | The lab notebook, Parts A–D (with all outputs and graphs) |
| `prepare_dataset.py` | Downloads the X-rays and writes them into per-class folders |

## What each part does

| Part | Contents |
|------|----------|
| **A** | Load the dataset, count the images in each class, display 6 sample X-rays with labels |
| **B** | Resize to 128 × 128, convert to arrays, normalise pixels to 0–1, assign labels, 80/20 stratified train-test split |
| **C** | Build a CNN (2 × [Conv2D + MaxPooling] → Flatten → Dense → sigmoid output), compile, train for 5 epochs, plot the accuracy graph |
| **D** | Predict on the test set, report accuracy, show the confusion matrix and classification report, display test images with actual vs predicted labels |

## Model

```
Conv2D(16, 3x3) + ReLU  ->  MaxPooling(2x2)
Conv2D(32, 3x3) + ReLU  ->  MaxPooling(2x2)
Flatten  ->  Dense(64, ReLU)  ->  Dense(1, sigmoid)
```

Optimizer `adam`, loss `binary_crossentropy`, 5 epochs, batch size 32.

## Results

| Metric | Value |
|--------|-------|
| Training samples | 1600 |
| Testing samples | 400 |
| Test accuracy | **94.50 %** (378 / 400 correct) |

Confusion matrix (rows = actual, columns = predicted):

```
                Predicted
                NORMAL  PNEUMONIA
Actual NORMAL      198          2
       PNEUMONIA    20        180
```

## Run

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow
jupyter notebook medical_image_cnn.ipynb
```

The notebook calls `prepare_dataset.py` itself the first time it is run, so the
dataset does not have to be downloaded manually.
