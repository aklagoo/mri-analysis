"""Utility functions for IO operations, metrics, and pre-processing."""
import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


IMG_H, IMG_W = 128, 128


def load_dataset(source: str):
    dataset = {}
    patients = [f"Patient_{i + 1}" for i in range(5)]
    for patient in patients:
        csv_path = os.path.join(source, f"{patient}_Labels.csv")
        dir_path = os.path.join(source, patient)
        df = load_dataset_single(csv_path, dir_path)
        dataset[patient] = df
    return dataset


def load_dataset_single(csv_path, dir_path):
    # Load paths and labels
    df = pd.read_csv(csv_path)
    df['Path'] = [os.path.join(dir_path, f"IC_{ic}_thresh.png")
                  for ic in df['IC']]
    df['Label'] = [1 if d > 0 else 0 for d in df['Label']]
    return df


def preprocess_batch(imgs: List[np.ndarray]) -> np.ndarray:
    """Performs padding, scaling and conversion to grayscale."""
    outs = []
    for img in imgs:
        img_ = np.zeros((IMG_H, IMG_W, 3))
        hh, ww, _ = img.shape
        h = (IMG_H - hh) // 2
        w = (IMG_W - ww) // 2
        img_[h:h + hh, w:w + ww] = img
        outs.append(img_[:, :, 0].reshape(-1))
    return np.stack(outs) if len(outs) else None


def calculate_metrics(preds, labels):
    """Calculate metrics from predictions and labels."""
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }