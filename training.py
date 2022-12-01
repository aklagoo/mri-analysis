"""
PROCESS:
1. Load all CSVs and associated images.
2. Load dataset and count

"""
import cv2
import glob
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from typing import List
from os.path import join as path_join
from utils import preprocess_batch, calculate_metrics


PATH_NEGATIVE = 'Dataset/Noise'
PATH_POSITIVE = 'Dataset/Unclassified'
TRAIN_SIZE = 0.7


def load_images_dir(dir_: str) -> List[np.ndarray]:
    """Extracts all images from a directory."""
    imgs = []
    for path in glob.glob(dir_):
        imgs.append(cv2.imread(path))
    return imgs


def load_dataset(path_positive, path_negative):
    # Extract pre-processed data
    x_pos = preprocess_batch(load_images_dir(path_join(path_positive, '*.png')))
    x_neg = preprocess_batch(load_images_dir(path_join(path_negative, '*.png')))

    # Perform train test splitting
    y_pos = np.ones(x_pos.shape[0])
    y_neg = np.zeros(x_neg.shape[0])
    x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(
        x_pos, y_pos, train_size=TRAIN_SIZE)
    x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(
        x_neg, y_neg, train_size=TRAIN_SIZE)

    # Group training and testing datasets
    x_train_ = np.concatenate((x_train_pos, x_train_neg))
    y_train_ = np.concatenate((y_train_pos, y_train_neg))
    x_test_ = np.concatenate((x_test_pos, x_test_neg))
    y_test_ = np.concatenate((y_test_pos, y_test_neg))
    x_train_, y_train_ = sklearn.utils.shuffle(x_train_, y_train_, random_state=0)
    x_test_, y_test_ = sklearn.utils.shuffle(x_test_, y_test_, random_state=0)

    return x_train_, y_train_, x_test_, y_test_


if __name__ == '__main__':
    # Load dataset
    print("Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(PATH_POSITIVE, PATH_NEGATIVE)

    # Train the SVM
    print("Training the model...")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)

    # Calculate metrics
    print("Evaluating metrics")
    preds_train = clf.predict(x_train)
    print(f"    Training metrics: {calculate_metrics(preds_train, y_train)}")
    preds_test = clf.predict(x_test)
    print(f"    Testing metrics: {calculate_metrics(preds_test, y_test)}")

    # Write model to file
    print("Saving the model")
    joblib.dump(clf, 'svm.joblib')

    print("Completed")
