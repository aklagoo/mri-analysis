"""Contains functions for performing predictions on a dataset."""
import joblib
import cv2
from sklearn.cluster import DBSCAN
import clustering as cls
import tqdm
import utils

SOURCE = "./PatientData-2/PatientData"
PATH_MODEL = "./svm.joblib"
PATH_OUTPUT = "./Dataset"
PATH_NOISE = "Noise"
PATH_UNCLASSIFIED = "Unclassified"


def process_single(path, template_, algorithm, classifier):
    slices = cls.get_filtered_slices(path, template_)
    ns, slices = cls.get_clusters(slices, algorithm)
    slices = [img for n, img in zip(ns, slices) if n]
    slices = utils.preprocess_batch(slices)
    if slices is None:
        return None
    preds = classifier.predict(slices)
    return preds


def process_patient(data, dbscan, classifier, template_, min_threshold):
    preds = []
    for path in tqdm.tqdm(data['Path']):
        single_preds = process_single(path, template_, dbscan, classifier)

        # If no images are found, predict noise.
        if single_preds is None:
            preds.append(0.)
            continue

        pos = (single_preds == 1).sum()
        neg = (single_preds == 0).sum()
        ratio = pos / (pos + neg)
        preds.append(1. if ratio >= min_threshold else 0.)

    return preds


if __name__ == '__main__':
    # Initialize objects
    dataset = utils.load_dataset(SOURCE)
    clf = joblib.load(PATH_MODEL)
    template = cv2.imread(cls.PATH_TEMPLATE)
    alg = DBSCAN(eps=cls.DBSCAN_EPS, min_samples=cls.DBSCAN_MIN_SAMPLES)

    # Calculate metrics
    for patient, df in dataset.items():
        print(f"Processing {patient}...")
        metrics = utils.calculate_metrics(process_patient(df, alg, clf,
                                                          template, 0.55))
        print(f"    Metrics: {metrics}")
