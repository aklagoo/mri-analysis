import csv
import cv2
import joblib
from sklearn.cluster import DBSCAN
from classification import PATH_MODEL, process_patient
from clustering import PATH_TEMPLATE, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from utils import load_dataset_single, calculate_metrics


if __name__ == '__main__':
    # Initialize objects
    df = load_dataset_single('test_Labels.csv',
                             'testPatient')
    clf = joblib.load(PATH_MODEL)
    template = cv2.imread(PATH_TEMPLATE)
    alg = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    # Calculate metrics
    preds = process_patient(df, alg, clf, template, 0.55)
    metrics = calculate_metrics(preds, df['Label'])

    # Write results
    df['Label'] = preds
    df.to_csv('Results.csv', columns=['IC', 'Label'], index=False)

    # Write metrics
    rows = [
        ["Accuracy", metrics['accuracy']],
        ["Precision", metrics['precision']],
        ["Sensitivity", metrics['sensitivity']],
        ["Specificity", metrics['specificity']],
    ]
    with open("Metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
