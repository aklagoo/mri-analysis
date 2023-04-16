"""Creates a training dataset."""
import os
import cv2
import tqdm
from sklearn.cluster import DBSCAN
from lib.clustering import get_filtered_slices, get_clusters, DBSCAN_MIN_SAMPLES,\
    DBSCAN_EPS
from lib.utils import load_dataset

SOURCE = "./PatientData-2/PatientData"
PATH_OUTPUT = "./Dataset"
PATH_NOISE = "Noise"
PATH_UNCLASSIFIED = "Unclassified"
PATH_TEMPLATE = "../data/template.png"


def process(df, template, algorithm, count):
    for label, path in tqdm.tqdm(zip(df['Label'], df['Path']), total=len(df)):
        slices = get_filtered_slices(path, template)
        ns, imgs = get_clusters(slices, algorithm)
        out_imgs = [img for n, img in zip(ns, imgs) if n]

        # Write images
        path_out = PATH_UNCLASSIFIED if label else PATH_NOISE
        dir_ = os.path.join(PATH_OUTPUT, path_out)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

        for img in out_imgs:
            path = os.path.join(dir_, "{0:02d}.png".format(count))
            cv2.imwrite(path, img)
            count += 1
        count += 99

    return count


def main():
    # Initialize the template and
    template = cv2.imread("./template.png")
    alg = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    # Load CSVs
    print("Loading CSVs and paths...")
    dataset = load_dataset(SOURCE)

    # Process and generate dataset
    count = 0
    for patient, df in dataset.items():
        print(f"Processing {patient}...")
        count = process(df, template, alg, count)

    print("Completed.")


if __name__ == '__main__':
    main()
