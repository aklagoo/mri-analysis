import glob
import os
import tqdm
import clustering
from cv2 import imread
from sklearn.cluster import DBSCAN


# Output paths
PATH_SLICES = "./Slices"
PATH_BOUND = "./Boundaries"
PATH_CLUSTERS = "./Clusters"


if __name__ == '__main__':
    # List all files
    files = glob.glob("./testPatient/*_thresh.png")

    # Load the template
    template = imread("./template.png")

    # Process each image
    print("Processing images...")
    for file in tqdm.tqdm(files):
        # Prepare output path
        a, i, _ = os.path.basename(file)[:-4].split('_')
        prefix = "{0}_{1:03d}".format(a, int(i))

        # Generate slices and write
        images = clustering.get_filtered_slices(file, template)
        clustering.write_images(images, PATH_SLICES, prefix)

        # Detect clusters and write
        alg = DBSCAN(eps=clustering.DBSCAN_EPS,
                     min_samples=clustering.DBSCAN_MIN_SAMPLES)
        counts, images_cl = clustering.get_clusters(images, alg)
        clustering.write_images(images_cl, PATH_CLUSTERS, prefix)
        clustering.write_cluster_counts(counts, PATH_CLUSTERS, prefix)

    print("Completed.")
