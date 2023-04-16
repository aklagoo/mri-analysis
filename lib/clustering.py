"""Contains brain scan slice and boundary extraction functions."""
import glob
import os.path
from collections import namedtuple
import numpy as np
from typing import List
import cv2
from sklearn.cluster import DBSCAN
import tqdm
import csv
import colorsys

DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS = 2

PATH_SLICES = "./Output/Slices"
PATH_BOUND = "./Output/Boundaries"
PATH_CLUSTERS = "./Output/Clusters"
PATH_TEMPLATE = "../data/template.png"

PARAM_FILTER_THRESHOLD = 100
PARAM_BOUND_THRESHOLD = 5

DRAW_CLUSTERS_WHITE = 0
DRAW_CLUSTERS_COLOR = 1
DRAW_CLUSTERS_SOURCE = 2

Point = namedtuple("Point", "x y")
Size = namedtuple("Size", "width height")


def generate_colors(n_):
    """Generates equally distributed colors."""
    if not n_:
        return []
    colors_ = []
    unit = 1 / n_
    for i in range(n_):
        color_ = colorsys.hsv_to_rgb(unit * i, 0.8, 1)
        colors_.append(np.array(color_) * 255)

    return colors_


def write_images(slices: List[np.ndarray], base_dir: str, prefix: str):
    """Writes a batch of images to an output path."""
    dir_ = os.path.join(base_dir, prefix)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    for i, slice_ in enumerate(slices):
        path = os.path.join(dir_, "{0:02d}.png".format(i))
        cv2.imwrite(path, slice_)


def write_cluster_counts(counts: List[int], base_dir: str, prefix: str):
    """Writes the counts of clusters to a CSV at a specified path."""
    # Generate output path
    dir_ = os.path.join(base_dir, prefix)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    path = os.path.join(dir_, "counts.csv")

    # Prepare output
    rows = zip([f"{i:02d}" for i in range(len(counts))], counts)

    # Write to CSV
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SliceNumber", "ClusterCount"])
        writer.writerows(rows)


def get_filtered_slices(file: str, template: np.ndarray) -> List[np.ndarray]:
    """Extracts all non-empty slices from a single image.

    Args:
        file: Path to image file.
        template: The template for Rs.

    Returns:
        List of image slices.
    """
    # Read the image and extract template width and height
    img = cv2.imread(file)
    _, tw, th = template.shape[::-1]

    # Detect all instances of 'R's
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res > 0.8)

    # Group coordinates by rows
    coordinates = []
    last_y = -1
    for x, y in zip(xs, ys):
        if y != last_y:
            coordinates.append([])
            last_y = y
        coordinates[-1].append(Point(x, y))

    # Calculate a default width of a slice for the last 'R' in each cell
    default_width = coordinates[0][1][0] - coordinates[0][0][0] - tw

    # Extract image slices
    slices = []
    for i, row in enumerate(coordinates[1:], 1):
        for j, current in enumerate(row):
            # Calculate the top left corner
            above = coordinates[i - 1][j]
            top_left = Point(above.x + tw, above.y + th)

            # Calculate the bottom right corner
            if j == len(coordinates[i]) - 1:
                bottom_right = Point(current.x + tw + default_width, current.y)
            else:
                right = coordinates[i][j + 1]
                bottom_right = Point(right.x, current.y)
            slice_ = img[top_left.y:bottom_right.y, top_left.x:bottom_right.x]

            # If the image doesn't contain much, get rid of it.
            if slice_.sum() < PARAM_FILTER_THRESHOLD:
                continue

            # Otherwise, append the image
            slices.append(slice_)

    return slices


def draw_boundaries(images: List[np.ndarray]):
    """Detects contours and draws boundaries on a batch of images.

    The boundaries are made on the passed images, modifying them in the process.

    Args:
        images: A batch of images.
    """
    for img in images:
        # Convert to B/W and apply thresholding
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, PARAM_BOUND_THRESHOLD, 255,
                                      cv2.THRESH_BINARY)

        # Detect and draw boundaries
        contours, hierarchy = cv2.findContours(image=img_thresh,
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
        _ = cv2.drawContours(img, contours, -1, color=(255, 255, 0),
                             thickness=1, lineType=cv2.LINE_AA)


def cluster(image_: np.ndarray, alg: DBSCAN, color: int = DRAW_CLUSTERS_WHITE, draw_valid: bool = True) -> (int, np.ndarray):
    """Detects clusters and returns a filtered image and the count of large
    clusters.

    This functions extracts non-gray pixels, performs clustering using DBScan
    and re-maps the clustered coordinates onto the base image, acting as a
    binary mask. The count of all clusters larger than 135 pixels is also
    calculated. The original image is not modified.

    Args:
        image_: A single image.
        alg: An instance of a DBSCAN algorithm object.

    Returns:
        The count of large clusters and a cluster-masked image extracted from
        the original image.
    """
    # Initialize the output image.
    image_clustered_ = np.zeros(image_.shape, dtype='uint8')

    # Extract colored points.
    coordinates = []
    for i, row in enumerate(image_):
        for j, px in enumerate(row):
            b, g, r = px
            if not (b == g == r) and px.sum() > 5:
                coordinates.append([i, j])

    # Perform clustering
    if len(coordinates) < 5:
        return 0, image_clustered_
    clustering = alg.fit(coordinates)

    # Group the clusters
    n_clusters = np.max(clustering.labels_) + 1
    clusters_ = [[] for _ in range(n_clusters)]
    for coord, label in zip(coordinates, clustering.labels_):
        if label != -1:
            clusters_[label].append(coord)

    # Count and store valid clusters
    valid = [cluster_ for cluster_ in clusters_ if len(cluster_) > 135]

    # Draw clusters
    clusters_ = valid if draw_valid else clusters_
    if color == DRAW_CLUSTERS_WHITE:
        draw_clusters_white(clusters_, image_clustered_)
    elif color == DRAW_CLUSTERS_COLOR:
        draw_clusters_color(clusters_, image_clustered_)
    else:
        draw_clusters_source(clusters_, image_clustered_)

    return len(valid), image_clustered_


def draw_clusters_source(clusters_, base_image_, image_):
    """Draws clusters on the base image as a masked version of the source."""
    for cluster_ in clusters_:
        for point in cluster_:
            i, j = point
            base_image_[i, j, :] = image_[i, j, :]


def draw_clusters_color(clusters_, base_image_):
    """Draws clusters on the base image as colored regions."""
    # Generate colors
    colors_ = generate_colors(len(clusters_))

    for idx, cluster_ in enumerate(clusters_):
        for point in cluster_:
            i, j = point
            base_image_[i, j, :] = colors_[idx]


def draw_clusters_white(clusters_, base_image_):
    """Draws clusters on the base image as white regions."""
    for cluster_ in clusters_:
        for point in cluster_:
            i, j = point
            base_image_[i, j, :] = [255, 255, 255]


def get_clusters(images_: List[np.ndarray], alg: DBSCAN, color: int = DRAW_CLUSTERS_WHITE, draw_valid: bool = True) -> (List[int], List[np.ndarray]):
    """Calculates clusters for a batch of images and returns images masked with
    clustered points and the count of large clusters.

        images_: A batch of images.
        alg: An instance of a DBSCAN algorithm object.

    Returns:
        An array of large cluster counts per image and an array of
        cluster-masked images.
    """
    images_clustered_ = []
    counts_ = []
    for i, image_ in enumerate(images_):
        count_, image_clustered = cluster(image_, alg, color, draw_valid)
        images_clustered_.append(image_clustered)
        counts_.append(count_)
    return counts_, images_clustered_


def main():
    # Extract list of files
    files = glob.glob("./PatientData/Data/*_thresh.png")

    # Load the template
    template = cv2.imread("./template.png")

    # Process the images
    print("Processing images...")
    for file in tqdm.tqdm(files):
        # Split path
        a, i, _ = os.path.basename(file)[:-4].split('_')
        prefix = "{0}_{1:03d}".format(a, int(i))

        # Generate slices and write
        images = get_filtered_slices(file, template)
        write_images(images, PATH_SLICES, prefix)

        # Detect clusters and write
        alg = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        counts, images_cl = get_clusters(images, alg)
        write_images(images_cl, PATH_CLUSTERS, prefix)
        write_cluster_counts(counts, PATH_CLUSTERS, prefix)

    print("Completed")


if __name__ == "__main__":
    main()
