"""Contains brain scan slice and boundary extraction functions."""
import glob
import os.path
from collections import namedtuple
import numpy as np
from typing import List
import cv2
import tqdm

PATH_SLICES = "./Output/Slices"
PATH_BOUND = "./Output/Boundaries"
PATH_TEMPLATE = "./template.png"

PARAM_FILTER_THRESHOLD = 100
PARAM_BOUND_THRESHOLD = 5

Point = namedtuple("Point", "x y")
Size = namedtuple("Size", "width height")


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


def write_images(slices: List[np.ndarray], base_dir: str, prefix: str):
    """Writes a batch of images to an output path."""
    for i, slice_ in enumerate(slices):
        dir_ = os.path.join(base_dir, prefix)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        path = os.path.join(dir_, "{0:02d}.png".format(i))
        cv2.imwrite(path, slice_)


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

        # Detect and draw boundaries
        draw_boundaries(images)
        write_images(images, PATH_BOUND, prefix)

    print("Completed")


if __name__ == "__main__":
    main()
