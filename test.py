import glob
import os

import tqdm

import brainExtraction
from cv2 import imread


PATH_SLICES = "./Slices"
PATH_BOUND = "./Boundaries"


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
        images = brainExtraction.get_filtered_slices(file, template)
        brainExtraction.write_images(images, PATH_SLICES, prefix)

        # Detect and draw boundaries
        brainExtraction.draw_boundaries(images)
        brainExtraction.write_images(images, PATH_BOUND, prefix)
    print("Completed.")
