import glob
import numpy as np
from typing import List
import cv2


pattern = np.array([
    [[255,255,255,],[255,255,255,],[255,255,255,],[255,255,255,],],
    [[255,255,255,],[0,0,0,],[0,0,0,],[255,255,255,],],
    [[255,255,255,],[255,255,255,],[255,255,255,],[255,255,255,],],
    [[255,255,255,],[0,0,0,],[255,255,255,],[0,0,0,],],
    [[255,255,255,],[0,0,0,],[0,0,0,],[255,255,255,],],
])
pattern_w, pattern_h = 4, 5


def get_slices(file: str) -> List[np.ndarray]:
    # Load image
    img = cv2.imread()


    # Detect all instances of the R
    matches = cv2.matchTemplate(img, pattern, cv2.TM_SQDIFF)
    poss = np.stack(*np.where(matches == 1)[::-1], axis=1)

    # Group and sort
    


def main():

    # Extract list of files
    files = glob.glob("./PatientData/Data/*_thresh.png")


    # Extract
    for file in files:

        slices = get_slices(file)



if __name__ == "__main__":

    main()

