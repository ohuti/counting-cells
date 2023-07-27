import os
import cv2 as cv

from utils.get_references import get_references
from utils.mask_region_of_interest import mask_region_of_interest

DIR_PATH = os.environ['DIR_PATH']


def main():
    result_dir = f'result/{DIR_PATH}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    files = os.listdir(DIR_PATH)
    for file in files:
        path = os.path.join(DIR_PATH, file)
        image = cv.imread(path)

        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        x_ref, y_ref = get_references(gray_img)
        masked_gray = mask_region_of_interest(gray_img, x_ref, y_ref)

        cv.line(masked_gray, (x_ref, 0), (x_ref, masked_gray.shape[0]), 0, 1)
        cv.line(masked_gray, (0, y_ref), (masked_gray.shape[1], y_ref), 0, 1)
        cv.putText(masked_gray, file, (10, 20), cv.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv.imshow('preview', masked_gray)
        cv.waitKey(0)


if __name__ == '__main__':
    main()
