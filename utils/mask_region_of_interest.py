import cv2 as cv
import numpy as np

from utils.find_white_pixel import find_white_pixel


def mask_region_of_interest(gray_img, x_ref, y_ref):
    x1_first_white, _ = find_white_pixel(gray_img, 'x', y_ref)
    x2_first_white, _ = find_white_pixel(gray_img, 'x', y_ref, invert=True)
    y1_first_white, _ = find_white_pixel(gray_img, 'y', x_ref)

    blank = np.zeros(gray_img.shape[:2], np.uint8)
    mask = cv.rectangle(blank, (x1_first_white, y1_first_white), (x2_first_white, gray_img.shape[0]), 255, -1)
    masked_gray = cv.bitwise_and(gray_img, mask)
    return masked_gray
