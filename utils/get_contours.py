import cv2 as cv
import numpy as np


def get_contours(masked_gray, coordinates):
    selected_contours = []
    for coordinate in coordinates:
        x1 = coordinate.get('x1')
        x2 = coordinate.get('x2')
        y1 = coordinate.get('y1')
        y2 = coordinate.get('y2')

        blank = np.zeros(masked_gray.shape[:2], np.uint8)
        mask = cv.rectangle(blank, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), 255, -1)
        
        sub_section = cv.bitwise_and(masked_gray, mask)
        sub_section = cv.GaussianBlur(sub_section, (3, 3), 0)
        
        thresh = cv.adaptiveThreshold(sub_section, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            ctr_area = cv.contourArea(contour)
            if ctr_area >= 15 and ctr_area <= 100:
                selected_contours.append(contour)
    return selected_contours
