import os
import math
import cv2 as cv
import numpy as np


DIR_PATH = os.environ['DIR_PATH']
WHITE_THRESHOLD = int(os.environ['WHITE_THRESHOLD'])
CHECK_PARALLEL_PIXELS = int(os.environ['CHECK_PARALLEL_PIXELS'])


def crop_outside_three_lines(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    col = math.floor(gray_img.shape[1]/4)
    y1_threshold = None
    for row in range(gray_img.shape[0]):
        for index in range(CHECK_PARALLEL_PIXELS):
            if gray_img[row][col + index] > WHITE_THRESHOLD:
                y1_threshold = row
                break
            if index == 0:
                continue
            elif gray_img[row][col - index] > WHITE_THRESHOLD:
                y1_threshold = row
                break
        if y1_threshold is not None and row - y1_threshold > 50:
            break


    row = math.floor(gray_img.shape[0]/2)
    x1_threshold = None
    for col in range(gray_img.shape[1]):
        for index in range(CHECK_PARALLEL_PIXELS):
            if gray_img[row + index][col] > WHITE_THRESHOLD:
                x1_threshold = col
                break
            if index == 0:
                continue
            elif gray_img[row - index][col] > WHITE_THRESHOLD:
                x1_threshold = col
                break
        if x1_threshold is not None and col - x1_threshold > 50:
            break

    x2_threshold = None
    for col in range(gray_img.shape[1]):
        true_col = gray_img.shape[1] - col - 1
        for index in range(CHECK_PARALLEL_PIXELS):
            if gray_img[row + index][true_col] > WHITE_THRESHOLD:
                x2_threshold = true_col
                break
            if index == 0:
                    continue
            elif gray_img[row - index][true_col] > WHITE_THRESHOLD:
                x2_threshold = true_col
                break
        if x2_threshold is not None and x2_threshold - true_col > 50:
            break

    return img[y1_threshold:gray_img.shape[0], x1_threshold:x2_threshold], gray_img[y1_threshold:gray_img.shape[0], x1_threshold:x2_threshold]

def mask_subsections_and_count_cells(gray_img):
    gray_copy = gray_img.copy()
    blured_img = cv.GaussianBlur(gray_img, (7,7), 0)

    _, thresh = cv.threshold(blured_img, 115, 255, cv.THRESH_BINARY)
    cv.imshow('thresh', thresh)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(gray_copy, contours, -1, (0, 255, 0), 2)
    cv.imshow('contours', gray_copy)

    rounded_contours = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

        if len(approx) < 3:
            rounded_contours.append(contour)

    return rounded_contours

def main():
    files = os.listdir(DIR_PATH)
    for file in files:
        path = os.path.join(DIR_PATH, file)
        img = cv.imread(path)

        print(f'handling file {path}')
        masked, masked_gray = crop_outside_three_lines(img)
        contours = mask_subsections_and_count_cells(masked_gray)
        print(f'Contours detected: {len(contours)}')

        cv.drawContours(masked, contours, -1, (0, 255, 0), 5)

        cv.imshow('result', masked)
        cv.waitKey(0)
        break

    # cv2.imshow('gray', gray)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # dilation_kernel = np.ones((3, 3), np.uint8)
    # eroded = cv2.erode(thresh, dilation_kernel, iterations=4)

    # masked = cv2.bitwise_and(gray, gray, mask=eroded)

    # blank = np.zeros(img.shape[:2], np.uint8)
    # mask = cv2.rectangle(blank, (58, 35), (555, img.shape[0]), 255, -1)
    # masked = cv2.bitwise_and(masked, masked, mask=mask)
    
    # cell_thresh = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    
    # contours, _ = cv2.findContours(cell_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contoured_img = img.copy()
    # cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 1)
    
    # cv2.imshow('A037 - 20220711_113715.bmp', img)
    # cv2.imshow('A037 - 20220711_113715.bmp with contours', contoured_img)


if __name__ == '__main__':
    main()
