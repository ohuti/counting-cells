import os
import math
import cv2 as cv
import numpy as np


WHITE_THRESHOLD = 185


def crop_outside_three_lines(gray_img):
    col = math.floor(gray_img.shape[1]/3)
    y1_threshold = None
    for row in range(gray_img.shape[0]):
        if gray_img[row][col] > WHITE_THRESHOLD:
            y1_threshold = row
        elif y1_threshold is not None and gray_img[row][col] < WHITE_THRESHOLD and row - y1_threshold > 50:
            break

    row = math.floor(gray_img.shape[0]/2)
    x1_threshold = None
    for col in range(gray_img.shape[1]):
        if gray_img[row][col] > WHITE_THRESHOLD:
            x1_threshold = col
        elif x1_threshold is not None and gray_img[row][col] < WHITE_THRESHOLD and col - x1_threshold > 50:
            break

    x2_threshold = None
    for col in range(gray_img.shape[1]):
        true_col = gray_img.shape[1] - col - 1
        if gray_img[row][true_col] > WHITE_THRESHOLD:
            x2_threshold = true_col
        elif x2_threshold is not None and gray_img[row][true_col] < WHITE_THRESHOLD and x2_threshold - true_col > 50:
            break

    blank = np.zeros(gray_img.shape[:2], dtype=np.uint8)
    mask = cv.rectangle(blank, (x1_threshold, y1_threshold), (x2_threshold, gray_img.shape[0]), 255, -1)

    print(f'x1_threshold: {x1_threshold}')
    print(f'x2_threshold: {x2_threshold}')
    print(f'y1_threshold: {y1_threshold}')
    
    return cv.bitwise_and(gray_img, mask)

def main():
    files = os.listdir('images/Animal 20/Eritrócito')

    for file in files:
        path = os.path.join('images/Animal 20/Eritrócito', file)
        img = cv.imread(path)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        print(f'handling file {path}')
        masked_gray = crop_outside_three_lines(gray)
        cv.imshow(f'grayscale', gray)
        cv.imshow(f'masked image', masked_gray)
        cv.waitKey(0)

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
