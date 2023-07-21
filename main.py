import cv2
import numpy as np

def main():
    img = cv2.imread('images/Animal 20/Eritrócito/A038 - 20220711_113718.bmp')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray', gray)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    dilation_kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, dilation_kernel, iterations=4)

    masked = cv2.bitwise_and(gray, gray, mask=eroded)

    blank = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.rectangle(blank, (58, 35), (555, img.shape[0]), 255, -1)
    masked = cv2.bitwise_and(masked, masked, mask=mask)
    
    cell_thresh = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    
    contours, _ = cv2.findContours(cell_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoured_img = img.copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 1)
    
    cv2.imshow('A037 - 20220711_113715.bmp', img)
    cv2.imshow('A037 - 20220711_113715.bmp with contours', contoured_img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
