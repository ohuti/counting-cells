import math
import cv2 as cv


if __name__ == '__main__':
    img = cv.imread('images/Animal 20/Eritrócito/A037 - 20220711_113715.bmp')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    col = math.floor(gray.shape[1]/3)
    whiteFound = False
    y1_threshold = None
    for row in range(gray.shape[0]):
        # print(f'Coordinates (x, y): ({row}, {col}) - Color: {gray[row][col]}')
        if gray[row][col] > 200:
            whiteFound = True
            y1_threshold = row
        elif gray[row][col] < 200 and whiteFound and row - y1_threshold > 50:
            break
    print(f'Y1 Threshold: {y1_threshold}')
    
    cv.imshow('gray', gray)
    cv.waitKey(0)
