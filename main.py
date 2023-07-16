import cv2
import numpy as np

def main():
    img = cv2.imread('images/Animal 20/Eritrócito/A037 - 20220711_113715.bmp')
    image_copy = img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('A037 - 20220711_113715.bmp', img)
    # cv2.imshow('A037 - 20220711_113715.bmp THRESHOLD', thresh)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
