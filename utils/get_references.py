import os
import math

X_REF_OFFSET = int(os.environ['X_REF_OFFSET'])
Y_REF_OFFSET = int(os.environ['Y_REF_OFFSET'])


def get_references(gray_img):
    y_ref = math.floor(gray_img.shape[0]/2) - 20
    x_ref = math.floor(gray_img.shape[1]/2) - 50
    return x_ref, y_ref
