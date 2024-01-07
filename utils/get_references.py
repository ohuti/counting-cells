import os
import math

X_REF_OFFSET = int(os.environ['X_REF_OFFSET'])
Y_REF_OFFSET = int(os.environ['Y_REF_OFFSET'])


def get_references(gray_img):
    x_ref = math.floor(gray_img.shape[1]/2) + X_REF_OFFSET
    y_ref = math.floor(gray_img.shape[0]/2) + Y_REF_OFFSET
    return x_ref, y_ref
