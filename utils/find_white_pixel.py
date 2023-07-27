import os
import math
import numpy as np


CHECK_PARALLEL_PIXELS = int(os.environ['CHECK_PARALLEL_PIXELS'])


def find_white_pixel(gray_img, axis: str, fixed_pos: int, invert = False, offset = 0):
    tons_of_gray = {}
    for row in range(gray_img.shape[0]):
        for col in range(gray_img.shape[1]):
            positive_offset = gray_img[row][col]
            if tons_of_gray.get(positive_offset) is None:
                tons_of_gray[positive_offset] = positive_offset
    
    tons_of_gray = list(tons_of_gray.values())
    tons_of_gray.sort()
    white_threshold = math.ceil(np.median(tons_of_gray))

    axis = axis.lower()
    if axis == 'y':
        variable_axis = gray_img.shape[0]
    elif axis == 'x':
        variable_axis = gray_img.shape[1]
    
    first_white_pos = None
    last_white_pos = None
    white_found = False
    for pos in range(variable_axis):
        if pos < 3:
            continue

        true_pos = pos if invert is False else variable_axis - pos - 1
        true_pos += offset
        if true_pos >= variable_axis:
            break
        for index in range(CHECK_PARALLEL_PIXELS):
            positive_offset = gray_img[fixed_pos + index][true_pos] if axis == 'x' else gray_img[true_pos][fixed_pos + index]
            negative_offset = gray_img[fixed_pos - index][true_pos] if axis == 'x' else gray_img[true_pos][fixed_pos - index]
            
            if (positive_offset >= white_threshold or negative_offset >= white_threshold) and white_found is False:
                white_found = True
                first_white_pos = true_pos
                break
            elif (white_threshold >= positive_offset or white_threshold >= negative_offset) and white_found is True:
                white_found = False
                last_white_pos = true_pos - 1 if invert is False else true_pos + 1
                break
        
        if last_white_pos is not None and abs(true_pos - last_white_pos) > 50:
            break

    return first_white_pos, last_white_pos
