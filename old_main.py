import os
import math
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime


DIR_PATH = os.environ['DIR_PATH']
WHITE_THRESHOLD = int(os.environ['WHITE_THRESHOLD'])
CHECK_PARALLEL_PIXELS = int(os.environ['CHECK_PARALLEL_PIXELS']) if int(os.environ['CHECK_PARALLEL_PIXELS']) > 0 else 1


def find_white_pixel(gray_img, axis: str, fixed_pos: int, invert = False):
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
        if pos < 5:
            continue

        for offset in range(CHECK_PARALLEL_PIXELS):
            index = offset if invert is False else (variable_axis - 1 - offset)
            positive_offset = gray_img[fixed_pos][pos + index] if axis == 'x' else gray_img[pos + index][fixed_pos]
            negative_offset = gray_img[fixed_pos][pos - index] if axis == 'x' else gray_img[pos - index][fixed_pos]
            
            if (positive_offset >= white_threshold or negative_offset >= white_threshold) and white_found is False:
                white_found = True
                first_white_pos = pos
            elif (white_threshold >= positive_offset or white_threshold >= negative_offset) and white_found is True:
                white_found = False
                last_white_pos = pos - 1
        
        if last_white_pos is not None and abs(pos - last_white_pos) > 50:
            break

    return first_white_pos, last_white_pos

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

def detect_subsections(gray_img):
    fixed_row_pos = math.floor(gray_img.shape[0]/3)
    fixed_col_pos = math.floor(gray_img.shape[1]/3)

    y_divisions = []
    for row in range(gray_img.shape[0]):
        for index in range(CHECK_PARALLEL_PIXELS):
            if gray_img[row - 1][fixed_col_pos + index] >= WHITE_THRESHOLD and gray_img[row][fixed_col_pos + index] < WHITE_THRESHOLD:
                if (row - 1) <= 10:
                    continue
                y_divisions.append(row)
                break
            if index == 0:
                continue
            elif gray_img[row - 1][fixed_col_pos - index] >= WHITE_THRESHOLD and gray_img[row][fixed_col_pos - index] < WHITE_THRESHOLD:
                if (row - 1) <= 10:
                    continue
                y_divisions.append(row)
                break
    
    x_divisions = []
    for col in range(gray_img.shape[1]):
        for index in range(CHECK_PARALLEL_PIXELS):
            if gray_img[fixed_row_pos + index][col - 1] >= WHITE_THRESHOLD and gray_img[fixed_row_pos + index][col] < WHITE_THRESHOLD:
                if (col - 1) <= 10:
                    continue
                x_divisions.append(col)
                break
            if index == 0:
                continue
            elif gray_img[fixed_row_pos - index][col - 1] >= WHITE_THRESHOLD and gray_img[fixed_row_pos - index][col] < WHITE_THRESHOLD:
                if (col - 1) <= 10:
                    continue
                x_divisions.append(col)
                break

    y_divisions.append(gray_img.shape[0])
    x_divisions.append(gray_img.shape[1])
    
    coordinates = []
    for y_index in range(len(y_divisions)):
        if y_index == 0:
            y1 = 0
            y2 = y_divisions[y_index]
        else:
            y1 = y_divisions[y_index - 1]
            y2 = y_divisions[y_index]
        
        for x_index in range(len(x_divisions)):
            if x_index == 0:
                x1 = 0
                x2 = x_divisions[x_index]
            else:
                x1 = x_divisions[x_index - 1]
                x2 = x_divisions[x_index]
        
            coordinates.append({ 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2 })
    return coordinates

def apply_masks_and_count_cells(gray_img, subsection_coordinates):
    selected_contours = []
    for coordinate in subsection_coordinates:
        x1 = coordinate.get('x1')
        x2 = coordinate.get('x2')
        y1 = coordinate.get('y1')
        y2 = coordinate.get('y2')

        blank = np.zeros(gray_img.shape[:2], np.uint8)
        mask = cv.rectangle(blank, (x1 + 5, y1 + 5), (x2 - 8, y2 - 8), 255, -1)
        masked = cv.bitwise_and(gray_img, mask)
        masked = cv.GaussianBlur(masked, (5, 5), 0)
        thresh = cv.adaptiveThreshold(masked, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)

        # dilation_kernel = np.ones((1, 1), np.uint8)
        # dilated_thresh = cv.dilate(thresh, dilation_kernel, iterations=3)

        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            ctr_area = cv.contourArea(contour)
            if ctr_area >= 15 and ctr_area <= 100:
                selected_contours.append(contour)

    return selected_contours


def main():
    start = datetime.now()
    result_dir = f'result/{DIR_PATH}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data = []
    files = os.listdir(DIR_PATH)
    for file in files:
        path = os.path.join(DIR_PATH, file)
        img = cv.imread(path)

        print(f'handling file {path}')
        masked, masked_gray = crop_outside_three_lines(img)

        subsection_coordinates = detect_subsections(masked_gray)
        contours = apply_masks_and_count_cells(masked_gray, subsection_coordinates)

        cv.drawContours(masked, contours, -1, (0, 255, 0), 1)

        filename = f'result/{path}'.replace('//', '/')
        cv.imwrite(filename, masked)
        data.append({
            'filename': filename,
            'cells counted': len(contours)
        })
    data_frame = pd.DataFrame(data)
    data_frame.to_excel(f'{result_dir}/counts.xlsx', index=False)
    end = datetime.now()
    print(f'Process finished successfully. Elapsed time: {end - start}')


if __name__ == '__main__':
    main()
