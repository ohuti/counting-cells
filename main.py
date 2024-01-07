import os
import cv2 as cv
import pandas as pd
from datetime import datetime

from utils.get_contours import get_contours
from utils.get_references import get_references
from utils.mask_region_of_interest import mask_region_of_interest
from utils.map_divisions_and_get_sub_section_coordinates import map_divisions_and_get_sub_section_coordinates

DIR_PATH = os.environ['DIR_PATH']


def main():
    start = datetime.now()
    result_dir = f'result/{DIR_PATH}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data = []
    files = os.listdir(DIR_PATH)
    # files = ['A003 - 20220718_123354.bmp']
    for file in files:
        file_start = datetime.now()
        path = os.path.join(DIR_PATH, file)
        image = cv.imread(path)

        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        x_ref, y_ref = get_references(gray_img)
        masked_gray = mask_region_of_interest(gray_img, x_ref, y_ref)
        coordinates = map_divisions_and_get_sub_section_coordinates(masked_gray, x_ref, y_ref)
        selected_contours = get_contours(masked_gray, coordinates)
        contour_image = image.copy()
        cv.drawContours(contour_image, selected_contours, -1, (0, 255, 0), 1)

        filename = f'result/{path}'.replace('//', '/')
        cv.imwrite(filename, contour_image)
        data.append({
            'filename': filename,
            'cells counted': len(selected_contours)
        })
        file_end = datetime.now()
        print(f'handling file {path}. Elapsed time {file_end - file_start}')

    data_frame = pd.DataFrame(data)
    data_frame.to_excel(f'{result_dir}/counts.xlsx', index=False)
    end = datetime.now()
    print(f'Process finished successfully. Elapsed time: {end - start}')


if __name__ == '__main__':
    main()
