from utils.find_white_pixel import find_white_pixel


def map_divisions_and_get_sub_section_coordinates(masked_gray, x_ref, y_ref):
    x_divisions = []
    image_end_found = None
    x_last = 0
    while image_end_found is None:
        x_first, x_last = find_white_pixel(masked_gray, 'x', y_ref, offset=x_last)
        if x_first is not None and x_last is not None:
            x_divisions.append({ 'x_first': x_first, 'x_last': x_last })
        if x_last is None or x_last > masked_gray.shape[1]:
            image_end_found = True

    y_divisions = []
    image_end_found = None
    y_last = 0
    while image_end_found is None:
        y_first, y_last = find_white_pixel(masked_gray, 'y', x_ref, offset=y_last)
        y_divisions.append({ 'y_first': y_first, 'y_last': y_last })
        if y_last is None or y_last > masked_gray.shape[1]:
            image_end_found = True

    coordinates = []
    for x_index, x_pos in enumerate(x_divisions):
        if x_index == 0:
            continue
        x1 = x_divisions[x_index - 1].get('x_last')
        x2 = x_pos.get('x_first')
        for y_index, y_pos in enumerate(y_divisions):
            if y_index == 0:
                continue
            y1 = y_divisions[y_index - 1].get('y_last')
            y2 = y_pos.get('y_first') if y_pos.get('y_first') is not None else masked_gray.shape[0]
            coordinates.append({ 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2 })
    return coordinates
