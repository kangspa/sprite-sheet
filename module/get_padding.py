def get_padding(padding):
    if len(padding) == 1:
        padding_top = padding_bottom = padding_left = padding_right = padding[0]
    elif len(padding) == 2:
        padding_top = padding_bottom = padding[0]
        padding_left = padding_right = padding[1]
    elif len(padding) == 3:
        padding_top = padding[0]
        padding_left = padding_right = padding[1]
        padding_bottom = padding[2]
    elif len(padding) == 4:
        padding_top, padding_right, padding_bottom, padding_left = padding
    else:
        raise ValueError("Padding array must have 1 to 4 values.")

    return padding_top, padding_right, padding_bottom, padding_left