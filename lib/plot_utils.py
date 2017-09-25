import numpy as np

import matplotlib.pyplot as plt


def create_canvas(size_x=100, size_y=100, colored=False):
    if colored:
        return np.zeros((size_x, size_y, 3))
    else:
        return np.zeros((size_x, size_y))


def draw_rect(np_arr, rect, color=1, thickness=0):
    """
    plots a rectangle into a numpy array
    coordinates of the rectangle are in [0,1]
    :param np_arr:
    :param rect:
    :param color:
    :param thickness:
    :return:
    """
    size_x, size_y = np_arr.shape
    cx, cy, dx, dy = rect
    min_x = int((cx - 0.5 * dx) * size_x)
    min_y = int((cy - 0.5 * dy) * size_y)
    max_x = int((cx + 0.5 * dx) * size_x)
    max_y = int((cy + 0.5 * dy) * size_y)

    if thickness > 0:
        rect[min_x:min_x + thickness, min_y:max_y] = color
        rect[max_x - thickness:max_x, min_y:max_y] = color
        rect[min_x:max_x, min_y:min_y + thickness] = color
        rect[min_x:max_x, max_y - thickness:max_y] = color
    else:
        rect[min_x:max_x, min_y:max_y] = color

    return rect


def plot_canvas(canvas):
    plt.imshow(canvas)
    plt.show()
