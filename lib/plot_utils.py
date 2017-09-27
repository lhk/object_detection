import numpy as np

import matplotlib.pyplot as plt


def create_canvas(size_x=100, size_y=100, colored=False):
    if colored:
        return np.zeros((size_x, size_y, 3))
    else:
        return np.zeros((size_x, size_y))


def draw_rect(canvas, rect, color=1, thickness=0):
    """
    plots a rectangle into a numpy array
    coordinates of the rectangle are in [0,1]
    :param canvas:
    :param rect:
    :param color:
    :param thickness:
    :return:
    """
    size_x, size_y = canvas.shape[:2]
    cx, cy, dx, dy = rect

    assert 0 <= cx <= 1, "coordinates must be in [0,1]"
    assert 0 <= cy <= 1, "coordinates must be in [0,1]"
    assert 0 <= dx <= 1, "coordinates must be in [0,1]"
    assert 0 <= dy <= 1, "coordinates must be in [0,1]"

    min_x = int((cx - 0.5 * dx) * size_x)
    min_y = int((cy - 0.5 * dy) * size_y)
    max_x = int((cx + 0.5 * dx) * size_x)
    max_y = int((cy + 0.5 * dy) * size_y)

    min_x = np.maximum(min_x, 0)
    min_y = np.maximum(min_y, 0)
    max_x = np.minimum(size_x - 1, max_x)
    max_y = np.minimum(size_y - 1, max_y)

    if thickness > 0:
        canvas[min_x:min_x + thickness, min_y:max_y] = color
        canvas[max_x - thickness:max_x, min_y:max_y] = color
        canvas[min_x:max_x, min_y:min_y + thickness] = color
        canvas[min_x:max_x, max_y - thickness:max_y] = color
    else:
        canvas[min_x:max_x, min_y:max_y] = color

    return canvas


def plot_canvas(canvas):
    plt.imshow(canvas)
    plt.show()
