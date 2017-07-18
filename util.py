"""
a collection of helper functions.
They support working with lists of objects and bounding boxes.
"""
import numpy as np

def split(objects):
    labels = [label for (label, _, _, _, _) in objects]
    coords = [(a, b, c, d) for (_, a, b, c, d) in objects]
    return labels, coords

def merge(labels, coords):
    objects = [(label, *coords) for (label, coords) in zip(labels, coords)]
    return objects
# the various augmentation functions have different requirements on layout
# yolo expects a format which encodes boxes as 1 vertex, width and height: x_center, y_center, width, height
# the augmentation needs two vertices: x_min, y_min, x_max, y_max
# helper functions for conversions:

def wh_to_minmax(objects):
    # convert from [x, y, w, h] to [x_min, y_min, x_max, y_max]
    objects = [(label, x-dx/2, y-dy/2, x+dx/2, y+dy/2) for (label, x, y, dx, dy) in objects]
    
    return objects

def minmax_to_wh(objects):
    labels, coords = split(objects)
    
    # convert from [x_min, y_min, x_max, y_max] to [x, y, w, h]
    new_coords = []
    for (x_min, y_min, x_max, y_max) in coords:
        dx = x_max - x_min
        dy = y_max - y_min
        cx = x_min + dx/2
        cy = y_min + dy/2
        new_coords.append((cx, cy, dx, dy))
        
    objects = [(label, *coords) for (label, coords) in zip(labels, new_coords)]
    return objects

def boxes_to_vertices(boxes):
    """
    Takes a list of bounding boxes and creates a list of vertices
    The output shape is [number of boxes, 4, 2]
    4 for the 4 vertices,
    2 for x/y
    :param boxes: Input tensor, must be 2D
    :return: output tensor, 3D
    """

    assert len(boxes.shape) == 2, "boxes must be a 2D tensor"
    assert boxes.shape[1] == 4, "boxes must be [:, 4] tensor"

    # look at the four vertices of each box
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    assert np.all(x_min < x_max), "coordinates must be given as [xmin, ymin, xmax, ymax]"
    assert np.all(y_min < y_max), "coordinates must be given as [xmin, ymin, xmax, ymax]"

    # create new axis to stack the x,y coordinates
    x_min = np.expand_dims(x_min, axis=-1)
    y_min = np.expand_dims(y_min, axis=-1)
    x_max = np.expand_dims(x_max, axis=-1)
    y_max = np.expand_dims(y_max, axis=-1)

    # stack the x,y coordinates to create the vertices
    # the resulting arrays are indexed [idx of box, idx of x or y]
    up_left = np.concatenate([x_min, y_min], axis=-1)
    up_right = np.concatenate([x_min, y_max], axis=-1)
    down_right = np.concatenate([x_max, y_max], axis=-1)
    down_left = np.concatenate([x_max, y_min], axis=-1)

    # now stack the vertices, along axis 1
    up_left = np.expand_dims(up_left, axis=1)
    up_right = np.expand_dims(up_right, axis=1)
    down_right = np.expand_dims(down_right, axis=1)
    down_left = np.expand_dims(down_left, axis=1)

    # create an array of all vertices, of all boxes
    # the shape is [number of boxes, number of vertices, number of coordinates]
    # ->  shape is [number of boxes, 4, 2]
    vertices = np.concatenate([up_left, up_right, down_right, down_left], axis=1)

    return vertices

def vertices_to_boxes(vertices):
    """
    Takes a list of vertices and converts them to bounding boxes
    :param vertices: Input tensor, must be 2D
    :return: output tensor, 2D
    """

    assert len(vertices.shape)==2, "vertices must be a 2D tensor"
    assert vertices.shape[1]==2, "vertices must be [:, 2] tensor"

    vertices = vertices.reshape((-1, 4, 2))

    x = vertices[:, :, 0]
    y = vertices[:, :, 1]

    x_min = x.min(axis=-1)
    x_max = x.max(axis=-1)
    y_min = y.min(axis=-1)
    y_max = y.max(axis=-1)

    x_min = np.expand_dims(x_min, axis=-1)
    x_max = np.expand_dims(x_max, axis=-1)
    y_min = np.expand_dims(y_min, axis=-1)
    y_max = np.expand_dims(y_max, axis=-1)

    boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=-1)

    return boxes

def filter_boxes(boxes):
    return [(label, x_min, y_min, x_max, y_max) for (label, x_min, y_min, x_max, y_max) in boxes 
                if x_max > 0 and y_max > 0 and x_min < 1 and y_min < 1]

def clip_boxes(boxes):
    labels, coords = split(boxes)
    coords = np.array(coords)
    coords = np.clip(coords, 0, 1)
    return merge(labels, coords)