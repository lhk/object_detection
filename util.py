"""
a collection of helper functions.
They support working with lists of objects and bounding boxes.
"""

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

def filter_boxes(boxes):
    return [(label, x_min, y_min, x_max, y_max) for (label, x_min, y_min, x_max, y_max) in boxes 
                if x_max > 0 and y_max > 0 and x_min < 1 and y_min < 1]

def clip_boxes(boxes):
    labels, coords = split(boxes)
    coords = np.array(coords)
    coords = np.clip(coords, 0, 1)
    return merge(labels, coords)