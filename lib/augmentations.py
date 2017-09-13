"""
This file is unfortunately in a patchwork state.
I need image augmentations, they are provided by keras.
But the object annotations need be kept in sync with the image. The exact same transformations need to be applied to image and boxes.

If a transformation is applied to the image, the bounding box is transformed too.
This is done by transforming the corners of the box into vectors and applying the transformation matrix to the vector.
Then a new box can be generated from the vectors.

My long term goal is to get this integrated into the keras sourcecode.
Actually, the random_transform_with_vertices is mostly taken from the keras source.
Until this is possible, the code is kept here.

A second part of this is the augment(...) function.
That function contains the interesting code and describes the augmentations applied to the data.
Unless you are interested in the internals of augmentations with bounding boxes, simply ignore everything else.
"""

import numpy as np
import cv2

from keras.preprocessing.image import *

from lib.utils.bbox import vertices_to_boxes, boxes_to_vertices
from lib.utils.object import split, merge

def random_transform_with_vertices(x, vertices,
                                   rg,
                                   wrg,
                                   hrg,
                                   zoom_range,
                                   row_axis=1, col_axis=2, channel_axis=0,
                                   fill_mode='nearest', cval=0.):
    """Randomly augment a single image tensor.
       Also augments a list of vertices
    # Arguments
        x: 3D tensor, single image.
        seed: random seed.

    # Returns
        A randomly transformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0

    # use composition of homographies
    # to generate final transform that needs to be applied


    # apply to image
    h, w = x.shape[row_axis], x.shape[col_axis]

    shift_x = np.random.uniform(-hrg, hrg)
    shift_y = np.random.uniform(-wrg, wrg)
    tx = shift_x * h
    ty = shift_y * w

    theta = np.pi / 180 * np.random.uniform(-rg, rg)

    assert 0<zoom_range[0] and 0<zoom_range[1], "zoom can not be 0"
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)


    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = rotation_matrix

    shift_matrix = np.array([[1, 0, tx],
                             [0, 1, ty],
                             [0, 0, 1]])
    transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    final_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, final_matrix, channel_axis,
                        fill_mode=fill_mode, cval=cval)

    vertices = np.dot(vertices, rotation_matrix[:2, :2])
    vertices = vertices - [shift_x, shift_y]
    vertices *= [1/zx, 1/zy]
    return x, vertices



def augment(img, objects, max_hsv_scale=[0.5,0.5,0.5], 
                          max_rotation=10, 
                          max_shift=0.1,
                          zoom_range=(0.75, 1.25)):
    
    labels, coords = split(objects)
    coords = np.array(coords)    
    vertices = boxes_to_vertices(coords)
    vertices = vertices.reshape((-1, 2))
    
    # move origin to [0.5, 0.5]
    vertices -= [0.5, 0.5]
    
    # apply image transformations
    
    img, vertices = random_transform_with_vertices(img, vertices,
                                                   max_rotation, 
                                                   max_shift, max_shift, 
                                                   zoom_range, 
                                                   row_axis=0, col_axis=1, channel_axis=2)
    
    vertices += [0.5, 0.5]
    coords = vertices_to_boxes(vertices)
    
    # clip to allowed range
    coords = np.clip(coords, 0, 1)
    
    # apply hsv shift
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.astype(np.float32)
    
    lower_hsv_scale = np.ones((3,)) - max_hsv_scale
    upper_hsv_scale = np.ones((3,)) + max_hsv_scale
    hsv_scale = np.random.uniform(lower_hsv_scale, upper_hsv_scale)
    img *= hsv_scale
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    # apply flip
    flip = np.random.rand()>0.5
    if flip:
        img = img[:,::-1]
        coords = [(x_min, 1-y_max, x_max, 1-y_min) for (x_min, y_min, x_max, y_max) in coords]
        
    objects = merge(labels, coords)
    
    # remove bounding boxes that are completely outside of the image
    # after clipping x_min = x_max or y_min = y_max for such an object
    objects = [(label, x_min, y_min, x_max, y_max) for (label, x_min, y_min, x_max, y_max) in objects if
              x_min != x_max and y_min != y_max]
    
    return img, objects