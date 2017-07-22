"""
helper functions to work with predictions
"""

import numpy as np
from activations import *

def get_probabilities(data,
                     out_x, out_y,
                     B, C):
    # this will only work on one image at once
    # in case you've provided a whole batch, we reshape to the proper dimension and drop the batch dimension
    # nevertheless you shouldn't pass in a whole batch, maybe that would mask other problems related to the dimensions
    data = data.reshape((-1, out_x, out_y, B, C+5))
    data = data[0]

    classes = data[:,:,:,5:]
    classes = softmax(classes)
    max_classes = classes.max(axis=-1)

    objectness = np_sigmoid(data[:,:,:,4])

    probs = max_classes * objectness
    
    return classes, objectness, probs

def extract_from_blob(blob, 
                     out_x, out_y,
                     B, C):
    """ extracts values from the blob fed to the loss function"""
    pointer = 0
    length = C
    f_labels = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length

    length = 1
    f_objectness = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length

    length = 1
    f_areas = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length

    length = 4
    f_boxes = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length

    length = 2
    f_upper_left_corner = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length

    length = 2
    f_lower_right_corner = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length
    
    return {"f_labels":f_labels, "f_objectness":f_objectness, 
            "f_areas": f_areas, "f_boxes":f_boxes, 
            "f_upper_left_corner":f_upper_left_corner, "f_lower_right_corner":f_lower_right_corner}