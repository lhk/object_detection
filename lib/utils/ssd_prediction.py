"""
helper functions to work with predictions.
The blob which is passed to the loss function can be inconvenient to work with. The extract_from_blob function converts this blob into a handy dictionary.
The get_probabilites() function takes a single prediction, 1 along the batch dimension, and sliced the objectness, class probabilities and joint object probability out of it.
"""

import numpy as np
from lib.utils.activations import *

def get_probabilities(data,
                     out_x, out_y,
                     B, C):
    # this will only work on one image at once
    # in case you've provided a whole batch, we reshape to the proper dimension and drop the batch dimension
    # nevertheless you shouldn't pass in a whole batch, maybe that would mask other problems related to the dimensions
    data = data.reshape((-1, out_x, out_y, B, C+5))
    assert data.shape[0] == 1, "this doesn't work on batches"
    
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
    length = 4
    f_boxes = (blob[:, :, :, pointer: pointer + length]).reshape((-1, out_x * out_y, B, length))
    pointer += length
    
    return {"f_labels":f_labels, "f_objectness":f_objectness, "f_boxes":f_boxes}