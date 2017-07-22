"""
This is the YOLO loss formulation.
The first part of the function is not very pretty. The correct predictions are given as one big blob of data.
The exact content is described in the generator.py file.
In this function, we slice the individual components out of the blob.

The loss has multiple components.
The output of the network is [batch_size, out_x, out_y, anchor boxes, data]
This is reshaped to a vector of cells [batch_size, out_x * out_y, anchor boxes, data]
-> each cell contains a list of anchor boxes and the data belonging to each anchor box.
Then, for every anchor box, we take a closer look at the data.

There is one trick:
The loss function needs information about the network.
These are curried in.
See the demo on how this works.
"""

import numpy as np
import tensorflow as tf

import inspect

from utils.activations import *

def curry(func, *args, **kwargs):
    assert inspect.getargspec(func)[1] == None, 'Currying can\'t work with *args syntax'
    assert inspect.getargspec(func)[2] == None, 'Currying can\'t work with *kwargs syntax'
    assert inspect.getargspec(func)[3] == None, 'Currying can\'t work with default arguments'

    if (len(args) + len(kwargs)) >= func.__code__.co_argcount:

        return func(*args, **kwargs)

    return (lambda *x, **y: curry(func, *(args + x), **dict(kwargs, **y)))

@curry
def loss_func(anchors,
        out_x, out_y,
        B, C,
        lambda_class, lambda_coords, lambda_obj, lambda_noobj, blob, output_layer):
    
    print("anchors", anchors)
    print("out_x", out_x)
    print("out_y", out_y)
    print("B", B)
    print("C", C)
    print("l_class", lambda_class)
    print("l_coords", lambda_coords)
    print("l_noobj", lambda_noobj)
    print("l_obj", lambda_obj)
    
    output_layer = tf.reshape(output_layer, (-1, out_x, out_y, B, (4 + 1 + C)))
    
    # the loss deals with many different components
    # I'm trying to keep an intuitive naming scheme
    # f_ is a value fed to the network
    # p_ is a value predicted by the network
    # l_ is a value prepared for the loss, they should be on the same scale as their f_ counterpart

    # getting the data out of the blob
    pointer = 0
    length = C
    f_labels = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    length = 1
    f_objectness = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    length = 1
    f_areas = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    length = 4
    f_boxes = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    length = 2
    f_upper_left_corner = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    length = 2
    f_lower_right_corner = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    # part 1: the coordinate predictions
    # slice the predicted coordinates out of the loss
    p_coords = output_layer[:, :, :, :, :4]
    p_coords = tf.reshape(p_coords, (-1, out_x * out_y, B, 4))

    # scale them with the anchor boxes
    # the anchor boxes need to be reshaped for proper broadcasting
    r_anchors = anchors.reshape((1, 1, B, 2))
    WH = np.reshape([out_x, out_y], [1, 1, 1, 2])
    
    # split into xy, wh
    # I'm prefacing the values described in the paper b_ (as in the paper)
    # and then I'm applying the rescaling: sqrt for wh, none for xy. those values are prefaced with s_
    p_coords_xy = p_coords[:, :, :, :2]
    b_coords_xy = tf_sigmoid(p_coords_xy)
    s_coords_xy = b_coords_xy

    p_coords_wh = p_coords[:, :, :, 2:]
    
    # width and height get a special treatment
    # they have exp as an activation function, are scaled with the anchors
    # and their output is not in relative coords but in [0, out_x] and [0, out_y]
    # TODO: changing this seems to break the network
    # if the division with WH is removed, the output is expected to be in [0,1]
    # this means that the network has to predict very small values
    # which seems to be problematic
    # this needs more work, maybe a different activation function 
    # training the network to predict values not in [0,1] has the disadvantage that it is not portable
    # different output resolutions will break the setup
    b_coords_wh = (tf.exp(p_coords_wh) * r_anchors) / WH
    s_coords_wh = tf.sqrt(b_coords_wh)

    l_coords = tf.concat([s_coords_xy, s_coords_wh], axis=-1)

    # for the area, the width and height need to be multiplied with W and H
    # they need to be in [0, out_x] and [0, out_y]
    a_coords_wh = b_coords_wh * WH

    # part 2: the masks
    # the complicated part is finding out, which anchor is responsible for the prediction
    # calculate the predicted areas, the output of the network is supposed to be in the [0,out_x], [0,out_y] range
    # no rescaling is necessary here
    p_areas = a_coords_wh[:, :, :, 0] * a_coords_wh[:, :, :, 1]
    p_areas = tf.maximum(p_areas, 0.)

    # get the upper left and lower right corners of the predicted rectangles
    # they correspond directly to the similar calculations in the data preprocessing
    p_upper_left_corner = p_coords_xy - 0.5 * a_coords_wh
    p_lower_right_corner = p_coords_xy + 0.5 * a_coords_wh

    # compare the predicted areas with the object's bbox
    # intersection over union is used for this
    # the anchor box with the highest intersection is responsible for predicting this object
    # only this box is supposed to have a high objectness, only the coordinates of this bounding box have to be accurate

    # the intersection is calculated just as the area is calculated in the generator
    # we look at the two rectangles: the prediction and the given rectangle
    # then we take the upper left and lower right corners
    # one of those corners will be more outward than the other, so we take a minimum
    # calculate intersection and IoU
    inner_upper_left = tf.maximum(p_upper_left_corner, f_upper_left_corner)
    inner_lower_right = tf.minimum(p_lower_right_corner, f_lower_right_corner)
    diag = inner_lower_right - inner_upper_left
    diag = tf.maximum(diag, 0.)
    intersection = diag[:, :, :, 0] * diag[:, :, :, 1]

    # these areas have been calculated in the generator
    f_areas = tf.reshape(f_areas, (-1, out_x * out_y, B))
    
    # IoU with smoothing to prevent division by zero
    union = f_areas + p_areas - intersection
    union = tf.maximum(union, 0.)
    eps=0.01
    IoU = intersection / (eps + union)

    # determine the best box
    # this provides a mask along the B dimension, which keeps only the responsible boxes
    best_IoU = tf.argmax(IoU, axis=-1)
    box_mask = tf.one_hot(best_IoU, depth=B, axis=-1)
    box_mask = tf.reshape(box_mask, (-1, out_x * out_y, B, 1))
    
    # now we know which box is responsible for the prediction in each cell
    # we still need to take the objectness into consideration
    # only if there actually is an object in the cell, the predictions are used for the loss
    # the objectness for each cell and for each box is fed in f_objectness
    # the mask will be 1 iff there is an object in the cell and this box is responsible for finding the object
    mask = tf.multiply(box_mask, f_objectness, name="mask")

    # calculate the loss terms and mask them
    # all the losses in yolo are simply squared error
    # and they are multiplied with a weighting factor

    # remember: l_ is the output from the network, prepared for the loss
    #           f_ is the corresponding value, fed into the network

    # coordinates
    loss_coords = l_coords - f_boxes
    loss_coords = tf.pow(loss_coords, 2)
    loss_coords = tf.multiply(loss_coords, mask, name="masked_loss_coords")
    loss_coords = tf.reduce_sum(loss_coords)
    loss_coords = lambda_coords * loss_coords

    # objectness
    # different factors for objects and no objects
    p_objectness = output_layer[:, :, :, :, 4]
    p_objectness = tf_sigmoid(p_objectness)
    l_objectness = tf.reshape(p_objectness, (-1, out_x * out_y, B, 1))

    r_objectness = tf.reshape(f_objectness, (-1, out_x * out_y, B, 1))

    loss_objectness = l_objectness - r_objectness
    loss_objectness = tf.pow(loss_objectness, 2)

    loss_obj = tf.multiply(loss_objectness, mask, name="masked_loss_obj")
    loss_obj = tf.reduce_sum(loss_obj)
    loss_obj = lambda_obj * loss_obj

    loss_noobj = tf.multiply(loss_objectness, 1 - mask, name="masked_loss_noobj")
    loss_noobj = tf.reduce_sum(loss_noobj)
    loss_noobj = lambda_noobj * loss_noobj

    # classification
    p_labels = output_layer[:, :, :, :, 5:]
    softmax = tf.nn.softmax(p_labels)
    l_labels = tf.reshape(softmax, (-1, out_x * out_y, B, C))

    loss_labels = l_labels - f_labels
    loss_labels = tf.pow(loss_labels, 2)
    loss_labels = tf.multiply(loss_labels, mask, name="masked_loss_labels")
    loss_labels = tf.reduce_sum(loss_labels)
    loss_labels = lambda_class * loss_labels

    # adding everything together
    total_loss = loss_coords + loss_obj + loss_noobj + loss_labels
    # total_loss = loss_obj + loss_noobj

    return total_loss