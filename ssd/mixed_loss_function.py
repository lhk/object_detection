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

from lib.utils.activations import *


# code taken from here: https://gist.github.com/Djexus/1193399/88a4ced30874876e561c8f1d480d83df38c20eca
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

    length = 4
    f_boxes = tf.reshape(blob[:, :, :, pointer: pointer + length], (-1, out_x * out_y, B, length))
    pointer += length

    # part 1: the coordinate predictions
    # slice the predicted coordinates out of the loss
    p_boxes = output_layer[:, :, :, :, :4]
    p_boxes = tf.reshape(p_boxes, (-1, out_x * out_y, B, 4))

    p_boxes_xy = p_boxes[:, :, :, :2]
    p_boxes_wh = p_boxes[:, :, :, 2:]

    b_boxes_xy = tf_sigmoid(p_boxes_xy)
    s_boxes_xy = b_boxes_xy
    s_boxes_wh = p_boxes_wh

    l_boxes = tf.concat([s_boxes_xy, s_boxes_wh], axis=-1)

    # as opposed to YOLO, the mask only relies on objectness
    # we have filtered the boxes by overlap in the generator
    # no calculation of IoU is necessary here
    mask = f_objectness

    # calculate the loss terms and mask them
    # all the losses in yolo are simply squared error
    # and they are multiplied with a weighting factor

    # remember: l_ is the output from the network, prepared for the loss
    #           f_ is the corresponding value, fed into the network

    # coordinates
    # TODO: replace mse with smooth L1
    loss_coords = l_boxes - f_boxes
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
    #total_loss = loss_coords + loss_obj + loss_noobj + loss_labels
    total_loss = loss_obj + loss_noobj

    return total_loss
