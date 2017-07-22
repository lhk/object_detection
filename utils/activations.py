"""
a collection of activation functions
"""

import numpy as np
import tensorflow as tf

def tf_sigmoid(t):
    return 1 / (1 + tf.exp(-t))

def np_sigmoid(x):
    return 1 / (1+ np.exp(-x))

def softmax(x, axis=-1):
    max_x = x.max(axis=axis)
    max_x = np.expand_dims(max_x, axis)
    x = x - max_x
    exp_x = np.exp(x)
    exp_x_sum = np.expand_dims(exp_x.sum(axis), axis)
    return exp_x / exp_x_sum