# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys

B = 5  # number of anchor boxes
C = 20  # number of classes

# ### parameters of the model
# 
# These are training parameters.
# The input and output resolution are important for setting up the boxes as loss for training.
# The lambdas are factors to weigh the different loss components against each other.




in_x = 256
in_y = 256

lambda_coords = 10
lambda_class = 2
lambda_obj = 5
lambda_noobj = 0.5

# ### Set up the training data
# Follow the guide on the darknet side to set up VOC:
# https://pjreddie.com/darknet/yolo/




# prepare a config for the augmentations
config = {}
config["max_hsv_scale"] = [0.1, 0.5, 0.5]
config["max_rotation"] = 10
config["max_shift"] = 0.05
config["zoom_range"] = (0.8, 1.2)

train_path = "/home/lars/data/darknet/VOC/train.txt"
test_path = "/home/lars/data/darknet/VOC/2007_test.txt"

# train_path = r"C:\Users\lhk\OneDrive\data\VOC\train.txt"
# test_path = r"C:\Users\lhk\OneDrive\data\VOC\2007_test.txt"

# iterator class to provide data to model.fit_generator

from ssd_playground.mixed_generator import Augmenter
batch_size = 32

# anchor boxes are taken from the tiny yolo voc config
#anchors = np.zeros((B, 2))
#anchors[:] = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
anchors = np.zeros((B, 2))
anchors[:] = [[0.9, 0.35], [0.8, 0.45], [0.6, 0.6], [0.45, 0.8], [0.35,0.9]]

# the anchors are given as width, height
# this doesn't work with numpy's layout
# we have to switch the x and y dimensions
temp = anchors[:, 0].copy()
anchors[:, 0] = anchors[:, 1]
anchors[:, 1] = temp



out_x = [32, 24, 16, 8]
out_y = [32, 24, 16, 8]
scale = [0.2, 0.4, 0.75, 1]

train_gen =  Augmenter(train_path, in_x, in_y, out_x, out_y, scale, anchors, B, C, batch_size)

np.random.seed(0)
while True:
    # test the generator
    batch = next(train_gen)
    imgs = batch[0]
    objects = batch[1]

    #plt.imshow(imgs[0, :, :])
    #plt.show()