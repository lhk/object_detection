"""
Keras is able to fit a model on a generator.
This allows you to:
 1. load batches dynamically - you don't have to keep your entire dataset in memory
 2. process/ prepare your data
 3. apply online augmentation
 
The following class does all of this. The constructor expects a path to the dataset.
A call to __next__(self) will:
 1. set up arrays to hold the data fed to the network
 2. loads data from the data_path
 3. augments the images and bounding boxes
 4. applies preprocessing
 5. fills the arrays to be fed to the network.
 
I'm trying to keep the parts well separated.
The data loading/augmentation is pretty much standard.
The filling of the arrays is complicated and prepares the YOLO loss formulation.
This is the interesting part of the code, you can compare with the loss function in loss.py

The class is an iterator.
Please note that I'm using only thread-local storage by allocating new variables for everything in the __next__ method.
The __next__ method is (hopefully) thread-safe.
This allows me to use the keras built-in support for multiple workers.
See the demo for training a model with this iterator.

The various arrays are all appended into one big blob of data.
That's not very nice. It's NOT a feature and NOT part of the loss.
It's a workaround to allow a loss function with the parameters (y_true, y_predicted).
Please note how the data is sliced into the blob at the end of the generator.
The loss function contains completely symmetrical code to slice the data out of the blob.
"""

import numpy as np
import cv2

from parser.parser import parse_image_label_pairs, parse_labels
from preprocessing import preprocess_yolo
from utils.object import wh_to_minmax, minmax_to_wh, split, merge
from augmentations import augment

class Augmenter:
    
    def __init__(self, data_path, 
                 in_x, in_y, out_x, out_y,
                 B, C, 
                 batch_size = 2, config = None):
        
        # misc
        self.data_path = data_path
        self.batch_size = batch_size
        
        # network input and output size
        self.in_x = in_x
        self.in_y = in_y
        self.out_x = out_x
        self.out_y = out_y
        
        # number of anchor boxes and classes
        self.B = B
        self.C = C
        
        # config for the augmentation
        if not config:
            config={}
            config["max_hsv_scale"]=[0.1, 0.5, 0.5]
            config["max_rotation"]=10
            config["max_shift"]=0.05
            config["zoom_range"]=(0.8,1.2)
        self.config = config
        
        # get a list of pairs of (image_path, label_path)
        self.image_label_pairs=parse_image_label_pairs(data_path)
        
    def __iter__(self):
        return self

    def __next__(self):
        
        # TODO: use self. for these
        data_path = self.data_path
        batch_size = self.batch_size
        config = self.config
        image_label_pairs= self.image_label_pairs
        
        in_x = self.in_x
        in_y = self.in_y
        out_x = self.out_x
        out_y = self.out_y
        
        B = self.B
        C = self.C
        
        # batch to store the images
        batch = np.zeros((batch_size, in_x, in_y, 3))

        # for training, we need to provide numpy arrays that can be fed to tensorflow
        # on the tensorflow side, the correct values are then compared with the predicted values
        # the predictions are different for each anchor box
        # the output of the network has the dimension [batch_size, out_x, out_y, anchor boxes, data]
        # this is reshaped to [batch_size, out_x * out_y, anchor boxes, data] -> a vector of anchor boxes for every batch
        # the data contains many different informations
        # I'm using the same format to feed data
        # this is inefficient, since some information is the same for all bounding boxes
        # but it makes broadcasting easier in the loss function

        # this is a one-hot encoding of the object
        labels = np.zeros([batch_size, out_x * out_y, B, C])

        # this stores whether there is an object in the cell
        objectness = np.zeros([batch_size, out_x * out_y, B, 1])

        #  for each bounding box, this stores [x, y, w, h]. x and y are relative to the center of the corresponding cell
        boxes = np.zeros([batch_size, out_x * out_y, B, 4])

        # this stores the object coordinates
        # coordinates will go from 0 to out_x and 0 to out_y
        # attention: this is only used for computing the corners and areas
        # it is not passed to the loss
        # the loss works on the boxes array
        coords = np.zeros([batch_size, out_x * out_y, B, 4])

        # corners of the bounding box
        upper_left_corner = np.zeros([batch_size, out_x * out_y, B, 2])
        lower_right_corner = np.zeros([batch_size, out_x * out_y, B, 2])

        # this stores the areas covered by the bounding boxes
        areas = np.zeros([batch_size, out_x * out_y, B, 1])

        # the container to store all of this
        # this is passed to the loss function, the different parts are then sliced out of this blob of data
        # the C+10 is the sum of the sizes of the last dimension of everything passed ot the loss
        blob = np.zeros((batch_size, out_x * out_y, B, C + 10))

        # fill the batch
        for b in range(batch_size):

            idx = np.random.randint(len(image_label_pairs))
            pair = image_label_pairs[idx]

            # load image and object list
            img = cv2.imread(pair[0])
            img = cv2.resize(img, (in_y, in_x)) # opencv wants (width, height) in numpy, y corresponds to width
            objects = parse_labels(pair[1])

            # yolo expects bounding boxes in the [x, y, w, h] format
            # this is the format in the label files
            # the augmentations need boxes in [x_min, y_min, x_max, y_max]
            # this is also the format used by tensorflow
            objects_minmax = wh_to_minmax(objects)

            # apply augmentations
            # the config dictionary is only used here
            img, objects_minmax = augment(img, objects_minmax, **config)

            # convert back to the format desired by yolo
            objects = minmax_to_wh(objects_minmax)

            # convert image to yolo input format
            img = preprocess_yolo(img)

            batch[b] = img

            # HERE IS THE YOLO SPECIFIC CODE
            
            # convert the objects to a new format, as expected by the loss
            processed_objects = []

            for obj in objects:
                # the label
                label = int(obj[0])

                # first process the x and y coordinates
                obj_x, obj_y = obj[1], obj[2]

                # this is supposed to be a percentage
                assert 0 <= obj_x <= 1, "x should be in [0,1]"
                assert 0 <= obj_y <= 1, "y should be in [0,1]"

                # convert to network coordinates, [0, out_x] and [0, out_y]
                obj_x = obj_x * out_x
                obj_y = obj_y * out_y

                # the coordinate should be relative to their cell
                rel_x = obj_x - np.floor(obj_x)
                rel_y = obj_y - np.floor(obj_y)

                # the number of the corresponding cell
                cell_number = np.floor(obj_x) * out_y + np.floor(obj_y)
                cell_number = int(cell_number)

                # now process the width and height
                size_x = obj[3]
                size_y = obj[4]

                assert 0 <= size_x <= 1, "width should be in [0,1]"
                assert 0 <= size_y <= 1, "height should be in [0,1]"

                # TODO: refactor this to allow arbitrary target functions
                # the loss works on the square root of width and height
                sqr_size_x = np.sqrt(size_x)
                sqr_size_y = np.sqrt(size_y)

                # now plug everything together to a new object for training
                processed_object = [cell_number, label, rel_x, rel_y, sqr_size_x, sqr_size_y]
                processed_objects.append(processed_object)


            # there will be only one object for every cell
            # during the loop, we will overwrite existing values of previous objects
            # no ordering is done
            # in theory, a small, distant object can overwrite a bigger, closer object
            # a person holding a cat? if both bounding boxes have their center in the same cell, the network is trained only on one
            for obj in processed_objects:
                cell_number, label, rel_x, rel_y, sqr_size_x, sqr_size_y = obj

                # maybe we have already processed an object for this cell
                # we will only consider one such object
                # overwrite the previous values
                labels[b, cell_number, :, :] = 0
                labels[b, cell_number, :, label] = 1

                # store the objectness
                objectness[b, cell_number, :, :] = 1

                # store the object sizes
                boxes[b, cell_number, :, :] = [rel_x, rel_y, sqr_size_x, sqr_size_y]

                # convert coordinates, they stay relative but we need to undo the square root
                # and scale the width and height by the network sizes
                # coordinates are given in the output activation map
                coords[b, cell_number, :, 0] = rel_x - 0.5 * sqr_size_x ** 2 * out_x
                coords[b, cell_number, :, 1] = rel_y - 0.5 * sqr_size_y ** 2 * out_y
                coords[b, cell_number, :, 2] = rel_x + 0.5 * sqr_size_x ** 2 * out_x
                coords[b, cell_number, :, 3] = rel_y + 0.5 * sqr_size_y ** 2 * out_y

            # for every cell, for every object, calculate the area
            # this is done by taking the upper left and lower right corner
            # they can then be substracted to get width and height
            # please note the difference between boxes and coordinates
            # the boxes are x,y,w,h
            # the coordinates are [x_min, y_min, x_max, y_max] -> the vertices of the box
            upper_left_corner[b] = coords[b, :, :, 0:2]
            lower_right_corner[b] = coords[b, :, :, 2:4]

            # calculate width and height
            width_height = lower_right_corner[b] - upper_left_corner[b];

            # calculate area
            area = width_height[:, :, 0] * width_height[:, :, 1]
            area = np.expand_dims(area, -1)

            areas[b, :] = area

        # asserts to make sure the arrays are correct
        assert np.all(areas>=0), "an object must not have negative area"

        # the huge blob of data
        data = [labels, objectness, areas, boxes, upper_left_corner, lower_right_corner]
        pointer = 0
        for item in data:
            length = item.shape[-1]
            blob[:, :, :, pointer:pointer + length] = item
            pointer += length

        assert pointer==blob.shape[-1], "data needs to fit exactly into the blob"
        assert not np.any(np.isnan(blob)), "no value should be nan"

        return batch, blob