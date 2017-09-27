"""
This will mix parts of the SSD and YOLO loss formulation
"""

import numpy as np
import cv2

from lib.parser.parser import parse_image_label_pairs, parse_labels

from lib.preprocessing import preprocess_yolo, preprocess_vgg16
from lib.utils.object import wh_to_minmax, minmax_to_wh, split, merge
from lib.augmentations import augment

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

from lib.plot_utils import *


# from lib.augmentations import augment


class Augmenter:

    def __init__(self, data_path,
                 in_x, in_y, out_x_list, out_y_list,
                 scale_list, anchors, B, C,
                 batch_size,
                 IoU_threshold=0.5, config=None):

        assert len(out_x_list) == len(out_y_list) == len(scale_list), "the lists need to have the same length"
        self.num_outputs = len(out_x_list)

        # misc
        self.data_path = data_path
        self.batch_size = batch_size

        # network input and output size
        self.in_x = in_x
        self.in_y = in_y
        self.out_x_list = out_x_list
        self.out_y_list = out_y_list

        # sizes and scale of the anchor boxes
        self.scale_list = scale_list
        self.anchors = anchors

        # number of anchor boxes and classes
        self.B = B
        self.C = C

        # threshold to attribute object to box
        self.IoU_threshold = IoU_threshold

        # config for the augmentation
        if not config:
            config = {}
            config["max_hsv_scale"] = [0.1, 0.5, 0.5]
            config["max_rotation"] = 10
            config["max_shift"] = 0.05
            config["zoom_range"] = (0.8, 1.2)
        self.config = config

        # get a list of pairs of (image_path, label_path)
        self.image_label_pairs = parse_image_label_pairs(data_path)

        self.setup_default_boxes()

    def setup_default_boxes(self):
        """
        computes default boxes for every output
        takes into consideration the given anchors and scales
        :return:
        """
        # network input and output size
        in_x = self.in_x
        in_y = self.in_y
        out_x_list = self.out_x_list
        out_y_list = self.out_y_list

        # sizes and scale of the anchor boxes
        scale_list = self.scale_list
        anchors = self.anchors

        # number of anchor boxes and classes
        B = self.B
        C = self.C

        self.default_boxes_list = []
        self.default_upper_left_corner_list = []
        self.default_lower_right_corner_list = []
        self.default_areas_list = []

        for i in range(self.num_outputs):
            out_x = out_x_list[i]
            out_y = out_y_list[i]
            scale = scale_list[i]

            # in order to calculate whether a box "contains" an object,
            # we use the jacard overlap
            # for that we need to have the default sizes of the predicted boxes
            # and this is computed for every scale and all the resolutions
            scaled_anchors = anchors * scale

            # calculating coordinates and areas for the default boxes
            default_boxes = np.zeros((B, 2))
            default_boxes[:] = scaled_anchors

            self.default_boxes_list.append(default_boxes)

            # getting the x and y sizes, necessary to compute the box coordinates
            default_size_x = default_boxes[:, 0]
            default_size_y = default_boxes[:, 1]

            # for every cell in the grid of output activations, we place the default boxes around the cell
            # the default_boxes need to be moved to the center of the cell
            # the corresponding grid of displacements is created here
            m_grid = np.meshgrid(np.arange(out_x), np.arange(out_y), sparse=False, indexing='ij')
            x_grid = m_grid[0]
            y_grid = m_grid[1]

            x_grid = x_grid.reshape(out_x, out_y, 1)
            y_grid = y_grid.reshape(out_x, out_y, 1)

            x_grid = x_grid / out_x
            y_grid = y_grid / out_y

            # grid to store the coordinates of the default boxes
            default_coords = np.zeros((out_x, out_y, B, 4))
            default_coords[:, :, :, 0] = x_grid - 0.5 * default_size_x
            default_coords[:, :, :, 1] = y_grid - 0.5 * default_size_y
            default_coords[:, :, :, 2] = x_grid + 0.5 * default_size_x
            default_coords[:, :, :, 3] = y_grid + 0.5 * default_size_y

            default_coords = np.clip(default_coords, 0, 1)

            default_upper_left_corner = np.zeros((out_x, out_y, B, 2))
            default_lower_right_corner = np.zeros((out_x, out_y, B, 2))
            default_upper_left_corner[:] = default_coords[:, :, :, 0:2]
            default_lower_right_corner[:] = default_coords[:, :, :, 2:4]

            self.default_upper_left_corner_list.append(default_upper_left_corner)
            self.default_lower_right_corner_list.append(default_lower_right_corner)

            # calculate width and height
            default_width_height = default_lower_right_corner - default_upper_left_corner

            # calculate area
            default_areas = np.zeros((out_x, out_y, B, 1))
            default_areas[:, :, :, 0] = default_width_height[:, :, :, 0] * default_width_height[:, :, :, 1]

            self.default_areas_list.append(default_areas)

            # creating flat lists that contain all boxes, corners, areas
            # in order to determine which part of the lists belongs to which output layer,
            # the indices at which new layers start are recorded

    def __iter__(self):
        return self

    def __next__(self):
        """
        create augmented training data
        :return: a list of images, a list of blobs that can be parsed by the loss functions
        """

        batch_size = self.batch_size

        # network input and output size
        in_x = self.in_x
        in_y = self.in_y
        out_x_list = self.out_x_list
        out_y_list = self.out_y_list

        # sizes and scale of the anchor boxes
        scale_list = self.scale_list
        anchors = self.anchors

        # number of anchor boxes and classes
        B = self.B
        C = self.C

        # threshold to attribute object to box
        IoU_threshold = self.IoU_threshold

        config = self.config

        # get a list of pairs of (image_path, label_path)
        image_label_pairs = self.image_label_pairs

        # this is the batch fed to the network
        # only this is actual input to the forward pass
        # the other numpy arrays are for the calculation of the loss
        batch = np.zeros((batch_size, in_x, in_y, 3))

        # first we load all the images for this batch
        # the objects within the image are just stored in a list
        # this is a list of lists: one list for each image
        object_list = []

        # fill the batch
        for b in range(batch_size):
            idx = np.random.randint(len(image_label_pairs))
            pair = image_label_pairs[idx]

            # load image and object list
            img = cv2.imread(pair[0])
            img = cv2.resize(img, (in_y, in_x))  # opencv wants (width, height) in numpy, y corresponds to width
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

            # store the objects for further processing
            object_list.append(objects)

            batch[b] = img

        # now we have a list of lists for the objects
        # for every sample in the batch, there is a list of objects contained in this sample
        # for every object, we need to know which layer is supposed to predict it
        # go through the list again and determine the best overlaps

        # the assigned objects store: for every layer for every sample the list of objects
        # the mapping is: index of output layer -> nested list of objects (for every sample there's a list)
        # assigned_objects[layernumber] = [[objects for this layer in sample 1], [... in sample 2], [sample 3], ...]
        assigned_objects_batch = {layer: [] for layer in range(self.num_outputs)}

        for i in range(batch_size):

            # looking at the objects for sample i
            objects = object_list[i]

            # for this sample, the list of objects assigned to each layer
            # this will be inserted in the assigned_objects_batch
            assigned_objects = {}
            for layer in range(self.num_outputs):
                assigned_objects[layer] = []

            # determine which layer is supposed to store this
            for object in objects:
                # deconstruct the object
                label, cx, cy, size_x, size_y = object

                assert 0 <= cx <= 1, "x should be in [0,1]"
                assert 0 <= cy <= 1, "y should be in [0,1]"

                assert 0 <= size_x <= 1, "width should be in [0,1]"
                assert 0 <= size_y <= 1, "height should be in [0,1]"

                # now create a numpy array to store the object coordinates
                gt_coords = np.zeros((4,))
                gt_coords[0] = cx - 0.5 * size_x
                gt_coords[1] = cy - 0.5 * size_y
                gt_coords[2] = cx + 0.5 * size_x
                gt_coords[3] = cy + 0.5 * size_y

                gt_upper_left_corner = np.zeros((2,))
                gt_lower_right_corner = np.zeros((2,))
                gt_upper_left_corner[:] = gt_coords[0:2]
                gt_lower_right_corner[:] = gt_coords[2:4]

                # calculate width and height
                gt_width_height = gt_lower_right_corner - gt_upper_left_corner;

                # calculate area
                gt_area = gt_width_height[0] * gt_width_height[1]

                # go through all layers, determine which cell holds this object
                # calculate IoU and store it
                # after having calculated all IoU values, determine the best
                IoUs = []
                for layer in range(self.num_outputs):
                    out_x = out_x_list[layer]
                    out_y = out_y_list[layer]

                    # convert to network coordinates, [0, out_x] and [0, out_y]
                    obj_x = cx * out_x
                    obj_y = cy * out_y

                    # the coordinate should be relative to their cell
                    rel_x = obj_x - np.floor(obj_x)
                    rel_y = obj_y - np.floor(obj_y)

                    # the number of the corresponding cell
                    x_idx = np.floor(obj_x)
                    y_idx = np.floor(obj_y)
                    x_idx = int(x_idx)
                    y_idx = int(y_idx)
                    cell_number = x_idx * out_y + y_idx
                    cell_number = int(cell_number)

                    # get the default boxes for this layer
                    default_upper_left_corners = self.default_upper_left_corner_list[layer]
                    default_lower_right_corners = self.default_lower_right_corner_list[layer]
                    default_areas = self.default_areas_list[layer]

                    # and select the entries for this cell
                    default_upper_left_corner = default_upper_left_corners[x_idx, y_idx]
                    default_lower_right_corner = default_lower_right_corners[x_idx, y_idx]
                    default_area = default_areas[x_idx, y_idx]

                    # now compare the areas of the default boxes and the ground truth boxes
                    inner_upper_left = np.maximum(gt_upper_left_corner, default_upper_left_corner)
                    inner_lower_right = np.minimum(gt_lower_right_corner, default_lower_right_corner)
                    diag = inner_lower_right - inner_upper_left
                    diag = np.maximum(diag, 0.)
                    intersection = diag[:, 0] * diag[:, 1]

                    # reshaping to align for broadcasting
                    intersection = intersection.reshape((B, 1))
                    default_area = default_area.reshape((B, 1))

                    # IoU with smoothing to prevent division by zero
                    union = default_area + gt_area - intersection
                    union = np.maximum(union, 0.)
                    eps = 0.01
                    IoU = intersection / (eps + union)

                    IoUs.append(IoU)

                # reshape IoUs to expose dimension for the layer
                IoUs = [IoU.reshape((1, B)) for IoU in IoUs]

                # convert this to a numpy array
                IoU_concatenated = np.concatenate(IoUs, axis=0)

                # determine the best fitting default box
                best_IoU = IoU_concatenated.argmax()
                best_IoU = np.unravel_index(best_IoU, IoU_concatenated.shape)

                best_layer = best_IoU[0]
                best_box = best_IoU[1]

                assigned_objects[best_layer].append((object, best_box))

            # write these objects into the outer layer
            for layer in range(self.num_outputs):
                assigned_objects_batch[layer].append(assigned_objects[layer])

        # for every output, there's an individual blob for the corresponding loss function
        blobs = []

        # go through all the outputs and fill their blobs
        for i in range(self.num_outputs):
            out_x = out_x_list[i]
            out_y = out_y_list[i]
            scale = scale_list[i]
            scaled_anchors = anchors * scale

            # what are the objects that have to be predicted by this layer ?
            assigned_objects = assigned_objects_batch[i]

            # this is a one-hot encoding of the labels of the objects
            labels = np.zeros([batch_size, out_x * out_y, B, C])

            # for every box: is there an object in this box
            objectness = np.zeros([batch_size, out_x * out_y, B, 1])

            # the ground_truth coordinates of the objects in this batch
            gt_coords = np.zeros([batch_size, out_x * out_y, B, 4])
            gt_upper_left_corner = np.zeros([batch_size, out_x * out_y, B, 2])
            gt_lower_right_corner = np.zeros([batch_size, out_x * out_y, B, 2])
            gt_areas = np.zeros([batch_size, out_x * out_y, B, 1])

            # the values to be predicted by the neural network
            # this is the regression target for the coordinate prediction
            # we predict 4 coordinates for every box
            target_coords = np.zeros([batch_size, out_x * out_y, B, 4])

            # the loss formulation accepts a single tensor
            # this blob contains all the data we need, in the loss function, we slice the following out of it:
            # a one-hot encoding of the C classes -> C entries per box
            # a binary encoding of objectness -> 1 entry per box
            # the coordinates to be predicted -> 4 entries per box
            #
            # this blob is also passed to the network, for the loss
            blob = np.zeros((batch_size, out_x * out_y, B, C + 5))

            # go through the batch dimension and fill the numpy arrays
            for b in range(batch_size):

                # the assigned_objects have to be predicted by this layer,
                # assigned_objects[b] correspond to the current sample in the batch
                objects = assigned_objects[b]

                # this is a preparation step which looks at every object and converts the gt data to a more usable format
                processed_objects = []
                for obj, box_index in objects:

                    # deconstruct the object
                    label, cx, cy, size_x, size_y = object
                    label = int(label)

                    assert 0 <= cx <= 1, "x should be in [0,1]"
                    assert 0 <= cy <= 1, "y should be in [0,1]"

                    assert 0 <= size_x <= 1, "width should be in [0,1]"
                    assert 0 <= size_y <= 1, "height should be in [0,1]"

                    # convert to network coordinates, [0, out_x] and [0, out_y]
                    obj_x = cx * out_x
                    obj_y = cy * out_y

                    # the coordinate should be relative to their cell
                    rel_x = obj_x - np.floor(obj_x)
                    rel_y = obj_y - np.floor(obj_y)

                    # the number of the corresponding cell
                    x_idx = np.floor(obj_x)
                    y_idx = np.floor(obj_y)
                    x_idx = int(x_idx)
                    y_idx = int(y_idx)
                    cell_number = x_idx * out_y + y_idx
                    cell_number = int(cell_number)

                    # now fill the arrays that will later be combined into the blob

                    # maybe we have already processed an object for this cell
                    # we will only consider one such object
                    # overwrite the previous values
                    labels[b, cell_number, :, :] = 0
                    labels[b, cell_number, :, label] = 1

                    # the index of the best default box has already been determined
                    objectness[b, cell_number, box_index, 0] = 1

                    xy_idx = np.s_[b, cell_number, :, :2]
                    wh_idx = np.s_[b, cell_number, :, 2:]

                    target_coords[xy_idx] = rel_x, rel_y

                    # width and height, scaled by default box, scaled by log
                    target_coords[wh_idx] = size_x, size_y

                    target_coords[wh_idx] = target_coords[wh_idx] / scaled_anchors
                    target_coords[wh_idx] = np.log(target_coords[wh_idx])

                    vis = True
                    if vis:
                        canvas = create_canvas(100, 100, True)
                        draw_rect(canvas, (cx, cy, size_x, size_y), (1, 0, 0), 1)
                        scaled_anchors = anchors * scale
                        box_idx = 0
                        wh = scaled_anchors[box_idx]
                        draw_rect(canvas, (x_idx/out_x, y_idx/out_y, *wh), (0,1,0), 1)
                        plot_canvas(canvas)
                        print("debug mark")

            # the huge blob of data
            data = [labels, objectness, target_coords]
            pointer = 0
            for item in data:
                length = item.shape[-1]
                blob[:, :, :, pointer:pointer + length] = item
                pointer += length

            assert pointer == blob.shape[-1], "data needs to fit exactly into the blob"
            assert not np.any(np.isnan(blob)), "no value should be nan"

            blobs.append(blob)

        batch = preprocess_input(batch)
        return batch, blobs
