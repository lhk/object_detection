"""
This will mix parts of the SSD and YOLO loss formulation
"""

import numpy as np
import cv2

from lib.parser.parser import parse_image_label_pairs, parse_labels

from lib.preprocessing import preprocess_yolo
from lib.utils.object import wh_to_minmax, minmax_to_wh, split, merge
#from lib.augmentations import augment


def generate(in_x, in_y, out_x, out_y, scale, anchors, B, C, batch_size, data_path, IoU_threshold=0.5, config=None):
    # todo: in_x, out_x and scale are maybe redundant. can I calculate one value from the other two ?

    # config for the augmentation
    if not config:
        config={}
        config["max_hsv_scale"]=[0.1, 0.5, 0.5]
        config["max_rotation"]=10
        config["max_shift"]=0.05
        config["zoom_range"]=(0.8,1.2)

    # get a list of pairs of (image_path, label_path)
    image_label_pairs=parse_image_label_pairs(data_path)


    while True:

        # just for debugging, if you need this to be deterministic
        #np.random.seed(0)

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

        # this stores the object coordinates
        # coordinates will go from 0 to out_x and 0 to out_y
        # attention: this is only used for computing the corners and areas
        # it is not passed to the loss
        # the loss works on the boxes array
        coords = np.zeros([batch_size, out_x * out_y, B, 4])

        # corners of the gt box
        gt_upper_left_corner = np.zeros([batch_size, out_x * out_y, B, 2])
        gt_lower_right_corner = np.zeros([batch_size, out_x * out_y, B, 2])

        # areas of the gt boxes
        gt_areas = np.zeros([batch_size, out_x * out_y, B, 1])

        # desired coordinate predictions
        # TODO: this is important, if you change this, you need to adapt the evaluation
        # I'm using the name from the SSD paper: g_hat is used for an L1 loss with the output of the layer
        g_hat = np.zeros([batch_size, out_x * out_y, B, 4])

        # the container to store all of this
        # this is passed to the loss function, the different parts are then sliced out of this blob of data
        # the C+10 is the sum of the sizes of the last dimension of everything passed ot the loss
        blob = np.zeros((batch_size, out_x * out_y, B, C + 5))

        # calculating coordinates and areas for the default boxes
        default_boxes = np.zeros((B, 2))
        default_boxes[:] = anchors * scale
        default_size_x = default_boxes[:, 0]
        default_size_y = default_boxes[:, 1]

        default_coords = np.zeros((B, 4))
        default_coords[:, 0] = 0 - 0.5 * default_size_x * out_x
        default_coords[:, 1] = 0 - 0.5 * default_size_y * out_y
        default_coords[:, 2] = 0 + 0.5 * default_size_x * out_x
        default_coords[:, 3] = 0 + 0.5 * default_size_y * out_y

        default_upper_left_corner = np.zeros((B, 2))
        default_lower_right_corner = np.zeros((B, 2))
        default_upper_left_corner[:] = default_coords[:,0:2]
        default_lower_right_corner[:] = default_coords[:,2:4]

        # calculate width and height
        default_width_height = default_lower_right_corner - default_upper_left_corner

        # calculate area
        default_area = default_width_height[:, 0] * default_width_height[:,  1]
        default_area = np.expand_dims(default_area, -1)

        default_areas = np.zeros((B, 1))
        default_areas[:] = default_area



        # fill the batch
        for b in range(batch_size):

            idx = np.random.randint(len(image_label_pairs))
            pair = image_label_pairs[idx]

            # load image and object list
            img = cv2.imread(pair[0])
            img = cv2.resize(img, (in_y, in_x)) # opencv wants (width, height) in numpy, y corresponds to width
            objects = parse_labels(pair[1])

            # convert image to yolo input format
            img = preprocess_yolo(img)

            batch[b] = img

            # this is a preparation step which looks at every object and converts the gt data to a more usable format
            # for every object we want to know
            # 1. the label
            # 2. the cell in which this object is present
            # 3. the offsets and sizes of the object
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
                x_idx = np.floor(obj_x)
                y_idx = np.floor(obj_y)
                cell_number = x_idx * out_y + y_idx
                cell_number = int(cell_number)

                # extract width and height
                size_x = obj[3]
                size_y = obj[4]

                assert 0 <= size_x <= 1, "width should be in [0,1]"
                assert 0 <= size_y <= 1, "height should be in [0,1]"

                # now fill some of the data arrays, coordinates are in [0, out_x]
                coords[b, cell_number, :, 0] = rel_x - 0.5 * size_x * out_x
                coords[b, cell_number, :, 1] = rel_y - 0.5 * size_y * out_y
                coords[b, cell_number, :, 2] = rel_x + 0.5 * size_x * out_x
                coords[b, cell_number, :, 3] = rel_y + 0.5 * size_y * out_y

                # plug all of this together
                processed_object = [cell_number, label, rel_x, rel_y, size_x, size_y, x_idx, y_idx]
                processed_objects.append(processed_object)



            # for every cell, for every object, calculate the area
            # this is done by taking the upper left and lower right corner
            # they can then be substracted to get width and height
            # please note the difference between boxes and coordinates
            # the boxes are x,y,w,h
            # the coordinates are [x_min, y_min, x_max, y_max] -> the vertices of the box
            gt_upper_left_corner[b] = coords[b, :, :, 0:2]
            gt_lower_right_corner[b] = coords[b, :, :, 2:4]

            # calculate width and height
            gt_width_height = gt_lower_right_corner[b] - gt_upper_left_corner[b];

            # calculate area
            gt_area = gt_width_height[:, :, 0] * gt_width_height[:, :, 1]
            gt_area = np.expand_dims(gt_area, -1)

            gt_areas[b, :] = gt_area

            # now compare the areas of the default boxes and the ground truth boxes
            inner_upper_left = np.maximum(gt_upper_left_corner, default_upper_left_corner)
            inner_lower_right = np.minimum(gt_lower_right_corner, default_lower_right_corner)
            diag = inner_lower_right - inner_upper_left
            diag = np.maximum(diag, 0.)
            intersection = diag[:, :, :, 0] * diag[:, :, :, 1]

            # align shapes to the layout of the gt areas
            default_areas = default_areas.reshape((1, 1, B, 1))
            intersection = intersection.reshape((batch_size, out_x * out_y, B, 1))

            # IoU with smoothing to prevent division by zero
            union = default_areas + gt_areas - intersection
            union = np.maximum(union, 0.)
            eps=0.01
            IoU = intersection / (eps + union)

            IoU_img = IoU.reshape((-1, out_x, out_y, B, 1))

            visualize = False
            if visualize:
                import matplotlib.pyplot as plt

                f, axes = plt.subplots(1, 2)
                axes[0].imshow(img)
                axes[1].imshow(IoU_img[b,:,:,0,0])
                plt.show()

            # attention: every cell in the output can predict at most 1 object
            # if there is more than one object in the cell, later objects will override earlier objects
            # we do the following
            # 1. fill the labels array with a one-hot vector for the classes
            # 2. store the objectness
            # 3. store the object sizes
            # 4. compare the object to the default boxes and determine which default boxes are active

            # TODO: the objectness determines wether predictions are considered in the loss function
            # in yolo, the prediction with the highest IoU will be chosen
            # that is chosen after the forward pass
            # in ssd the boxes are chosen offline, here in the code
            # in the loss function, no IoU will be computed
            # that means that we no longer pass objectness = 1 for every box
            # but rather determine which boxes have sufficient overlap first
            for obj in processed_objects:
                cell_number, label, rel_x, rel_y, size_x, size_y, x_idx, y_idx = obj

                # maybe we have already processed an object for this cell
                # we will only consider one such object
                # overwrite the previous values
                labels[b, cell_number, :, :] = 0
                labels[b, cell_number, :, label] = 1

                # store the objectness
                objectness[b, cell_number, :, :] = IoU[b, cell_number, :, :] > IoU_threshold
                objectness[b, cell_number, IoU[B, cell_number].argmax(), :]=1

                # the target for the bounding box regression
                # x and y coordinates: an offset to the cell's center, scaled by default box width and height
                g_hat[b, cell_number, :, 0] = rel_x
                g_hat[b, cell_number, :, 1] = rel_y

                # width and height, scaled by default box, scaled by log
                g_hat[b, cell_number, :, 2] = size_x
                g_hat[b, cell_number, :, 3] = size_y

                g_hat[b, cell_number, :, 2:3] *= (out_x, out_y)
                g_hat[b, cell_number, :, 2:3] /= anchors
                g_hat[b, cell_number, :, 2:3] = np.log(g_hat[b, cell_number, :, 2:3])


                #g_hat[b, cell_number, :, 2] = np.log(size_x / default_boxes[:, 0])
                #g_hat[b, cell_number, :, 3] = np.log(size_y / default_boxes[:, 1])

                # visualization
                visualize = False
                if visualize:
                    if np.any(objectness[b, cell_number, :, :] == 1):
                        idxes = np.where(objectness[b, cell_number, :, :] == 1)
                        for (i,j) in zip(idxes[0], idxes[1]):
                            box = idxes[i][j]

                            temp = np.zeros((out_x ,out_x))
                            x_min = coords[b, cell_number, box, 0] + x_idx
                            y_min = coords[b, cell_number, box, 1] + y_idx
                            x_max = coords[b, cell_number, box, 2] + x_idx
                            y_max = coords[b, cell_number, box, 3] + y_idx

                            x_min = int(x_min)
                            y_min = int(y_min)
                            x_max = int(x_max)
                            y_max = int(y_max)

                            temp[x_min:x_max, y_min:y_max]=1

                            import matplotlib.pyplot as plt

                            f, axes = plt.subplots(2, 2)
                            axes[0, 0].imshow(img)
                            axes[0, 1].imshow(IoU_img[b, :, :, 0, 0])
                            axes[1, 0].imshow(temp)
                            plt.show()

                            print("found one")


        # asserts to make sure the arrays are correct
        assert np.all(gt_areas>=0), "an object must not have negative area"

        # the huge blob of data
        data = [labels, objectness, g_hat]
        pointer = 0
        for item in data:
            length = item.shape[-1]
            blob[:, :, :, pointer:pointer + length] = item
            pointer += length

        assert pointer==blob.shape[-1], "data needs to fit exactly into the blob"
        assert not np.any(np.isnan(blob)), "no value should be nan"

        yield batch, blob