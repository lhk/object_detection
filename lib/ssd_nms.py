"""
The output of YOLO contains probabilities but no discrete object detections.
This file helps to look for all detections above a certain threshold.
In a next step, non-max-suppression needs to be applied to these detections.
The nms algorithm is here, too.
"""
import numpy as np
import tensorflow as tf

from lib.utils.ssd_prediction import extract_from_blob, get_probabilities
from lib.utils.activations import np_sigmoid, softmax

def get_detections(predictions, threshold, anchors, out_x, out_y, in_x, in_y, B, C):
    
    predictions = predictions.reshape((-1, out_x, out_y, B, C+5))
    assert predictions.shape[0] == 1, "this doesn't work on batches"
    
    predictions = predictions[0] # drop the batch dimension
    
    # get the data out of the predictions
    classes, objectness, probs = get_probabilities(predictions, out_x, out_y, B, C)
    
    # probs is along the B dimension
    # for every cell in the output activation map, get the best bounding box score
    max_probs = probs.max(axis=-1)
    
    thresholded = max_probs > threshold
    # which coordinates are bigger than the threshold ?
    xy = np.where(thresholded)
    
    detections=[]
    # look at all the coordinates found by the thresholding
    for row, col in zip(xy[0], xy[1]):

        # for this coordinate, find the box with the highest objectness
        current_probs = objectness[row, col]
        box_idx = np.argmax(current_probs)
        box = predictions[row, col, box_idx]

        default_box = anchors[box_idx]
        df_w = default_box[0]
        df_h = default_box[1]

        # get the predicted coordinates, convert them to percent
        # this is the same code as in the generator and the loss function
        # the network learns to predict coordinates encoded in this way
        p_x = (row + box[0]*df_w) / out_x
        p_y = (col + box[1]*df_h) / out_y
        p_dx = (np.exp(box[2])) * df_w / out_x
        p_dy = (np.exp(box[3])) * df_h / out_y

        # resize the predicted coordinates to the input resolution
        min_x = int ((p_x - p_dx/2.) * in_x)
        max_x = int ((p_x + p_dx/2.) * in_x)
        min_y = int ((p_y - p_dy/2.) * in_y)
        max_y = int ((p_y + p_dy/2.) * in_y)

        # clip them to the image size
        min_x = max(min_x, 0)
        max_x = min(max_x, in_x)
        min_y = max(min_y, 0)
        max_y = min(max_y, in_y)

        # get the highest class prediction
        current_classes = classes[row, col, box_idx]
        label = np.argmax(current_classes)


        detections.append((label, min_x, max_x, min_y, max_y, current_probs.max()))
        
    return detections

def apply_nms(detections, session, iou_threshold=0.2):
    # sort the detections
    #create a dictionary that maps labels to detections and their confidence scores
    label_dict={}

    for detection in detections:

        label, min_x, max_x, min_y, max_y, score = detection

        if label in label_dict:
            label_dict[label].append(((min_x, min_y, max_x, max_y), score))
        else:
            label_dict[label] = [((min_x, min_y, max_x, max_y), score)]
            
    # create a new dictionary. Again, it maps labels to detections
    # but the detections are now filtered with non-max suppression
    nms = {}

    for label in label_dict:
        boxes = [box for (box, score) in label_dict[label]]
        scores = [score for (box, score) in label_dict[label]]

        # tensorflow has a built-in algorithm for non-max suppresion
        # the result is an array of indexes into the list of boxes
        # those indices are the chosen / retained boxes
        # unfortunately, the list is a tensor
        # we need a session to evaluate the tensor
        # at the very top of this notebook we have created this session
        idx = tf.image.non_max_suppression(boxes, scores, 5, iou_threshold=iou_threshold)
        idx = session.run(idx)

        # boxes we keep
        boxes = [boxes[i] for i in idx]

        nms[label] = boxes
        
    return nms
        
def idx_to_name(detections, names):
    # convert a mapping from idx -> value
    # to name -> value
    # by making a lookup in names list
    detections = {names[key]: value for key,value in detections.items()}
    
    return detections