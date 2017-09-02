"""
the different networks need different preprocessing.
if necessary, a preprocessing function should be accompanied with a corresponding postprocessing.
If the output of the preprocessing can't be drawn with pyplot, there should be a postprocessing function.
"""
import numpy as np

def preprocess_vgg16(img):
    # vgg16 expects image in BGR format
    # and a shift is applied
    img = img.astype(np.float32)
    offset = [103.939, 116.779,123.68]
    img -= offset
    return img

def postprocess_vgg16(img):
    img = img.copy()
    
    # undo the color shift
    offset = [103.939, 116.779,123.68]
    img += offset
    
    # convert to rgb
    img = img[:,:,:,::-1]
    
    # matplotlib expects colors in [0,1]
    img /= 255
    
    return img

def preprocess_yolo(img):
    # yolo expects images in RGB
    # and color values in [0,1]
    img = img.astype(np.float32)
    img = img[:,:,::-1]
    img /= 255
    return img