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
out_x = 32
out_y = 32

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

#train_path = r"C:\Users\lhk\OneDrive\data\VOC\train.txt"
#test_path = r"C:\Users\lhk\OneDrive\data\VOC\2007_test.txt"

# iterator class to provide data to model.fit_generator
from ssd.generator import generate
batch_size = 64

# anchor boxes are taken from the tiny yolo voc config
anchors = np.zeros((B, 2))
anchors[:] = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]

# the anchors are given as width, height
# this doesn't work with numpy's layout
# we have to switch the x and y dimensions
temp = anchors[:, 0].copy()
anchors[:, 0] = anchors[:, 1]
anchors[:, 1] = temp

scale = 0.7
data_path = train_path


train_gen =  generate(in_x, in_y, out_x, out_y, scale, anchors, B, C, batch_size, data_path)

# test the generator
batch = next(train_gen)
imgs = batch[0]
objects = batch[1]

plt.imshow(imgs[0, :, :])

# # Loss function
# 
# The loss function makes use of currying. Therefore this code is a little complicated.
# Keras expects a loss in this format loss(y_true, y_pred).
# 
# But the loss_func in loss.py needs to know additional parameters such as the network size.
# I'm feeding that data by currying the loss_func and providing the additional parameters now.
# The result is a function with two remaining parameters and a signature as expected by keras.
# 
# This currying can go very wrong, if you mix up the order of the parameters.
# If the loss function is called, it prints the parameters it has been given.
# Be sure to check this.
# Look at model.compile.




from loss_function import loss_func

meta_data = [anchors, out_x, out_y, B, C, lambda_class, lambda_coords, lambda_obj, lambda_noobj]
loss = loss_func(*meta_data)

# # Training the model
# Compile with the custom loss, set up a few callbacks and train.




from keras.optimizers import Adam, SGD

# check this: are the parameters correct ?
detection_model.compile(Adam(lr=0.00005), loss)


# detection_model.compile(SGD(lr=1e-4, momentum=0.9, decay = 1e-7), loss)





# taken from the keras source
# if the learning rate is too fast, NaNs can occur, stop the training in this case
class TerminateOnNaN(Callback):
    def __init__(self):
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        self.seen += 1

        logs = logs or {}
        loss = logs.get('loss')

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                print("logs: ", logs)

                self.model.stop_training = True


nan_terminator = TerminateOnNaN()

# train in small steps and append histories
# if training is interrupted, the history array still contains usable data
import time

histories = []
times = []
for i in range(20):
    history = detection_model.fit_generator(train_gen, 6400 // batch_size,
                                            epochs=5,
                                            callbacks=[nan_terminator],
                                            validation_data=val_gen,
                                            validation_steps=1600 // batch_size,
                                            # use_multiprocessing=False)
                                            workers=4,
                                            max_queue_size=24)
    histories.append(history)
    times.append(time.time())

# ### Plot the test / val loss
# As you can see, the model reaches about 1000 for validation loss.
# Then it overfits.
# 
# This number can't be interpreted correctly. It depends on the size of the network and the batch.
# A solution would be to take the mean in the loss instead of summing all components.
# But that would mess with the learning rate.
# 
# I'm evaluating the pretrained model against the validation generator.
# Surprisingly, the new model reaches better scores.
# A possible explanation: The original yolo doesn't use rotations as augmentations. The validation generator uses rotations.
# Or the number of samples from the validation set was simply too small




losses = []
val_losses = []

for item in histories:
    losses.extend(item.history["loss"])
    val_losses.extend(item.history["val_loss"])

plt.plot(losses)
plt.plot(val_losses)
plt.legend(["train", "val"])
plt.title("loss")
plt.show()

extraction_model.compile(Adam(lr=0.0001), loss)

len(histories)

# how does the reference model deal with this loss function ?
extraction_model.evaluate_generator(val_gen, 12800 // batch_size, max_queue_size=20, workers=4,
                                    use_multiprocessing=False)

# # Evaluation

# ### Two important helper functions to work with the data
# with get_probabilities you can extract the predicted classes, objectness and the combined probability from the output of the network
# 
# the extract_from_blob helps with the blob of data fed to the keras loss.
# This blob is hard to read, so the function slices the individual parts out of it and converts them to a dictionary




# del extract_from_blob
# del get_probabilities
from utils.prediction import extract_from_blob, get_probabilities

test_gen = val_gen
# get some sample data
batch = next(test_gen)
img = batch[0].copy()

plt.imshow(img[0])
plt.show()

# feed the data to the model
predictions = detection_model.predict(batch[0])
predictions.shape

# ### Comparing given objectness with confidence of the network




# extract the given objectness for this image
loss_dict = extract_from_blob(batch[1], out_x, out_y, B, C)

# read the given objectness out of the loss dictionary
f_objectness = loss_dict["f_objectness"].reshape((-1, out_x, out_y, B))

# get the data out of the predictions
classes, objectness, probs = get_probabilities(predictions[0], out_x, out_y, B, C)

# probs is along the B dimension
# for every cell in the output activation map, get the best bounding box score
max_probs = probs.max(axis=-1)

threshold = 0.3
thresholded = max_probs > threshold

f, axes = plt.subplots(1, 3, figsize=(10, 10))

axes[0].imshow(f_objectness[0, :, :, 0])
axes[0].set_title("given objectness")

axes[1].imshow(max_probs)
axes[1].set_title("confidence")

axes[2].imshow(thresholded)
axes[2].set_title("thresholded")
plt.show()

# ### Getting the predicted bounding boxes




from nms import get_detections, apply_nms, idx_to_name

detections = get_detections(predictions[0], threshold, anchors, out_x, out_y, in_x, in_y, B, C)

print("number of detections: ", len(detections))

# ## Non-Max Suppression
# Sometimes yolo will predict the same object in more than one cell.
# This happens mostly for very big objects where the center of the object is not clear.
# 
# We need non-max suppression to remove overlapping bounding boxes.
# 
# We apply the non-max suppression to each label separately.




# taken from the yolo repository
names = ["aeroplane",
         "bicycle",
         "bird",
         "boat",
         "bottle",
         "bus",
         "car",
         "cat",
         "chair",
         "cow",
         "diningtable",
         "dog",
         "horse",
         "motorbike",
         "person",
         "pottedplant",
         "sheep",
         "sofa",
         "train",
         "tvmonitor"]

nms = apply_nms(detections, sess)

nms = idx_to_name(nms, names)

print("we found the following boxes after non-max suppression")
print(nms)

# ### Plotting the output
# I'm using opencv to draw rectangles around all detections
# and to write the name in text onto the image.
# 
# The image has a very low resolution.
# For the output it is upscaled.
# The main reason for this is to allow high-res text.




img = batch[0][0]
output_img = img.copy()
dim_x, dim_y = output_img.shape[:2]
factor = 5
output_img = cv2.resize(output_img, (dim_y * factor, dim_x * factor))

for label in nms:
    boxes = nms[label]
    for box in boxes:
        min_x, min_y, max_x, max_y = box
        min_x *= factor
        min_y *= factor
        max_x *= factor
        max_y *= factor

        cv2.rectangle(output_img, (min_y, min_x), (max_y, max_x), (0, 1, 0), 10)
        # cv2.rectangle(output_img,(min_y-100, min_x-100),(min_y + 100, min_x+100),(0,1,0),-1)
        cv2.putText(output_img, label, (min_y, min_x), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.5, color=(1, 1, 1),
                    thickness=12)

plt.figure(figsize=(10, 10))
plt.imshow(output_img)
plt.show()
