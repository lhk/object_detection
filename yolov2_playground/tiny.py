import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2

import sys

sys.path.insert(0, "/home/lars/libraries/keras/")
import keras

assert keras.__version__[0] == "2", "we work on version 2 of keras"

from keras.layers import Input
from keras.layers import BatchNormalization, SpatialDropout2D
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model

import keras.backend as K

from keras.callbacks import Callback, ModelCheckpoint

# ### allow dynamic memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# ### Load the pretrained model

extraction_model = load_model("models/tiny_yolo_voc.h5")

# the pretrained weights shouldn't be updated any more
# I'm only using them for feature extraction
for layer in extraction_model.layers:
    layer.trainable = False

from keras.layers.merge import Concatenate

block3 = extraction_model.get_layer(name="max_pooling2d_4").output
block4 = extraction_model.get_layer(name="max_pooling2d_5").output
block5 = extraction_model.get_layer(name="max_pooling2d_6").output

block3_resized = MaxPool2D((2,2), name="e_maxpool1")(block3)
block4_resized = block4
block5_resized = block5

shape3 = [int(dim) for dim in block3_resized.shape[1:3]]
shape4 = [int(dim) for dim in block4_resized.shape[1:3]]
shape5 = [int(dim) for dim in block5_resized.shape[1:3]]

assert shape3 == shape4, "resolution must be identical"
assert shape4 == shape5, "resolution must be identical"

#extracted_features = Concatenate(axis=-1)([
#    block3_resized,
#    block4_resized,
#    block5_resized])
extracted_features = block5_resized
extracted_features.shape.as_list()

B = 5  # number of anchor boxes
C = 20  # number of classes



# build a model for the head
head_input = Input((None, None, 512))

# block 1
conv = Conv2D(1024, 3,
              padding="same",
              use_bias=False,
              kernel_regularizer=keras.regularizers.l2(0.0005),
              name="head_conv1")(head_input)
conv = BatchNormalization(name="head_bnorm1")(conv)
conv = LeakyReLU(0.1, name="head_lrelu1")(conv)
conv = SpatialDropout2D(0.3)(conv)

# block 2
conv = Conv2D(1024, 3,
              padding="same",
              use_bias=False,
              kernel_regularizer=keras.regularizers.l2(0.0005),
              name="head_conv2")(conv)
conv = BatchNormalization(name="head_bnorm2")(conv)
conv = LeakyReLU(0.1, name="head_lrelu2")(conv)
conv = SpatialDropout2D(0.15)(conv)

# output
conv = Conv2D(B * (C + 5), 1,
              padding="same",
              use_bias=True,
              kernel_regularizer=keras.regularizers.l2(0.0005),
              name="head_conv3")(conv)
head_output = conv
head_model = Model(inputs=head_input, outputs=head_output)

detection_head = head_model(extracted_features)

detection_model = Model(inputs=extraction_model.input, outputs=[detection_head])
detection_model.summary()
# ### parameters of the model
#
# These are training parameters.
# The input and output resolution are important for setting up the boxes as loss for training.
# The lambdas are factors to weigh the different loss components against each other.


input_tensor = detection_model.input

in_x = int(input_tensor.shape[1])
in_y = int(input_tensor.shape[2])

out_x_list=[]
out_y_list=[]

for output_tensor in detection_model.outputs:
    out_x = int(output_tensor.shape[1])
    out_y = int(output_tensor.shape[2])

    out_x_list.append(out_x)
    out_y_list.append(out_y)


scale_list = [0.1]
assert len(out_x_list)==len(out_y_list)==len(scale_list), "specific number of outputs"
num_outputs = len(scale_list)

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
# iterator class to provide data to model.fit_generator
# from ssd.ssd_generator import generate
from yolov2_playground.mixed_generator import Augmenter
from lib.preprocessing import preprocess_yolo, postprocess_yolo

# anchor boxes are taken from the tiny yolo voc config
anchors = np.zeros((B, 2))
anchors[:] =[[1.08,1.19],  [3.42,4.41],  [6.63,11.38],  [9.42,5.11],  [16.62,10.52]]

# the anchors are given as width, height
# this doesn't work with numpy's layout
# we have to switch the x and y dimensions
temp = anchors[:,0].copy()
anchors[:,0]=anchors[:,1]
anchors[:,1]= temp

batch_size = 64

train_gen =  Augmenter(train_path, in_x, in_y, out_x_list, out_y_list, scale_list, anchors, B, C, batch_size, preprocess_yolo)
test_gen =  Augmenter(test_path, in_x, in_y, out_x_list, out_y_list, scale_list, anchors, B, C, batch_size, preprocess_yolo)


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

# from ssd.ssd_loss_function import loss_func
from yolov2_playground.mixed_loss_function import loss_func

loss_functions=[]
for i in range(num_outputs):
    out_x = out_x_list[i]
    out_y = out_y_list[i]

    meta_data = [anchors, out_x, out_y, B, C, lambda_class, lambda_coords, lambda_obj, lambda_noobj]
    loss = loss_func(*meta_data)

    loss_functions.append(loss)

# # Training the model
# Compile with the custom loss, set up a few callbacks and train.


from keras.optimizers import Adam, SGD

from keras.models import model_from_json
from keras.callbacks import  ModelCheckpoint
# check this: are the parameters correct ?

training = True
if training:
    detection_model.compile(Adam(lr=0.00001), loss=loss_functions)


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


    # callbacks for the model
    nan_terminator = TerminateOnNaN()
    checkpoint_callback = ModelCheckpoint("models/checkpoints/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5",
                                          monitor='val_loss', verbose=0, save_best_only=False,
                                          save_weights_only=True, mode='auto', period=2)

    training_results = detection_model.fit_generator(generator = train_gen,
                                            steps_per_epoch = 30,
                                            epochs=200,
                                            callbacks=[nan_terminator, checkpoint_callback],
                                            validation_data=test_gen,
                                            validation_steps=30,
                                            # use_multiprocessing=False)
                                            workers=6,
                                            max_queue_size=3*batch_size)


    plt.plot(training_results.history["loss"])
    plt.plot(training_results.history["val_loss"])
    plt.legend(["train", "val"])
    plt.title("loss")
    plt.show()

    print("finished training")

else:
    # with open("models/detection_model.json") as json_file:
    #    json_string = json_file.read()
    #    detection_model = model_from_json(json_string)
    detection_model.load_weights("models/checkpoints/weights.99-540.66-535.19.hdf5")

# # Evaluation

# ### Two important helper functions to work with the data
# with get_probabilities you can extract the predicted classes, objectness and the combined probability from the output of the network
#
# the extract_from_blob helps with the blob of data fed to the keras loss.
# This blob is hard to read, so the function slices the individual parts out of it and converts them to a dictionary


from lib.utils.activations import softmax, np_sigmoid

# del extract_from_blob
# del get_probabilities
from lib.utils.ssd_prediction import extract_from_blob, get_probabilities
from lib.name_table import names

np.random.seed(0)

indices = {}

from tqdm import tqdm
for i in tqdm(range(50)):
    # get some sample data
    batch = next(test_gen)
    imgs = batch[0]
    imgs = postprocess_yolo(imgs)

    blobs = batch[1]

    # feed the data to the model
    predictions = detection_model.predict(imgs)

    # list of all detections
    detections=[]

    for batch_index in range(batch_size):

        # process every output individually
        for i in range(num_outputs):

            prediction = predictions[i]
            out_x = out_x_list[i]
            out_y = out_y_list[i]
            scale = scale_list[i]

            blob = blobs[i]

            # look at the prediction corresponding to this batch
            prediction = prediction.reshape((-1, out_x, out_y, B, C + 5))
            prediction = prediction[batch_index]
            img = imgs[batch_index]

            # ### Comparing given objectness with confidence of the network

            # extract the given objectness for this image
            loss_dict = extract_from_blob(blob, out_x, out_y, B, C)

            # read the given data out of the blob
            # I add an f_ to everything that was fed to the network
            f_objectness = loss_dict["f_objectness"].reshape((-1, out_x, out_y, B))
            f_labels = loss_dict["f_labels"].reshape((-1, out_x, out_y, B, C))

            classes = prediction[:, :, :, 5:]
            classes = softmax(classes)
            max_classes = classes.max(axis=-1)

            objectness = np_sigmoid(prediction[:, :, :, 4])

            probs = max_classes * objectness

            # probs is along the B dimension
            # for every cell in the output activation map, get the best bounding box score
            max_probs = probs.max(axis=-1)

            threshold = 0.3
            thresholded = max_probs > threshold

            # which coordinates are bigger than the threshold ?
            batch_row_col = np.where(thresholded)

            f, axes = plt.subplots(1, 4, figsize=(10, 10))

            contains_object = f_objectness[batch_index].max()
            print("objectness:" + str(contains_object))

            # let's look at the objects we are given here
            object_indices = np.where(f_objectness[batch_index]==1)
            for x,y,b in zip(object_indices[0], object_indices[1], object_indices[2]):
                # label is encoded as a one-hot vector at this position
                label = np.argmax(f_labels[batch_index, x, y, b])
                label = int(label)
                name = names[label]
                print(name)

                class_vector = classes[x,y,b]
                class_idx = np.argmax(class_vector)

                print("highest prediction is: " + str(names[class_idx]))

            axes[0].imshow(f_objectness[batch_index].sum(axis=-1))
            axes[0].set_title("given objectness")

            axes[1].imshow(max_probs)
            axes[1].set_title("confidence")

            axes[2].imshow(thresholded)
            axes[2].set_title("thresholded")

            axes[3].imshow(img)
            axes[3].set_title("original image")
            plt.show()

            debug_mark = 0
