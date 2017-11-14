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

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import Callback, ModelCheckpoint

# ### allow dynamic memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# ### Load the pretrained model

extraction_model = VGG16(include_top=False, input_shape=(448, 448, 3))


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


block4_conv1 = extraction_model.get_layer(name="block4_conv3")
pool1 = MaxPool2D()(block4_conv1.output)
head1 = head_model(pool1)

block4_conv3 = extraction_model.get_layer(name="block4_conv3")
pool1 = MaxPool2D()(block4_conv3.output)
pool2 = MaxPool2D()(pool1)
head2 = head_model(pool2)

block5_conv1 = extraction_model.get_layer(name="block5_conv3")
pool1 = MaxPool2D()(block5_conv1.output)
pool2 = MaxPool2D()(pool1)
head3 = head_model(pool2)

block5_conv3 = extraction_model.get_layer(name="block5_conv3")
pool1 = MaxPool2D()(block5_conv3.output)
pool2 = MaxPool2D()(pool1)
pool3 = MaxPool2D()(pool2)
head4 = head_model(pool3)

#block8_pool1 = MaxPool2D()(block7_pool1)
#head_8_1 = head_model(block8_pool1)


#detection_model = Model(inputs=extraction_model.input, outputs=[head1, head2, head3, head4])
detection_model = Model(inputs=extraction_model.input, outputs=[head1, head2, head3])
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


#scale_list = [0.2, 0.4, 0.75, 1]
scale_list = [0.2, 0.4, 0.75]
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
from ssd_playground.mixed_generator import Augmenter

anchors = np.zeros((B, 2))
anchors[:] = [[0.9, 0.35], [0.8, 0.45], [0.6, 0.6], [0.45, 0.8], [0.35,0.9]]

batch_size = 12

train_gen =  Augmenter(train_path, in_x, in_y, out_x_list, out_y_list, scale_list, anchors, B, C, batch_size)
test_gen =  Augmenter(test_path, in_x, in_y, out_x_list, out_y_list, scale_list, anchors, B, C, batch_size)


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
from ssd.mixed_loss_function import loss_func

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
    detection_model.compile(Adam(lr=0.00003), loss=loss_functions)


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
    checkpoint_callback = ModelCheckpoint("models/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                          monitor='val_loss', verbose=0, save_best_only=False,
                                          save_weights_only=True, mode='auto', period=1)

    history = detection_model.fit_generator(generator = train_gen,
                                            steps_per_epoch = 400,
                                            epochs=100,
                                            callbacks=[nan_terminator, checkpoint_callback],
                                            validation_data=test_gen,
                                            validation_steps=20,
                                            # use_multiprocessing=False)
                                            workers=6,
                                            max_queue_size=30)

    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["train", "val"])
    plt.title("loss")
    plt.show()

    detection_model.save_weights("models/vgg16_1.h5")

    print("finished training")

else:
    # with open("models/detection_model.json") as json_file:
    #    json_string = json_file.read()
    #    detection_model = model_from_json(json_string)
    detection_model.load_weights("models/vgg16_1.h5")

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

np.random.seed(0)

indices = {}

from tqdm import tqdm
for i in tqdm(range(50)):
    # get some sample data
    batch = next(test_gen)
    imgs = batch[0]
    blobs = batch[1]

    # feed the data to the model
    predictions = detection_model.predict(imgs)

    # list of all detections
    detections=[]

    # process every output individually
    for i in range(num_outputs):

        prediction = predictions[i]
        out_x = out_x_list[i]
        out_y = out_y_list[i]
        scale = scale_list[i]

        blob = blobs[i]

        # we only look at the first prediction
        prediction = prediction.reshape((-1, out_x, out_y, B, C + 5))
        prediction = prediction[0]
        img = imgs[0]

        # ### Comparing given objectness with confidence of the network

        # extract the given objectness for this image
        loss_dict = extract_from_blob(blob, out_x, out_y, B, C)

        # read the given objectness out of the loss dictionary
        f_objectness = loss_dict["f_objectness"].reshape((-1, out_x, out_y, B))

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

        axes[0].imshow(f_objectness[0, :, :, 0])
        axes[0].set_title("given objectness")

        axes[1].imshow(max_probs)
        axes[1].set_title("confidence")

        axes[2].imshow(thresholded)
        axes[2].set_title("thresholded")

        axes[3].imshow(img)
        axes[3].set_title("original image")
        plt.show()
