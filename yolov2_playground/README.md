# object_detection
This is a port of [YOLO](https://pjreddie.com/darknet/) to tensorflow and keras.

There are many existing ports, such as:
 - https://github.com/allanzelener/YAD2K
 - https://github.com/thtrieu/darkflow

These ports are much more polished and feature complete.

I'm using YOLO for a student project and wanted to really understand how it works.
This repository is intended as a tutorial and a reference.
I've tried to keep the code minimal and well documented.

This port doesn't contain:
 - a parser for cfg files
 - a method to load yolo weights
 - some of the more complicated yolo data augmentations (resizing the model)

The port contains:
 - parsing of training data annotation files
 - data augmentations
 - generator to set up batches to train on
 - loss function
 - evaluation

You can use it to train a network for object detection.
A complete demo is contained in the demo.ipynb notebook.

In this notebook, I'm taking a keras port of the tiny-yolo architecture.
The last 3 convolutions are reset.
Then the network is retrained.
This is meant to simulate the usual workflow: You use a pretrained model to extract features and adapt it to a new task.
I don't have the resources to train an extraction model on imagenet, so I've taken an existing model.
It was created with YAD2K from official YOLO config and weights.
Many thanks to the authors of YAD2K.


# sample results

### Motorbikes and person:

![motorbikes and person](http://i.imgur.com/bzpdub5.png "")

### A dog with a saddle:

![a dog with a saddle](http://i.imgur.com/wbr1fNP.png "")

### Persons, cat, potted plant:

![persons, cat and potted plant](http://i.imgur.com/ZGARTfG.png "")
