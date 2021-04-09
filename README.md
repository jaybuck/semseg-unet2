# U-Net Deep Learning Models for Crack Detection

## Summary

The Fully Convolutional Network (FCN) models that Prenav is using for
Crack Detection (in 2019 and early 2020) work reasonably well, but
are challenging to understand when we work to figure out why they make
the errors that they do. We wanted to use a simpler Deep Learning
model architecture that is easier to understand in order to build
better insights in order to really improve the precision and recall
of the crack detection models.

The U-Net architecture is ideal for this. And the code we used as a
starting point is a tutorial so is designed to be understandable.
We started with this TensorFlow tutorial:

<https://www.tensorflow.org/tutorials/images/segmentation>

We modified it:

* For the crack detection task
* To use OpenImageIO to read and process exr files
* To output useful images showing:
    - The concrete
    - The pixels the human labelers labeled as cracks
    - The pixels the model classified as cracks.

Though our focus was using a simpler Deep Learning model architecture
to understand what the models are doing in the crack detection task,
we found that the trained U-Net models give us good precision and
recall on this task as well. Definitely a win.

## Introduction

Discuss what the code does.
In particular, note that it operates on image tiles, not entire images.

## Prerequisites

The basic prerequisites to run the U-Net Python code that trains U-Net models and
then uses them to predict the class of pixels in new images are:

* A computer running linux. We have tested under Ubuntu 16.04. Ubuntu 18.04 should work well too.
* At least 64 GB of RAM
* An Nvidia GPU with at least 8 GB of RAM, along with:
  * Nvidia drivers
    * CUDA >= 10.0
* Python 3
* TensorFlow 1.15
* OpenImageIO, aka oiio

All of the packages in the environment used for these experiments are
listed in the file `requirements_unet_all.txt`

## Contents of the semantic_segmentation_unet directory

* `src` : Directory that holds Python code for training, testing and evaluating U-Net models
for the crack detection task.
* `run_*.sh` : Example scripts for
    - U-Net model training
    - Using a trained model for prediction on images
    - Evaluating key metrics such as precision and recall on labeled test images.


    


