""" TensorFlow Keras data generator for reading Pillow images. """
import sys
import os
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image

from tensorflow.keras.utils import Sequence

from common_utils import listdir_files, get_numpy_var_info, color_img


class PilDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, filenames, imgpath_dict,
                 to_fit=True, batch_size=32, dim=(224, 224),
                 n_channels=3, n_classes=2, shuffle=True, pos_weight=1.0, verbosity=0):
        """Initialization

        :param filenames: list of all image ids (file basenames) to use in the generator
        :param imgpath_dict: list of image and label mask tuples
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.img_filenames = filenames
        self.imgpath_dict = imgpath_dict
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pos_weight = pos_weight
        self.verbosity = verbosity
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.img_filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        if self.verbosity > 4:
            print('\r__getitem__ batch {}'.format(index), end='\r')

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        filenames_tmp = [self.img_filenames[k] for k in indexes]

        # Generate data
        X, y, pixweights = self._generate_Xy(filenames_tmp)

        if self.to_fit:
            return X, y, pixweights
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.img_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def get_batch_ids(self, batch_index):
        indexes = self.indexes[batch_index * self.batch_size: min(len(self.indexes), (batch_index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.img_filenames[k] for k in indexes]
        return list_IDs_temp


    def _generate_Xy(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images and
        """
        # Initialization
        n_pixels = self.dim[0] * self.dim[1]
        this_batch_size = min(self.batch_size, len(list_IDs_temp))
        X = np.zeros((this_batch_size, *self.dim, self.n_channels))
        y = np.zeros((this_batch_size, n_pixels, 2), dtype=int)
        pixelweights = np.zeros((this_batch_size, n_pixels), dtype=float)
        # print('generate_Xy: y: ', get_numpy_var_info(y))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load image file
            img_path, label_mask_path = self.imgpath_dict[ID]
            img_data = np.array(Image.open(img_path))
            label_data = np.array(Image.open(label_mask_path))

            X[i, ] = img_data

            if label_data.shape[0] == img_data.shape[0]:
                gt_bg = (label_data <= 128).astype(np.uint8)
                gt_fg = (label_data > 128).astype(np.uint8)
                gt_sum = np.sum(gt_fg)
                # print('gt_sum: ', gt_sum)
                gt = np.stack((gt_bg, gt_fg), axis=2)
                gt_flat = gt.reshape((n_pixels, 2))
                weight = 1.0*(gt_bg == 1) + self.pos_weight*(gt_fg == 1)
                # print('generate_Xy: weight: ', get_numpy_var_info(weight))
                weight = np.reshape(weight, (n_pixels,))
                # print('generate_Xy: gt_flat: ', get_numpy_var_info(gt_flat))
                # print('generator: gt ', get_numpy_var_info(gt))
                y[i, ] = gt_flat
                pixelweights[i, ] = weight

        return X, y, pixelweights

