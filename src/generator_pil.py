""" TensorFlow Keras data generator for reading Pillow images. """
import sys
import os
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image

from tensorflow.keras.utils import Sequence

from common_utils import listdir_files, get_numpy_var_info, get_imagebuf_info, \
    nparray_to_image_buf, concatenate_images, select_channels, \
    convert_to_rgb_image, color_img

# ToDo:
# Rename imgpath_dict as something showing this is a dict[filename] -> path
# Like pathdict
# Need to pass in filepaths of label masks.
# Perhaps pathdict[filename] -> (pngpath, labelmaskpath)


class PilDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, filenames, imgpath_dict,
                 to_fit=True, batch_size=32, dim=(224, 224),
                 n_channels=3, n_classes=2, shuffle=True, pos_weight=1.0, verbosity=0):
        """Initialization

        :param filenames: list of all image ids (file basenames) to use in the generator
        :param imgpath_dict: list of image imgpath_dict (file names)
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
            img_path = self.imgpath_dict[ID]
            exr_data = self._load_exr_image(img_path)
            # print('_generate_Xy {}    shape: {}'.format(self.imgpath_dict[ID], exr_data.shape))
            exr_channels = exr_data.shape[-1]
            if exr_channels < self.n_channels:
                print('Warning: image has too few channels: {}'.format(ID))
                continue
            image_pixels = exr_data[:, :, :self.n_channels]
            X[i, ] = image_pixels

            if exr_channels >= self.n_channels + 1:
                label_pixels = exr_data[:, :, self.n_channels]
                gt_bg = (label_pixels <= 0.5).astype(np.uint8)
                gt_fg = (label_pixels > 0.5).astype(np.uint8)
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


    def _load_exr_image(self, img_path, img_format=oiio.FLOAT, dst_width=224, dst_height=224):
        """Load grayscale image

        :param img_path: path to image to load
        :return: loaded image
        """
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgbuf = oiio.ImageBuf(img_path)
        img_spec = imgbuf.spec()
        src_width = img_spec.width
        src_height = img_spec.height

        # Extract crop out of image if it is larger than dst width and height.
        if img_spec.width != dst_width or img_spec.height != dst_height:
            # print('_load_exr: {}     w {}  h {}  c {}   format {}  channelnames: {}'.format(img_path, img_spec.width, img_spec.height, img_spec.nchannels, img_spec.format, img_spec.channelnames))
            # rgb_buf = convert_to_rgb_image(imgbuf)
            # rgb_spec = rgb_buf.spec()
            # print('    rgb: w {}  h {}  c {}   format {}  channelnames: {}'.format(rgb_spec.width, rgb_spec.height, rgb_spec.nchannels, rgb_spec.format, rgb_spec.channelnames))

            # resized_buf = oiio.ImageBuf(oiio.ImageSpec(dst_width, dst_height, img_spec.nchannels, img_spec.format))
            # resized_buf = oiio.ImageBuf(oiio.ImageSpec(dst_width, dst_height, 3, rgb_spec.format))
            # oiio.ImageBufAlgo.resize(resized_buf, rgb_buf)
            # # resized_buf = oiio.ImageBufAlgo.resize(imgbuf, roi=oiio.ROI(0, 224, 0, 224, 0, 1, 0, 3))
            # resized_spec = resized_buf.spec()
            # # print('    resized: w {}  h {}  c {}   format {}'.format(resized_spec.width, resized_spec.height, resized_spec.nchannels, resized_spec.format))
            # rgb_img1 = convert_to_rgb_image(resized_buf)

            # Let's try cropping parts out of the source image:
            x_begin = random.randint(0, src_width - dst_width)
            x_end = x_begin + dst_width
            y_begin = random.randint(0, src_height - dst_height)
            y_end = y_begin + dst_height
            # print('    crop param: {} {} {} {}'.format(x_begin, x_end, y_begin, y_end))
            # crop_buf = oiio.ImageBufAlgo.cut(rgb_buf, oiio.ROI(x_begin, x_end, y_begin, y_end))
            # crop_spec = crop_buf.spec()
            # print('    crop: w {}  h {}  c {}   format {}  channelnames: {}'.format(crop_spec.width, crop_spec.height, crop_spec.nchannels, crop_spec.format, crop_spec.channelnames))

            img_data = np.array(imgbuf.get_pixels(img_format))
            image_data = img_data[y_begin:y_end, x_begin:x_end, :]
        else:
            image_data = np.array(imgbuf.get_pixels(img_format))
        return image_data
