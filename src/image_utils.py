# Utility functions to handle images

import os
import numpy as np
from functools import reduce
import scipy.misc
import yaml
import glob

import numpy as np
from PIL import Image


def get_image_info(img: Image):
    info_str = 'image:  size: {}    mode: {} format: {} info: {}'.format(img.size,
                                                                         img.mode,
                                                                         img.format,
                                                                         img.info)
    img_data = np.array(img)
    info_str = info_str + "np_data_info: " + get_numpy_var_info(img_data)
    return info_str


def save_output_exr(out_path, img_buf, output_prob, channel_name='class_prob'):
    """
    Save the img_buf and output_prob mask as an exr file
    :param out_path: file path to write to
    :param img_buf: oiio.ImageBuf of original image
    :param output_prob: 2-d ndarray of the output probabilities
    :param channel_name: (default: 'class_prob') name of the channel that output_prob goes into
    """
    # output_prob_buf = nparray_to_image_buf(output_prob.reshape(*output_prob.shape, 1), img_buf_type=oiio.UINT8)
    # class_prob = select_channels_by_index(output_prob_buf, 0, channel_name)
    # out_exr = concatenate_images(img_buf, class_prob)
    # out_exr.set_write_format(oiio.HALF)
    # out_exr.write(out_path)
    print("save_output_exr not implemented")


def save_output_jpg(out_path, img_buf, output_prob, thresh=.5):
    """
    Save the img_buf and output_prob mask as a jpg file
    :param out_path: file path to write to
    :param image_buf: PIL Image of original image
    :param output_prob: 2-d ndarray of the output probabilities
    :param thresh: (default: .5) threshold for the output_prob
    :return:
    """
    img_data = np.array(img_buf).astype(np.uint8)
    colored_data = color_img(img_data, output_prob.reshape(*output_prob.shape, 1), thresh, [255, 0, 0])
    Image.fromarray(colored_data).save(out_path)


def color_img(rgb_data, mask, thresh, color):
    """
    Color the image by color for pixels where the mask is greater than thresh
    :param rgb_data: 3-d ndarray (w x h x numchannels)
    :param mask: 2-d ndarray (w x h)
    :param thresh: threshold
    :param color: array-like of length 3 indicating (r, g, b) pixel values
    :return: colored image
    """
    mask = mask > thresh
    mask = np.dot(mask, np.array(color, ndmin=2))
    rgb_data = np.maximum(rgb_data, mask)
    return rgb_data


def select_channels(img, *channel_names):
    """
    Selects the channel names as listed in channel_names and returns an oiio.ImageBuf object with only those channels
    in it.
    :param img: oiio.ImageBuf object
    :param channel_names: list of channel names
    :return: oiio.ImageBuf object containing only the channels specified in channel_names
    """
    # dst = oiio.ImageBuf()
    # oiio.ImageBufAlgo.channels(dst, img, tuple(channel_names))
    # return dst
    print("select_channels not implemented")
    

def select_channels_by_index(img, channel_indexes, channel_names):
    """
    Selects the channels as specified by channel_indexes in img and outputs an oiio.ImageBuf object with those
    channels named by channel_names
    :param img: oiio.ImageBuf object
    :param channel_indexes: list of channel indexes or a single int
    :param channel_names: list of channel names (must be the same length as channel_indexes or a single string
    :return: oiio.ImageBuf object
    """
    # if isinstance(channel_indexes, int) and isinstance(channel_names, str):
    #     channel_indexes = (channel_indexes,)
    #     channel_names = (channel_names,)
    # else:
    #     assert len(channel_indexes) == len(channel_names), \
    #         "channel_indexes (len={}) must have same length as channel_names (len={})".format(
    #             len(channel_indexes), len(channel_names))
    #     channel_indexes = tuple(channel_indexes)
    #     channel_names = tuple(channel_names)
    # dst = oiio.ImageBuf()
    # oiio.ImageBufAlgo.channels(dst, img, channelorder=channel_indexes, newchannelnames=channel_names)
    # return dst
    print("select_channels_by_index not implemented")


def compute_interval_bounds(begin, desired_length, max_length):
    """
    Computes the beginning and end of an interval bound given that the interval is at most max_length.  It is assumed
    that begin is >= 0.  If begin and begin + desired_length is between [0, max_length), then this pair is returned.
    If begin + desired_length is greater than max_length, then begin is shifted to ensure that desired_length can
    fit.
    :param begin:
    :param desired_length:
    :param max_length:
    :return: (begin, end) interval bounds.  The interval is begin inclusive to end exclusive.  i.e., [begin, end)
    """
    end = begin + desired_length
    if end <= max_length:
        return begin, end
    return max(0, max_length - desired_length), max_length


def crop_bounds_generator(img_shape, crop_shape, stride_step, row_offset=0, col_offset=0):
    """
    Returns a generator that returns the bounds (row_begin, row_end, col_begin, col_end).
    :param img_shape: 2-tuple of the shape of the image (num_rows, num_cols)
    :param crop_shape: 2-tuple of the shape of the crops (num_rows, num_cols)
    :param stride_step: 2-tuple of the steps to take for generating crops (vertical_stride, horizontal_stride)
    :param row_offset: (default: 0) starting offset for row
    :param col_offset: (default: 0) starting offset for column
    :return: returns a generator that yields 4-tuple of (row_begin, row_end, col_begin, col_end)
    """
    def generator():
        row_begin = row_offset
        row_end = 0
        while row_begin < img_shape[0] and row_end < img_shape[0]:
            row_begin, row_end = compute_interval_bounds(row_begin, crop_shape[0], img_shape[0])
            col_begin = col_offset
            col_end = 0
            while col_begin < img_shape[1] and col_end < img_shape[1]:
                col_begin, col_end = compute_interval_bounds(col_begin, crop_shape[1], img_shape[1])
                yield row_begin, row_end, col_begin, col_end
                col_begin += stride_step[1]
            row_begin += stride_step[0]
    return generator
