"""Functions needed by unet2.py, unet_predict.py and others"""
import os
import sys
from pathlib import Path
from functools import reduce

import numpy as np

def get_numpy_var_info(npvar: np.ndarray):
    info_str = 'dtype: {}  shape: {}    mean: {} min: {} max: {}'.format(npvar.dtype, npvar.shape,
                                                                        npvar.mean(), npvar.min(), npvar.max())
    return info_str


def listdir_files(dirname, extensions=['.png']):
    filename_dict = {}
    basepath = Path(dirname)
    dir_entries = basepath.iterdir()
    for item in dir_entries:
        if item.is_file() and item.suffix.lower() in extensions:
            filename_dict[item.name] = str(item)
    return filename_dict


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


# def nparray_to_image_buf(array, img_buf_type=oiio.UINT8, channel_names=None):
#     """
#     Convert the numpy array to an oiio.ImageBuf object
#     :param array: numpy array (should a 3-d array where the last dimension is the pixel channels)
#     :param img_buf_type: (default: oiio.UINT8)
#     :param channel_names: (default: None) to specify explicit channel names
#     :return: oiio.ImageBuf object
#     """
#     if len(array.shape) == 2:
#         array = np.reshape(array, (array.shape[0], array.shape[1], 1))
#     width = array.shape[1]
#     height = array.shape[0]
#     # depth = array.shape[2] if len(array.shape) == 3 else 1
#     depth = array.shape[2]
#     # print('nparray_to_image_buf: width {}   height {}   depth {}  img_buf_type {}'.format(width, height, depth, img_buf_type))
#     dst = oiio.ImageBuf(oiio.ImageSpec(width, height, depth, img_buf_type))
#     if channel_names:
#         assert isinstance(channel_names, tuple), 'channel_names must be tuple'
#         assert len(channel_names) == dst.spec().nchannels, \
#             'channel_names must be a tuple of length {}'.format(dst.spec().nchannels)
#         dst.spec().channelnames = channel_names
#     if not dst.set_pixels(oiio.ROI(), array):
#         raise RuntimeError('Error creating ImageBuf: {}'.format(dst.geterror()))
#     return dst


def color_img(rgb_img, mask, thresh, color):
    """
    Color the image by color for pixels where the mask is greater than thresh
    :param rgb_img: 3-d ndarray (w x h x numchannels)
    :param mask: 2-d ndarray (w x h)
    :param thresh: threshold
    :param color: array-like of length 3 indicating (r, g, b) pixel values
    :return: colored image
    """
    max_pix = rgb_img.max(axis=1)
    max_pixel = max_pix.max(axis=0)
    # print('color_img: rgb_img in:  dtype: {}   shape: {}  max: {}'.format(rgb_img.dtype, rgb_img.shape, max_pixel))

    mask = mask > thresh
    mask = mask.astype(np.uint8)
    # print('color_img:  mask:  mean: {} max: {}  dtype: {}   shape: {}'.format(mask.mean(), mask.max(), mask.dtype, mask.shape))

    color2 = np.array(color, ndmin=2)
    # print('color2 dtype: {}   shape: {}'.format(color2.dtype, color2.shape))

    mask2 = np.dot(mask, np.array(color, ndmin=2))
    # print('mask2 dtype: {}   shape: {}'.format(mask2.dtype, mask2.shape))

    rgb_img = np.maximum(rgb_img, mask2)
    rgb_img = rgb_img.astype(np.uint8)
    # print('rgb_img dtype: {}   shape: {}'.format(rgb_img.dtype, rgb_img.shape))

    return rgb_img
