#!/usr/bin/env python
"""
Run trained unet model on labeled exr images to get pixel predictions.
"""
import sys
import os
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import tensorflow.compat.v2 as tf
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from common_utils import listdir_files, get_numpy_var_info, get_imagebuf_info, \
    nparray_to_image_buf, concatenate_images, select_channels, \
    convert_to_rgb_image, color_img

from generator_pil import PilDataGenerator

Verbosity = 0

INPUT_SIZE = 224
OUTPUT_CHANNELS = 2


if __name__ == '__main__':
    ################################################################################
    # Arg parsing
    #
    parser = argparse.ArgumentParser(description='Unet model for crack detection')

    # Optional command-line args
    parser.add_argument('-b', '--batchsize', type=int, default=16,
                        help='Mini-batch size')
    parser.add_argument('--valbatchsize', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--maxbatches', type=int, default=2000,
                        help='Max number of batches to run')

    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Mini-batch size')

    parser.add_argument('-m', '--model_path', default='model.h5',
                        help='Path of model file')
    parser.add_argument('--testdir', default='test_data',
                        help='Directory holding validation exr images')
    parser.add_argument('--output_dir', default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    args = parser.parse_args()
    batch_size = args.batchsize
    batch_size_val = args.valbatchsize
    max_batches = args.maxbatches
    model_path = args.model_path
    test_dir = args.testdir
    output_dir = args.output_dir
    prob_thresh = args.thresh

    Verbosity = args.verbosity

    do_output_norm = False
    #do_output_norm = True

    png_dir = output_dir + '_png'
    plt_dir = output_dir + '_plot'

    prob_label = 'class_prob'
    int_thresh = int(prob_thresh * 256)

    homedir = os.path.expanduser('~')

    # Setup batch generator for images to run model on.
    img_filepaths = listdir_files(test_dir)
    img_filenames = list(img_filepaths.keys())
    img_filenames.sort()
    n_img = len(img_filenames)

    examples_length = n_img
    steps_per_epoch = max(1, examples_length // batch_size_val)
    print('-' * 80)
    print('Number of examples: {}    steps_per_epoch: {}'.format(examples_length, steps_per_epoch))
    print('=' * 80)

    img_generator = PilDataGenerator(img_filenames, img_filepaths, shuffle=False,
                                     to_fit=False, batch_size=batch_size_val, verbosity=2)

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    # Load the trained model.
    model = tf.keras.models.load_model(model_path)

    ############################################################################
    # Run the model
    #

    # Iterate across batches of examples
    # and run the model on each batch.

    n_written = 0

    for batch_index in range(steps_per_epoch):
        if batch_index >= max_batches:
            break
        images = img_generator.__getitem__(batch_index)
        batch_ids = img_generator.get_batch_ids(batch_index)
        print('\nbatch number {}\n'.format(batch_index))
        print('images shape: ', images.shape)
        preds = model.predict(images)
        for i, pred in enumerate(preds):
            id = batch_ids[i]
            print('\nimage file: {}'.format(id))
            image_filepath = img_filepaths[id]
            exr_buf = oiio.ImageBuf(image_filepath)
            exr_spec = exr_buf.spec()
            exr_width = exr_spec.width
            exr_height = exr_spec.height
            exr_nchannels = exr_spec.nchannels
            exr_format = exr_spec.format
            exr_channelnames = exr_spec.channelnames
            # print(get_imagebuf_info(exr_buf))

            fname_stem = os.path.splitext(id)[0]
            output_filename = '{}.exr'.format(fname_stem)
            output_filepath = os.path.join(output_dir, output_filename)
            png_filename = '{}.png'.format(fname_stem)
            png_filepath = os.path.join(png_dir, png_filename)

            # Convert prediction numpy array to oiio ImageBuf
            # Then concatenate the image channels with the prediction as a channel.
            #
            # Get the predicted score that pixel belongs to a crack:

            pred = pred.reshape(INPUT_SIZE, INPUT_SIZE, OUTPUT_CHANNELS)
            bg_channel = pred[:, :, 0]
            model_score = pred[:, :, 1]

            if Verbosity > 2:
                print('pred ', get_numpy_var_info(pred))
                print('bg_channel ', get_numpy_var_info(bg_channel))
                print('model_score ', get_numpy_var_info(model_score))

            if do_output_norm:
                score_min = model_score.min()
                model_score = model_score - score_min
                score_max = model_score.max()
                model_score = model_score / score_max
                print('class_prob after norm: ', get_numpy_var_info(model_score))

            # Turn that into class id of each pixel, 1 == crack
            pred_mask = np.argmax(pred, axis=-1)
            # print('pred mask ', get_numpy_var_info(pred_mask))
            pred_mask = pred_mask.astype(np.uint8)

            pred_img = Image.fromarray(model_score)



            exr_arr = np.array(exr_buf.get_pixels(oiio.FLOAT))
            dense_label_arr = None
            if 'dense_label' in exr_channelnames:
                dense_label_arr = exr_arr[:, :, exr_nchannels-1]
                dense_label_int = 255 * dense_label_arr
            # Change class_prob to shape (h, w, 1) array for np.append()
            class_prob1 = model_score[:, :, np.newaxis]
            # Append class_prob to exr_arr
            exr_prob_arr = np.append(exr_arr, class_prob1, axis=2)
            # print('exr_prob_arr ', get_numpy_var_info(exr_prob_arr))

            #pred_prob_buf = oiio.ImageBufAlgo.channel_append(exr_buf, pred_img)
            # pred_prob_buf = concatenate_images(exr_buf, pred_img)
            exr_prob_spec0 = oiio.ImageSpec(exr_width, exr_height, exr_nchannels + 1, exr_format)
            exr_pred_channelnames = list(exr_channelnames)
            exr_pred_channelnames.append(prob_label)
            exr_prob_spec0.channelnames = exr_pred_channelnames
            exr_prob_buf = oiio.ImageBuf(exr_prob_spec0)
            exr_prob_buf.set_pixels(oiio.ROI(), exr_prob_arr)
            exr_prob_spec = exr_prob_buf.spec()
            # print('exr_prob_spec ', get_imagebuf_info(exr_prob_buf))
            exr_prob_buf.write(output_filepath)

            # For viewing convenience, overlay the prediction on top of image and write to file.
            orig_rgb_buf = convert_to_rgb_image(exr_buf)
            rgb_arr = np.array(orig_rgb_buf.get_pixels(oiio.UINT8))
            # print('rgb_arr ', get_numpy_var_info(rgb_arr))

            # mask_buf = select_channels(pred_prob_buf, prob_label)
            # mask_arr = np.array(mask_buf.get_pixels(oiio.UINT8))

            # pred_arr = pred_mask.astype(np.float)
            pred_arr_int = model_score * 256.0
            pred_arr_int1 = pred_arr_int[:, :, np.newaxis]
            pred_max = np.max(pred_arr_int)

            # Color the label pixels blue.
            dense_label_int1 = dense_label_int[:, :, np.newaxis]
            rgb_dense_arr = color_img(rgb_arr, dense_label_int1, int_thresh, [0, 0, 255])

            # Color the prediction pixels red.
            rgb_pred_arr = color_img(rgb_dense_arr, pred_arr_int1, int_thresh, [255, 0, 0])
            rgb_pred_buf = nparray_to_image_buf(rgb_pred_arr, img_buf_type=oiio.UINT8, channel_names=('R', 'G', 'B'))
            rgb_pred_buf.write(png_filepath)
            n_written += 1
            print('-'*80)
            print('i {}  n_written {}   rgb_pred_buf {}'.format(i, n_written, png_filepath))
            print('-'*80 + '\n')

            # And also use matplotlib to show image and model output side by side.
            plot_info = [('Image', rgb_arr)]
            if dense_label_arr is not None:
                plot_info.append(('Labels', dense_label_int))
            plot_info.append(('Model Score', model_score))
            # plot_info.append(('Predicted Crack Pixels', pred_arr_int))
            plot_info.append(('Predicted Crack', rgb_pred_arr))

            plot_filename = '{}.png'.format(fname_stem)
            plot_filepath = os.path.join(plt_dir, plot_filename)

            n_plots = len(plot_info)
            plot_index = 1
            plt.figure(figsize=(12, 5))
            plt.clf()
            plt.suptitle('{}'.format(fname_stem))

            for plot_title, img in plot_info:
                plt.subplot(1, n_plots, plot_index)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.title(plot_title)
                plot_index += 1

            plt.savefig(plot_filepath)
            plt.close()

    print('Done')
