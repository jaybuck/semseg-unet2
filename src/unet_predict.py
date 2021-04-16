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

import tensorflow as tf

from common_utils import listdir_files, get_numpy_var_info
from image_utils import get_image_info, color_img
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
    test_imgdir = os.path.join(test_dir, "png")
    test_labeldir = os.path.join(test_dir, "mask")

    imgpath_dict = listdir_files(test_imgdir)
    labelpath_dict = listdir_files(test_labeldir)

    test_file_dict = {}
    for fname in imgpath_dict:
        if fname in labelpath_dict:
            test_file_dict[fname] = tuple([imgpath_dict[fname], labelpath_dict[fname]])

    test_filenames = list(test_file_dict.keys())
    n_img = len(test_filenames)
    print('Test images:  dir: {}  Number of images: {}'.format(test_imgdir, n_img))

    examples_length = n_img
    steps_per_epoch = max(1, examples_length // batch_size_val)
    print('-' * 80)
    print('Number of examples: {}    steps_per_epoch: {}'.format(examples_length, steps_per_epoch))
    print('=' * 80)

    img_generator = PilDataGenerator(test_filenames, test_file_dict, shuffle=False,
                                     to_fit=True, batch_size=batch_size_val, verbosity=Verbosity)

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
        images, masks, _ = img_generator.__getitem__(batch_index)
        batch_ids = img_generator.get_batch_ids(batch_index)
        print('\nbatch number {}\n'.format(batch_index))
        print('images shape: ', images.shape)
        preds = model.predict(images)
        for i, pred in enumerate(preds):
            id = batch_ids[i]
            print('\nimage file: {}'.format(id))
            image_filepath, mask_filepath = test_file_dict[id]
            img = Image.open(image_filepath)
            img_arr = np.array(img)
            img_width = img.width
            img_height = img.height
            img_nchannels = img_arr.shape[-1]
            img_format = img.format
            img_channelnames = img.getbands()
            # print("img info:", get_image_info(img))

            fname_stem = os.path.splitext(id)[0]
            output_filename = '{}.png'.format(fname_stem)
            output_filepath = os.path.join(output_dir, output_filename)
            png_filepath = os.path.join(png_dir, output_filename)

            # Read label mask
            labelmask_img = Image.open(mask_filepath)
            labelmask_arr = np.array(labelmask_img)
            labelmask_width = labelmask_img.width
            labelmask_height = labelmask_img.height
            labelmask_nchannels = labelmask_arr.shape[-1]
            labelmask_format = labelmask_img.format
            labelmask_channelnames = labelmask_img.getbands()
            # print("mask info:", get_image_info(labelmask_img))

            # Convert prediction numpy array to PIL Image
            # Then overlay the prediction onto the image channels.
            #
            # Get the predicted score that pixel belongs to a crack:
            pred = pred.reshape(INPUT_SIZE, INPUT_SIZE, OUTPUT_CHANNELS)
            bg_channel = pred[:, :, 0]
            model_score = pred[:, :, 1]

            if Verbosity > 2:
                print('pred ', get_numpy_var_info(pred))
                print('bg_channel ', get_numpy_var_info(bg_channel))

            if Verbosity > 1:
                print('model_score ', get_numpy_var_info(model_score))

            if do_output_norm:
                score_min = model_score.min()
                model_score = model_score - score_min
                score_max = model_score.max()
                model_score = model_score / score_max
                print('class_prob after norm: ', get_numpy_var_info(model_score))

            # Turn that into class id of each pixel, 1 == crack
            pred_mask = np.argmax(pred, axis=-1)
            print('pred mask ', get_numpy_var_info(pred_mask))
            pred_mask = pred_mask.astype(np.uint8)
            # Convert to Image
            score_int = (model_score * 255).astype(np.uint8)
            pred_img = Image.fromarray(score_int)
            pred_img.save(output_filepath)

            labelmask_int = (255 * labelmask_arr).astype(np.uint8)
            # Change class_prob to shape (h, w, 1) array for np.append()
            class_prob1 = model_score[:, :, np.newaxis]

            # For viewing convenience, overlay the prediction on top of image and write to file.
            pred_arr_int = model_score * 255.0
            pred_arr_int1 = pred_arr_int[:, :, np.newaxis]
            pred_max = np.max(pred_arr_int)

            # Color the label pixels blue.
            labelmask_int1 = labelmask_int[:, :, np.newaxis]
            rgb_dense_arr = color_img(img_arr, labelmask_int1, int_thresh, [0, 0, 255])

            # Color the prediction pixels red.
            rgb_pred_arr = color_img(rgb_dense_arr, pred_arr_int1, int_thresh, [255, 0, 0])
            rgb_pred_img = Image.fromarray(rgb_pred_arr)
            rgb_pred_img.save(png_filepath)
            n_written += 1
            print('-'*80)
            print('i {}  n_written {}   rgb_pred_img {}'.format(i, n_written, png_filepath))
            print('-'*80 + '\n')

            # And also use matplotlib to show image and model output side by side.
            plot_info = [('Image', img_arr)]
            if labelmask_arr is not None:
                plot_info.append(('Labels', labelmask_int))
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
