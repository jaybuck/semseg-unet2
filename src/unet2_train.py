#!/usr/bin/env python
"""
Build and train simple unet model.
"""
import sys
import os
from pathlib import Path

import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import tensorflow.compat.v2 as tf
import tensorflow as tf

from IPython.display import clear_output

from common_utils import listdir_files, get_numpy_var_info

from generator_pil import PilDataGenerator

tf.compat.v1.enable_eager_execution()

Verbosity = 0

IMAGE_SIDE = 512
INPUT_SIZE = 224
OUTPUT_CHANNELS = 2

UPSAMPLE_KERNEL_SIZE = 3

sample_image = None
sample_mask = None


def display(display_list, suffix=0):
    plt.figure(figsize=(8, 6))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    fname = '_display_{}.png'.format(suffix)
    plt.savefig(fname)
    plt.close()


def create_prob_img(pred):
    # print('create_prob_img pred type: ', type(pred))

    # print('create_prob_img pred: ', get_numpy_var_info(pred))
    pred0 = pred[0]
    pred_mat = pred0.reshape(INPUT_SIZE, INPUT_SIZE, OUTPUT_CHANNELS)
    # print('create_prob_img pred_mat: ', get_numpy_var_info(pred_mat))
    class_prob = pred_mat[:, :, 1]
    # print('1 class_prob: ', get_numpy_var_info(class_prob))
    class_prob = class_prob[..., np.newaxis]
    # print('2 class_prob: ', get_numpy_var_info(class_prob))
    return class_prob


def create_mask(pred):
    print('create_mask pred type: ', type(pred))

    print('create_mask pred: ', get_numpy_var_info(pred))
    pred0 = pred[0]
    pred_mat = pred0.reshape(INPUT_SIZE, INPUT_SIZE, OUTPUT_CHANNELS)
    print('create_mask pred_mat: ', get_numpy_var_info(pred_mat))
    pred_mask = np.argmax(pred_mat, axis=-1)
    print('1 pred_mask: ', get_numpy_var_info(pred_mask))
    pred_mask = pred_mask[..., np.newaxis]
    pred_mask = pred_mask.astype(np.uint8)
    print('2 pred_mask: ', get_numpy_var_info(pred_mask))
    print('3 pred_mask[0]: ', get_numpy_var_info(pred_mask[0]))
    return pred_mask


def show_predictions(generator=None, num=1, suffix='_'):
    if generator:
        images, masks, _ = generator.__getitem__(0)
        i = 0
        for image, mask in zip(images, masks):
            mask1 = mask[:, 1]
            mask1 = mask1.reshape((INPUT_SIZE, INPUT_SIZE, 1))
            pred = model.predict(image[tf.newaxis, ...])
            filesuffix = '{}_{}'.format(suffix, i)
            # display([image, mask1, create_mask(pred)], filesuffix)
            display([image, mask1, create_prob_img(pred)], filesuffix)

            i += 1
            if i > num:
                break
    else:
        pred = model.predict(sample_image[tf.newaxis, ...])
        # print('show_predictions: pred ', get_numpy_var_info(pred))
        sample_mask1 = sample_mask.copy()
        sample_mask1 = sample_mask1[:, 1]
        sample_mask1 = sample_mask1.reshape((INPUT_SIZE, INPUT_SIZE, 1))
        display([sample_image, sample_mask1, create_prob_img(pred)], suffix)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(suffix='e{}'.format(epoch))


class WriteHistoryCallback(tf.keras.callbacks.History):
    def __init__(self, save_filepath):
        super().__init__()
        self._path = save_filepath

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        out_epoch = self.epoch
        out_history = self.history
        with open(self._path, 'w') as f:
            f.write('epoch = {}\n'.format(out_epoch))
            f.write('history = {}\n'.format(out_history))


class PlotHistory(tf.keras.callbacks.History):
    def __init__(self, save_filepath):
        super().__init__()
        self._path = save_filepath
        self._lines = []

    def add_plot(self, key, label, linespec):
        """
        Adds a plot of the received key shown as label on the plot.
        """
        self._lines.append((key, label, linespec))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        out_epoch = self.epoch
        out_history = self.history

        plt.figure(1)
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel('Stat')
        plt.subplot(1, 1, 1)
        for key, label, linespec in self._lines:
            plt.plot(out_epoch, out_history[key], linespec, label=label)
        plt.legend()
        plt.grid(True)
        plt.savefig(self._path)
        plt.close()

DEFAULT_PLOT_HISTORY_SPEC_WITH_VALIDATION = [
    ('loss', 'Training loss', 'r.:'),
    ('val_loss', 'Validation loss', 'b.:')
]

DEFAULT_PLOT_HISTORY_SPEC_WITHOUT_VALIDATION = [
    ('loss', 'Training loss', 'r.:')
]


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False, name='upsample'):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.01)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False,
                                        name='{}_Conv2DTranspose'.format(name)))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization(name='{}_BatchNorm'.format(name)))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5, name='{}_Dropout'.format(name)))

    result.add(tf.keras.layers.ReLU(name='{}_ReLU'.format(name)))
    return result


def unet_model(output_channels, down_stack, up_stack):
    inputs = tf.keras.layers.Input(shape=[INPUT_SIZE, INPUT_SIZE, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    print('Creating upsample layers')
    for up, skip in zip(up_stack, skips):
        print('skip: ', skip)
        x = up(x)
        print('upsampled layer:', x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last_convtranspose = tf.keras.layers.Conv2DTranspose(
        output_channels, UPSAMPLE_KERNEL_SIZE, strides=2, padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name='last_convtranspose')

    x = last_convtranspose(x)

    softmax = tf.keras.layers.Softmax(name='softmax')(x)

    output = tf.keras.layers.Reshape((-1, output_channels), name='reshaped_softmax')(softmax)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    ################################################################################
    # Arg parsing
    #
    parser = argparse.ArgumentParser(description='Unet model for crack detection')

    # Optional command-line args
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Max number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=16,
                        help='Mini-batch size')
    parser.add_argument('--valbatchsize', type=int, default=256,
                        help='Mini-batch size')
    parser.add_argument('-l', '--learnrate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--posweight', type=float, default=1.0,
                        help='Training weight for positive (crack) pixels')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    parser.add_argument('-t', '--traindir', default='train_data',
                        help='Directory holding training exr images')
    parser.add_argument('--testdir', default='test_data',
                        help='Directory holding validation exr images')
    parser.add_argument('--logdir', default='logs',
                        help='Log directory')
    parser.add_argument('--summariesdir', default='keras_logs',
                        help='TensorBoard Summaries directory')

    args = parser.parse_args()
    nepochs = args.epochs
    batch_size = args.batchsize
    batch_size_val = args.valbatchsize
    learnrate = args.learnrate
    pos_weight = args.posweight

    train_dir = args.traindir
    test_dir = args.testdir
    log_dir = args.logdir
    summaries_dir = args.summariesdir

    Verbosity = args.debugLevel

    homedir = os.path.expanduser('~')

    model_filename = 'unetmodel'
    model_path = os.path.join(log_dir, model_filename)
    
    checkpoint_filename = 'unetmodel_checkpoint'
    checkpoint_path = os.path.join(log_dir, checkpoint_filename)

    history_path = os.path.join(log_dir, 'history.txt')

    buffer_size = 1000

    OUTPUT_CHANNELS = 2

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    ############################################################################
    # Get training and validation datasets
    #

    train_imgdir = os.path.join(train_dir, "png")
    train_labeldir = os.path.join(train_dir, "mask")

    train_imgpath_dict = listdir_files(train_imgdir)
    train_labelpath_dict = listdir_files(train_labeldir)

    train_file_dict = {}
    for fname in train_imgpath_dict:
        if fname in train_labelpath_dict:
            train_file_dict[fname] = tuple(train_imgpath_dict[fname], train_labelpath_dict[fname])

    train_filenames = list(train_file_dict.keys())
    n_train = len(train_filenames)
    print('Training:  dir: {}  Number of images: {}'.format(train_imgdir, n_train))

    fname0 = train_filenames[0]
    img = Image.open(fname0)
    print(f"Image fname {fname}  ")
    print('image size: ', img.size)
    print('image mode: ', img.mode)
    print('image format: ', img.format)
    print('image info: ', img.info)

    sys.exit(0)


    test_imgdir = os.path.join(test_dir, "png")
    test_labeldir = os.path.join(test_dir, "mask")

    val_imgpath_dict = listdir_files(test_imgdir)
    val_labelpath_dict = listdir_files(test_labeldir)

    val_file_dict = {}
    for fname in val_imgpath_dict:
        if fname in val_labelpath_dict:
            val_file_dict[fname] = tuple(val_imgpath_dict[fname], val_labelpath_dict[fname])

    val_filenames = list(val_file_dict.keys())
    n_val = len(val_filenames)
    print('Validation:  dir: {}  Number of images: {}'.format(test_dir, n_val))

    train_length = n_train
    steps_per_epoch = train_length // batch_size
    print('TRAIN_LENGTH: {}    STEPS_PER_EPOCH: {}'.format(train_length, steps_per_epoch))

    train_generator = PilDataGenerator(train_filenames, train_file_dict,
                                       to_fit=True, batch_size=batch_size, pos_weight=pos_weight, verbosity=2)

    val_generator = PilDataGenerator(val_filenames, val_file_dict,
                                     to_fit=True, batch_size=batch_size_val, pos_weight=pos_weight, verbosity=2)

    i = 0
    for images, masks, pixweights in train_generator:
        print('images shape: ', images.shape)
        print('masks shape: ', masks.shape)
        sample_image, sample_mask = images[0], masks[0]
        print('sample_image shape: {}   mask shape: {}'.format(sample_image.shape, sample_mask.shape))
        print('images batch: ', get_numpy_var_info(images))
        print('masks batch: ', get_numpy_var_info(masks))
        print('pixweights batch: ', get_numpy_var_info(pixweights))

        i += 1
        if i >= 1:
            break

    if Verbosity >= 1:
        print('Display sample image')
        sample_mask1 = sample_mask.copy()
        sample_mask1 = sample_mask1[:, 1]
        sample_mask1 = sample_mask1.reshape((INPUT_SIZE, INPUT_SIZE, 1))
        display([sample_image, sample_mask1], suffix='sampleimage')

    val_images, val_masks, val_pixweights = val_generator.__getitem__(0)
    print('val_images shape: {}    val_masks shape: {}'.format(val_images.shape, val_masks.shape))

    # print('Exiting')
    # sys.exit(0)

    ############################################################################
    # Define model
    #
    print('Defining unet model')
    base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 112 x 112
        'block_3_expand_relu',   # 56 x 56
        'block_6_expand_relu',   # 28 x 28
        'block_13_expand_relu',  # 14 x 14
        'block_16_project',      # 7 x 7
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    print('Creating the downsampling part of the unet')
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    # down_stack.trainable = False
    down_stack.trainable = True
    print('down_stack summary:')
    down_stack.summary()

    stringlist = []
    down_stack.summary(print_fn=stringlist.append)
    base_model_summary = "\n".join(stringlist)
    basemodelinfo_filename = 'basemodelinfo.txt'
    basemodelinfo_path = os.path.join(log_dir, basemodelinfo_filename)
    with open(basemodelinfo_path, 'w') as modelinfo_file:
        modelinfo_file.write(base_model_summary + '\n')
        for i, layer in enumerate(base_model.layers):
            modelinfo_file.write('{}  {}\n'.format(i, layer.name))
    
    # Decoder / upsampler is simply a series of upsample blocks implemented in TensorFlow.
    up_stack = [
        upsample(512, UPSAMPLE_KERNEL_SIZE, name='upsample_1'),  # 7x7 -> 14x14
        upsample(256, UPSAMPLE_KERNEL_SIZE, name='upsample_2'),  # 14x14 -> 28x28
        upsample(128, UPSAMPLE_KERNEL_SIZE, name='upsample_3'),  # 28x28 -> 56x56
        upsample(64, UPSAMPLE_KERNEL_SIZE, name='upsample_4'),   # 56x56 -> 112x112
    ]

    model = unet_model(OUTPUT_CHANNELS, down_stack, up_stack)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learnrate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learnrate),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    model.summary()
    stringlist = []
    model.summary(print_fn=stringlist.append)
    unet_model_summary = "\n".join(stringlist)
    fullmodelinfo_filename = 'unetmodelinfo.txt'
    fullmodelinfo_path = os.path.join(log_dir, fullmodelinfo_filename)
    with open(fullmodelinfo_path, 'w') as modelinfo_file:
        modelinfo_file.write(unet_model_summary + '\n\n')
        for i, layer in enumerate(model.layers):
            modelinfo_file.write('{}  {}\n'.format(i, layer.name))

    tf.keras.utils.plot_model(model, show_shapes=True)

    # Setup callbacks
    callback_list = [DisplayCallback()]

    callback_list.append(
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                           save_best_only=True, mode='min',
                                           verbose=1))

    callback_list.append(WriteHistoryCallback(history_path))


    plot_history_spec = DEFAULT_PLOT_HISTORY_SPEC_WITH_VALIDATION
    if plot_history_spec:
        plot_history = PlotHistory(os.path.join(log_dir, 'history.png'))
        for key, label, linespec in plot_history_spec:
            plot_history.add_plot(key, label, linespec)
        callback_list.append(plot_history)

    show_predictions(suffix='beforetraining')

    show_predictions(val_generator,  suffix='beforetraining_valgenerator')

    #############################################################################
    # Train the model
    #

    EPOCHS = nepochs
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = n_val // batch_size // VAL_SUBSPLITS

    model_history = model.fit(train_generator, epochs=EPOCHS,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=(val_images, val_masks, val_pixweights),
                              callbacks=callback_list)

    model.save(model_path)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    print('Plotting train and validation loss')
    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.savefig('_trainloss.png')
    plt.close()

    print('show_predictions...')
    show_predictions(val_generator, 8, suffix='aftertraining')

    print('Done')
