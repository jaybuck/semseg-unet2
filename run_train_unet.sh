#!/bin/bash
# Run unet_train.py to train a U-Net model for crack detection
#
# This is basically a template.
# You need to put a copy of this into the directory you will use to
# train a U-Net model and modify it to provide a record of how you
# trained the model.
# Yes, it would be better to build an experiment management system
# (a lightweight one) that would write artifacts to record how a
# particular model was built.
# Seriously, you should do that.
# But, in the meantime, to use this script modify these shell script variables:
#
# PROJECTDIR : github directory holding the semseg_unet code tree.
#
# UNET_TRAINTEST_DIR : The base dir of all of your u-net semantic segmentation
#     train and inference runs.
#
# MODEL_NAME : The name of the directory holding the files created
#     during model training.
#
# IMAGEDIR : The base dir of the directory tree holding the exr images
#     you will use to train, val and test your models.

set -e

export PRENAV_VERBOSITY="0"
if [ "$#" -gt 0 ]
then
    export PRENAV_VERBOSITY="$1"
fi

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "BASEDIR: " $BASEDIR

# Modify this:
# The name of the directory holding the files created during model training.
MODEL_NAME=20210404_training_example

# Modify this:
# To point to the base of the directory tree holding the
# unet semantic segmentation code.
PROJECTDIR=~/github/semset_unet2

SRCDIR=${PROJECTDIR}/src

PROG="${SRCDIR}/unet2_train.py"
echo "PROG:  $PROG"

# Modify this:
# To point to the top of your directory holding your
# U-Net training and inference runs.
UNET_TRAINTEST_DIR=/mnt/efs0/work/PN/trainruns

# Modify this:
# To point to the top of your directory holding the
# directories of image tiles you will use to train, validate and test
# your U-Net models
IMAGEDIR=/mnt/efs0/data1/PN/semseg/cracks/training

RESULTS_DIR="${BASEDIR}"
LOG_DIR="${BASEDIR}/logs"

# Directory holding the trained model
MODEL_DIR="${UNET_TRAINTEST_DIR}/model/${MODEL_NAME}"

# Newly trained model
MODEL_PATH="${MODEL_DIR}/logs/unet_model"
echo "MODEL_PATH: $MODEL_PATH"

# TRAIN_DIR="${IMAGEDIR}/imagetiles_train_224_10000"
TRAIN_DIR="${IMAGEDIR}/pngmask_train"
TEST_DIR="${IMAGEDIR}/pngmask_val"

NEPOCHS=3

echo "Running ${PROG}"
set -x

${PROG} --v ${PRENAV_VERBOSITY} \
        --epochs ${NEPOCHS} \
        --logdir ${LOG_DIR} \
        --batchsize 32 \
        --valbatchsize 3200 \
        --learnrate 0.000001 \
        --posweight 2.0 \
        --traindir ${TRAIN_DIR} \
        --testdir ${TEST_DIR}
