#!/bin/bash
# Run unet_predict.py using trained U-Net model

set -e

export PRENAV_VERBOSITY="0"
if [ "$#" -gt 0 ]
then
    export PRENAV_VERBOSITY="$1"
fi

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "BASEDIR: " $BASEDIR


# An example of a well-trained U-Net for crack detection:
# MODEL_NAME=20200312_posweight2

# The directory holding an example model from quick tests:
MODEL_NAME=MODEL_NAME=20210404_training_example

PROJECTDIR=~/github/semseg-unet2

SRCDIR=${PROJECTDIR}/src

PROG="${SRCDIR}/unet_predict.py"
echo "PROG:  $PROG"

UNET_TRAINTEST_DIR=/mnt/efs0/work/PN/trainruns

IMAGEDIR=/mnt/efs0/data1/PN/semseg/cracks/training
if [[ "$HOME" =~ .*"Users".* ]]; then
  echo "HOME is " $HOME
  IMAGEDIR=/Users/data/PN/semseg/cracks/training/png
fi
echo "IMAGEDIR:  $IMAGEDIR"

RESULTS_DIR="${BASEDIR}"
LOG_DIR="${BASEDIR}/logs"

# Directory holding the trained model
MODEL_DIR="${UNET_TRAINTEST_DIR}/model/${MODEL_NAME}"

# Filepath of trained model
#MODEL_PATH="${MODEL_DIR}/logs/model_checkpoint.h5"
MODEL_PATH="${LOG_DIR}/logs/unetmodel"
echo "MODEL_PATH: $MODEL_PATH"

# The usual validation set of 224x224 image tiles
# TEST_DIR="${IMAGEDIR}/imagetiles_val_224"

# A smaller set for quick tests
TEST_DIR="${IMAGEDIR}/png_val_1000"

echo "Running ${PROG}"
set -x
${PROG} --model_path ${MODEL_PATH} \
        --thresh 0.8 \
        --output_dir "${RESULTS_DIR}/predictions" \
        --verbosity ${PRENAV_VERBOSITY} \
        --testdir ${TEST_DIR}
