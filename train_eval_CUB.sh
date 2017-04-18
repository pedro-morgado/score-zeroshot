#!/usr/bin/env bash

# Prepare CUB dataset
if [ ! -d 'CUB_200_2011' ] && [ ! -f 'CUB_200_2011.tgz' ]; then
    # Download CUB dataset
    wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
fi
if [ ! -d 'CUB_200_2011' ]; then
    # Extract files
    echo 'Expanding tgz file...'
    tar xfz CUB_200_2011.tgz
    rm CUB_200_2011.tgz
    rm attributes.txt
    echo 'Done!'
fi
IMG_DIR='CUB_200_2011/images'

# Prepare LMDBs (score_train.py and score_eval.py expect LMDBs to be at 'LMDBs/CUB')
# Script prep_lmdbs.sh creates LMDBs for images, attributes, hierarchies and word2vec
./prep_lmdbs.sh CUB ${IMG_DIR}

# Download GoogLeNet caffemodel
if [ ! -f CNNs/inception_v1.caffemodel ]; then
    wget -O CNNs/inception_v1.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
fi

# Train SCoRe with attributes
MODEL_DIR='train_dir'
if [ -d ${MODEL_DIR} ]; then
    rm -r ${MODEL_DIR}
fi
mkdir ${MODEL_DIR}
python score_train.py ${MODEL_DIR} CUB Attributes GoogLeNet -g 0.01 -c 1.0 --iters 1000 --init_lr 0.001

# Evaluate SCoRe model
python score_eval.py ${MODEL_DIR} CUB Attributes GoogLeNet
