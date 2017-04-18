#!/usr/bin/env bash
if [ ! -f CNNs/alexnet.caffemodel ]; then
    wget -O CNNs/alexnet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
fi
if [ ! -f CNNs/inception_v1.caffemodel ]; then
    wget -O CNNs/inception_v1.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
fi
if [ ! -f CNNs/vgg19.caffemodel ]; then
    wget -O CNNs/vgg19.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
fi
