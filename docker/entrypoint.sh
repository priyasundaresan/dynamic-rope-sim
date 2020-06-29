#!/bin/bash
set -e

function use_pytorch_dense_correspondence()
{
    source /code/docker/setup_environment.sh
}

export -f use_pytorch_dense_correspondence

exec "$@"

cd /code
cp /code/dense_correspondence/pytorch-segmentation-detection/vision/torchvision/models/resnet.py /bin/2.83/python/lib/python3.7/site-packages/torchvision/models/resnet.py
