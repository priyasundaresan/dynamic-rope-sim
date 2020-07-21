export DC_SOURCE_DIR=/code/dense_correspondence
export MRCNN_SOURCE_DIR=/code/mrcnn_bbox/tools
export KP_DIR=/code/keypoints_dir

export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/pytorch-segmentation-detection
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/tools
export PYTHONPATH=$PYTHONPATH:$KP_DIR/src
export PYTHONPATH=$PYTHONPATH:$MRCNN_SOURCE_DIR
export PATH=$PATH:$DC_SOURCE_DIR/bin
export PATH=$PATH:$DC_SOURCE_DIR/modules/dense_correspondence_manipulation/scripts
use_director(){
    export PATH=$PATH:~/director/bin
}

export -f use_director
