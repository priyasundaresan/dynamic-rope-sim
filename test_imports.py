import bpy
import numpy as np
from math import pi
import json
import cv2
import copy
from PIL import Image
from torchvision import transforms
import scipy
from sklearn.neighbors import NearestNeighbors
import matplotlib
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset, compute_ap
import imageio
import os
import sys

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/tools"))
sys.path.insert(0, os.path.join(os.getcwd(), "mrcnn_bbox/tools"))

from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder
from image_utils import * 

from rigidbody_rope import *

from predict import BBoxFinder, PredictionConfig

if __name__ == '__main__':
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = 'mrcnn_bbox/networks/knot_network_1000/mask_rcnn_knot_cfg_0007.h5'
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    img = imageio.imread('images/000149_rgb.png')
    bbox_predictor.predict(img)
    
    #base_dir = 'dense_correspondence/networks'
    #network_dir = 'rope_400_cyl_rot_16'
    #dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    #dcn.eval()
    #with open('dense_correspondence/cfg/dataset_info.json', 'r') as f:
    #    dataset_stats = json.load(f)
    #dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    #cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    #f1 = "reference_images/knot_reference.png"
    #with open('reference_images/knot_reference.json', 'r') as f:
    #    ref_annots = json.load(f)
    #    pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
    #    hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
    #    pixels = [pull, hold]
    #for i in range(200,400,70):
    #    #f2 = "rope_400_cyl_rot/processed/images/%06d_rgb.png"%i
    #    f2 = "images/%06d.png"%i
    #    cf.load_image_pair(f1, f2)
    #    cf.compute_descriptors()
    #    best_matches = cf.find_k_best_matches(pixels, 50, mode="median")
    #    vis = cf.show_side_by_side()
