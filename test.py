import pprint
import json
import sys
import os
import cv2
import numpy as np
import copy
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/tools"))

from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder
from image_utils import * 

if __name__ == '__main__':
    base_dir = 'dense_correspondence/networks'
    network_dir = 'rope_400_cyl_rot_16'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('dense_correspondence/cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    f1 = "reference_images/knot_reference.png"
    with open('reference_images/knot_reference.json', 'r') as f:
        ref_annots = json.load(f)
        pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
        hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
        pixels = [pull, hold]
    for i in range(1,2):
        f2 = "rope_400_cyl_rot/processed/images/%06d_rgb.png"%i
        cf.load_image_pair(f1, f2)
        cf.compute_descriptors()
        best_matches = cf.find_k_best_matches(pixels, 50, mode="median")
        vis = cf.show_side_by_side()
