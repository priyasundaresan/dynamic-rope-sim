import bpy
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(os.getcwd(), "../dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "../dense_correspondence/tools"))
sys.path.insert(0, os.path.join(os.getcwd(), "../mrcnn_bbox/tools"))

from untangle_utils import *
from render import find_knot
from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder
from torchvision import transforms
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset, compute_ap
from predict import BBoxFinder, PredictionConfig

def load_cf(dense_correspondence_dir, network_name, image_width, image_height):
    network_dir = os.path.join(dense_correspondence_dir, 'networks')
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(network_dir, network_name), \
        model_param_file=os.path.join(network_dir, network_name, '003501.pth'))
    with open('%s/cfg/dataset_info.json'%dense_correspondence_dir, 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev, image_width=image_width, image_height=image_height)
    return cf

def load_ref_nets(path_to_refs, dense_correspondence_dir, bbox_dir):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    left_end = [ref_annots["reid_left_x"], ref_annots["reid_left_y"]]
    right_end = [ref_annots["reid_right_x"], ref_annots["reid_right_y"]]
    ends = [left_end, right_end]
    crop_pull = [ref_annots["crop_pull_x"], ref_annots["crop_pull_y"]]
    crop_hold = [ref_annots["crop_hold_x"], ref_annots["crop_hold_y"]]
    reid_ref = os.path.join(path_to_refs, 'reid_ref.png')
    pull_ref = os.path.join(path_to_refs, 'pull_ref.png')
    hold_ref = os.path.join(path_to_refs, 'hold_ref.png')
    refs = {"ends": {"ref_path": reid_ref, "pixels": ends}, \
            "hold": {"ref_path": hold_ref, "pixels": crop_hold}, \
            "pull": {"ref_path": pull_ref, "pixels": crop_pull}}
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = '%s/%s/mask_rcnn_knot_cfg_0010.h5'%(bbox_dir, ref_annots["bbox"])
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    nets = {"ends_cf": load_cf(dense_correspondence_dir, ref_annots["ends_cf"], 640, 480), \
                       "pull_cf": load_cf(dense_correspondence_dir, ref_annots["pull_cf"], 80, 60), \
                       "hold_cf": load_cf(dense_correspondence_dir, ref_annots["hold_cf"], 50, 50), \
                       "bbox": bbox_predictor}
    return nets, refs

class Hierarchical(object):
    def __init__(self, path_to_refs, dense_correspondence_dir, bbox_net_dir, params):
        nets, refs = load_ref_nets(path_to_refs, dense_correspondence_dir, bbox_net_dir)
        self.ends_cf = nets["ends_cf"]
        self.pull_cf = nets["pull_cf"]
        self.hold_cf = nets["hold_cf"]
        self.bbox_finder = nets["bbox"]
        self.path_to_ends_ref = refs["ends"]["ref_path"]
        self.path_to_hold_ref = refs["hold"]["ref_path"]
        self.path_to_pull_ref = refs["pull"]["ref_path"]
        self.ends_ref_pixels = refs["ends"]["pixels"]
        self.hold_ref_pixels = refs["hold"]["pixels"]
        self.pull_ref_pixels = refs["pull"]["pixels"]
        self.action_count = 0
        self.rope_length = params["num_segments"]

    def undone_check(self):
        pass

    def undo(self, start_frame, render=False, render_offset=0):
        pass
    
    def reidemeister(self, start_frame, render=False, render_offset=0):
        pass

if __name__ == '__main__':
    BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    DESCRIPTOR_DIR = os.path.join(BASE_DIR, 'dense_correspondence')
    BBOX_DIR = os.path.join(BASE_DIR, 'mrcnn_bbox', 'networks')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = Hierarchical(path_to_refs, DESCRIPTOR_DIR, BBOX_DIR, params)
