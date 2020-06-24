import bpy
import numpy as np
import sys
import os
import cv2
import imageio
# BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
sys.path.append(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(BASE_DIR, "dense_correspondence/tools"))
sys.path.insert(0, os.path.join(BASE_DIR, "mrcnn_bbox/tools"))

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
            "hold": {"ref_path": hold_ref, "pixels": [crop_hold]}, \
            "pull": {"ref_path": pull_ref, "pixels": [crop_pull]}}
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = '%s/%s/mask_rcnn_knot_cfg_0010.h5'%(bbox_dir, ref_annots["bbox"])
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    nets = {"ends_cf": load_cf(dense_correspondence_dir, ref_annots["ends_cf"], 640, 480), \
                       "pull_cf": load_cf(dense_correspondence_dir, ref_annots["pull_cf"], 50, 50), \
                       "hold_cf": load_cf(dense_correspondence_dir, ref_annots["hold_cf"], 80, 60), \
                       "bbox": bbox_predictor}
    return nets, refs

def descriptor_matches(cf, path_to_ref_img, path_to_curr_img, pixels, curr_frame):
    cf.load_image_pair(path_to_ref_img, path_to_curr_img)
    cf.compute_descriptors()
    best_matches, _ = cf.find_k_best_matches(pixels, 50, mode="median")
    vis = cf.show_side_by_side(plot=False)
    if not "%06d_desc.png" % curr_frame in os.listdir("preds/"):
        cv2.imwrite("preds/%06d_desc.png" % curr_frame, vis)
    else:
        i = 2
        while "%06d_desc_%d.png" % (curr_frame, i) in os.listdir("preds/"):
            i += 1
        cv2.imwrite("preds/%06d_desc_%01d.png" % (curr_frame, i), vis)
    return best_matches

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
        self.hold_box_width = 50
        self.hold_box_height = 50
        self.rope_length = params["num_segments"]
        self.max_frame_count = 50000

    def find_pull_hold(self, start_frame, render_offset=0, depth=0):
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return None, None
        img_num = start_frame-render_offset
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (img_num)
        path_to_curr_img_crop = "images/%06d_crop.png" % (img_num)
        path_to_curr_img_depth_crop = "images_depth/%06d_crop.png" % (img_num)
        path_to_curr_img_hold_crop = "images/%06d_hold_crop.png" % (img_num)
        path_to_curr_img_depth_hold_crop = "images_depth/%06d_hold_crop.png" % (img_num)
        img = cv2.imread(path_to_curr_img)
        img_depth = cv2.imread(path_to_curr_img_depth)
        crop, rescale_factor, (x_off, y_off) = crop_and_resize(box, img)
        crop_depth, _, _ = crop_and_resize(box, img_depth)
        cv2.imwrite("images/%06d_crop.png" % (img_num), crop)
        cv2.imwrite("images_depth/%06d_crop.png" % (img_num), crop_depth)
        cv2.imwrite("./preds/%06d_bbox.png" % (img_num), crop)

        path_to_curr_hold_img = path_to_curr_img_depth_crop if depth else path_to_curr_img_crop
        hold_crop_pixel = descriptor_matches(self.hold_cf, self.path_to_hold_ref, path_to_curr_hold_img, \
                                self.hold_ref_pixels, start_frame-render_offset)[0]
        hold_pixel = pixel_crop_to_full(np.array([hold_crop_pixel]), rescale_factor, x_off, y_off)[0]
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        hold_x, hold_y = hold_pixel
        hold_box_width, hold_box_height = self.hold_box_width, self.hold_box_height
        hold_box = [hold_x-hold_box_width//2, hold_y-hold_box_height//2, hold_x+hold_box_width//2, hold_y+hold_box_height//2]
        hold_crop_rgb, hold_rescale_factor, (hold_x_off, hold_y_off) = crop_and_resize(hold_box, img, aspect=(50,50))
        hold_crop_depth, hold_rescale_factor, (hold_x_off, hold_y_off) = crop_and_resize(hold_box, img_depth, aspect=(50,50))
        cv2.imwrite("images/%06d_hold_crop.png" % (img_num), hold_crop_rgb)
        cv2.imwrite("images_depth/%06d_hold_crop.png" % (img_num), hold_crop_depth)

        path_to_curr_pull_img = path_to_curr_img_depth_hold_crop if depth else path_to_curr_img_hold_crop
        pull_crop_pixel = descriptor_matches(self.pull_cf, self.path_to_pull_ref, path_to_curr_pull_img, self.pull_ref_pixels, \
                            start_frame-render_offset)[0]
        hold_pixel = pixel_crop_to_full(np.array([hold_crop_pixel]), rescale_factor, x_off, y_off)[0]
        pull_pixel = pixel_crop_to_full(np.array([pull_crop_pixel]), hold_rescale_factor, hold_x_off, hold_y_off)[0]
        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        return pull_pixel, hold_pixel

    def bbox_untangle(self, start_frame, render_offset=0):
        path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
        curr_img = imageio.imread(path_to_curr_img)
        boxes = self.bbox_finder.predict(curr_img, plot=False)
        boxes = sorted(boxes, key=lambda box: box[1], reverse=True)
        if len(boxes) == 0:
            return None, 0
        return boxes[0] # ASSUME first box is knot to be untied

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
        #if start_frame > self.max_frame_count:
        #    return True
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return True
        path_to_curr_img = "images/%06d_rgb.png"%(start_frame-render_offset)
        end2_pixel, end1_pixel = descriptor_matches(self.ends_cf, self.path_to_ends_ref, path_to_curr_img, \
                            self.ends_ref_pixels, start_frame-render_offset)
        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        return undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)

    def undo(self, start_frame, render=False, render_offset=0):
        pull_pixel, hold_pixel = self.find_pull_hold(start_frame, render_offset=render_offset)
        if pull_pixel is None:
            return start_frame, None, None, None
        dx = pull_pixel[0] - hold_pixel[0]
        dy = pull_pixel[1] - hold_pixel[1]
        action_vec = [dx, dy, 6] # 6 is arbitrary for dz
        action_vec /= np.linalg.norm(action_vec)

        print("hold", hold_pixel)
        print("pull", pull_pixel)
        path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
        img = cv2.imread(path_to_curr_img)
        img = cv2.circle(img, tuple(hold_pixel), 5, (255, 0, 0), 2)
        img = cv2.circle(img, tuple(pull_pixel), 5, (0, 0, 255), 2)
        img = cv2.arrowedLine(img, tuple(pull_pixel), (pull_pixel[0]+dx*5, pull_pixel[1]+dy*5), (0, 255, 0), 2)
        cv2.imwrite("./preds/%06d_action.png" % (start_frame-render_offset), img)
        #cv2.imshow("action", img)
        #cv2.waitKey(0)

        hold_idx = pixels_to_cylinders([hold_pixel])
        pull_idx = pixels_to_cylinders([pull_pixel])
        end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, render=render, render_offset=render_offset)
        self.action_count += 1
        return end_frame, pull_pixel, hold_pixel, action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):
        path_to_curr_img = "images/%06d_rgb.png"%(start_frame-render_offset)
        end2_pixel, end1_pixel = descriptor_matches(self.ends_cf, self.path_to_ends_ref, path_to_curr_img, \
                            self.ends_ref_pixels, start_frame-render_offset)
        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        middle_frame = reidemeister_right(start_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

        path_to_curr_img = "images/%06d_rgb.png"%(middle_frame-1-render_offset)
        end2_pixel, end1_pixel = descriptor_matches(self.ends_cf, self.path_to_ends_ref, path_to_curr_img, \
                            self.ends_ref_pixels, middle_frame-1-render_offset)
        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        self.action_count += 2
        return reidemeister_left(middle_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

if __name__ == '__main__':
    # BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
    DESCRIPTOR_DIR = os.path.join(BASE_DIR, 'dense_correspondence')
    BBOX_DIR = os.path.join(BASE_DIR, 'mrcnn_bbox', 'networks')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = Hierarchical(path_to_refs, DESCRIPTOR_DIR, BBOX_DIR, params)
