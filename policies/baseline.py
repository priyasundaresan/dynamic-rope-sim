import bpy
import numpy as np
import sys
import os
import cv2
import imageio
# BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
sys.path.append(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mrcnn_bbox/tools"))

from untangle_utils import *
from render import find_knot
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset, compute_ap
from predict import BBoxFinder, PredictionConfig

def load_bbox(path_to_refs, bbox_net_dir):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = '%s/%s/mask_rcnn_knot_cfg_0010.h5'%(bbox_net_dir, ref_annots["bbox"])
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    return bbox_predictor

class Heuristic(object):
    # HOLD = HIGHEST DEPTH PIXEL, PULL = 15px LEFT OF THAT, ORACLE REID
    def __init__(self, path_to_refs, bbox_net_dir, params):
        self.bbox_finder = load_bbox(path_to_refs, bbox_net_dir)
        self.action_count = 0
        self.max_action_count = 15
        self.rope_length = params["num_segments"]

    def bbox_untangle(self, start_frame, render_offset=0):
        path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
        curr_img = imageio.imread(path_to_curr_img)
        boxes = self.bbox_finder.predict(curr_img, plot=False)
        # sorting by confidence
        # boxes = sorted(boxes, key=lambda box: box[1], reverse=True)
        # sort right to left
        boxes = sorted(boxes, key=lambda box: max(box[0][0], box[0][2]), reverse=True)
        if len(boxes) == 0:
            return None, 0
        return boxes[0] # ASSUME first box is knot to be untied

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
        # if self.action_count > self.max_action_count:
        #     return True
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return True
        end2_idx = self.rope_length-1
        end1_idx = -1
        ret = undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)
        # if ret:
        #     self.num_knots -= 1
        return ret

    def find_pull_hold(self, start_frame, render_offset=0):
        # Hold = highest depth pixel
        # Pull = 15 px left of that
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return None, None
        path_to_curr_depth = "images_depth/%06d_rgb.png" % (start_frame-render_offset)
        img_depth = cv2.imread(path_to_curr_depth)
        crop, rescale_factor, (x_off, y_off) = crop_and_resize(box, img_depth)
        cv2.imwrite("./preds/%06d_crop_depth.png" % (start_frame-render_offset), crop)
        cv2.imwrite("./preds/%06d_bbox.png" % (start_frame-render_offset), crop)

        y,x,_ = np.where(crop==np.amax(crop))
        highest_depth_px = np.vstack((x,y)).T
        highest_depth_px = np.sort(highest_depth_px, axis=0) # Of the highest depth pixels, take the rightmost one to be the hold
        hold_crop_pixel = highest_depth_px[-1]
        pull_crop_pixel = [hold_crop_pixel[0]-15, hold_crop_pixel[1]]
        hold_pixel = pixel_crop_to_full(np.array([hold_crop_pixel]), rescale_factor, x_off, y_off)[0]
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        pull_pixel = pixel_crop_to_full(np.array([pull_crop_pixel]), rescale_factor, x_off, y_off)[0]
        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        return pull_pixel, hold_pixel

    def undo(self, start_frame, render=False, render_offset=0):
        pull_pixel, hold_pixel = self.find_pull_hold(start_frame, render_offset=render_offset)
        dx = pull_pixel[0] - hold_pixel[0]
        dy = pull_pixel[1] - hold_pixel[1]
        action_vec = [dx, dy, 6] # 6 is arbitrary for dz
        action_vec /= np.linalg.norm(action_vec)

        path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
        img = cv2.imread(path_to_curr_img)
        img = cv2.circle(img, tuple(hold_pixel), 5, (255, 0, 0), 2)
        img = cv2.circle(img, tuple(pull_pixel), 5, (0, 0, 255), 2)
        img = cv2.arrowedLine(img, tuple(pull_pixel), (pull_pixel[0]+dx*5, pull_pixel[1]+dy*5), (0, 255, 0), 2)
        cv2.imwrite("./preds/%06d_action.png" % (start_frame-render_offset), img)

        hold_idx = pixels_to_cylinders([hold_pixel])
        pull_idx = pixels_to_cylinders([pull_pixel])
        end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, render=render, render_offset=render_offset)
        self.action_count += 1
        return end_frame, pull_pixel, hold_pixel, action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):
        middle_frame = reidemeister_right(start_frame, -1, self.rope_length-1, render=render, render_offset=render_offset)
        end_frame = reidemeister_left(middle_frame, -1, self.rope_length-1, render=render, render_offset=render_offset)
        self.action_count += 2
        return end_frame

if __name__ == '__main__':
    BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BBOX_DIR = os.path.join(BASE_DIR, 'mrcnn_bbox', 'networks')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = Random(path_to_refs, BBOX_DIR, params)
