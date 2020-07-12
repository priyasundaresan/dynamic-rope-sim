import bpy
import numpy as np
import sys
import os
import cv2
import imageio
import torch
from PIL import Image
# BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
sys.path.append(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mrcnn_bbox/tools"))
sys.path.insert(0, os.path.join(BASE_DIR, "keypoints_dir/src"))

from untangle_utils import *
from render import find_knot
from model import Keypoints
from prediction_simple import Prediction
# from prediction import Prediction
from dataset import transform
from torchvision import transforms
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset, compute_ap
from predict import BBoxFinder, PredictionConfig

def load_kp(kp_dir, network_name, image_width, image_height, num_classes):
    network_dir = os.path.join(kp_dir, 'checkpoints', network_name)
    keypoints = Keypoints(num_classes, img_height=image_height, img_width=image_width)
    keypoints.load_state_dict(torch.load(os.path.join(network_dir, 'model_2_1_199.pth'), map_location='cpu'))
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints = keypoints.cuda()
    prediction = Prediction(keypoints, num_classes, image_height, image_width, None, None, use_cuda)
    return prediction

def load_nets(path_to_refs, kp_dir, bbox_dir):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = '%s/%s/mask_rcnn_knot_cfg_0010.h5'%(bbox_dir, ref_annots["bbox"])
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    nets = {"ends_kp": load_kp(kp_dir, ref_annots["ends_kp"], 640, 480, 4), \
            # "knot_kp": load_kp(kp_dir, ref_annots["knot_kp"], 640, 480, 4), \
            # JENN: uncomment when using bbox training
            "knot_kp": load_kp(kp_dir, ref_annots["knot_kp"], 80, 60, 2), \
            "bbox": bbox_predictor}
    return nets

def kp_matches(prediction, path_to_curr_img, curr_frame, num_classes, use_cuda=0):
    img = Image.open(path_to_curr_img)
    img = np.array(img)
    img_t = transform(img)
    if use_cuda:
        img_t = img_t.cuda()
    result, keypoints = prediction.predict(img_t)
    prediction.plot(img, result, keypoints, image_id=curr_frame)

    print("KEYPOINTS", keypoints)
    # TODO: later return end1, end2 pixels if ends_cf OR hold, pull pixels if knot_cf
    # currently returns end1, pull, hold, end2
    return keypoints

class Hierarchical_kp(object):
    def __init__(self, path_to_refs, kp_dir, bbox_net_dir, params):
        nets = load_nets(path_to_refs, kp_dir, bbox_net_dir)
        self.ends_kp = nets["ends_kp"]
        self.knot_kp = nets["knot_kp"]
        self.bbox_finder = nets["bbox"]
        self.action_count = 0
        self.rope_length = params["num_segments"]

    def find_pull_hold(self, start_frame, render_offset=0, depth=0):
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return None, None
        img_num = start_frame-render_offset
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (img_num)
        path_to_curr_img_crop = "images/%06d_crop.png" % (img_num)
        path_to_curr_img_depth_crop = "images_depth/%06d_crop.png" % (img_num)
        img = cv2.imread(path_to_curr_img)
        img_depth = cv2.imread(path_to_curr_img_depth)
        crop, rescale_factor, (x_off, y_off) = crop_and_resize(box, img)
        crop_depth, _, _ = crop_and_resize(box, img_depth)
        cv2.imwrite("images/%06d_crop.png" % (img_num), crop)
        cv2.imwrite("images_depth/%06d_crop.png" % (img_num), crop_depth)
        cv2.imwrite("./preds/%06d_bbox.png" % (img_num), crop)

        path_to_curr_hold_img = path_to_curr_img_depth_crop if depth else path_to_curr_img_crop
        # @JENN FIX
        # for current global view cf:
        # _, pull_pixel, hold_pixel, _ = kp_matches(self.knot_kp, path_to_curr_img, start_frame-render_offset, 4)
        # print("PULL", pull_pixel)
        # print("HOLD", hold_pixel)
        # uncomment for bbox cf:
        pull_crop_pixel, hold_crop_pixel = kp_matches(self.knot_kp, path_to_curr_img_crop, start_frame-render_offset, 2)
        hold_pixel = pixel_crop_to_full(np.array([hold_crop_pixel]), rescale_factor, x_off, y_off)[0]
        pull_pixel = pixel_crop_to_full(np.array([pull_crop_pixel]), rescale_factor, x_off, y_off)[0]
        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        return pull_pixel, hold_pixel

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
        box, confidence = self.bbox_untangle(start_frame, render_offset=render_offset)
        if box is None:
            return True
        path_to_curr_img = "images/%06d_rgb.png"%(start_frame-render_offset)
        # @JENN FIX
        # for current global view cf:
        end1_pixel, _, _, end2_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 4)
        # uncomment for bbox cf:
        # end2_pixel, end1_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 2)
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
        # @JENN FIX
        # for current global view cf:
        end1_pixel, _, _, end2_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 4)
        # uncomment for bbox cf:
        # end2_pixel, end1_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 2)
        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        middle_frame = reidemeister_right(start_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

        path_to_curr_img = "images/%06d_rgb.png"%(middle_frame-1-render_offset)
        # @JENN FIX
        # for current global view cf:
        end1_pixel, _, _, end2_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 4)
        # uncomment for bbox cf:
        # end2_pixel, end1_pixel = kp_matches(self.ends_kp, path_to_curr_img, start_frame-render_offset, 2)
        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        self.action_count += 2
        return reidemeister_left(middle_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

if __name__ == '__main__':
    # BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
    KP_DIR = os.path.join(BASE_DIR, 'keypoints_dir')
    BBOX_DIR = os.path.join(BASE_DIR, 'mrcnn_bbox', 'networks')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = Hierarchical_kp(path_to_refs, KP_DIR, BBOX_DIR, params)
