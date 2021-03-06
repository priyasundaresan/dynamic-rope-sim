import bpy
import numpy as np
import sys
import os
import cv2
import imageio
import torch
from PIL import Image
import colorsys
import tensorflow as tf
# BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
sys.path.append(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mrcnn_bbox/tools"))
sys.path.insert(0, os.path.join(BASE_DIR, "keypoints_cls"))
sys.path.insert(0, os.path.join(BASE_DIR, "keypoints_cls/src"))

from untangle_utils import *
from render import find_knot
from torchvision import transforms
from keras.models import Model, load_model
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset, compute_ap
from predict import BBoxFinder, PredictionConfig
from datetime import datetime
from PIL import Image
import numpy as np

def load_nets(path_to_refs, network_dir, bbox_dir, use_cuda=0):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    keypoints = KeypointsGauss(4, img_height=480, img_width=640)
    model_path = os.path.join(network_dir, "checkpoints", ref_annots["multi_head_kpt"])
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints.load_state_dict(torch.load(os.path.join(model_path, os.listdir(model_path)[0])))
        keypoints = keypoints.cuda()
    else:
        keypoints.load_state_dict(torch.load(os.path.join(model_path, os.listdir(model_path)[0]), map_location='cpu'))

    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = '%s/%s/mask_rcnn_knot_cfg_0010.h5'%(bbox_dir, ref_annots["bbox"])
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    return keypoints, bbox_predictor

def preds(model, path_to_curr_img, curr_frame, use_cuda=0):
    prediction = Prediction(model, 4, 480, 640, use_cuda)
    transform = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread(path_to_curr_img)
    img_t = transform(img)
    if use_cuda:
        img_t = img_t.cuda()
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    keypoints = []
    for i in range(4):
        h = heatmap[0][i]
        pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
        keypoints.append([pred_x, pred_y])
    prediction.plot(img, heatmap, image_id=curr_frame)
    print("KEYPONTS", keypoints)
    return keypoints

class MultiHead_KPT(object):
    def __init__(self, path_to_refs, network_dir, bbox_net_dir, params):
        self.use_cuda = torch.cuda.is_available()
        net, bbox = load_nets(path_to_refs, network_dir, bbox_net_dir, use_cuda=self.use_cuda)
        self.network = net
        self.bbox_finder = bbox
        self.action_count = 0
        self.rope_length = params["num_segments"]
        self.end1_kp_idx = 3
        self.end2_kp_idx = 0
        self.pull_kp_idx = 1
        self.hold_kp_idx = 2
        self.step = 4

    def find_pull_hold(self, start_frame, render_offset=0, depth=0):
        img_num = max(floor((start_frame-render_offset)/self.step)-1, 0)
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (img_num)

        keypoints = preds(self.network, path_to_curr_img, start_frame-render_offset, use_cuda=self.use_cuda)
        pull_pixel = keypoints[self.pull_kp_idx]
        hold_pixel = keypoints[self.hold_kp_idx]
        print("PULL PIXEL", pull_pixel)
        print("HOLD PIXEL", hold_pixel)

        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        return pull_pixel, hold_pixel

    def bbox_untangle(self, start_frame, render_offset=0):
        path_to_curr_img = "images/%06d_rgb.png" % (floor((start_frame-render_offset)/self.step)-1)
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
        img_num = max(floor((start_frame-render_offset)/self.step)-1,0)
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        keypoints = preds(self.network, path_to_curr_img, start_frame-render_offset, use_cuda=self.use_cuda)
        end1_pixel = keypoints[self.end1_kp_idx]
        end2_pixel = keypoints[self.end2_kp_idx]

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
        img_num = max(floor((start_frame-render_offset)/self.step)-1,0)
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
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
        # end1 is the right endpoint
        img_num = max(floor((start_frame-render_offset)/self.step)-1, 0)
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        keypoints = preds(self.network, path_to_curr_img, start_frame-render_offset, use_cuda=self.use_cuda)
        end1_pixel = keypoints[self.end1_kp_idx]
        end2_pixel = keypoints[self.end2_kp_idx]

        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        middle_frame = reidemeister_right(start_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

        img_num = floor((middle_frame-render_offset)/self.step) -1
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        keypoints = preds(self.network, path_to_curr_img, start_frame-render_offset, use_cuda=self.use_cuda)
        end1_pixel = keypoints[self.end1_kp_idx]
        end2_pixel = keypoints[self.end2_kp_idx]

        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        self.action_count += 2
        return reidemeister_left(middle_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

if __name__ == '__main__':
    # BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
    KP_DIR = os.path.join(BASE_DIR, 'keypoints_cls')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = BC(path_to_refs, NETWORK_DIR, params)
