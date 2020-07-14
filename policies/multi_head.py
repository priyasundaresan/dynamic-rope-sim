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

from untangle_utils import *
from render import find_knot
from dataset import transform
from torchvision import transforms
from keras.models import Model, load_model

def load_nets(path_to_refs, network_dir):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    # return load_model(os.path.join(network_dir, ref_annots["multi_head"], "saved-model-20-380.131.hdf5"), {'tf': tf})
    return load_model(os.path.join(network_dir, ref_annots["multi_head"], "two_head_untangling.hdf5"), {'tf': tf})

def img_reshape(input_img):
    img_dim = [640,480,3]
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

def plot_pred_actions(img, act_pred):
    pull_loc, drop_loc, hold_loc = act_pred.astype(int)
    cv2.circle(img, tuple(hold_loc), 3, (255,0,0), -1)
    cv2.arrowedLine(img, tuple(pull_loc), tuple(drop_loc), (0,255,0), 2)
    return img

def plot_predictions(img, pred):
    vis = img.copy()
    #(keypoints, reid, terminate) = pred
    (keypoints, terminate) = pred
    keypoints = keypoints.reshape((4,2)).astype(int)
    #reid = reid[0].argmax()
    terminate = terminate[0].argmax()
#    if reid:
#        cv2.putText(vis, "Take Reid", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (255, 255, 255), 2)
    if terminate:
        cv2.putText(vis, "Undone", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (255, 255, 255), 2)
    for i, (u, v) in enumerate(keypoints):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(keypoints), 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis,(int(u), int(v)), 4, (R, G, B), -1)
    return vis

def preds(model, path_to_curr_img, curr_frame):
    img = cv2.imread(path_to_curr_img)
    input_img = img_reshape(img)
    pred = model.predict(input_img)
    vis = plot_predictions(img, pred)
    if '%05d.jpg'%curr_frame not in os.listdir('preds'):
        cv2.imwrite('preds/%05d.jpg'%curr_frame, vis)
    else:
        cv2.imwrite('preds/%05d_2.jpg'%curr_frame, vis)
    return pred

class MultiHead(object):
    def __init__(self, path_to_refs, network_dir, params):
        net = load_nets(path_to_refs, network_dir)
        self.network = net
        self.action_count = 0
        self.rope_length = params["num_segments"]
        self.end1_kp_idx = 0
        self.end2_kp_idx = 3
        self.pull_kp_idx = 1
        self.hold_kp_idx = 2

    def find_pull_hold(self, start_frame, render_offset=0, depth=0):
        img_num = start_frame-render_offset
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (img_num)

        pred = preds(self.network, path_to_curr_img, start_frame-render_offset)
        (keypoints, terminate) = pred
        keypoints = keypoints[0]
        pull_pixel = keypoints[2*self.pull_kp_idx:2*self.pull_kp_idx+2]
        hold_pixel = keypoints[2*self.hold_kp_idx:2*self.hold_kp_idx+2]

        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        return pull_pixel, hold_pixel

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
        # JENN FIX THIS
        pred = preds(self.network, path_to_curr_img, start_frame-render_offset)
        (keypoints, terminate) = pred
        return terminate

    def bbox_untangle(self, start_frame, render_offset=0):
        # JENN assume there is at least one knot at first
        return True, None
        # pred = preds(self.bc_network, path_to_curr_img, start_frame-render_offset)
        # (keypoints, terminate) = pred
        # return None if terminate else True, None

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
        return end_frame+50, pull_pixel, hold_pixel, action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):
        path_to_curr_img = "images/%06d_rgb.png"%(start_frame-render_offset)
        pred = preds(self.network, path_to_curr_img, start_frame-render_offset)
        (keypoints, terminate) = pred
        keypoints = keypoints[0]
        end1_pixel = keypoints[2*self.end1_kp_idx:2*self.end1_kp_idx+2]
        end2_pixel = keypoints[2*self.end2_kp_idx:2*self.end2_kp_idx+2]

        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        middle_frame = reidemeister_right(start_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

        path_to_curr_img = "images/%06d_rgb.png"%(middle_frame-1-render_offset)
        pred = preds(self.network, path_to_curr_img, start_frame-render_offset)
        (keypoints, terminate) = pred
        keypoints = keypoints[0]
        end1_pixel = keypoints[2*self.end1_kp_idx:2*self.end1_kp_idx+2]
        end2_pixel = keypoints[2*self.end2_kp_idx:2*self.end2_kp_idx+2]

        end2_idx = pixels_to_cylinders([end2_pixel])
        end1_idx = pixels_to_cylinders([end1_pixel])
        self.action_count += 2
        return reidemeister_left(middle_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

if __name__ == '__main__':
    # BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
    KP_DIR = os.path.join(BASE_DIR, 'multi_head')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = BC(path_to_refs, NETWORK_DIR, params)
