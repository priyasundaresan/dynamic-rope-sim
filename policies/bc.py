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

from untangle_utils import *
from render import find_knot
from dataset import transform
from torchvision import transforms
from keras.models import Model, load_model

def load_nets(path_to_refs, network_dir):
    with open('%s/ref.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    return load_model(os.path.join(network_dir, ref_annots["bc"]))

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

def preds(model, path_to_curr_img, curr_frame):
    img = cv2.imread(path_to_curr_img)
    input_img = img_reshape(img)
    act_pred = model.predict(input_img).reshape((3,2))
    pull_loc, drop_loc, hold_loc = act_pred.astype(int)
    vis = plot_pred_actions(img, act_pred)
    cv2.imwrite('preds/%05d.jpg'%curr_frame, vis)
    return act_pred

class BC(object):
    def __init__(self, path_to_refs, network_dir, params):
        net = load_nets(path_to_refs, network_dir)
        self.bc_network = net
        self.action_count = 0
        self.rope_length = params["num_segments"]

    def find_pull_hold(self, start_frame, render_offset=0, depth=0):
        img_num = start_frame-render_offset
        path_to_curr_img = "images/%06d_rgb.png" % (img_num)
        path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (img_num)

        pull_pixel, drop_pixel, hold_pixel = preds(self.bc_network, path_to_curr_img, start_frame-render_offset)
        pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
        hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))
        drop_pixel = (int(drop_pixel[0]), int(drop_pixel[1]))
        return pull_pixel, hold_pixel, drop_pixel

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
        # JENN FIX THIS
        return False

    def undo(self, start_frame, render=False, render_offset=0):
        pull_pixel, hold_pixel, drop_pixel = self.find_pull_hold(start_frame, render_offset=render_offset)
        if pull_pixel is None:
            return start_frame, None, None, None
        dx = drop_pixel[0] - pull_pixel[0]
        dy = drop_pixel[1] - pull_pixel[1]
        action_vec = [dx, dy, 6] # 6 is arbitrary for dz

        hold_idx = pixels_to_cylinders([hold_pixel])
        pull_idx = pixels_to_cylinders([pull_pixel])
        end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, render=render, render_offset=render_offset, scale_action=False)
        self.action_count += 1
        return end_frame, pull_pixel, hold_pixel, action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):
        # just do the undo action predicted
        return self.undo(start_frame, render=render, render_offset=render_offset)[0]

if __name__ == '__main__':
    # BASE_DIR = '/Users/priyasundaresan/Desktop/blender/dynamic-rope'
    BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
    KP_DIR = os.path.join(BASE_DIR, 'bc_networks')
    path_to_refs = os.path.join(BASE_DIR, 'references', 'capsule')
    with open("../rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = BC(path_to_refs, NETWORK_DIR, params)
