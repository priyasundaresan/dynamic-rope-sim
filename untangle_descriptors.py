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
import argparse

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/tools"))
sys.path.insert(0, os.path.join(os.getcwd(), "mrcnn_bbox/tools"))

from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder
from knots import *
from render import *
from image_utils import *
from rigidbody_rope import *
from knots import *
from untangle_utils import *

from predict import BBoxFinder, PredictionConfig

def descriptor_matches(cf, path_to_ref_img, pixels, curr_frame, crop=False, depth=False, hold=None):
    dir = "images_depth/" if depth else "images/"
    path_to_curr_img = dir+"%06d_crop.png" % curr_frame if crop else dir+"%06d_rgb.png" % curr_frame
    path_to_ref_img = path_to_ref_img[1] if crop and depth else path_to_ref_img[0]
    cf.load_image_pair(path_to_ref_img, path_to_curr_img)
    cf.compute_descriptors()
    if crop:
        # vis = cf.show_side_by_side(pixels=[pixels, best_matches], plot=False)
        best_matches, _ = cf.find_k_best_matches(pixels, 50, mode="median", hold=hold)
        vis = cf.show_side_by_side(plot=False)
    else:
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

def reidemeister_descriptors(start_frame, cf, path_to_ref_img, ref_end_pixels, render=False, render_offset=0):
    end2_pixel, end1_pixel = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, start_frame-render_offset)
    end2_idx = pixels_to_cylinders([end2_pixel])
    end1_idx = pixels_to_cylinders([end1_pixel])
    middle_frame = reidemeister_right(start_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

    end2_pixel, end1_pixel = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, middle_frame-1-render_offset)
    end2_idx = pixels_to_cylinders([end2_pixel])
    end1_idx = pixels_to_cylinders([end1_pixel])
    return reidemeister_left(middle_frame, end1_idx, end2_idx, render=render, render_offset=render_offset)

def reidemeister_oracle(start_frame, render=False, render_offset=0):
    middle_frame = reidemeister_right(start_frame, -1, 49, render=render, render_offset=render_offset)
    return reidemeister_left(middle_frame, -1, 49, render=render, render_offset=render_offset)


def bbox_untangle(start_frame, bbox_detector, render_offset=0):
    path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
    curr_img = imageio.imread(path_to_curr_img)
    boxes = bbox_predictor.predict(curr_img, plot=False)
    # undo furthest right box first
    # boxes = sorted(boxes, key=lambda box: box[0][2], reverse=True)
    # sort boxes by confidence
    boxes = sorted(boxes, key=lambda box: box[1], reverse=True)
    if len(boxes) == 0:
        return None, 0
    return boxes[0] # ASSUME first box is knot to be untied

def undone_check_descriptors(start_frame, ends_cf, path_to_ref_full_img, ref_end_pixels, prev_pull, prev_hold, prev_action_vec, render_offset=0):
    # get endpoints from cf
    end2_pixel, end1_pixel = descriptor_matches(ends_cf, path_to_ref_full_img, ref_end_pixels, start_frame-render_offset)
    end2_idx = pixels_to_cylinders([end2_pixel])
    end1_idx = pixels_to_cylinders([end1_pixel])
    return undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)

def undone_check_oracle(start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
    # ground truth endpoints
    end2_idx = 49
    end1_idx = -1
    return undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)

def undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=0):
    piece = "Cylinder"

    hold_idx = pixels_to_cylinders([prev_hold])
    pull_idx = pixels_to_cylinders([prev_pull])
    pull_cyl = get_piece(piece, pull_idx)
    hold_cyl = get_piece(piece, hold_idx)

    end_idx = end1_idx # we are always undoing the right side
    print("pull_idx", pull_idx)
    print("end_idx", end_idx)
    end_cyl = get_piece(piece, end_idx)
    end_loc = end_cyl.matrix_world.translation
    hold_loc = hold_cyl.matrix_world.translation
    pull_loc = pull_cyl.matrix_world.translation

    prev_action_vec = prev_action_vec[:-1]/np.linalg.norm(prev_action_vec[:-1])
    end_hold_vec = np.array(end_loc - hold_loc)[:-1]/np.linalg.norm(np.array(end_loc - hold_loc)[:-1])
    print("action_vec", prev_action_vec)
    print("end_loc - hold_loc", end_hold_vec)
    print("dot", np.dot(prev_action_vec, end_hold_vec))
    if np.dot(prev_action_vec, end_hold_vec) > 0.7:
        return True
    return False

def find_pull_hold(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render_offset=0):
    box, confidence = bbox_untangle(start_frame, bbox_detector, render_offset=render_offset)
    print("BOX",box)
    if box is None:
        return None, None
    path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
    path_to_curr_img_depth = "images_depth/%06d_rgb.png" % (start_frame-render_offset)
    img = cv2.imread(path_to_curr_img)
    img_depth = cv2.imread(path_to_curr_img_depth)
    crop, rescale_factor, (x_off, y_off) = crop_and_resize(box, img)
    crop_depth, _, _ = crop_and_resize(box, img_depth)
    cv2.imwrite("images/%06d_crop.png" % (start_frame-render_offset), crop)
    cv2.imwrite("images_depth/%06d_crop.png" % (start_frame-render_offset), crop_depth)
    cv2.imwrite("./preds/%06d_bbox.png" % (start_frame-render_offset), crop)

    if not type(cf) == list:
        pull_crop_pixel, hold_crop_pixel = descriptor_matches(cf, path_to_ref_img, ref_crop_pixels, start_frame-render_offset, crop=True)
    else:
        # hold_crop_pixel = descriptor_matches(cf[1], path_to_ref_img, [ref_crop_pixels[1]], start_frame-render_offset, crop=True, depth=True)[0]
        hold_crop_pixel = descriptor_matches(cf[1], path_to_ref_img, [ref_crop_pixels[1]], start_frame-render_offset, crop=True)[0]
        holds = [hold_crop_pixel]
        pull_crop_pixel = descriptor_matches(cf[0], path_to_ref_img, [ref_crop_pixels[0]], start_frame-render_offset, crop=True, hold=holds)[0]

    # transform this pick and hold into overall space (scale and offset)
    pull_pixel, hold_pixel = pixel_crop_to_full(np.array([pull_crop_pixel, hold_crop_pixel]), rescale_factor, x_off, y_off)
    pull_pixel = (int(pull_pixel[0]), int(pull_pixel[1]))
    hold_pixel = (int(hold_pixel[0]), int(hold_pixel[1]))

    return pull_pixel, hold_pixel

def take_undo_action_descriptors(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render=False, render_offset=0, pixels=None):
    if pixels is None:
        pull_pixel, hold_pixel = find_pull_hold(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render_offset=render_offset)
    else:
        pull_pixel, hold_pixel = pixels
    if pull_pixel is None:
        return start_frame, None, None, None
    # calculate action vec
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
    return end_frame, pull_pixel, hold_pixel, action_vec

def take_undo_action_oracle(start_frame, render=False, render_offset=0):
    pull_idx, hold_idx, action_vec = find_knot(50)
    action_vec /= np.linalg.norm(action_vec)
    end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, render=render, render_offset=render_offset)
    pull_pixel, hold_pixel = cyl_to_pixels([pull_idx, hold_idx])
    return end_frame, pull_pixel[0], hold_pixel[0], action_vec


def run_untangling_rollout(params, crop_cf, ends_cf, path_to_ref_imgs, ref_pixels, bbox_predictor, chain=False, render=True, armature=0, policy=1, fig8=False):
    set_animation_settings(7000)
    piece = "Cylinder"
    last = params["num_segments"]-1
    undone=True

    index = 2
    if armature == 1:
        index = 6
    elif armature == 2:
        index = 10

    ref_end_pixels = ref_pixels[index:index+2]
    ref_crop_pixels = ref_pixels[index+2:index+4]

    if armature == 1:
        path_to_ref_full_img = [os.path.join(path_to_ref_img, 'armature_reid_ref.png')]
        path_to_ref_crop_img_d = os.path.join(path_to_ref_img, 'armature_crop_d_ref.png')
        path_to_ref_crop_img_rgb = os.path.join(path_to_ref_img, 'armature_crop_ref.png')
        path_to_ref_crop_img = [path_to_ref_crop_img_rgb, path_to_ref_crop_img_d]

    elif armature == 2:
        path_to_ref_full_img = [os.path.join(path_to_ref_img, 'braid_reid_ref.png')]
        path_to_ref_crop_img_d = os.path.join(path_to_ref_img, 'braid_crop_d_ref.png')
        path_to_ref_crop_img_rgb = os.path.join(path_to_ref_img, 'braid_crop_ref.png')
        path_to_ref_crop_img = [path_to_ref_crop_img_rgb, path_to_ref_crop_img_d]
    else:
        path_to_ref_full_img = [os.path.join(path_to_ref_img, 'reid_ref.png')]
        path_to_ref_crop_img_d = os.path.join(path_to_ref_img, 'crop_d_ref.png')
        path_to_ref_crop_img_rgb = os.path.join(path_to_ref_img, 'crop_ref.png')
        path_to_ref_crop_img = [path_to_ref_crop_img_rgb, path_to_ref_crop_img_d]

    if fig8:
        knot_end_frame = tie_figure_eight(params, render=False)
    else:
        knot_end_frame = tie_pretzel_knot(params, render=False)
    knot_end_frame = random_perturb(knot_end_frame, params)
    render_offset = knot_end_frame
    render_frame(knot_end_frame, render_offset=render_offset, step=1)

    # Policy = 0: oracle (ground truth)
    # Policy = 1: with descriptors
    # Policy = 2: (TODO) bounding box random action
    if policy==1:
        reid_end = reidemeister_descriptors(knot_end_frame, ends_cf, path_to_ref_full_img, ref_end_pixels, render=True, render_offset=render_offset)
    elif policy==0:
        reid_end = reidemeister_oracle(knot_end_frame, render=True, render_offset=render_offset)
    undo_end_frame = reid_end
    bbox, _ = bbox_untangle(undo_end_frame, bbox_predictor, render_offset=render_offset)
    while bbox is not None:
    # for _ in range(2):
        undone = False
        i = 0
        while not undone and i < 15:
        # while i < 2:
            try: # if the rope goes out of frame, take a reid move
                if policy==1:
                    undo_end_frame, pull, hold, action_vec = take_undo_action_descriptors(undo_end_frame, bbox_predictor, crop_cf, path_to_ref_crop_img, ref_crop_pixels, render=True, render_offset=render_offset)
                elif policy==0:
                    undo_end_frame, pull, hold, action_vec = take_undo_action_oracle(undo_end_frame, render=True, render_offset=render_offset)
                if pull is not None:
                    if policy==1:
                        undone = undone_check_descriptors(undo_end_frame, ends_cf, path_to_ref_full_img, ref_end_pixels, pull, hold, action_vec, render_offset=render_offset)
                    elif policy==0:
                        undone = undone_check_oracle(undo_end_frame, pull, hold, action_vec, render_offset=render_offset)
                else:
                    break
            except:
                if policy==1:
                    reid_end = reidemeister_descriptors(undo_end_frame, ends_cf, path_to_ref_full_img, ref_end_pixels, render=True, render_offset=render_offset)
                elif policy==0:
                    reid_end = reidemeister_oracle(undo_end_frame, render=True, render_offset=render_offset)
                undo_end_frame = reid_end
            i += 1
        if policy==1:
            reid_end = reidemeister_descriptors(undo_end_frame, ends_cf, path_to_ref_full_img, ref_end_pixels, render=True, render_offset=render_offset)
        elif policy==0:
            reid_end = reidemeister_oracle(undo_end_frame, render=True, render_offset=render_offset)
        undo_end_frame = reid_end
        bbox, _ = bbox_untangle(undo_end_frame, bbox_predictor, render_offset=render_offset)
        if undo_end_frame > 5000: # hard limit of ~25 actions
            return
    # undo_end_frame, pull, hold, action_vec = take_undo_action_descriptors(undo_end_frame, bbox_predictor, crop_cf, path_to_ref_crop_img, ref_crop_pixels, render=True, render_offset=render_offset)

def parse_annots(path_to_ref_img):
    with open(path_to_ref_img+'/ref_pixels.json', 'r') as f:
        ref_annots = json.load(f)
    pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
    hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
    left_end = [ref_annots["reid_left_x"], ref_annots["reid_left_y"]]
    right_end = [ref_annots["reid_right_x"], ref_annots["reid_right_y"]]
    crop_pull = [ref_annots["crop_pull_x"], ref_annots["crop_pull_y"]]
    crop_hold = [ref_annots["crop_hold_x"], ref_annots["crop_hold_y"]]
    armature_left_end = [ref_annots["armature_reid_left_x"], ref_annots["armature_reid_left_y"]]
    armature_right_end = [ref_annots["armature_reid_right_x"], ref_annots["armature_reid_right_y"]]
    armature_crop_pull = [ref_annots["armature_crop_pull_x"], ref_annots["armature_crop_pull_y"]]
    armature_crop_hold = [ref_annots["armature_crop_hold_x"], ref_annots["armature_crop_hold_y"]]
    braid_left_end = [ref_annots["braid_reid_left_x"], ref_annots["braid_reid_left_y"]]
    braid_right_end = [ref_annots["braid_reid_right_x"], ref_annots["braid_reid_right_y"]]
    braid_crop_pull = [ref_annots["braid_crop_pull_x"], ref_annots["braid_crop_pull_y"]]
    braid_crop_hold = [ref_annots["braid_crop_hold_x"], ref_annots["braid_crop_hold_y"]]
    ref_pixels = [pull, hold, left_end, right_end, crop_pull, crop_hold,
        armature_left_end, armature_right_end, armature_crop_pull, armature_crop_hold,
        braid_left_end, braid_right_end, braid_crop_pull, braid_crop_hold]

    return ref_pixels

def load_cf(base_dir, network_dir, crop=False):
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('dense_correspondence/cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    image_width = 80 if crop else 640
    image_height = 60 if crop else 480
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev, image_width=image_width, image_height=image_height)
    path_to_ref_img = "reference_images_bbox_crop"
    ref_pixels = parse_annots(path_to_ref_img)
    return cf, path_to_ref_img, ref_pixels

if __name__ == '__main__':
    if not os.path.exists("./preds"):
        os.makedirs('./preds')
    else:
        os.system('rm -r ./preds')
        os.makedirs('./preds')
    base_dir = 'dense_correspondence/networks'

    argv = sys.argv
    fig8 = "--fig8" in argv

    armature = 0 # 1 for chord, 2 for braid
    split_pull_hold = 1

    # Policy = 0: oracle (ground truth)
    # Policy = 1: with descriptors
    # Policy = 2: (TODO) bounding box random actions
    policy = 1

    network_dirs = {"chord": {"ends": 'armature_ends',
                            "local": 'armature_local_2knots',
                            "local_pull": "armature_local_2knots_2x",
                            "local_hold": "armature_local_2knots_2x_hold",
                            "bbox": "armature_1200"},
                    "braid": {"ends": 'braid_ends',
                            "local": 'braid_local_2knots',
                            "local_pull": "bbox_capsule_l_pull_rgb", # FIX THIS
                            "local_hold": "bbox_braid_2_hold_d",
                            "bbox": "braid_bbox_network"},
                    "capsule": {"ends": 'ends',
                            "local": "crop_capsule_offset1",
                            # "local_pull": "chord_local_uncenter_rgb_pull",
                            "local_pull": "bbox_capsule_l_pull_rgb",
                            "local_hold": "looser_bbox_cap_hold_rgb",
                            # "local_hold": "bbox_capsule_2_hold_d",
                            "bbox": "bbox_larger_mult"}}
                            # "bbox": "bbox_capsule_mult_looser"}}

    network_dir_dict = network_dirs["chord"] if armature == 1 else network_dirs["braid"]
    network_dir_dict = network_dirs["capsule"] if armature == 0 else network_dir_dict
    ends_network_dir = network_dir_dict["ends"]
    local_network_dir = network_dir_dict["local"]
    local_pull_network_dir = network_dir_dict["local_pull"]
    local_hold_network_dir = network_dir_dict["local_hold"]
    bbox_network_dir = network_dir_dict["bbox"]

    crop_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, local_network_dir, crop=True)
    try:
        local_pull_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, local_pull_network_dir, crop=True)
        local_hold_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, local_hold_network_dir, crop=True)
    except:
        pass
    ends_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, ends_network_dir)

    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = 'mrcnn_bbox/networks/{}/mask_rcnn_knot_cfg_0010.h5'.format(bbox_network_dir)
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    # make_rope(params)
    make_capsule_rope(params)
    if armature:
        rig_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    if split_pull_hold:
        crop_cf = [local_pull_cf, local_hold_cf]
    run_untangling_rollout(params, crop_cf, ends_cf, path_to_ref_img, ref_pixels, bbox_predictor, render=True, armature=armature, policy=policy, fig8=fig8)
