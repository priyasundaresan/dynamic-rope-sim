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

def set_animation_settings(anim_end):
    # Sets up the animation to run till frame anim_end (otherwise default terminates @ 250)
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def set_render_settings(engine, render_size):
    # Set rendering engine, dimensions, colorspace, images settings
    if not os.path.exists("./images"):
        os.makedirs('./images')
    else:
        os.system('rm -r ./images')
    scene = bpy.context.scene
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    if engine == 'BLENDER_WORKBENCH':
        scene.render.display_mode
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_render_samples = 1

def get_piece(piece_name, piece_id):
    # Returns the piece with name piece_name, index piece_id
    if piece_id == 0 or piece_id == -1:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def take_action(obj, frame, action_vec, animate=True):
    # Keyframes a displacement for obj given by action_vec at given frame
    curr_frame = bpy.context.scene.frame_current
    dx,dy,dz = action_vec
    if animate != obj.rigid_body.kinematic:
        # We are "picking up" a dropped object, so we need its updated location
        obj.location = obj.matrix_world.translation
        obj.rotation_euler = obj.matrix_world.to_euler()
        obj.keyframe_insert(data_path="location", frame=curr_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def find_knot(num_segments, chain=False, depth_thresh=0.4, idx_thresh=3, pull_offset=3):
    piece = "Torus" if chain else "Cylinder"
    cache = {}

    # Make a single pass, store the xy positions of the cylinders
    for i in range(num_segments):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        key = tuple((x,y))
        val = {"idx":i, "depth":z}
        cache[key] = val
    neigh = NearestNeighbors(2, 0)
    planar_coords = list(cache.keys())
    neigh.fit(planar_coords)
    # Now traverse and look for the under crossing
    for i in range(num_segments):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        match_idxs = neigh.kneighbors([(x,y)], 2, return_distance=False) # 1st neighbor is always identical, we want 2nd
        nearest = match_idxs.squeeze().tolist()[1:][0]
        x1,y1 = planar_coords[nearest]
        curr_cyl, match_cyl = cache[(x,y)], cache[(x1,y1)]
        depth_diff = match_cyl["depth"] - curr_cyl["depth"]
        idx_diff = abs(match_cyl["idx"] - curr_cyl["idx"])
        if depth_diff > depth_thresh and idx_diff > idx_thresh:
            pull_idx = i + pull_offset # Pick a point slightly past under crossing to do the pull
            dx = planar_coords[pull_idx][0] - x
            dy = planar_coords[pull_idx][1] - y
            hold_idx = match_cyl["idx"]
            action_vec = [7*dx, 7*dy, 6] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
    return -1, last, [0,0,0] # Didn't find a pull/hold

def render_frame(frame, render_offset=0, step=2, num_annotations=400, filename="%06d_rgb.png", folder="images"):
    # Renders a single frame in a sequence (if frame%step == 0)
    frame -= render_offset
    if frame%step == 0:
        scene = bpy.context.scene
        index = frame//step
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)


def tie_knot(params, chain=False, render=False):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)
    for i in range(last+1):
        obj = get_piece(piece, i if i != 0 else -1)
        take_action(obj, 1, (0,0,0), animate=(i==0 or i==last))

    # Wrap endpoint one circularly around endpoint 2
    take_action(end2, 80, (10,0,0))
    take_action(end1, 80, (-15,5,0))
    take_action(end1, 120, (-1,-7,0))
    take_action(end1, 150, (3,0,-4))
    take_action(end1, 170, (0,2.5,0))

    # Thread endpoint 1 through the loop (downward)
    take_action(end1, 180, (0,0,-2))

    # Pull to tighten knot
    take_action(end1, 200, (5,0,2))
    take_action(end2, 200, (0,0,0))
    take_action(end1, 230, (7,0,5))
    take_action(end2, 230, (-7,0,0))

    # Now, we "drop" the rope; no longer animated and will move only based on rigid body physics
    #toggle_animation(end1, 240, False)
    #toggle_animation(end2, 240, False)

    take_action(end1, 260, (-1,0,-1))
    take_action(end2, 260, (1,0,-1))
    toggle_animation(end1, 280, False)
    toggle_animation(end2, 280, False)

    ## Reidemeister
    for step in range(1, 350):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)
    return 350

def pixels_to_cylinders(pixels):
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    render_size = (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            )
    cyl_pixels = []
    indices = list(range(50))
    for i in indices:
        cyl = get_piece("Cylinder", i if i != 0 else -1)
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, cyl.matrix_world.translation)
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        cyl_pixels.append(pixel)
    neigh = NearestNeighbors(1, 0)
    neigh.fit(cyl_pixels)
    match_idxs = neigh.kneighbors(pixels, 1, return_distance=False) # 1st neighbor is always identical, we want 2nd
    return match_idxs.squeeze()
    #nearest = match_idxs.squeeze().tolist()[1:][0]
    #return nearest

def descriptor_matches(cf, path_to_ref_img, pixels, curr_frame, crop=False):
    path_to_curr_img = "images/%06d_crop.png" % curr_frame if crop else "images/%06d_rgb.png" % curr_frame
    cf.load_image_pair(path_to_ref_img, path_to_curr_img)
    cf.compute_descriptors()
    best_matches, _ = cf.find_k_best_matches(pixels, 50, mode="median")
    cf.show_side_by_side()
    # return pixels_to_cylinders(best_matches)
    return best_matches

def reidemeister_descriptors(start_frame, cf, path_to_ref_img, ref_end_pixels, render=False, render_offset=0):
    piece = "Cylinder"
    # end2_idx, end1_idx = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, start_frame-render_offset)
    end2_pixel, end1_pixel = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, start_frame-render_offset)
    end2_idx = pixels_to_cylinders([end2_pixel])
    end1_idx = pixels_to_cylinders([end1_pixel])
    end2 = get_piece(piece, end2_idx)

    middle_frame = start_frame+50
    end_frame = middle_frame+50
    take_action(end2, middle_frame, (-8-end2.matrix_world.translation[0],0,0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)

    # end2_idx, end1_idx = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, middle_frame-1-render_offset)
    end2_pixel, end1_pixel = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, start_frame-render_offset)
    end2_idx = pixels_to_cylinders([end2_pixel])
    end1_idx = pixels_to_cylinders([end1_pixel])

    end1 = get_piece(piece, end1_idx)
    toggle_animation(end2, middle_frame, False)
    take_action(end1, end_frame, (10-end1.matrix_world.translation[0],0,0))

    # Drop the ends
    #toggle_animation(end1, end_frame, False)
    toggle_animation(end1, end_frame, False)

    settle_time = 50
    for step in range(middle_frame, end_frame+settle_time):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)
    return end_frame

def bbox_untangle(start_frame, bbox_detector, render_offset=0):
    path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
    curr_img = imageio.imread(path_to_curr_img)
    boxes = bbox_predictor.predict(curr_img)
    return boxes[0] # ASSUME first box is knot to be untied

def undone_check(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, hold_pos, render_offset=0, thresh=4):
    pull_pixel, hold_pixel = find_pull_hold(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render_offset=render_offset)
    if np.linalg.norm(hold_pos - hold_pixel) < thresh: # threshold for when a crossing is undone
        return False
    return True

def crop_and_resize(box, img, aspect=(320,240)):
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    box_width = y_max - y_min
    box_height = x_max - x_min

    # resize this crop to be 320x240
    new_width = int((box_height*aspect[1])/aspect[0])
    offset = new_width - box_width
    x_min -= int(offset/2)
    x_max += offset - int(offset/2)

    crop = img[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, aspect)
    return resized

def pixel_crop_to_full(box_pixel, box, aspect=(320,240)):
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    box_width = y_max - y_min
    box_height = x_max - x_min

    # resize this crop to be 320x240
    new_width = int((box_height*aspect[1])/aspect[0])
    offset = new_width - box_width
    x_min -= int(offset/2)
    x_max += offset - int(offset/2)

    box_pixel_x, box_pixel_y = box_pixel
    full_pixel_x = int(box_pixel_x/aspect[0] * new_width) + x_min
    full_pixel_y = int(box_pixel_y/aspect[1] * box_height) + y_min
    return [full_pixel_x, full_pixel_y]

def find_pull_hold(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render_offset=0):
    box, confidence = bbox_untangle(start_frame, bbox_detector, render_offset=render_offset)
    path_to_curr_img = "images/%06d_rgb.png" % (start_frame-render_offset)
    img = cv2.imread(path_to_curr_img)
    # scale box and save to "images/%06d_crop.png" % curr_frame
    crop = crop_and_resize(box, img)
    cv2.imwrite("images/%06d_crop.png" % (start_frame-render_offset), crop)

    pull_crop_pixel, hold_crop_pixel = descriptor_matches(cf, path_to_ref_img, ref_crop_pixels, start_frame-render_offset, crop=True)
    # transform this pick and hold into overall space (scale and offset)
    pull_pixel = pixel_crop_to_full(pull_crop_pixel, box)
    hold_pixel = pixel_crop_to_full(hold_crop_pixel, box)

    return pull_pixel, hold_pixel

def take_undo_action_descriptors(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render=False, render_offset=0):
    piece = "Cylinder"

    pull_pixel, hold_pixel = find_pull_hold(start_frame, bbox_detector, cf, path_to_ref_img, ref_crop_pixels, render_offset=render_offset)
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
    cv2.imshow("action", img)
    cv2.waitKey(0)

    hold_idx = pixels_to_cylinders([hold_pixel])
    pull_idx = pixels_to_cylinders([pull_pixel])
    pull_cyl = get_piece(piece, pull_idx)
    hold_cyl = get_piece(piece, hold_idx)

    ## Undoing
    take_action(hold_cyl, start_frame + 100, (0,0,0))
    for step in range(start_frame, start_frame+10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)
    take_action(pull_cyl, start_frame + 100, action_vec)

    ## Release both pull, hold
    toggle_animation(pull_cyl, start_frame + 100, False)
    toggle_animation(hold_cyl, start_frame + 100, False)

    # Let the rope settle after the action, so we can know where the ends are afterwards
    for step in range(start_frame + 10, start_frame + 200):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)

    return start_frame+200, pull_pixel, hold_pixel, action_vec

def random_loosen(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1

    pick, hold, _ = find_knot(params["num_segments"])
    if random.random() < 0.5:
        pick = random.choice(range(10, 40))
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    dx = np.random.uniform(0,1)*random.choice((-1,1))
    dy = np.random.uniform(0,1)*random.choice((-1,1))
    dz = np.random.uniform(0.75,1.75)

    mid_frame = start_frame + 50
    end_frame = start_frame + 100

    take_action(hold_cyl, mid_frame, (0,0,0))
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    toggle_animation(hold_cyl, mid_frame, False)
    for step in range(start_frame + 10, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)
    return end_frame

def run_untangling_rollout(params, full_cf, crop_cf, path_to_ref_imgs, ref_pixels, bbox_predictor, chain=False, render=True):
    set_animation_settings(7000)
    piece = "Cylinder"
    last = params["num_segments"]-1

    ref_knot_pixels = ref_pixels[:2]
    ref_end_pixels = ref_pixels[2:4]
    ref_crop_pixels = ref_pixels[4:]

    path_to_ref_full_img = os.path.join(path_to_ref_img, 'reid_ref.png')
    path_to_ref_crop_img = os.path.join(path_to_ref_img, 'crop_ref.png')

    #render_offset = 350 # length of a knot action
    knot_end_frame = tie_knot(params, render=False)
    render_offset = knot_end_frame
    render_frame(knot_end_frame, render_offset=render_offset, step=1)
    reid_end = reidemeister_descriptors(knot_end_frame, full_cf, path_to_ref_full_img, ref_end_pixels, render=True, render_offset=render_offset)

    # take undo actions
    undo_end_frame, _, hold, _ = take_undo_action_descriptors(reid_end, bbox_predictor, crop_cf, path_to_ref_crop_img, ref_crop_pixels, render=True, render_offset=render_offset)
    while not undone_check(undo_end_frame, bbox_predictor, crop_cf, path_to_ref_crop_img, ref_crop_pixels, hold, render_offset=render_offset):
        undo_end_frame, _, hold, _ = take_undo_action_descriptors(undo_end_frame, bbox_predictor, crop_cf, path_to_ref_crop_img, ref_crop_pixels, render=True, render_offset=render_offset)

def load_cf(base_dir, network_dir, crop=False):
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('dense_correspondence/cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    image_width = 320 if crop else 640
    image_height = 240 if crop else 480
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev, image_width=image_width, image_height=image_height)
    path_to_ref_img = "reference_images"
    with open('reference_images/ref_pixels.json', 'r') as f:
        ref_annots = json.load(f)
    pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
    hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
    left_end = [ref_annots["reid_left_x"], ref_annots["reid_left_y"]]
    right_end = [ref_annots["reid_right_x"], ref_annots["reid_right_y"]]
    crop_pull = [ref_annots["crop_pull_x"], ref_annots["crop_pull_y"]]
    crop_hold = [ref_annots["crop_hold_x"], ref_annots["crop_hold_y"]]
    ref_pixels = [pull, hold, left_end, right_end, crop_pull, crop_hold]
    return cf, path_to_ref_img, ref_pixels

if __name__ == '__main__':
    base_dir = 'dense_correspondence/networks'
    network_dir = 'full_length_corr'
    full_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, network_dir)

    network_dir = 'crop_s2_blur7-10_unzoomed'
    crop_cf, path_to_ref_img, ref_pixels = load_cf(base_dir, network_dir, crop=True)

    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = 'mrcnn_bbox/networks/knot_network_1000/mask_rcnn_knot_cfg_0007.h5'
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    run_untangling_rollout(params, full_cf, crop_cf, path_to_ref_img, ref_pixels, bbox_predictor, render=True)
