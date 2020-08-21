import bpy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import os
#import imageio
import cv2
sys.path.append(os.getcwd())
from knots import *
from render import *
#from image_utils import *
from rigidbody_rope import *
from knots import *

def render_frame(frame, render_offset=0, step=2, num_annotations=400, filename="%06d_rgb.png", folder="images"):
    # Renders a single frame in a sequence (if frame%step == 0)
    frame -= render_offset
    if frame%step == 0:
        scene = bpy.context.scene
        index = frame//step
        render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)

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
    #match_idxs = neigh.kneighbors(pixels, 1, return_distance=False) # 1st neighbor is always identical, we want 2nd
    print('PIXELS:', pixels)
    two_match_idxs = neigh.kneighbors(pixels, 2, return_distance=False)
    idx1, idx2 = two_match_idxs.squeeze()
    cyl_1, cyl_2 = get_piece("Cylinder", idx1), get_piece("Cylinder", idx2)
    match = idx1
    pixel_dist = np.linalg.norm(np.array(cyl_pixels[idx1]) - np.array(cyl_pixels[idx2]))
    thresh = 10
    print("pixeldist", pixel_dist)
    if pixel_dist < thresh:
        if cyl_2.matrix_world.translation[2] > cyl_1.matrix_world.translation[2]:
            match = idx2
    return match

def cyl_to_pixels(cyl_indices):
    pixels = []
    scene = bpy.context.scene
    render_size = (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            )
    for i in cyl_indices:
        cyl = get_piece("Cylinder", i if i != 0 else -1)
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, cyl.matrix_world @ cyl.location)
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        pixels.append([pixel])
    return pixels

def crop_and_resize(box, img, aspect=(80,60)):
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    box_width = x_max - x_min
    box_height = y_max - y_min

    # resize this crop to be 320x240
    new_width = int((box_height*aspect[0])/aspect[1])
    offset = new_width - box_width
    x_min -= int(offset/2)
    x_max += offset - int(offset/2)

    crop = img[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, aspect)
    rescale_factor = new_width/aspect[0]
    offset = (x_min, y_min)
    return resized, rescale_factor, offset

def pixel_crop_to_full(pixels, crop_rescale_factor, x_offset, y_offset):
    global_pixels = pixels * crop_rescale_factor
    global_pixels[:, 0] += x_offset
    global_pixels[:, 1] += y_offset
    return global_pixels

def random_perturb(start_frame, params, render=False, render_offset=0):
    piece = "Cylinder"

    pick, hold, _ = find_knot(params["num_segments"])
    if random.random() < 0.5:
        pick = random.choice(range(10, 40))
    pull_cyl = get_piece(piece, pick)

    dx = np.random.uniform(0,0.3)*random.choice((-1,1))
    dy = np.random.uniform(0,0.3)*random.choice((-1,1))
    dz = np.random.uniform(0.5,1)

    mid_frame = start_frame + 30
    end_frame = start_frame + 60

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    for step in range(start_frame, end_frame+1):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)
    return end_frame

def take_undo_action(start_frame, pull_idx, hold_idx, norm_action_vec, render=False, render_offset=0, scale_action=True, hold_flag=True):
    piece = "Cylinder"
    action_vec = norm_action_vec*3 if scale_action else norm_action_vec
    pull_cyl = get_piece(piece, pull_idx)
    hold_cyl = get_piece(piece, hold_idx)

    ## Undoing
    if abs(hold_idx - pull_idx) > 5 and hold_flag:
        take_action(hold_cyl, start_frame + 100, (0,0,0))
    for step in range(start_frame, start_frame+10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)
    take_action(pull_cyl, start_frame + 100, action_vec)

    ## Release both pull, hold
    toggle_animation(pull_cyl, start_frame + 100, False)
    toggle_animation(hold_cyl, start_frame + 100, False)

    settle_time = 10
    # Let the rope settle after the action, so we can know where the ends are afterwards
    for step in range(start_frame + 10, start_frame + 200 + settle_time+1):
        bpy.context.scene.frame_set(step)
        if render:
            # render_frame(step, render_offset=render_offset, step=1)
            render_frame(step, render_offset=render_offset, step=4)

    return start_frame+200+settle_time, action_vec

def reidemeister_right(start_frame, end1_idx, end2_idx, render=False, render_offset=0):
    piece = "Cylinder"
    end1 = get_piece(piece, end1_idx)
    end2 = get_piece(piece, end2_idx)
    middle_up_frame = start_frame+50
    middle_move_frame = middle_up_frame+70
    end_frame = middle_move_frame+50
    take_action(end1, middle_up_frame, (0,0,2))
    take_action(end1, middle_move_frame, (17-end1.matrix_world.translation[0],0-end1.matrix_world.translation[1],0))
    take_action(end1, end_frame, (0,0,-2))
    for step in range(start_frame, end_frame+1):
        bpy.context.scene.frame_set(step)
        if render:
            # render_frame(step, render_offset=render_offset, step=1)
            render_frame(step, render_offset=render_offset, step=4)
    toggle_animation(end1, end_frame, False)
    return end_frame

def reidemeister_left(start_frame, end1_idx, end2_idx, render=False, render_offset=0):
    piece = "Cylinder"
    end_frame = start_frame + 70
    end2 = get_piece(piece, end2_idx)
    take_action(end2, end_frame, (-8-end2.matrix_world.translation[0],0-end2.matrix_world.translation[1],0))
    # Drop the ends
    toggle_animation(end2, end_frame, False)

    settle_time = 10
    for step in range(start_frame, end_frame+settle_time+1):
        bpy.context.scene.frame_set(step)
        if render:
            # render_frame(step, render_offset=render_offset, step=1)
            render_frame(step, render_offset=render_offset, step=4)
    return end_frame+settle_time

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
