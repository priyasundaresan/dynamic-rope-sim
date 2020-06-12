import bpy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import os

sys.path.append(os.getcwd())

from knots import *
from render import *
from image_utils import *
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
    for step in range(start_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)
    return end_frame
