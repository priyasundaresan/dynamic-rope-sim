import bpy
import numpy as np
from math import pi
import json
import cv2
import numpy as np
import copy
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import os
import sys

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/tools"))

from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder
from image_utils import * 

from rigidbody_rope import *

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
    if not os.path.exists("./images_depth"):
        os.makedirs('./images_depth')
    else:
        os.system('rm -r ./images_depth')
        os.makedirs('./images_depth')
    if not os.path.exists("./image_masks"):
        os.makedirs('./image_masks')
    else:
        os.system('rm -r ./image_masks')
        os.makedirs('./image_masks')
    scene = bpy.context.scene
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    #scene.view_settings.exposure = 0.8
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
    if piece_id == -1:
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
        render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)

def render_mask(mask_filename, depth_filename, index):
    # NOTE: this method is still in progress
    scene = bpy.context.scene
    saved = scene.render.engine
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1
    scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]
    norm_node = tree.nodes.new(type="CompositorNodeNormalize")
    inv_node = tree.nodes.new(type="CompositorNodeInvert")
    math_node = tree.nodes.new(type="CompositorNodeMath")
    math_node.operation = 'CEIL' # Threshold the depth image
    composite = tree.nodes.new(type = "CompositorNodeComposite")

    links.new(render_node.outputs["Depth"], inv_node.inputs["Color"])
    links.new(inv_node.outputs[0], norm_node.inputs[0])
    links.new(norm_node.outputs[0], composite.inputs["Image"])

    scene.render.filepath = depth_filename % index
    bpy.ops.render.render(write_still=True)

    links.new(norm_node.outputs[0], math_node.inputs[0])
    links.new(math_node.outputs[0], composite.inputs["Image"])

    scene.render.filepath = mask_filename % index
    bpy.ops.render.render(write_still=True)
    # Clean up 
    scene.render.engine = saved
    for node in tree.nodes:
        if node.name != "Render Layers":
            tree.nodes.remove(node)
    scene.use_nodes = False

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
    toggle_animation(end1, 240, False)
    toggle_animation(end2, 240, False)

    ## Reidemeister
    for step in range(1, 350):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)
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

def descriptor_matches(cf, path_to_ref_img, pixels, curr_frame):
    path_to_curr_img = "images/%06d_rgb.png" % curr_frame
    cf.load_image_pair(path_to_ref_img, path_to_curr_img)
    cf.compute_descriptors()
    best_matches, _ = cf.find_k_best_matches(pixels, 50, mode="median")
    cf.show_side_by_side()
    return pixels_to_cylinders(best_matches)

def reidemeister_descriptors(start_frame, cf, path_to_ref_img, ref_end_pixels, render=False, render_offset=0):
    piece = "Cylinder"
    end2_idx, end1_idx = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, start_frame)
    end2 = get_piece(piece, end2_idx)

    middle_frame = start_frame+25
    end_frame = start_frame+50
    take_action(end2, middle_frame, (-9-end2.matrix_world.translation[0],0,0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)

    end2_idx, end1_idx = descriptor_matches(cf, path_to_ref_img, ref_end_pixels, middle_frame-1)
    end1 = get_piece(piece, end1_idx)
    take_action(end1, end_frame, (11-end1.matrix_world.translation[0],0,0))

    # Drop the ends
    toggle_animation(end1, end_frame, False)
    toggle_animation(end2, end_frame, False)

    for step in range(middle_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, step=1)
    return end_frame

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

def run_untangling_rollout(params, cf, path_to_ref_img, ref_pixels, chain=False, render=True):
    set_animation_settings(7000)
    piece = "Cylinder"
    last = params["num_segments"]-1

    ref_knot_pixels = ref_pixels[:2]
    ref_end_pixels = ref_pixels[2:]

    knot_end_frame = tie_knot(params, render=False)
    render_frame(knot_end_frame, step=1)
    reidemeister_descriptors(knot_end_frame, cf, path_to_ref_img, ref_end_pixels, render=True)

if __name__ == '__main__':
    base_dir = 'dense_correspondence/networks'
    network_dir = 'full_length_corr'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('dense_correspondence/cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    path_to_ref_img = "reference_images/reid_ref.png"
    with open('reference_images/ref_pixels.json', 'r') as f:
        ref_annots = json.load(f)
        pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
        hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
        left_end = [ref_annots["reid_left_x"], ref_annots["reid_left_y"]]
        right_end = [ref_annots["reid_right_x"], ref_annots["reid_right_y"]]
        ref_pixels = [pull, hold, left_end, right_end]

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    run_untangling_rollout(params, cf, path_to_ref_img, ref_pixels, render=True)
