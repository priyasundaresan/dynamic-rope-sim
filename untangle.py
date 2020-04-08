import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from sklearn.neighbors import NearestNeighbors

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
        os.makedirs('./images')
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
        obj.keyframe_insert(data_path="location", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def find_knot(params, chain=False, thresh=0.4, pull_offset=3):

    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1
    cache = {}

    # Make a single pass, store the xy positions of the cylinders
    for i in range(last):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        key = tuple((x,y))
        val = {"idx":i, "depth":z}
        cache[key] = val
    neigh = NearestNeighbors(2, 0)
    planar_coords = list(cache.keys())
    neigh.fit(planar_coords)
    # Now traverse and look for the under crossing
    for i in range(last):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        match_idxs = neigh.kneighbors([(x,y)], 2, return_distance=False) # 1st neighbor is always identical, we want 2nd
        nearest = match_idxs.squeeze().tolist()[1:][0]
        x1,y1 = planar_coords[nearest]
        curr_cyl, match_cyl = cache[(x,y)], cache[(x1,y1)]
        depth_diff = match_cyl["depth"] - curr_cyl["depth"]
        if depth_diff > thresh:
            pull_idx = i + pull_offset # Pick a point slightly past under crossing to do the pull
            dx = planar_coords[pull_idx][0] - x
            dy = planar_coords[pull_idx][1] - y
            hold_idx = match_cyl["idx"]
            action_vec = [7*dx, 7*dy, 6] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
    return -1, last, [0,0,0] # Didn't find a pull/hold

def render_frame(frame, step=2, filename="%06d.png", folder="images"):
    # Renders a single frame in a sequence (if frame%step == 0)
    if frame%step == 0:
        scene = bpy.context.scene
        scene.render.filepath = os.path.join(folder, filename) % (frame//step)
        bpy.ops.render.render(write_still=True)

def knot_test(params, chain=False, render=False):
    set_animation_settings(800)
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1

    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)
    # Allow endpoints to be keyframe-animated at the start
    # Pin the two endpoints initially

    for i in range(last+1):
        obj = get_piece(piece, i if i != 0 else -1)
        if i == 0 or i == last:
            take_action(obj, 1, (0,0,0), animate=True)
        else:
            take_action(obj, 1, (0,0,0), animate=False)

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
    take_action(end1, 375, (5,0,0))
    for step in range(350, 375):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)
    take_action(end2, 400, (-5,0,0))

    # Drop the ends
    toggle_animation(end1, 400, False)
    toggle_animation(end2, 400, False)

    for step in range(375, 400):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)

    pick, hold, action_vec = find_knot(params)
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    ## Undoing
    take_action(hold_cyl, 500, (0,0,0))
    for step in range(400, 410):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)
    take_action(pull_cyl, 500, action_vec)

    ## Release both pull, hold
    toggle_animation(pull_cyl, 500, False)
    toggle_animation(hold_cyl, 500, False)

    # Let the rope settle after the action, so we can know where the ends are afterwards
    for step in range(410, 600):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)

    # Reidemeister #2
    take_action(end1, 625, (7,0,0))
    for step in range(600, 625):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)
    take_action(end2, 650, (-7,0,0))

    toggle_animation(end1, 650, False)
    toggle_animation(end2, 650, False)

    for step in range(625, 700):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    knot_test(params, render=True)
