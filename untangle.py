import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from sklearn.neighbors import NearestNeighbors

def set_animation_settings(anim_end):
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def get_piece(piece_name, piece_id):
    if piece_id == -1:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def toggle_animation(obj, frame, animate):
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def take_action(obj, frame, action_vec, animate=True):
    curr_frame = bpy.context.scene.frame_current
    dx,dy,dz = action_vec
    if animate != obj.rigid_body.kinematic:
        obj.location = obj.matrix_world.translation
        obj.keyframe_insert(data_path="location", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def find_knot(params, chain=False, thresh=0.4, pull_offset=3):

    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1
    cache = {}

    for i in range(last):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        key = tuple((x,y))
        val = {"idx":i, "depth":z}
        cache[key] = val
    neigh = NearestNeighbors(2, 0)
    planar_coords = list(cache.keys())
    neigh.fit(planar_coords)
    for i in range(last):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        match_idxs = neigh.kneighbors([(x,y)], 2, return_distance=False) # 1st neighbor is always identical, we want 2nd
        nearest = match_idxs.squeeze().tolist()[1:][0]
        curr_cyl, match_cyl = cache[(x,y)], cache[planar_coords[nearest]]
        depth_diff = match_cyl["depth"] - curr_cyl["depth"]
        if depth_diff > thresh:
            pull_idx = i + pull_offset
            hold_idx = match_cyl["idx"]
            return pull_idx, hold_idx
    return -1,last

def knot_test(params, chain=False):
    set_animation_settings(700)
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
    take_action(end1, 375, (5,0,0))
    for step in range(350, 375):
        bpy.context.scene.frame_set(step)
    take_action(end2, 400, (-5,0,0))

    toggle_animation(end1, 400, False)
    toggle_animation(end2, 400, False)

    for step in range(375, 400):
        bpy.context.scene.frame_set(step)

    #find_knot(params)
    #pick, hold = 16, 26
    pick, hold = find_knot(params)
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    ## Undoing
    take_action(hold_cyl, 500, (0,0,0))
    for step in range(400, 410):
        bpy.context.scene.frame_set(step)
    take_action(pull_cyl, 500, (-6,-2,3))


    ### Release both pull, hold
    toggle_animation(pull_cyl, 500, False)
    toggle_animation(hold_cyl, 500, False)

    for step in range(410, 600):
        bpy.context.scene.frame_set(step)
    take_action(end1, 625, (7,0,0))
    for step in range(600, 625):
        bpy.context.scene.frame_set(step)
    take_action(end2, 650, (-7,0,0))

    toggle_animation(end1, 650, False)
    toggle_animation(end2, 650, False)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    make_table(params)
    knot_test(params)
