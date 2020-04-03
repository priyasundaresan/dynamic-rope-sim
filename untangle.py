import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

def set_animation_settings(anim_end):
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def get_piece(piece_name, piece_id):
    if piece_id == -1:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def keyframe_kinematic(obj, frame, toggle_kinematic=False):
    if toggle_kinematic:
        obj.rigid_body.kinematic = not obj.rigid_body.kinematic
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def take_action(obj, frame, action_vec, toggle_kinematic=False, save_kinematic=False, save_location=False):
    curr_frame = bpy.context.scene.frame_current
    if save_location:
        obj.location = obj.matrix_world.translation
        obj.keyframe_insert(data_path="location", frame=curr_frame)
    dx,dy,dz = action_vec
    if save_kinematic:
        keyframe_kinematic(obj, frame if not save_location else curr_frame, toggle_kinematic=toggle_kinematic)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def knot_test(params, chain=False):
    set_animation_settings(800)
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1

    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)
    # Allow endpoints to be keyframe-animated at the start
    # Pin the two endpoints initially
    keyframe_kinematic(end1, 1, toggle_kinematic=True)
    keyframe_kinematic(end2, 1, toggle_kinematic=True)

    for i in range(last+1):
        obj = get_piece(piece, i if i != 0 else -1)
        take_action(obj, 1, (0,0,0), save_kinematic=True)

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
    keyframe_kinematic(end1, 240, toggle_kinematic=True)
    keyframe_kinematic(end2, 240, toggle_kinematic=True)

    # Reidemeister
    for step in range(1, 350):
        bpy.context.scene.frame_set(step)
    take_action(end1, 375, (5,0,0), toggle_kinematic=True, save_kinematic=True, save_location=True)
    for step in range(350, 375):
        bpy.context.scene.frame_set(step)
    take_action(end2, 400, (-5,0,0), toggle_kinematic=True, save_kinematic=True, save_location=True)

    keyframe_kinematic(end1, 400, toggle_kinematic=True)
    keyframe_kinematic(end2, 400, toggle_kinematic=True)

    for step in range(375, 400):
        bpy.context.scene.frame_set(step)

    #pick, hold = find_knot(params, chain=chain)
    pick, hold = 16, 26
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    # Undoing
    take_action(hold_cyl, 500, (0,0,0), toggle_kinematic=True, save_kinematic=True, save_location=True)
    for step in range(400, 410):
        bpy.context.scene.frame_set(step)
    take_action(pull_cyl, 500, (-6,-2,3), toggle_kinematic=True, save_kinematic=True, save_location=True)

    # Release both pull, hold
    keyframe_kinematic(pull_cyl, 500, toggle_kinematic=True)
    keyframe_kinematic(hold_cyl, 500, toggle_kinematic=True)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    make_table(params)
    knot_test(params)
