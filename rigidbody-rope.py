import bpy
import os
import json
import time
import sys
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
import random
from random import sample
from sklearn.cluster import KMeans
import bmesh

'''Usage: blender -P rigidbody-rope.py'''

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def make_rope(params):
    segment_radius = params["segment_radius"]
    num_segments = params["num_segments"]
    bpy.ops.mesh.primitive_cylinder_add(location=(segment_radius*num_segments,0,0))
    bpy.ops.transform.resize(value=(segment_radius, segment_radius, segment_radius))
    cylinder = bpy.context.object
    cylinder.rotation_euler = (0, np.pi/2, 0)
    bpy.ops.rigidbody.object_add()
    cylinder.rigid_body.mass = params["segment_mass"]
    cylinder.rigid_body.friction = params["segment_friction"]
    cylinder.rigid_body.linear_damping = params["linear_damping"]
    cylinder.rigid_body.angular_damping = params["angular_damping"] # NOTE: this makes the rope a lot less wiggly
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 20
    for i in range(num_segments-1):
        bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value":(-2*segment_radius, 0, 0)})
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.connect(con_type='POINT', connection_pattern='CHAIN_DISTANCE')
    bpy.ops.object.select_all(action='DESELECT')

def make_chain(params):
    '''
    Join multiple toruses together to make chain shape (http://jayanam.com/chains-with-blender-tutorial/)
    TODOS: fix torus vertex selection to make oval shape
           fix ARRAY selection after making empty mesh
    '''
    # hacky fix from https://www.reddit.com/r/blenderhelp/comments/dnb56f/rendering_python_script_in_background/
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.screen.screen_full_area(override)
                break
    scale = params["chain_scale"]
    chain_len = params["chain_len"] # chain length doubles for every 1 of chain_len
    center_start = (1.8*scale*(2**(chain_len+1))/2) + 2
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), major_radius=1*scale, minor_radius=0.25*scale, abso_major_rad=1.25*scale, abso_minor_rad=0.75*scale)
    # bpy.ops.object.editmode_toggle()
    link = bpy.context.active_object
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    verts_to_move = [0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287]

    for i in verts_to_move:
        link.data.vertices[i].select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    # bpy.ops.transform.translate(value=(0, 1, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.transform.translate(value=(0*scale, 1*scale, 0*scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.editmode_toggle()

    first_link = bpy.context.active_object
    link_len = 1.9 * scale

    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, link_len, 0), "orient_type":'LOCAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'LOCAL', "constraint_axis":(False, True, False), "mirror":True, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":False, "use_accurate":False})
    bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

    for i in range(chain_len):
        bpy.ops.object.select_all(action='SELECT')
        move_len = 2**i * link_len * 2
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, move_len, 0), "orient_type":'LOCAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'LOCAL', "constraint_axis":(False, True, False), "mirror":True, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":False, "use_accurate":False})

    bpy.ops.object.select_all(action='SELECT')
    # first_link.select_set(True)
    # bpy.context.scene.context = 'PHYSICS'
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.object_settings_copy()

    bpy.ops.transform.translate(value=(0, -center_start, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 1000

    #
    # bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0*scale, 1.8*scale, 0*scale))
    # bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    #
    # bpy.context.view_layer.objects.active = self.rope
    # bpy.ops.object.modifier_add(type='ARRAY')
    # bpy.context.object.modifiers["Array"].use_relative_offset = False
    # bpy.context.object.modifiers["Array"].use_object_offset = True
    # bpy.context.object.modifiers["Array"].offset_object = bpy.data.objects["Empty"]
    # self.rope_asymm.modifiers["Array"].count = 100
    # bpy.ops.object.modifier_add(type='CURVE')
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.join()

def make_table(params):
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,-5))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 0.7
    bpy.ops.object.select_all(action='DESELECT')

def chain_knot_test(params):
    # Makes a knot by a hardcoded trajectory
    # Press Spacebar once the Blender script loads to run the animation
    end2 = bpy.data.objects['Torus']
    end1 = bpy.data.objects['Torus.%03d'%((2**(params["chain_len"]+1))-1)]

    # Allow endpoints to be keyframe-animated
    end1.rigid_body.enabled = False
    end1.rigid_body.kinematic = True
    end2.rigid_body.enabled = False
    end2.rigid_body.kinematic = True

    # Pin the two endpoints initially
    end1.keyframe_insert(data_path="location", frame=1)
    end2.keyframe_insert(data_path="location", frame=1)

    # Wrap endpoint one circularly around endpoint 2
    end2.location[0] += 10
    end1.location[0] -= 15
    end1.location[1] += 5
    end1.keyframe_insert(data_path="location", frame=80)
    end2.keyframe_insert(data_path="location", frame=80)
    end1.location[0] -= 1
    end1.location[1] -= 7
    end1.keyframe_insert(data_path="location", frame=120)
    end1.location[0] += 3
    end1.location[2] -= 4
    end1.keyframe_insert(data_path="location", frame=150)
    end1.location[1] += 2.5
    end1.keyframe_insert(data_path="location", frame=170)

    # Thread endpoint 1 through the loop (downward)
    end1.location[2] -= 2
    end1.keyframe_insert(data_path="location", frame=180)

    # Pull to tighten knot
    end1.location[0] += 5
    end1.location[2] += 2
    end1.keyframe_insert(data_path="location", frame=200)
    end2.keyframe_insert(data_path="location", frame=200)

    end1.location[0] += 7
    end1.location[2] += 5
    end2.location[0] -= 7
    end1.keyframe_insert(data_path="location", frame=230)
    end2.keyframe_insert(data_path="location", frame=230)

def find_knot(params, chain=False):
    # this returns a pick and hold point for a rope with a single knot
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1

    depths = {}
    for i in range(last):
        piece_name = '%s.%03d'%(piece, i) if i != 0 else piece
        depths[str(i)] = bpy.data.objects[piece_name].matrix_world.translation[2]

    occlusions = []
    for d in depths.keys():
        if depths[d] > np.mean([h for h in depths.values()]):
            occlusions.append(int(d))

    clus = KMeans(n_clusters=2)
    clus.fit(np.array(occlusions).reshape((-1, 1)))
    labels = clus.labels_

    pick_and_hold = []
    for l in np.unique(labels):
        l_pieces = [occlusions[i] for i in range(len(labels)) if labels[i] == l]
        l_depths = [depths[str(i)] for i in l_pieces]
        for piece_index in l_pieces:
            if depths[str(piece_index)] == np.max(l_depths):
                pick_and_hold.append(piece_index)
    hold = max(pick_and_hold)
    pick = min(pick_and_hold)
    return pick, hold

def knot_test(params, chain=False):
    # Resources:
    # Dynamically animate/un-animate: https://blender.stackexchange.com/questions/130889/insert-keyframe-for-rigid-body-properties-for-object-python-script-blender
    # https://blenderartists.org/t/make-a-rigid-body-end-up-in-a-particular-position/634204/2

    # Makes a knot by a hardcoded trajectory
    # Press Spacebar once the Blender script loads to run the animation
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1
    anim_end = 800
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

    end1 = bpy.data.objects[piece]
    end2 = bpy.data.objects['%s.%03d'%(piece, last)]

    for i in range(last):
        if i != 0:
            bpy.data.objects['%s.%03d'%(piece, i)].keyframe_insert(data_path="rigid_body.kinematic")

    # Set up cylinders that will be used to undo a crossing
    # pull_cyl = bpy.data.objects['%s.015'%(piece)]
    # # pull_cyl = bpy.data.objects['%s.%03d'%(piece, pick)]
    # pull_cyl.rigid_body.kinematic = False
    # pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # pull_cyl.keyframe_insert(data_path="location", frame=1)
    #
    # hold_cyl = bpy.data.objects['%s.026'%(piece)]
    # # hold_cyl = bpy.data.objects['%s.%03d'%(piece, hold)]
    # hold_cyl.rigid_body.kinematic = False
    # hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # hold_cyl.keyframe_insert(data_path="location", frame=1)

    # Allow endpoints to be keyframe-animated at the start
    end1.rigid_body.kinematic = True # This means end1 is manually animable
    end2.rigid_body.kinematic = True
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=1)

    # Pin the two endpoints initially
    end1.keyframe_insert(data_path="location", frame=1)
    end2.keyframe_insert(data_path="location", frame=1)

    # Wrap endpoint one circularly around endpoint 2
    end2.location[0] += 10
    end1.location[0] -= 15
    end1.location[1] += 5
    end1.keyframe_insert(data_path="location", frame=80)
    end2.keyframe_insert(data_path="location", frame=80)
    end1.location[0] -= 1
    end1.location[1] -= 7
    end1.keyframe_insert(data_path="location", frame=120)
    end1.location[0] += 3
    end1.location[2] -= 4
    end1.keyframe_insert(data_path="location", frame=150)
    end1.location[1] += 2.5
    end1.keyframe_insert(data_path="location", frame=170)

    # Thread endpoint 1 through the loop (downward)
    end1.location[2] -= 2
    end1.keyframe_insert(data_path="location", frame=180)

    # Pull to tighten knot
    end1.location[0] += 5
    end1.location[2] += 2
    end1.keyframe_insert(data_path="location", frame=200)
    end2.keyframe_insert(data_path="location", frame=200)

    end1.location[2] += 5
    end1.location[0] += 7
    end2.location[0] -= 7
    end1.keyframe_insert(data_path="location", frame=230)
    end2.keyframe_insert(data_path="location", frame=230)

    # Now, we "drop" the rope; no longer animated and will move only based on rigid body physics
    end1.rigid_body.kinematic = False # This means end1 is not animable, will move based on physics solver
    end2.rigid_body.kinematic = False
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=240)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=240)

    # Now, I want to try re-picking up the end of the rope after it's been dropped
    # Turns out this is a little tricky because we need to know where end1 ended up after the drop happened
    # Otherwise, when you just translate end1 and keyframe it, it thinks its still at its pre-drop position
    # so end1 first quickly jumps from its post-drop position to pre-drop pose and then does the motion
    # This is the "snapping" effect where it quickly jumps to an outdated location (like the softbody!!)
    # We really just want it to move from its post-drop position to the translation

    # Workaround: I pick  a frame when I think the rope has settled, get the updated location of end1, and
    # set it back to kinematic (controllable by animation)

    # Note: I step through the animation up to frame 350 cuz that's about when the rope settles after being dropped
    # I had to use end1.matrix_world.translation to get the updated world coordinate of end1 as the sim progresses
    # because end1.location does NOT give the up-to-date location taking into account physics

    for step in range(1, 351):
        bpy.context.scene.frame_set(step)
        #print(end1.matrix_world.translation) # does update properly :)
        #print(end1.location) # does NOT update properly

    # Reidemeister
    end1.rigid_body.kinematic = True
    # Next line is critical - without it, the rope "snaps" back to starting position at frame 1 because its location is not up to date with how the simulation progressed after the drop; try uncommmenting to see what I mean
    end1.location = end1.matrix_world.translation
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=350)
    end1.keyframe_insert(data_path="location", frame=350)
    end1.location[0] = 9
    end1.keyframe_insert(data_path="location", frame=375)

    end2.rigid_body.kinematic = True
    end2.location = end2.matrix_world.translation
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=375)
    end2.keyframe_insert(data_path="location", frame=375)
    end2.location[0] = -9
    end2.keyframe_insert(data_path="location", frame=400)

    end1.rigid_body.kinematic = False
    end2.rigid_body.kinematic = False
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=400)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=400)

    for step in range(351, 401):
        bpy.context.scene.frame_set(step)

    pick, hold = find_knot(params, chain=chain)
    # pull_cyl = bpy.data.objects['%s.015'%(piece)]
    pull_cyl = bpy.data.objects['%s.%03d'%(piece, pick)]
    pull_cyl.rigid_body.kinematic = False
    # pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # pull_cyl.keyframe_insert(data_path="location", frame=1)

    # hold_cyl = bpy.data.objects['%s.026'%(piece)]
    hold_cyl = bpy.data.objects['%s.%03d'%(piece, hold)]
    hold_cyl.rigid_body.kinematic = False
    # hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # hold_cyl.keyframe_insert(data_path="location", frame=1)


    # Undoing
    hold_cyl.rigid_body.kinematic = True
    hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=400)
    hold_cyl.location = hold_cyl.matrix_world.translation
    # We want the hold cylinder to stay in place during the pull
    hold_cyl.keyframe_insert(data_path="location", frame=400)
    hold_cyl.keyframe_insert(data_path="location", frame=500)

    # Doing this because if I try to keyframe the location of hold and pull at frame 400, it causes "snapping"
    for step in range(401, 410):
        bpy.context.scene.frame_set(step)

    pull_cyl.rigid_body.kinematic = True
    pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=410)
    pull_cyl.location = pull_cyl.matrix_world.translation
    pull_cyl.keyframe_insert(data_path="location", frame=410)
    # Pull
    pull_cyl.location[2] += 3
    pull_cyl.location[0] -= 6
    pull_cyl.location[1] -= 2
    pull_cyl.keyframe_insert(data_path="location", frame=500)

    # Release both pull, hold
    pull_cyl.rigid_body.kinematic = False
    pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=500)
    hold_cyl.rigid_body.kinematic = False
    hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=500)


def coil_test(params, chain=False):

    # Allow endpoints to be keyframe-animated
    if not chain:
        end1 = bpy.data.objects['Cylinder']
    else:
        end1 = bpy.data.objects['Torus']
    end1.rigid_body.enabled = False
    end1.rigid_body.kinematic = True
    end1.keyframe_insert(data_path="location", frame=1)

    anim_start = 120
    anim_end = 600
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

    # Helix equation:
    # x = rcos(t), y = rsin(t), z = ct
    r = 1.15
    c = -0.75
    start_height = 19
    t0 = 0
    tn = 10*np.pi
    timesteps = 50
    for t in np.linspace(t0, tn, timesteps):
        x = r*np.cos(t)
        y = r*np.sin(t)
        z = c*t + start_height
        end1.location = x,y,z
        end1.keyframe_insert(data_path="location", frame=anim_start+(t+1)*(float(scene.frame_end-anim_start)/timesteps))

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    # make_chain(params)
    make_table(params)
    # chain_knot_test(params)
    knot_test(params)
    # knot_test(params, chain=True)
    # coil_test(params)
    # coil_test(params, chain=True)
