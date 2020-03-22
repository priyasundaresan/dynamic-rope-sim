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
    cylinder.rigid_body.angular_damping = params["angular_damping"]
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 20
    for i in range(num_segments-1):
        bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value":(-2*segment_radius, 0, 0)})
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.connect(con_type='POINT', connection_pattern='CHAIN_DISTANCE')
    bpy.ops.object.select_all(action='DESELECT')

def make_table(params):
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,-5))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 0.7
    bpy.ops.object.select_all(action='DESELECT')

def knot_test(params):
    # Resources: 
    # Dynamically animate/un-animate: https://blender.stackexchange.com/questions/130889/insert-keyframe-for-rigid-body-properties-for-object-python-script-blender
    # https://blenderartists.org/t/make-a-rigid-body-end-up-in-a-particular-position/634204/2

    # Makes a knot by a hardcoded trajectory
    # Press Spacebar once the Blender script loads to run the animation
    anim_end = 800
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

    end1 = bpy.data.objects['Cylinder']
    end2 = bpy.data.objects['Cylinder.%03d'%(params["num_segments"]-1)]

    # Allow endpoints to be keyframe-animated at the start
    end1.rigid_body.kinematic = True
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
    end1.rigid_body.kinematic = False
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

    end1.rigid_body.kinematic = True 
    end1.location = end1.matrix_world.translation # This line is critical - without it, the rope "snaps" back to starting position at frame 1 because its location is not up to date with how the simulation progressed after the drop; try uncommmenting to see what I mean
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=350)
    end1.keyframe_insert(data_path="location", frame=350) 
    end1.location[2] += 20
    end1.keyframe_insert(data_path="location", frame=500)

def coil_test(params):

    # Allow endpoints to be keyframe-animated
    end1 = bpy.data.objects['Cylinder']
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
    make_table(params)
    knot_test(params)
    #coil_test(params)
