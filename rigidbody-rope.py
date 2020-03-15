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
    cylinder.rigid_body.linear_damping = params["damping"]
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
    bpy.ops.object.select_all(action='DESELECT')

def action_test(params):
    # Makes a knot by a hardcoded trajectory
    # Press Spacebar once the Blender script loads to run the animation
    end1 = bpy.data.objects['Cylinder']
    end2 = bpy.data.objects['Cylinder.%03d'%(params["num_segments"]-1)]

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

    # Pull endpoint 1 up and through
    end1.location[0] += 5
    end1.location[2] += 2
    end1.keyframe_insert(data_path="location", frame=200)
    end1.location[0] += 5
    end1.keyframe_insert(data_path="location", frame=230)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    make_table(params)
    action_test(params)
