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
from random import sample
import bmesh

'''Usage: blender -b -P cloth-blender.py'''

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

def make_table():
    # Generate table surface
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    return bpy.context.object

def add_hook(obj, vertex_index):
    obj.data.vertices[vertex_index].select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.object.hook_add_newob()
    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj.data.vertices[vertex_index].select = False
    hook = bpy.context.object
    return hook

def make_rope(params):
    radius = params["radius"]
    location = (params["location_x"], params["location_y"], params["location_z"])
    bpy.ops.curve.primitive_nurbs_path_add(radius=radius, location=location)
    bpy.ops.object.convert(target='MESH')
    rope = bpy.context.object
    pinned_group = bpy.context.object.vertex_groups.new(name='Pinned')
    e1, e2 = 0, len(rope.data.vertices)-1
    ends = (e1, e2)
    pinned_group.add(ends, 1.0, 'ADD')
    hook1 = add_hook(rope, e1)
    hook2 = add_hook(rope, e2)
    bpy.ops.object.modifier_add(type='SOFT_BODY')
    softbody = rope.modifiers["Softbody"].settings
    softbody.goal_spring = params["stiffness"]
    softbody.goal_default = params["goal_strength"]
    softbody.vertex_group_goal = "Pinned"
    softbody.use_self_collision = True
    softbody.ball_size = params["ball_size"]
    softbody.goal_friction = params["damping"]
    bpy.ops.object.modifier_add(type='SKIN')
    bpy.ops.object.modifier_add(type='SUBSURF')
    rope.modifiers["Subdivision"].levels = params["subdiv_viewport"]
    for v in rope.data.skin_vertices[0].data:
        v.radius = params["vertex_radius"], params["vertex_radius"]
    return rope, hook1, hook2


if __name__ == '__main__':
    clear_scene()
    make_table()
    #make_rope(1, (0,0,0.75))
    with open("rope_params.json", "r") as f:
        rope_params = json.load(f)
    rope, hook1, hook2 = make_rope(rope_params)
