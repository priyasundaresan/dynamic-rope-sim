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
    #bpy.ops.mesh.subdivide(number_cuts=1) # Tune this number for cloth detail
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=1, quadcorner='INNERVERT')
    bpy.ops.object.editmode_toggle()
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

    return [bpy.data.objects['Cylinder.%03d' % (i) if i>0 else "Cylinder"] for i in range(num_segments)]

def make_rope_v2(params):
    segment_radius = params["segment_radius"]
    num_segments = params["num_segments"]
    bend_stiffness = 0.1
    bend_damping = 0.1
    twist_stiffness = 0 #1.0
    twist_damping = 0.5

    stretch_limit = segment_radius * 0.1
    stretch_stiffness = 100
    stretch_damping = 10.0

    links = []

    # Create all the links of the rope
    for i in range(num_segments):
        # See: https://blender.stackexchange.com/questions/26890/how-can-i-make-a-pill-shape-capsule
        if hasattr(bpy.ops.mesh, 'primitive_round_cube_add'):
            bpy.ops.mesh.primitive_round_cube_add(
                radius=segment_radius-0.01, arc_div=12,
                size=(0, 0, segment_radius*4),
                location=(segment_radius*num_segments - 2*i*segment_radius,0,0),
                rotation=(0, np.pi/2, 0))
        else:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=segment_radius, depth=segment_radius*2,
                location=(segment_radius*num_segments - 2*i*segment_radius,0,0),
                rotation=(0, np.pi/2, 0))
        
        cylinder = bpy.context.object
        cylinder.name = ('Cylinder.%03d' % (i)) if i > 0 else 'Cylinder'
        bpy.ops.rigidbody.object_add()
        cylinder.rigid_body.mass = params["segment_mass"]
        cylinder.rigid_body.friction = params["segment_friction"]
        #cylinder.rigid_body.linear_damping = params["linear_damping"]
        #cylinder.rigid_body.angular_damping = params["angular_damping"] # NOTE: this makes the rope a lot less wiggly
        links.append(cylinder)

    # Create spring connections between the links
    for i in range(1,num_segments):
        bpy.ops.object.empty_add(type='ARROWS', radius=1, location=(segment_radius*(num_segments + 1 - 2*i), 0, 0))
        bpy.ops.rigidbody.constraint_add()
        joint = bpy.context.object
        joint.name = 'joint_' + str(i-1) + ':' + str(i)
        joint.rigid_body_constraint.type = 'GENERIC_SPRING'
        joint.rigid_body_constraint.object1 = links[i-1]
        joint.rigid_body_constraint.object2 = links[i]

        # limit the allowed linear translation:
        joint.rigid_body_constraint.use_limit_lin_x = True
        joint.rigid_body_constraint.use_limit_lin_y = True
        joint.rigid_body_constraint.use_limit_lin_z = True
        joint.rigid_body_constraint.limit_lin_x_lower = -stretch_limit
        joint.rigid_body_constraint.limit_lin_x_upper = 0
        joint.rigid_body_constraint.limit_lin_y_lower = 0
        joint.rigid_body_constraint.limit_lin_y_upper = 0
        joint.rigid_body_constraint.limit_lin_z_lower = 0
        joint.rigid_body_constraint.limit_lin_z_upper = 0

        # Make the rope stretchy
        joint.rigid_body_constraint.use_spring_x = stretch_limit > 0 and stretch_stiffness > 0
        joint.rigid_body_constraint.spring_stiffness_x = stretch_stiffness
        joint.rigid_body_constraint.spring_damping_x = stretch_damping

        # set spring constraints on rotation.
        # rotations about y and z control how the rope bends
        # rotations about x control how the rope twists
        joint.rigid_body_constraint.use_spring_ang_x = twist_stiffness > 0
        joint.rigid_body_constraint.use_spring_ang_y = bend_stiffness > 0
        joint.rigid_body_constraint.use_spring_ang_z = bend_stiffness > 0
        joint.rigid_body_constraint.spring_stiffness_ang_x = twist_stiffness
        joint.rigid_body_constraint.spring_damping_ang_x = twist_damping
        joint.rigid_body_constraint.spring_stiffness_ang_y = bend_stiffness
        joint.rigid_body_constraint.spring_damping_ang_y = bend_damping
        joint.rigid_body_constraint.spring_stiffness_ang_z = bend_stiffness
        joint.rigid_body_constraint.spring_damping_ang_z = bend_damping

    bpy.context.scene.rigidbody_world.steps_per_second = 500 # note: lower values (e.g., 120) cause an explosion + bus error
    bpy.context.scene.rigidbody_world.solver_iterations = 1000

    return links

def make_rope_v3(params):
    # This method relies on an STL file that contains a mesh for a
    # capsule.  The capsule cannot be non-unformly scaled without
    # distorting the end caps.  So instead we compute the rope_length
    # based on the param's radius and num_segments, and compute the
    # number of segments composed of the capsules that we need
    radius = params["segment_radius"]
    rope_length = radius * params["num_segments"] * 2 * 0.9 # HACKY -- shortening the rope artificially by 10% for now
    num_segments = int(rope_length / radius)
    separation = radius*1.1 # HACKY - artificially increase the separation to avoid link-to-link collision
    link_mass = params["segment_mass"] # TODO: this may need to be scaled
    link_friction = params["segment_friction"]

    # Parameters for how much the rope resists twisting
    twist_stiffness = 20
    twist_damping = 10

    # Parameters for how much the rope resists bending
    bend_stiffness = 0
    bend_damping = 5

    num_joints = int(radius/separation)*2+1
    loc0 = rope_length/2

    # Create the first link from the STL. In the filename: 12 = number
    # of radial subdivisions, 8 = number of length-wise subdivisions,
    # 1 = radius, 2 = height..
    bpy.ops.import_mesh.stl(filepath="capsule_12_8_1_2.stl")
    link0 = bpy.context.object
    link0.name = "link_0"
    bpy.ops.transform.resize(value=(radius, radius, radius))
    # The link has Z-axis up, and the origin is in its center.
    link0.rotation_euler = (0, pi/2, 0)
    link0.location = (loc0, 0, 0)
    bpy.ops.rigidbody.object_add()
    link0.rigid_body.mass = link_mass
    link0.rigid_body.friction = link_friction
    # switch the collision_shape to CAPSULE--this is both faster and
    # more accurate (for an actual capsule) than the default.
    link0.rigid_body.collision_shape = 'CAPSULE'
    
    links = [link0]
    for i in range(1,num_segments):
        # copy link0 to create each additional link
        linki = link0.copy()
        linki.data = link0.data.copy()
        linki.name = "link_" + str(i)
        linki.location = (loc0 - i*separation, 0, 0)
        bpy.context.collection.objects.link(linki)

        links.append(linki)

        # Create a GENERIC_SPRING connecting this link to the previous.
        bpy.ops.object.empty_add(type='ARROWS', radius=radius*2, location=(loc0 - (i-0.5)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-1) + ':' + str(i)
        rbc = joint.rigid_body_constraint
        # connect the two links
        rbc.object1 = links[i-1]
        rbc.object2 = links[i]
        # disable translation from the joint.  Note: we can consider
        # making a "stretchy" rope by setting
        # limit_lin_x_{lower,upper} to a non-zero range.
        rbc.use_limit_lin_x = True
        rbc.use_limit_lin_y = True
        rbc.use_limit_lin_z = True
        rbc.limit_lin_x_lower = 0
        rbc.limit_lin_x_upper = 0
        rbc.limit_lin_y_lower = 0
        rbc.limit_lin_y_upper = 0
        rbc.limit_lin_z_lower = 0
        rbc.limit_lin_z_upper = 0
        if twist_stiffness > 0 or twist_damping > 0:
            rbc.use_spring_ang_x = True
            rbc.spring_stiffness_ang_x = twist_stiffness
            rbc.spring_damping_ang_x = twist_damping
        if bend_stiffness > 0 or bend_damping > 0:
            rbc.use_spring_ang_y = True
            rbc.use_spring_ang_z = True
            rbc.spring_stiffness_ang_y = bend_stiffness
            rbc.spring_stiffness_ang_z = bend_stiffness
            rbc.spring_damping_ang_y = bend_damping
            rbc.spring_damping_ang_z = bend_damping

    # After creating the rope, we connect every link to the link 1
    # separated by a joint that has no constraints.  This prevents
    # collision detection between the pairs of rope points.
    for i in range(2, num_segments):
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=radius*1.5, location=(loc0 - (i-1)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-2) + ':' + str(i)
        joint.rigid_body_constraint.object1 = links[i-2]
        joint.rigid_body_constraint.object2 = links[i]

    # the following parmaeters seem sufficient and fast for using this
    # rope.  steps_per_second can probably be lowered more to gain a
    # little speed.
    bpy.context.scene.rigidbody_world.steps_per_second = 1000
    bpy.context.scene.rigidbody_world.solver_iterations = 100
        
    return links

    
def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    #bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0), rotation=(36*np.pi/180, -65*np.pi/180, 18*np.pi/180))
    bpy.ops.object.camera_add(location=(2,0,28), rotation=(0,0,0))
    #bpy.ops.object.camera_add(location=(11,-33,7.5), rotation=(radians(80), 0, radians(16.5)))
    bpy.context.scene.camera = bpy.context.object

def make_table(params):
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,-5))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    #table.rigid_body.friction = 0.7
    table.rigid_body.friction = 0.8
    bpy.ops.object.select_all(action='DESELECT')

def tie_knot_with_fixture(end, fixture):
    end.rigid_body.kinematic = True
    end.keyframe_insert(data_path="location", frame=1)
    end.keyframe_insert(data_path="rotation_euler", frame=1)

    # Create a sequence of [ frame, angle, location ] for the end
    # we're moving to tie the knot.
    key_frames = [
        [ 20, -160, (-12, -1.5, -2) ],
        [ 80,  -160, (0, -1.5, -3.0) ],
        [ 110, -180, (4, -0.75, -3) ],
        [ 120, -270, (4, 0, -3) ],
        [ 130, -360, (4, 1, -3) ],
        [ 180, -450, (0, 1, -3) ],
        [ 200, -360, (0, -1, -3) ],
        [ 250, -360, (0, -3, -3) ],
        [ 330, -360, (0, -3, 14) ] ]

    # Apply the sequence as a set of keyframes to the end of the rope
    # we're pulling.
    for fno, rot, loc in key_frames:
        bpy.context.scene.frame_current = fno
        end.location = loc
        end.rotation_euler = (rot*pi/180, 90*pi/180, 0)
        end.keyframe_insert(data_path="location", frame=fno)
        end.keyframe_insert(data_path="rotation_euler", frame=fno)

    # move the fixture out of the way at frames 250 to 270
    bpy.context.scene.frame_current = 250
    fixture.keyframe_insert(data_path="location", frame=250)
    fixture.keyframe_insert(data_path="rotation_euler", frame=250)
    bpy.context.scene.frame_current = 270
    fixture.location = (0, 4, -1.5)
    fixture.rotation_euler = (radians(30), 0, 0)
    fixture.keyframe_insert(data_path="location", frame=270)
    fixture.keyframe_insert(data_path="rotation_euler", frame=270)

    # reset blender to show the first frame
    bpy.context.scene.frame_current = 1
    
if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    links = make_rope_v3(params)
    make_table(params)

    # create a fixture for tying the knot
    bpy.ops.mesh.primitive_cube_add(location=(0,0,-1.5))
    bpy.ops.transform.resize(value=(1,2,0.25))
    bpy.ops.rigidbody.object_add()
    fixture = bpy.context.object
    fixture.rigid_body.friction = 0.5
    fixture.rigid_body.kinematic = True

    # Extend the number for frames in the animation and the rigid-body
    # simulation.
    frame_end = 500
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    # A little hacky... add extra weight to the last segment to have
    # it tighten the knot more when the opposite end is lifted
    links[0].rigid_body.mass *= 100
    tie_knot_with_fixture(links[-1], fixture)

    add_camera_light()
