import bpy
from math import *
import mathutils
from mathutils import *
import numpy as np

import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

# def clear_scene():
#     '''Clear existing objects in scene'''
#     for block in bpy.data.meshes:
#         if block.users == 0:
#             bpy.data.meshes.remove(block)
#     for block in bpy.data.materials:
#         if block.users == 0:
#             bpy.data.materials.remove(block)
#     for block in bpy.data.textures:
#         if block.users == 0:
#             bpy.data.textures.remove(block)
#     for block in bpy.data.images:
#         if block.users == 0:
#             bpy.data.images.remove(block)
#     bpy.ops.object.mode_set(mode='OBJECT')
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete()

# def set_animation_settings(anim_end):
#     # Sets up the animation to run till frame anim_end (otherwise default terminates @ 250)
#     scene = bpy.context.scene
#     scene.frame_end = anim_end
#     scene.rigidbody_world.point_cache.frame_end = anim_end

def get_piece(piece_name, piece_id):
    # Returns the piece with name piece_name, index piece_id
    # if piece_id == -1:
    if piece_id == -1 or piece_id == 0:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def import_mesh(filepath, axis, scale=1):
    bpy.ops.import_mesh.stl(filepath=filepath, axis_up=axis)
    bpy.ops.rigidbody.object_add()
    c = bpy.context.object
    c.rigid_body.type = 'PASSIVE'
    if scale != 1:
        bpy.ops.transform.resize(value=(scale, scale, scale))
        bpy.ops.object.transform_apply(scale=True)
    return c

def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    loc = obj.matrix_world.translation
    obj.rigid_body.kinematic = animate
    obj.location = loc
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)
    obj.keyframe_insert(data_path="location", frame=frame)

def add_camera_light():
	bpy.ops.object.light_add(type='SUN', radius=1, location=(26,-26,20))
	bpy.ops.object.camera_add(location=(25,-25,20), rotation=(radians(60),0,radians(40)))
	bpy.context.scene.camera = bpy.context.object


class YumiGripper():

	def parent_no_transform(self, parent_obj, child_obj):
		child_obj.parent = parent_obj
		bpy.context.view_layer.update()
		child_obj.matrix_world = parent_obj.matrix_world.inverted() @ child_obj.matrix_world

	def add_child(self, parent, child, name):
		const = child.constraints.new(type='CHILD_OF')
		const.name = name
		const.target = parent
		return const

	def keyframe_gripper(self, frame):
		self.gripper_base.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.claw1.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.claw2.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.gripper_base.keyframe_insert(data_path="location", frame=frame)
		self.claw1.keyframe_insert(data_path="location", frame=frame)
		self.claw2.keyframe_insert(data_path="location", frame=frame)

	def __init__(self, gripper_base, finger1, finger2):

		#Get mesh for gripper and place in propper position
		self.gripper_base = gripper_base
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.gripper_base.location = (0,0,4)
		self.gripper_base.rotation_euler = (0,0,radians(90))
		#self.gripper_base.rigid_body.kinematic = True


		self.claw1 = finger1
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.claw1.location = (0.09,0.38,2.7)
		self.claw1.rotation_euler = (0, 0, radians(90))
		#self.claw1.rigid_body.kinematic = True

		self.claw2 = finger2
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.claw2.location = (-0.07,-0.38,2.7)
		self.claw2.rotation_euler = (0, 0, radians(90))
		#self.claw2.rigid_body.kinematic = True
		###########

		self.parent_no_transform(self.gripper_base, self.claw1)
		self.parent_no_transform(self.gripper_base, self.claw2)

		bpy.context.view_layer.objects.active = self.gripper_base

	def hold(self,cylinder, end_frame):
		self.pick_place(cylinder, end_frame, (0,0,0), pick_height=0)

	def pick_place(self, cylinder, end_frame, move_vector, pick_height=2):

		start_frame = bpy.context.scene.frame_current

		### GET CYLINDER
		bpy.context.view_layer.update()
		toggle_animation(cylinder, start_frame, True)
		cylinder.keyframe_insert(data_path = "location", frame=start_frame)

		cyl_rot_z = cylinder.matrix_world.to_euler().z
		pick_coord = cylinder.matrix_world.translation + Vector((0,0,1.75))
		pick_x, pick_y, pick_z = pick_coord

		drop_coord = pick_coord + Vector(move_vector)
		drop_x, drop_y, drop_z = drop_coord
		
		close_amount = (0.2,0,0)
		self.gripper_base.rotation_euler = (0, 0, cyl_rot_z)
		self.gripper_base.location = Vector((pick_x, pick_y, pick_z + 6))
		self.keyframe_gripper(start_frame)

		### INITIAL CHILDING BEGIN
		con = self.add_child(self.gripper_base, cylinder, str(cylinder))
		con.influence = 0.0
		con.keyframe_insert(data_path="influence", frame=start_frame)
		### INITIAL CHILDING END

		frames = end_frame - start_frame

		#MOVE TO ABOVE ROPE
		pick_up_frame = int(0.16*frames) + start_frame
		self.gripper_base.location = Vector((pick_x, pick_y, pick_z))
		self.keyframe_gripper(pick_up_frame)
		###########

		########## INFLUENCE HIGH CHILD OF 
		con.influence = 0.0
		con.keyframe_insert(data_path="influence", frame=pick_up_frame+1)
		cylinder.keyframe_insert(data_path="rotation_euler", frame=pick_up_frame+1)
		cylinder.keyframe_insert(data_path="location", frame=pick_up_frame+1)

		bpy.context.view_layer.update()
		cylinder.matrix_world = self.gripper_base.matrix_world.inverted() @ cylinder.matrix_world
		cylinder.keyframe_insert(data_path="rotation_euler", frame=pick_up_frame+2)
		cylinder.keyframe_insert(data_path="location", frame=pick_up_frame+2)

		con.influence = 1.0
		con.keyframe_insert(data_path="influence", frame=pick_up_frame+2)
		########## INFLUENCE HIGH CHILD OF

		#CLAMP DOWN
		clamp_frame = int(0.24*frames) + start_frame
		self.claw1.location = self.claw1.location - Vector(close_amount)
		self.claw2.location = self.claw2.location + Vector(close_amount)
		self.keyframe_gripper(clamp_frame)
		###########

		#LIFT, MOVE, and PLACE
		lift_frame = int(0.32*frames) + start_frame
		move_frame = int(0.6*frames) + start_frame
		place_frame = int(0.8*frames) + start_frame


		self.gripper_base.location = Vector((pick_x, pick_y, pick_z + pick_height))
		self.keyframe_gripper(lift_frame)

		self.gripper_base.location = Vector((drop_x, drop_y, drop_z + pick_height))
		self.keyframe_gripper(move_frame)

		self.gripper_base.location = Vector((drop_x, drop_y, drop_z))
		self.keyframe_gripper(place_frame)
		############

		#OPEN CLAMP
		release_frame = int(0.92*frames) + start_frame
		self.claw1.location = self.claw1.location + Vector(close_amount)
		self.claw2.location = self.claw2.location - Vector(close_amount)
		self.keyframe_gripper(release_frame)
		############# 

		########## INFLUENCE LOW CHILD OF 
		cylinder.keyframe_insert(data_path="location", frame=release_frame-1)
		cylinder.keyframe_insert(data_path="rotation_euler", frame=release_frame-1)

		con.influence = 1.0
		con.keyframe_insert(data_path="influence", frame=release_frame-1)

		bpy.context.view_layer.update()
		mat = cylinder.matrix_world

		con.influence = 0.0
		con.keyframe_insert(data_path="influence", frame=release_frame)

		cylinder.matrix_world = mat
		cylinder.keyframe_insert(data_path="rotation_euler", frame=release_frame)
		cylinder.keyframe_insert(data_path = "location", frame=release_frame)
		########## INFLUENCE LOW CHILD OF

		self.gripper_base.location = Vector((drop_x, drop_y, drop_z + 6))
		self.gripper_base.keyframe_insert(data_path="location", frame=end_frame)

		toggle_animation(cylinder, end_frame, False)
