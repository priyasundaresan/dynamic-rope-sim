import bpy
from math import *
import mathutils
from mathutils import *
import numpy as np

import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

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

def set_animation_settings(anim_end):
    # Sets up the animation to run till frame anim_end (otherwise default terminates @ 250)
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def get_piece(piece_name, piece_id):
    # Returns the piece with name piece_name, index piece_id
    # if piece_id == -1:
    if piece_id == -1 or piece_id == 0:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def render_frame(frame, step=2, filename="%06d.png", folder="images"):
	if frame%step == 0:
		scene = bpy.context.scene
		scene.render.filepath = os.path.join(folder, filename) % (frame//step)
		bpy.ops.render.render(write_still=True)

def import_mesh(filepath, axis, scale=1):
    bpy.ops.import_mesh.stl(filepath=filepath, axis_up=axis)
    bpy.ops.rigidbody.object_add()
    c = bpy.context.object
    # c.rigid_body.type = 'PASSIVE'
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
		self.gripper_base_armature.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.claw1_armature.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.claw2_armature.keyframe_insert(data_path="rotation_euler", frame=frame)
		self.gripper_base_armature.keyframe_insert(data_path="location", frame=frame)
		self.claw1_armature.keyframe_insert(data_path="location", frame=frame)
		self.claw2_armature.keyframe_insert(data_path="location", frame=frame)

	def __init__(self, gripper_base, finger1, finger2):
		

		#Get mesh for gripper and place in propper position
		self.gripper_base = gripper_base
		# self.gripper_base = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/BASE.STL", "-Z", scale = 0.02)
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.gripper_base.location = (0,0,4)
		self.gripper_base.rotation_euler = (0,0,radians(90))
		self.gripper_base.rigid_body.kinematic = True


		self.claw1 = finger1
		# self.claw1 = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/Finger1.STL", "-Z", scale = 0.02)
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.claw1.location = (0.09,0.38,2.7)
		self.claw1.rotation_euler = (0, 0, radians(90))
		self.claw1.rigid_body.kinematic = True

		self.claw2 = finger2
		# self.claw2 = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/Finger2.STL", "-Z", scale = 0.02)
		bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
		self.claw2.location = (0,-0.38,2.7)
		self.claw2.rotation_euler = (0, 0, radians(90))
		self.claw2.rigid_body.kinematic = True
		###########

		#Make armature portion for gripper 
		bpy.ops.object.armature_add(location=self.gripper_base.location, rotation=self.gripper_base.rotation_euler)
		self.gripper_base_armature = bpy.context.object

		bpy.ops.object.armature_add(location=self.claw1.location, rotation=(0,3.14159,0))
		self.claw1_armature = bpy.context.object
		self.parent_no_transform(self.gripper_base_armature, self.claw1_armature)

		bpy.ops.object.armature_add(location=self.claw2.location, rotation=(0,3.14159,0))
		self.claw2_armature = bpy.context.object
		self.parent_no_transform(self.gripper_base_armature, self.claw2_armature)
		###########

		#Rig armature to meshes 
		self.parent_no_transform(self.gripper_base_armature, self.gripper_base)
		self.parent_no_transform(self.claw1_armature, self.claw1)
		self.parent_no_transform(self.claw2_armature, self.claw2)
		###########

		self.keyframe_gripper(0)

		bpy.context.view_layer.objects.active = self.gripper_base

	def hold(self,cylinder, end_frame):
		self.pick_place(cylinder, end_frame, (0,0,0), pick_height=0)

	def pick_place(self, cylinder, end_frame, move_vector, pick_height=2):

		start_frame = bpy.context.scene.frame_current

		start_height = 6

		### GET CYLINDER
		bpy.context.view_layer.update()
		toggle_animation(cylinder, start_frame, True)

		cylinder.keyframe_insert(data_path="rotation_euler", frame=start_frame)
		cylinder.keyframe_insert(data_path="location", frame=start_frame)

		cyl_rot_z = cylinder.matrix_world.to_euler().z
		cyl_tilt = cylinder.matrix_world.to_euler().x
		pick_coord = cylinder.matrix_world.translation + Vector((0,0,1.75))
		pick_x, pick_y, pick_z = pick_coord

		drop_coord = pick_coord + Vector(move_vector)
		drop_x, drop_y, drop_z = drop_coord
		
		close_amount = (0.15,0,0)
		self.gripper_base_armature.rotation_euler = (cyl_tilt, 0, cyl_rot_z)
		self.gripper_base_armature.location = Vector((pick_x, pick_y, pick_z + start_height))
		self.keyframe_gripper(start_frame)

		### INITIAL CHILDING BEGIN
		bpy.context.view_layer.update()

		con = self.add_child(self.gripper_base_armature, cylinder, str(cylinder))
		con.influence = 0.0
		con.keyframe_insert(data_path="influence", frame=start_frame)
		### INITIAL CHILDING END

		frames = end_frame - start_frame

		#MOVE TO ABOVE ROPE
		pick_up_frame = int(0.16*frames) + start_frame
		self.gripper_base_armature.location = Vector((pick_x, pick_y, pick_z))
		self.keyframe_gripper(pick_up_frame)
		###########
		
		########## INFLUENCE HIGH CHILD OF 
		con.influence = 0.0
		con.keyframe_insert(data_path="influence", frame=pick_up_frame+1)
		bpy.context.view_layer.update()
		cylinder.keyframe_insert(data_path="rotation_euler", frame=pick_up_frame+1)
		cylinder.keyframe_insert(data_path="location", frame=pick_up_frame+1)

		bpy.context.view_layer.update()
		cylinder.matrix_world = self.gripper_base_armature.matrix_world.inverted() @ cylinder.matrix_world
		cylinder.keyframe_insert(data_path="rotation_euler", frame=pick_up_frame+2)
		cylinder.keyframe_insert(data_path="location", frame=pick_up_frame+2)

		con.influence = 1.0
		con.keyframe_insert(data_path="influence", frame=pick_up_frame+2)
		########## INFLUENCE HIGH CHILD OF

		#CLAMP DOWN
		clamp_frame = int(0.24*frames) + start_frame
		self.claw1_armature.location = self.claw1_armature.location - Vector(close_amount)
		self.claw2_armature.location = self.claw2_armature.location + Vector(close_amount)
		self.keyframe_gripper(clamp_frame)
		###########

		#LIFT, MOVE, and PLACE
		lift_frame = int(0.32*frames) + start_frame
		move_frame = int(0.6*frames) + start_frame
		place_frame = int(0.8*frames) + start_frame

		self.gripper_base_armature.location = Vector((pick_x, pick_y, pick_z + pick_height))
		self.keyframe_gripper(lift_frame)

		self.gripper_base_armature.location = Vector((drop_x, drop_y, drop_z + pick_height))
		self.keyframe_gripper(move_frame)

		self.gripper_base_armature.location = Vector((drop_x, drop_y, drop_z))
		self.keyframe_gripper(place_frame)
		############

		#OPEN CLAMP
		release_frame = int(0.92*frames) + start_frame
		self.claw1_armature.location = self.claw1_armature.location + Vector(close_amount)
		self.claw2_armature.location = self.claw2_armature.location - Vector(close_amount)
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

		self.gripper_base_armature.location = Vector((drop_x, drop_y, drop_z + start_height))
		self.gripper_base_armature.keyframe_insert(data_path="location", frame=end_frame)

		toggle_animation(cylinder, end_frame, False)
		

def animation_test():
	bpy.context.scene.render.fps = 24
	last = params["num_segments"]
	piece = "Cylinder"

	for i in range(last):
		obj = get_piece("Cylinder", i)
		obj.keyframe_insert(data_path="rotation_euler", frame=0)
		obj.keyframe_insert(data_path="location", frame = 0)

	base_a = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/ACTION_BASE.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger1_a = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/ACTION_Finger1.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger2_a = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/ACTION_Finger2.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")

	base_h = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/HOLD_BASE.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger1_h = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/HOLD_Finger1.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger2_h = import_mesh("/Users/vainaviv/Documents/GitHub/dynamic-rope-sim/GripperSTL/HOLD_Finger2.STL", "-Z", scale = 0.02)
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")

	gripper_action = YumiGripper(base_a, finger1_a, finger2_a)
	gripper_hold = YumiGripper(base_h, finger1_h, finger2_h)

	c25 = get_piece(piece, 25)
	c35 = get_piece(piece, 35)

	gripper_action.pick_place(c25, 200, (-2,4,0))
	gripper_hold.hold(c35, 200)

	for i in range(last):
		obj = get_piece("Cylinder", i)
		toggle_animation(obj, 200, False)

	for step in range(1, 300):
		bpy.context.scene.frame_set(step)
		render_frame(step)

	base = bpy.data.objects["ACTION BASE"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger1 = bpy.data.objects["ACTION Finger1"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger2 = bpy.data.objects["ACTION Finger2"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	gripper_action = YumiGripper(base, finger1, finger2)

	base = bpy.data.objects["HOLD BASE"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger1 = bpy.data.objects["HOLD Finger1"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	finger2 = bpy.data.objects["HOLD Finger2"]
	bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
	gripper_hold = YumiGripper(base, finger1, finger2)

	c14 = get_piece(piece, 14)
	gripper_action.pick_place(c35, 500, (-3,4,0), pick_height=0)
	gripper_hold.hold(c14, 500)

	for i in range(last):
		obj = get_piece("Cylinder", i)
		toggle_animation(obj, 500, False)

	for step in range(300, 550):
		bpy.context.scene.frame_set(step)
		render_frame(step)

if __name__ == '__main__':
	with open("rigidbody_params.json", "r") as f:
		params = json.load(f)
	clear_scene()
	make_capsule_rope(params)
	make_table(params)
	#rig_rope(params) # UNCOMMENT TO SEE CYLINDER REPR
	add_camera_light()
	animation_test()
	set_animation_settings(600) 

