import bpy
import numpy as np
import os
from mathutils import Vector
import random
import sys
sys.path.append(os.getcwd())

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def add_camera_light(dr=True):
    if dr:
        light_data = bpy.data.lights.new(name="Light", type='POINT')
        light_data.energy = 300
        light_object = bpy.data.objects.new(name="LightObj", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = (0, 0, 5)
    else:
        bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,20))
    bpy.ops.object.camera_add(location=(2,0,28), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object

def randomize_light():
    #print(list(bpy.data.objects))
    scene = bpy.context.scene
    scene.view_settings.exposure = random.uniform(2.5,3.5)
    light_data = bpy.data.lights['Light']
    light_data.color = tuple(np.random.uniform(0,1,3))
    light_data.energy = np.random.uniform(300,500)
    light_data.shadow_color = tuple(np.random.uniform(0,1,3))
    light_obj = bpy.data.objects['LightObj']
    light_obj.data.color = tuple(np.random.uniform(0.7,1,3))
    #light_obj.location += Vector(np.random.uniform(-0.25,0.25,3).tolist())
    light_obj.location = Vector(np.random.uniform(-4,4,3).tolist())
    light_obj.location[2] = np.random.uniform(4,7)
    light_obj.rotation_euler[0] += np.random.uniform(-np.pi/4, np.pi/4)
    light_obj.rotation_euler[1] += np.random.uniform(-np.pi/4, np.pi/4)
    light_obj.rotation_euler[2] += np.random.uniform(-np.pi/4, np.pi/4)

def randomize_camera():
    scene = bpy.context.scene
    bpy.ops.view3d.camera_to_view_selected()
    dx = np.random.uniform(-0.05,0.05)
    dy = np.random.uniform(-0.05,0.05)
    dz = np.random.uniform(-1,1)
    bpy.context.scene.camera.location += Vector((dx,dy,dz))
    bpy.context.scene.camera.rotation_euler = (0, 0, np.random.uniform(-np.pi/4, np.pi/4))

def randomize_rig(rig, mode="capsule", ):
    if mode=="capsule":
        rig = bpy.data.objects['BezierCircle']
        new_scale = np.random.uniform(0.5,1.25)
        rig.scale = (new_scale, new_scale, new_scale)
    else:
        pass
        #circle = bpy.data.objects["Circle.003"]
        #sx,sy,sz = circle.scale
        #sy = np.random.uniform(0.06,0.15)
        #sy = sx
        #circle.scale = (sx,sy,sz)
        #new_off = np.random.uniform(11,13)
        #new_iter = np.random.uniform(14,16)
        #rig.modifiers["Screw"].screw_offset = new_off
        #rig.modifiers["Screw"].iterations = new_iter

def pattern(obj, texture_filename):
    '''Add image texture to object (don't create new materials, just overwrite the existing one if there is one)'''
    if '%sTexture' % obj.name in bpy.data.materials: 
        mat = bpy.data.materials['%sTexture'%obj.name]
    else:
        mat = bpy.data.materials.new(name="%sTexture"%obj.name)
        mat.use_nodes = True
    if "Image Texture" in mat.node_tree.nodes:
        texImage = mat.node_tree.nodes["Image Texture"]
    else:
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage.image = bpy.data.images.load(texture_filename)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    mat.specular_intensity = np.random.uniform(0, 0.3)
    mat.roughness = np.random.uniform(0.5, 1)
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

def texture_randomize(obj, textures_folder):
    rand_img_path = random.choice(os.listdir(textures_folder))
    img_filepath = os.path.join(textures_folder, rand_img_path)
    pattern(obj, img_filepath)

def color_randomize(obj, color=None):
    if color is None:
        r,g,b = np.random.uniform(0,1,3)
    else:   
        r,g,b = np.array(color) + np.random.standard_normal(3)/10.0
    color = [r,g,b,1]
    if '%sColor' % obj.name in bpy.data.materials:
        mat = bpy.data.materials['%sColor'%obj.name]
    else:
        mat = bpy.data.materials.new(name="%sColor"%obj.name)
        mat.use_nodes = False
    mat.diffuse_color = color
    mat.specular_intensity = np.random.uniform(0, 0.1)
    mat.roughness = np.random.uniform(0.5, 1)
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    set_viewport_shading('MATERIAL')
    
