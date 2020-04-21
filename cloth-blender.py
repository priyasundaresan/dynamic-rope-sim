import bpy
import os
import json
import time
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
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    return bpy.context.object

def make_cloth():
    '''Create cloth and generate new state'''
    # Generate a rigid cloth, add cloth and collision physics
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.editmode_toggle()
    #bpy.ops.mesh.subdivide(number_cuts=25) # Tune this number for cloth detail
    bpy.ops.mesh.subdivide(number_cuts=10) # Tune this number for cloth detail
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels=3 # Smooths the cloth so it doesn't look blocky
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    #bpy.context.object.modifiers["Solidify"].thickness = 0.075
    bpy.context.object.modifiers["Solidify"].thickness = 0.1
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True
    return bpy.context.object

def generate_cloth_state(cloth):
    # Move cloth slightly above the table and simulate a drop
    # Pinned group is the vertices that should not fall
    if cloth is None:
        cloth = make_cloth()
    dx = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dy = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dz = np.random.uniform(0.4,0.8,1)
    cloth.location = (dx,dy,dz)
    cloth.rotation_euler = (0, 0, random.uniform(0, np.pi)) # fixed z, rotate only about x/y axis slightly
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])
    pinned_group = bpy.context.object.vertex_groups.new(name='Pinned')
    n = random.choice(range(1,4)) # Number of vertices to pin
    subsample = sample(range(len(cloth.data.vertices)), n)
    pinned_group.add(subsample, 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    # Episode length = 30 frames
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = 30 # Roughly when the cloth settles
    return cloth

def reset_cloth(cloth):
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def pattern(obj, texture_filename):
    '''Add image texture to object'''
    mat = bpy.data.materials.new(name="ImageTexture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_filename)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    obj.data.materials.append(mat)

def colorize(obj, color):
    '''Add color to object'''
    mat = bpy.data.materials.new(name="Color")
    mat.use_nodes = False
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    set_viewport_shading('MATERIAL')

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    bpy.ops.object.camera_add(location=(0,0,8), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object

def render(filename, engine, episode, cloth, render_size, annotations=None, num_annotations=0):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = "./images/{}".format(filename)
    scene.view_settings.exposure = 1.3
    if engine == 'BLENDER_WORKBENCH':
        scene.render.display_mode
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_render_samples = 1
    for frame in range(0, scene.frame_end):
        # Render 10 images per episode (episode is really 30 frames)
        if frame%3==0:
            index = ((scene.frame_end - scene.frame_start)*episode + frame)//3 
            render_mask("image_masks/%06d_visible_mask.png", index)
            scene.render.filepath = filename % index
            bpy.ops.render.render(write_still=True)
            if annotations is not None:
                annotations = annotate(cloth, index, annotations, num_annotations, render_size)
        # TODO: this is kind of a hack for now, must increment frame by one or cloth looks weird
        # Baking the simulation seems too time-consuming..., so for now just stepping through the frames
        scene.frame_set(frame)
    return annotations

def render_mask(filename, index):
    # NOTE: this method is still in progress
    scene = bpy.context.scene
    saved = scene.render.engine
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1
    scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]
    norm_node = tree.nodes.new(type="CompositorNodeNormalize")
    inv_node = tree.nodes.new(type="CompositorNodeInvert")
    math_node = tree.nodes.new(type="CompositorNodeMath")
    math_node.operation = 'CEIL' # Threshold the depth image
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs["Depth"], inv_node.inputs["Color"])
    links.new(inv_node.outputs[0], norm_node.inputs[0])
    links.new(norm_node.outputs[0], math_node.inputs[0])
    links.new(math_node.outputs[0], composite.inputs["Image"])
    scene.render.filepath = filename % index
    bpy.ops.render.render(write_still=True)
    # Clean up 
    scene.render.engine = saved
    for node in tree.nodes:
        if node.name != "Render Layers":
            tree.nodes.remove(node)
    scene.use_nodes = False

def annotate(cloth, frame, mapping, num_annotations, render_size):
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_deformed = cloth.evaluated_get(depsgraph)
    #vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)[::len(list(cloth_deformed.data.vertices))//num_annotations]] 
    vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)[::num_annotations]] 
    pixels = []
    for i in range(len(vertices)):
        v = vertices[i]
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        pixels.append([pixel])
    mapping[frame] = pixels
    return mapping

def render_dataset(num_episodes, filename, num_annotations, texture_filepath='', render_width=640, render_height=480, color=None):
    # Remove anything in scene 
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    render_scale = scene.render.resolution_percentage / 100
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    clear_scene()
    # Make the camera, lights, table, and cloth only ONCE
    add_camera_light()
    table = make_table()
    cloth = make_cloth()
    if texture_filepath != '':
        engine = 'BLENDER_EEVEE'
        pattern(cloth, texture_filepath)
    elif color:
        engine = 'BLENDER_WORKBENCH'
        colorize(cloth, color)
    annot = {}
    for episode in range(num_episodes):
        reset_cloth(cloth) # Restores cloth to flat state
        cloth = generate_cloth_state(cloth) # Creates a new deformed state
        annot = render(filename, engine, episode, cloth, render_size, annotations=annot, num_annotations=num_annotations) # Render, save ground truth
    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(annot, outfile, sort_keys=True, indent=2)
    
if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs('./images')
    else:
        os.system('rm -r ./images')
        os.makedirs('./images')
    if not os.path.exists("./image_masks"):
        os.makedirs('./image_masks')
    else:
        os.system('rm -r ./image_masks')
        os.makedirs('./image_masks')
    texture_filepath = 'textures/cloth.jpg'
    #texture_filepath = 'textures/qr.png'
    green = (0,0.5,0.5,1)
    filename = "images/%06d_rgb.png"
    episodes = 1 # Note each episode has 10 rendered frames 
    num_annotations = 200 # Pixelwise annotations per image
    #render_dataset(episodes, filename, num_annotations, color=green)
    start = time.time()
    render_dataset(episodes, filename, num_annotations, texture_filepath=texture_filepath)
    print("Render time:", time.time() - start)
