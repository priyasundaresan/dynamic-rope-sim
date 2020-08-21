import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from sklearn.neighbors import NearestNeighbors
import knots
import xml.etree.cElementTree as ET
from xml.dom import minidom
from render_bbox import *
from dr_utils import *

def set_animation_settings(anim_end):
    # Sets up the animation to run till frame anim_end (otherwise default terminates @ 250)
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def set_render_settings(engine, render_size):
    # Set rendering engine, dimensions, colorspace, images settings
    if os.path.exists("./images"):
        os.system('rm -r ./images')
    os.makedirs('./images')
    if os.path.exists("./images_depth"):
        os.system('rm -r ./images_depth')
    os.makedirs('./images_depth')
    if os.path.exists("./image_masks"):
        os.system('rm -r ./image_masks')
    os.makedirs('./image_masks')
    if os.path.exists("./annots"):
        os.system('rm -r ./annots')
    os.makedirs('./annots')
    scene = bpy.context.scene
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    #scene.view_settings.exposure = 1.3
    #scene.view_settings.exposure = 0.8
    if engine == 'BLENDER_WORKBENCH':
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1

def create_pixelannot_xml(annotation_idx, pixel):
    scene = bpy.context.scene
    annotation = ET.Element('annotation')
    obj = ET.SubElement(annotation, 'hold_pixel')
    x,y = pixel
    ET.SubElement(obj, 'x').text = str(x)
    ET.SubElement(obj, 'y').text = str(y)
    tree = ET.ElementTree(annotation)
    xml_file_name = "./annots/%05d.xml" % annotation_idx
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(xml_file_name, "w") as f:
        f.write(xmlstr)

def annotate(frame, mapping, num_annotations, knot_only=False, end_only=False, export_hold_pixel=False, offset=1):
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    pixels = []
    if knot_only:
        annot_list = []
        pull, hold, _ = find_knot(50)
        indices = list(range(pull-offset,pull-offset+2))
        #indices = list(range(pull-offset, pull+offset+1)) + list(range(hold-offset, hold+offset+1))
        # indices = list(range(pull-offset, pull+offset+1))
        # indices = list(range(hold-offset, hold+offset+1))
        box_offset = 4
        pull_idx_min, pull_idx_max = max(0,pull-box_offset), min(49,pull+box_offset+1)
        hold_idx_min, hold_idx_max = max(0,hold-box_offset), min(49,hold+box_offset+1)
    elif end_only:
        indices = list(range(4)) + list(range(46,50))
    else:
        indices = list(range(50))
    for i in indices:
        cyl = get_piece("Cylinder", i if i != 0 else -1)
        cyl_verts = list(cyl.data.vertices)
        step_size = len(indices)*len(cyl_verts)//num_annotations
        #step_size = 1
        vertex_coords = [cyl.matrix_world @ v.co for v in cyl_verts][::step_size]
        for i in range(len(vertex_coords)):
            v = vertex_coords[i]
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
            pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
            pixels.append([pixel])
    if knot_only and export_hold_pixel:
        hold_cyl = get_piece("Cylinder", hold)
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, hold_cyl.location)
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        create_pixelannot_xml(frame, pixel)
    mapping[frame] = pixels

def get_piece(piece_name, piece_id):
    # Returns the piece with name piece_name, index piece_id
    if piece_id == -1 or piece_id == 0 or piece_id is None:
        return bpy.data.objects['%s' % (piece_name)]
    return bpy.data.objects['%s.%03d' % (piece_name, piece_id)]

def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def take_action(obj, frame, action_vec, animate=True):
    # Keyframes a displacement for obj given by action_vec at given frame
    curr_frame = bpy.context.scene.frame_current
    dx,dy,dz = action_vec
    if animate != obj.rigid_body.kinematic:
        # We are "picking up" a dropped object, so we need its updated location
        obj.location = obj.matrix_world.translation
        obj.rotation_euler = obj.matrix_world.to_euler()
        obj.keyframe_insert(data_path="location", frame=curr_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def find_knot(num_segments, chain=False, depth_thresh=0.4, idx_thresh=3, pull_offset=3,knot_idx=None):

    piece = "Torus" if chain else "Cylinder"
    cache = {}

    # Make a single pass, store the xy positions of the cylinders
    for i in range(num_segments):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        key = tuple((x,y))
        val = {"idx":i, "depth":z}
        cache[key] = val
    neigh = NearestNeighbors(2, 0)
    planar_coords = list(cache.keys())
    neigh.fit(planar_coords)
    # Now traverse and look for the under crossing
    idx_list = range(knot_idx[0], knot_idx[1]) if not knot_idx is None else range(num_segments)
    for i in idx_list:
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        match_idxs = neigh.kneighbors([(x,y)], 2, return_distance=False) # 1st neighbor is always identical, we want 2nd
        nearest = match_idxs.squeeze().tolist()[1:][0]
        x1,y1 = planar_coords[nearest]
        curr_cyl, match_cyl = cache[(x,y)], cache[(x1,y1)]
        depth_diff = match_cyl["depth"] - curr_cyl["depth"]
        idx_diff = abs(match_cyl["idx"] - curr_cyl["idx"])
        if depth_diff > depth_thresh and idx_diff > idx_thresh:
            pull_idx = i + pull_offset # Pick a point slightly past under crossing to do the pull
            dx = planar_coords[pull_idx][0] - x
            dy = planar_coords[pull_idx][1] - y
            hold_idx = match_cyl["idx"]
            SCALE_X = 1
            SCALE_Y = 1
            Z_OFF = 2
            action_vec = [SCALE_X*dx, SCALE_Y*dy, Z_OFF] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
    return 16, 25, [0,0,0] # Didn't find a pull/hold

def center_camera(randomize=True, flip=False):
    # move camera to above knot location as crop
    knot_pull, knot_hold, _ = find_knot(50)
    if knot_pull is -1 or knot_hold is None:
        return
    hold_cyl = get_piece("Cylinder", -1 if knot_hold == 0 else knot_hold)
    pull_cyl = get_piece("Cylinder", -1 if knot_pull == 0 else knot_pull)
    hold_loc = hold_cyl.matrix_world.translation
    pull_loc = pull_cyl.matrix_world.translation
    camera_x = (hold_loc[0] + pull_loc[0])/2
    camera_y = (hold_loc[1] + pull_loc[1])/2
    camera_z = 1
    offset = 0.2
    # offset = 0.05
    dx = np.random.uniform(-offset, offset) if randomize else 0
    dy = np.random.uniform(-offset, offset) if randomize else 0
    dz = np.random.uniform(0.75, 2.5) if randomize else 0 # Tweaked this to be higher, see more of the knot
    # check that the Cylinder.049 on left and Cylinder on right
    cyl_0_loc = get_piece("Cylinder", -1).matrix_world.translation
    cyl_49_loc = get_piece("Cylinder", 49).matrix_world.translation
    if cyl_0_loc[0] < cyl_49_loc[0]:
        bpy.context.scene.camera.rotation_euler = (0, 0, np.pi)

    rot = np.random.uniform(-np.pi/16, np.pi/16)

    x_rot = np.random.uniform(-np.pi/64, np.pi/64)
    y_rot = np.random.uniform(-np.pi/64, np.pi/64)
    #bpy.context.scene.camera.rotation_euler = (x_rot, y_rot, rot) # Allowing non-planar rotation a little bit too
    bpy.context.scene.camera.rotation_euler = (0, 0, rot) # Allowing non-planar rotation a little bit too

    # reset camera location:
    bpy.context.scene.camera.location = (camera_x+dx, camera_y+dy, camera_z+dz)
    return

def randomize_camera():
    ANGLE_DIVS = 65
    xrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    yrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    zrot = np.random.uniform(-pi/6, pi/6) 
    xoffset = 0.5
    yoffset = 0.5
    zoffset = 0.5
    dx = np.random.uniform(-xoffset, xoffset)
    dy = np.random.uniform(-yoffset, yoffset)
    dz = np.random.uniform(-zoffset, zoffset)
    bpy.context.scene.camera.rotation_euler = (xrot, yrot, zrot)
    piece = "Cylinder"
    mid_rope = get_piece(piece, 25)
    x,y,z = mid_rope.matrix_world.translation
    #bpy.context.scene.camera.location = Vector((x,y,np.random.uniform(15,25))) + Vector((dx, dy, dz))
    #bpy.context.scene.camera.location = Vector((x,y,np.random.uniform(15,25))) + Vector((dx, dy, dz))
    #bpy.context.scene.camera.location = Vector((x,y,np.random.uniform(13,24))) + Vector((dx, dy, dz))
    bpy.context.scene.camera.location = Vector((x,y,np.random.uniform(23,25))) + Vector((dx, dy, dz))
    #bpy.context.scene.camera.location = Vector((2,0,25)) + Vector((dx, dy, dz))

def render_frame(frame, render_offset=0, step=2, num_annotations=540, filename="%06d_rgb.png", folder="images", annot=True, mapping=None):
    global rig
    # Renders a single frame in a sequence (if frame%step == 0)
    frame -= render_offset

    # DOMAIN RANDOMIZATION
    randomize_camera()
    randomize_light()
    table = bpy.data.objects["Plane"]
    if random.random() < 0.33:
        texture_randomize(table, 'dr_data/val2017')
    elif random.random() < 0.66:
        texture_randomize(table, 'dr_data/fabrics')
    else:
        color_randomize(table)

    color = (np.random.uniform(0.7,1.0),np.random.uniform(0.6,1.0),np.random.uniform(0.6,1.0))
    color_randomize(rig, color=color)

    if frame%step == 0:
        scene = bpy.context.scene

        index = frame//step
        render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)
        if annot:
            annotate(index, mapping, num_annotations)

def render_mask(mask_filename, depth_filename, index):
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
    links.new(norm_node.outputs[0], composite.inputs["Image"])

    scene.render.filepath = depth_filename % index
    bpy.ops.render.render(write_still=True)

    links.new(norm_node.outputs[0], math_node.inputs[0])
    links.new(math_node.outputs[0], composite.inputs["Image"])

    scene.render.filepath = mask_filename % index
    bpy.ops.render.render(write_still=True)
    # Clean up
    scene.render.engine = saved
    for node in tree.nodes:
        if node.name != "Render Layers":
            tree.nodes.remove(node)
    scene.use_nodes = False

def take_undo_action_oracle(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):
    piece = "Cylinder"
    pull_idx, hold_idx, action_vec = find_knot(50)
    action_vec = np.array(action_vec) + np.random.uniform(-0.5, 0.5, 3)
    action_vec /= np.linalg.norm(action_vec)
    action_vec *= 2
    pull_cyl = get_piece(piece, pull_idx)
    hold_cyl = get_piece(piece, hold_idx)
    end_frame = start_frame + 100
    take_action(hold_cyl, end_frame, (0,0,0))

    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        #render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        if render and (abs(step-start_frame) < 5 or abs(step-(start_frame+10)) < 5):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1

    take_action(pull_cyl, end_frame, action_vec)
    ## Release both pull, hold
    toggle_animation(pull_cyl, end_frame, False)
    toggle_animation(hold_cyl, end_frame, False)
    settle_time = 30
    # Let the rope settle after the action, so we can know where the ends are afterwards
    for step in range(start_frame + 10, end_frame+settle_time):
        bpy.context.scene.frame_set(step)
        #render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        if render and (abs(step-(start_frame+10)) < 2 or abs(step-(end_frame+settle_time)) < 2):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1
    return end_frame+settle_time, render_offset

def random_loosen(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1

    pick, hold, _ = find_knot(params["num_segments"])
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    dx = np.random.uniform(0.6,0.8)*random.choice((-1,1))
    dy = np.random.uniform(0.6,0.8)*random.choice((-1,1))
    dz = np.random.uniform(1,2)

    mid_frame = start_frame + 50
    end_frame = start_frame + 100

    take_action(hold_cyl, mid_frame, (0,0,0))
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        if render and (abs(step-start_frame) < 5 or abs(step-(start_frame+10)) < 5):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    take_action(hold_cyl, mid_frame, (-dx,-dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    toggle_animation(hold_cyl, mid_frame, False)
    for step in range(start_frame + 10, end_frame):
        bpy.context.scene.frame_set(step)
        if render and (abs(step-(start_frame+10)) < 2 or abs(step-end_frame) < 2):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1
    return end_frame, render_offset

def reidemeister(params, start_frame,render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)

    middle_frame = start_frame+25
    end_frame = start_frame+75
    take_action(end2, middle_frame, (-6-end2.matrix_world.translation[0],np.random.uniform(-2,2),0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
    take_action(end1, end_frame, (9-end1.matrix_world.translation[0],np.random.uniform(-2,2),0))

    # Drop the ends
    toggle_animation(end1, end_frame, False)
    toggle_animation(end2, end_frame, False)

    for step in range(middle_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
    return end_frame

def flip_knot(start_frame, render=False, render_offset=0):
    piece = "Cylinder"

    pick, hold, _ = find_knot(params["num_segments"])
    if random.random() < 0.5:
        pick = random.choice(range(10, 40))
    pull_cyl = get_piece(piece, pick)

    dx = 0
    dy = np.random.uniform(-2,-2)
    dz = np.random.uniform(1,1)

    mid_frame = start_frame + 30
    end_frame = start_frame + 60

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    for step in range(start_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset)
    return end_frame

def random_end_pick_place(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)

    dx1 = np.random.uniform(-8,0)
    dy1 = np.random.uniform(-2,2.1)
    dz1 = np.random.uniform(4,6)
    #dz1 = np.random.uniform(0,0)

    dx2 = np.random.uniform(0,5)
    dy2 = np.random.uniform(-2,2.1)
    dz2 = np.random.uniform(4,6)
    #dz2 = np.random.uniform(0,0)

    middle_frame = start_frame+25
    end_frame = start_frame+50
    end1_first = random.random() > 0.5

    take_action(end1, middle_frame, (dx1,dy1,dz1))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        # if render and step % render_freq == 0:
        if render and (abs(step-start_frame) < 5 or abs(step-middle_frame) < 5):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1

    take_action(end2, end_frame, (dx2,dy2,dz2))
    toggle_animation(end1, middle_frame, False)
    toggle_animation(end2, end_frame-10, False)

    for step in range(middle_frame, end_frame+20):
        bpy.context.scene.frame_set(step)

        if render and end_frame-step < 20 and (abs(step-end_frame) < 2 or abs(step-middle_frame) < 2):
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
        elif render:
            render_offset += 1
    # return end_frame, pick, render_offset
    return end_frame, render_offset

def generate_dataset(params, iters=1, chain=False, render=False):

    set_animation_settings(15000)
    piece = "Cylinder"
    last = params["num_segments"]-1
    mapping = {}

    render_offset = 0
    num_loosens = 4# For each knot, we can do num_loosens loosening actions
    for i in range(iters):
        #clear_scene()
        #make_capsule_rope(params)
        #rig = rig_rope(params, braid=random.choice((0,1)))

        num_knots = 1
        # knot_end_frame = knots.tie_pretzel_knot(params, render=False)
        # if random.random() < 0.5:
        #     knot_end_frame = flip_knot(knot_end_frame, render=False)
        if i%2==0:
           knot_end_frame = knots.tie_pretzel_knot(params, render=False)
        elif i%2==1:
           knot_end_frame = knots.tie_figure_eight(params, render=False)
        reid_end_frame = reidemeister(params, knot_end_frame, render=False, mapping=mapping)
        render_offset += reid_end_frame

        loosen_start = reid_end_frame
        for i in range(num_loosens):
            loosen_end_frame, offset = take_undo_action_oracle(params, loosen_start, render=render, render_offset=render_offset, mapping=mapping)
            loosen_start = loosen_end_frame
            render_offset = offset
        render_offset -= loosen_end_frame
        # Delete all keyframes to make a new knot and reset the frame counter
        bpy.context.scene.frame_set(0)
        for a in bpy.data.actions:
            bpy.data.actions.remove(a)

    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(mapping, outfile, sort_keys=True, indent=2)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_capsule_rope(params)
    rig = rig_rope(params, braid=1)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    generate_dataset(params, iters=20, render=True)

#    os.mkdir('./cap_pull')
#    os.system('mv ./images ./cap_pull')
#    os.system('mv ./images_depth ./cap_pull')
#    os.system('mv ./image_masks ./cap_pull')
