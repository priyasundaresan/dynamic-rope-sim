import bpy
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from dr_utils import *
from sklearn.neighbors import NearestNeighbors
from knots import tie_pretzel_knot, tie_stevedore, tie_figure_eight, tie_double_pretzel

def set_animation_settings(anim_end):
    # Sets up the animation to run till frame anim_end (otherwise default terminates @ 250)
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

def set_render_settings(engine, render_size):
    # Set rendering engine, dimensions, colorspace, images settings
    if not os.path.exists("./images"):
        os.makedirs('./images')
    else:
        os.system('rm -r ./images')
    if not os.path.exists("./annots"):
        os.makedirs('./annots')
    else:
        os.system('rm -r ./annots')
        os.makedirs('./annots')
    scene = bpy.context.scene
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    #scene.view_settings.exposure = 0.8
    if engine == 'BLENDER_WORKBENCH':
        scene.render.display_mode
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='JPEG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_render_samples = 1
        scene.render.image_settings.file_format='JPEG'

def create_labimg_xml(annotation_idx, annotation_list):
    scene = bpy.context.scene
    annotation = ET.Element('annotation')
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(scene.render.resolution_x)
    ET.SubElement(size, 'height').text = str(scene.render.resolution_y)
    ET.SubElement(size, 'depth').text = str(3)
    for annot in annotation_list:
        xmin, ymin, xmax, ymax = annot
        object = ET.SubElement(annotation, 'object')
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    tree = ET.ElementTree(annotation)
    xml_file_name = "./annots/%05d.xml" % annotation_idx
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(xml_file_name, "w") as f:
        f.write(xmlstr)
    #tree.write(xml_file_name, pretty_print=True)

def find_knot_cylinders(num_segments, chain=False, num_knots=1):
    piece = "Torus" if chain else "Cylinder"
    cache = {}
    curr_z = get_piece(piece, -1).matrix_world.translation[2]
    dz_thresh = 0.2
    dzs = []
    for i in range(num_segments):
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        dz = abs(z - curr_z)
        dzs.append(dz)
        curr_z = z
    dzs = np.round(dzs, 1)
    if num_knots == 1:
        nonzero = np.where(dzs>0.2)
        start_idx, end_idx = np.amin(nonzero), np.amax(nonzero)
        result = [[start_idx, end_idx]]
    else:
        nonzero = np.where(dzs>0.2)[0]
        split_idx, x, dx = 0, nonzero[0], 0
        for i in range(len(nonzero)):
            dx_curr = nonzero[i] - x
            if dx_curr > dx:
                dx = dx_curr
                split_idx = i
                x = nonzero[i]
        s1, e1 = np.amin(nonzero[:split_idx]), np.amax(nonzero[:split_idx])
        s2, e2 = np.amin(nonzero[split_idx:]), np.amax(nonzero[split_idx:])
        result = [[s1,e1],[s2,e2]]
    return result

def annotate(frame, offset=4, num_knots=1):
    # knot_only = True:  means only record the under, over crossings
    # knot_only = False:  means record annotations for full rope
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    render_size = (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            )
    annot_list = []
    knots = find_knot(50, num_knots=num_knots)
    for i, knot in enumerate(knots):
        if num_knots == 1:
            start_idx, end_idx = find_knot_cylinders(50, num_knots=num_knots)[i]
            indices = list(range(max(0,start_idx-offset), min(50, end_idx+offset)))
        else:
            pull, hold, _ = knot
            pull_idx_min, pull_idx_max = max(0,pull-offset), min(49,pull+offset+1)
            hold_idx_min, hold_idx_max = max(0,hold-offset), min(49,hold+offset+1)
            indices = list(range(pull_idx_min, pull_idx_max)) + list(range(hold_idx_min, hold_idx_max))
        min_x = scene.render.resolution_x
        max_x = 0
        min_y = scene.render.resolution_y
        max_y = 0
        for i in indices:
            cyl = get_piece("Cylinder", i if i != 0 else -1)
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, cyl.matrix_world.translation)
            x, y = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        min_x -= np.random.randint(10, 12)
        min_y -= np.random.randint(10, 12)
        max_x += np.random.randint(10, 12)
        max_y += np.random.randint(10, 12)
        print("Width: %d, Height: %d"%(max_x - min_x, max_y - min_y))
        annot_list.append([min_x,min_y,max_x,max_y])
    create_labimg_xml(frame, annot_list)

def get_piece(piece_name, piece_id):
    # Returns the piece with name piece_name, index piece_id
    if piece_id == -1:
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

def find_knot(num_segments, chain=False, num_knots=1, depth_thresh=0.43, idx_thresh=3, pull_offset=3):
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
    knots  = []
    #for i in range(num_segments):
    i = 0
    while i < num_segments - pull_offset:
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
            action_vec = [7*dx, 7*dy, 6] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            knots.append([pull_idx, hold_idx, action_vec])
            if len(knots) == num_knots:
                break
            i += 25
            continue
            #return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
        i += 1
    return knots
    #return -1, last, [0,0,0] # Didn't find a pull/hold


def randomize_camera():
    #ANGLE_DIVS = 55
    ANGLE_DIVS = 35
    xrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    yrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    zrot = np.random.uniform(-pi/6, pi/6) 
    xoffset = 1.5
    yoffset = 1.5
    zoffset = 1.5
    #zoffset = 10
    dx = np.random.uniform(-(xoffset+3.5), xoffset)
    dy = np.random.uniform(-yoffset, yoffset)
    dz = np.random.uniform(-18, 1)
    bpy.context.scene.camera.rotation_euler = (xrot, yrot, zrot)
    bpy.context.scene.camera.location = Vector((2,0,28)) + Vector((dx, dy, dz))

#def randomize_camera():
#    #rot = np.random.uniform(-pi/12, pi/12)
#    rot = np.random.uniform(-pi/6, pi/6) + random.choice((0, np.pi))
#    xoffset = 0.2
#    yoffset = 0.2
#    zoffset = 0.2
#    dx = np.random.uniform(-xoffset, xoffset)
#    dy = np.random.uniform(-yoffset, yoffset)
#    dz = np.random.uniform(-zoffset, zoffset)
#    bpy.context.scene.camera.rotation_euler = (0, 0, rot)
#    bpy.context.scene.camera.location = Vector((2,0,28)) + Vector((dx, dy, dz))
    
def render_frame(frame, render_offset=0, step=5, filename="%05d.jpg", folder="images", annot=True, num_knots=1, mapping=None):
    # Renders a single frame in a sequence (if frame%step == 0)
    global rig
    randomize_light()
    randomize_rig(rig)
    table = bpy.data.objects["Plane"]
    if random.random() < 0.33:
        texture_randomize(table, 'dr_data/val2017')
    elif random.random() < 0.66:
        texture_randomize(table, 'dr_data/fabrics')
    else:
        color_randomize(table)
    color = (np.random.uniform(0.7,1.0),np.random.uniform(0.7,1.0),np.random.uniform(0.7,1.0))
    color_randomize(rig, color=color)
    randomize_camera()

    frame -= render_offset
    if frame%step == 0:
        scene = bpy.context.scene
        index = frame//step
        #render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)
        if annot:
            annotate(index, num_knots=num_knots)

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


def reidemeister(params, start_frame, render=False, render_offset=0, annot=True, num_knots=1, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)

    middle_frame = start_frame+50
    end_frame = start_frame+100

    take_action(end1, middle_frame, (np.random.uniform(9,11)-end1.matrix_world.translation[0],np.random.uniform(-3,3),0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)
    take_action(end2, end_frame, (np.random.uniform(-6,-8)-end2.matrix_world.translation[0],np.random.uniform(-3,3),0))
    # Drop the ends

    toggle_animation(end1, middle_frame, False)
    #toggle_animation(end1, end_frame, False)
    toggle_animation(end2, end_frame, False)

    for step in range(middle_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)
    return end_frame

def take_undo_action_oracle(params, start_frame, render=False, render_offset=0, annot=True, num_knots=1, mapping=None):
    piece = "Cylinder"
    pull_idx, hold_idx, action_vec = find_knot(50)[0]
    action_vec = np.array(action_vec) + np.random.uniform(-0.5, 0.5, 3)
    action_vec /= np.linalg.norm(action_vec)
    action_vec *= 2
    pull_cyl = get_piece(piece, pull_idx if pull_idx else -1)
    hold_cyl = get_piece(piece, hold_idx if hold_idx else -1)
    end_frame = start_frame + 100
    take_action(hold_cyl, end_frame, (0,0,0))

    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)

    take_action(pull_cyl, end_frame, action_vec)
    ## Release both pull, hold
    toggle_animation(pull_cyl, end_frame, False)
    toggle_animation(hold_cyl, end_frame, False)
    settle_time = 30
    # Let the rope settle after the action, so we can know where the ends are afterwards
    for step in range(start_frame + 10, end_frame+settle_time):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)
    return end_frame+settle_time


def random_loosen(params, start_frame, render=False, render_offset=0, annot=True, num_knots=1, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1

    knots = find_knot(params["num_segments"], num_knots=num_knots)
    pick, hold, _ = knots[random.choice(range(len(knots)))]
    #if random.random() < 0.5:
    #    pick = random.choice(range(10, 40))
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    dx = np.random.uniform(0,2.5)*random.choice((-1,1))
    dy = np.random.uniform(0,2.5)*random.choice((-1,1))
    #dz = np.random.uniform(0.5,1)
    #dz = np.random.uniform(0.75,2.25)
    #dz = np.random.uniform(0.75,1.75)
    #dz = np.random.uniform(0.75,1.25)
    dz = np.random.uniform(1.5,3.5)

    mid_frame = start_frame + 50
    end_frame = start_frame + 100

    take_action(hold_cyl, mid_frame, (0,0,0))
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    toggle_animation(hold_cyl, mid_frame, False)
    for step in range(start_frame + 10, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping, num_knots=num_knots)
    return end_frame

def generate_dataset(params, chain=False, render=False):

    set_animation_settings(15000)
    piece = "Cylinder"
    last = params["num_segments"]-1
    mapping = None
    
    #knot_end_frame = tie_pretzel_knot(params, render=False)
    #reid_start = knot_end_frame
    #for i in range(1): 
    ## NOTE: each iteration renders 75 images, ~45 is about 3500 images for generating a training dset
    #    reid_end_frame = reidemeister(params, reid_start, render=render, render_offset=knot_end_frame, mapping=mapping)
    #    reid_start = random_loosen(params, reid_end_frame, render=render, render_offset=knot_end_frame, mapping=mapping)

    render_offset = 0
    num_loosens = 4
    for i in range(30):
        num_knots = 1
        if i%6==0:
            knot_end_frame = tie_pretzel_knot(params, render=False)
        elif i%6==1:
            knot_end_frame = tie_figure_eight(params, render=False)
        elif i%6==2:
            knot_end_frame = tie_stevedore(params, render=False)
        else:
            knot_end_frame = tie_double_pretzel(params, render=False)
            num_knots = 2
        render_offset += knot_end_frame
        reid_end_frame = reidemeister(params, knot_end_frame, render=render, render_offset=render_offset, num_knots=num_knots, mapping=mapping)
        #loosen_end_frame = random_loosen(params, reid_end_frame, render=render, render_offset=render_offset, num_knots=num_knots, mapping=mapping)
        start = reid_end_frame
        for i in range(num_loosens):
            loosen_end_frame = take_undo_action_oracle(params, start, render=render, render_offset=render_offset, num_knots=num_knots, mapping=mapping)
            start = loosen_end_frame
        render_offset -= loosen_end_frame
        bpy.context.scene.frame_set(0)
        for a in bpy.data.actions:
            bpy.data.actions.remove(a)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    #make_rope(params)
    make_capsule_rope(params)
    rig = rig_rope(params, mode="cable")
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    start = time.time()
    generate_dataset(params, render=True)
    end = time.time()
    print("time", end-start)
