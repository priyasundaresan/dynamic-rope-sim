import bpy
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from sklearn.neighbors import NearestNeighbors
from knots import tie_pretzel_knot, tie_stevedore, tie_figure_eight, tie_double_pretzel
from untangle_utils import *
from dr_utils import *

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
    if not os.path.exists("./actions"):
        os.makedirs('./actions')
    else:
        os.system('rm -r ./actions')
        os.makedirs('./actions')
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
        scene.render.image_settings.file_format='JPEG'
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_render_samples = 1

def save_actions(annotation_idx, annotation_list):
    np_annotations = np.array(annotation_list)
    np.save('actions/%05d.npy'%annotation_idx, np_annotations)

def find_knot(num_segments, chain=False, depth_thresh=0.35, idx_thresh=3, pull_offset=3):

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
    for i in range(num_segments):
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

def annotate(frame, offset=4, annotate_undone_check=False, annotate_end_episode=False):
    # knot_only = True:  means only record the under, over crossings
    # knot_only = False:  means record annotations for full rope
    scene = bpy.context.scene
    render_size = (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            )
    annot_list = []
    pull_idx, hold_idx, action_vec = find_knot(50)
    if action_vec != [0,0,0]:
        action_vec /= np.linalg.norm(action_vec)
        action_vec *= 2.5
    undone_flag = int(annotate_undone_check)
    end_episode_flag = int(annotate_end_episode)
    end1_loc = get_piece("Cylinder", 50-1).matrix_world.translation
    pull_loc = get_piece("Cylinder", pull_idx if pull_idx else -1).matrix_world.translation
    hold_loc = get_piece("Cylinder", hold_idx if hold_idx else -1).matrix_world.translation
    end2_loc = get_piece("Cylinder", -1).matrix_world.translation
    locs = [end1_loc, pull_loc, hold_loc, end2_loc]
    annotations = [] # [[x1,y1],[x2,y2],...
    for loc in locs:
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, loc)
        x, y = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        annotations += [x,y]
    annotations += [undone_flag]
    annotations += [end_episode_flag]
    save_actions(frame, annotations)

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

def randomize_camera():
    ANGLE_DIVS = 65
    xrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    yrot = np.random.uniform(-pi/ANGLE_DIVS, pi/ANGLE_DIVS) 
    zrot = np.random.uniform(-pi/6, pi/6) 
    #xoffset = 2
    #yoffset = 2
    #zoffset = 2
    xoffset = 1
    yoffset = 1
    zoffset = 1
    dx = np.random.uniform(-xoffset, xoffset)
    dy = np.random.uniform(-yoffset, yoffset)
    dz = np.random.uniform(-zoffset, zoffset)
    bpy.context.scene.camera.rotation_euler = (xrot, yrot, zrot)
    bpy.context.scene.camera.location = Vector((2,0,28)) + Vector((dx, dy, dz))
    
def render_frame(frame, render_offset=0, step=10, filename="%05d.jpg", folder="images", annot=True, annotate_undone_check=False, annotate_end_episode=False):
    # DOMAIN RANDOMIZE
    global rig
    randomize_camera()
    #randomize_rig(rig, mode="braid")
    #randomize_rig(rig)
    #randomize_light()
    #table = bpy.data.objects["Plane"]
    #if random.random() < 0.33:
    #    texture_randomize(table, 'dr_data/val2017')
    #elif random.random() < 0.66:
    #    texture_randomize(table, 'dr_data/fabrics')
    #else:
    #    color_randomize(table)
    #color = (np.random.uniform(0.7,1.0),np.random.uniform(0.7,1.0),np.random.uniform(0.7,1.0))
    #color_randomize(rig, color=color)

    # Renders a single frame in a sequence (if frame%step == 0)
    frame -= render_offset
    if frame%step == 0:
        scene = bpy.context.scene
        index = frame//step
        #render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = os.path.join(folder, filename) % index
        bpy.ops.render.render(write_still=True)
        if annot:
            annotate(index, annotate_undone_check=annotate_undone_check, annotate_end_episode=annotate_end_episode)

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


def reidemeister(params, start_frame, render=False, render_offset=0, annot=True):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)

    middle_frame = start_frame+50
    end_frame = start_frame+100

    #take_action(end1, middle_frame, (np.random.uniform(9,11)-end1.matrix_world.translation[0],np.random.uniform(-3,3),0))
    take_action(end1, middle_frame, (11-end1.matrix_world.translation[0],np.random.uniform(-3,3),0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot)
    #take_action(end2, end_frame, (np.random.uniform(-6,-8)-end2.matrix_world.translation[0],np.random.uniform(-3,3),0))
    take_action(end2, end_frame, (-8-end2.matrix_world.translation[0],np.random.uniform(-3,3),0))
    # Drop the ends

    toggle_animation(end1, middle_frame, False)
    #toggle_animation(end1, end_frame, False)
    toggle_animation(end2, end_frame, False)

    for step in range(middle_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot)
    return end_frame

def take_undo_action_oracle(params, start_frame, render=False, render_offset=0, annot=True):
    piece = "Cylinder"
    pull_idx, hold_idx, action_vec = find_knot(50)
    #action_vec = np.array(action_vec) + np.random.uniform(-0.5, 0.5, 3)
    action_vec = np.array(action_vec) + np.random.uniform(-1,1,3)
    action_vec /= np.linalg.norm(action_vec)
    action_vec *= 2.5
    pull_cyl = get_piece(piece, pull_idx if pull_idx else -1)
    hold_cyl = get_piece(piece, hold_idx if hold_idx else -1)
    #end_frame = start_frame + 100
    end_frame = start_frame + 130
    take_action(hold_cyl, end_frame, (0,0,0))

    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        #render_frame(step, render_offset=render_offset, annot=annot)
        if render and (abs(step-start_frame) < 5 or abs(step-(start_frame+10)) < 5):
            render_frame(step, render_offset=render_offset, annot=annot)
        elif render:
            render_offset += 1

    take_action(pull_cyl, end_frame, action_vec)
    ## Release both pull, hold
    toggle_animation(pull_cyl, end_frame, False)
    toggle_animation(hold_cyl, end_frame, False)
    settle_time = 30

    for step in range(start_frame + 10, end_frame+settle_time):
        bpy.context.scene.frame_set(step)
        #render_frame(step, render_offset=render_offset, annot=annot)
        if render and (abs(step-(start_frame+10)) < 2 or abs(step-(end_frame+settle_time)) < 2):
            render_frame(step, render_offset=render_offset, annot=annot)
        elif render:
            render_offset += 1
    return end_frame+settle_time, render_offset, pull_idx, hold_idx, action_vec


def random_end_pick_place(params, start_frame, render=False, render_offset=0, annot=True):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)
    
    dx1 = np.random.uniform(-5,0)
    dy1 = np.random.uniform(-2,2)
    dz1 = np.random.uniform(4,6)
    #dz1 = np.random.uniform(0,0)

    dx2 = np.random.uniform(0,5)
    dy2 = np.random.uniform(-2,2)
    dz2 = np.random.uniform(4,6)
    #dz2 = np.random.uniform(0,0)

    middle_frame = start_frame+25
    end_frame = start_frame+50
    end1_first = random.random() > 0.5

    take_action(end1, middle_frame, (dx1,dy1,dz1))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot)
    take_action(end2, end_frame, (dx2,dy2,dz2))
    toggle_animation(end1, middle_frame, False)
    toggle_animation(end2, end_frame-10, False)

    for step in range(middle_frame, end_frame+20):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot)
    return end_frame+20

def undone_check(pull_idx, hold_idx, prev_action_vec):
    piece = "Cylinder"
    pull_cyl = get_piece(piece, pull_idx if pull_idx else -1)
    hold_cyl = get_piece(piece, hold_idx if hold_idx else -1)

    end_cyl = get_piece(piece, -1)
    end_loc = end_cyl.matrix_world.translation
    hold_loc = hold_cyl.matrix_world.translation
    pull_loc = pull_cyl.matrix_world.translation

    prev_action_vec = prev_action_vec[:-1]/np.linalg.norm(prev_action_vec[:-1])
    end_hold_vec = np.array(end_loc - hold_loc)[:-1]/np.linalg.norm(np.array(end_loc - hold_loc)[:-1])
    thresh = 3
    dist = np.linalg.norm(end_loc - hold_loc)
    #if np.dot(prev_action_vec, end_hold_vec) > 0.7 and dist < thresh:
    if np.dot(prev_action_vec, end_hold_vec) > 0.75:
        return True
    return False

def generate_dataset(iters, params, chain=False, render=False):

    set_animation_settings(15000)
    piece = "Cylinder"
    last = params["num_segments"]-1

    render_offset = 0
    #num_loosens = 4
    num_loosens = 6
    for i in range(iters):
        print("Iter %d of %d" % (i,iters))
        num_knots = 1
        if i%3==0:
            knot_end_frame = tie_pretzel_knot(params, render=False)
        elif i%3==1:
            knot_end_frame = tie_figure_eight(params, render=False)
        else:
            knot_end_frame = tie_double_pretzel(params, render=False)
            num_knots = 2
        knot_end_frame = random_end_pick_place(params, knot_end_frame, render=False)
        render_offset += knot_end_frame
        reid_end_frame = reidemeister(params, knot_end_frame, render=render, render_offset=render_offset)
        #perturb_end_frame = random_perturb(reid_end_frame, params, render=False)
        #render_offset += perturb_end_frame - reid_end_frame
        #start = perturb_end_frame

        start = reid_end_frame
        #for i in range(num_loosens):

        undone = False
        while not undone:
            #annotate_reid = i==(num_loosens-1)
            loosen_end_frame, offset, pull_idx, hold_idx, action_vec = take_undo_action_oracle(params, \
                                            start, \
                                            render=render, \
                                            render_offset=render_offset, \
                                            )
            start = loosen_end_frame
            render_offset = offset
            undone = undone_check(pull_idx,hold_idx,action_vec)
        # Take one more loosening action for good measure
        loosen_end_frame, offset, pull_idx, hold_idx, action_vec = take_undo_action_oracle(params, \
                                            start, \
                                            render=render, \
                                            render_offset=render_offset, \
                                            )
        render_offset = offset
        #render_offset -= loosen_end_frame
        loosen_buffer = loosen_end_frame + 60
        for step in range(loosen_end_frame, loosen_buffer):
            bpy.context.scene.frame_set(step)
            render_frame(step, render_offset=render_offset, annot=True, annotate_undone_check=True, annotate_end_episode=(num_knots==1)) # Notice annotate_undone_check set to True
        render_offset -= loosen_buffer
        bpy.context.scene.frame_set(0)
        for a in bpy.data.actions:
            bpy.data.actions.remove(a)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_capsule_rope(params)
    #rig = rig_rope(params, braid=1)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    start = time.time()
    iters = 10
    generate_dataset(iters, params, render=True)
    end = time.time()
    print("time", end-start)
