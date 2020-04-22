import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *
from sklearn.neighbors import NearestNeighbors

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
    if not os.path.exists("./images_depth"):
        os.makedirs('./images_depth')
    else:
        os.system('rm -r ./images_depth')
        os.makedirs('./images_depth')
    if not os.path.exists("./image_masks"):
        os.makedirs('./image_masks')
    else:
        os.system('rm -r ./image_masks')
        os.makedirs('./image_masks')
    scene = bpy.context.scene
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    #scene.view_settings.exposure = 0.8
    if engine == 'BLENDER_WORKBENCH':
        #scene.render.display_mode
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_render_samples = 1

def annotate(frame, mapping, num_annotations, knot_only=True, offset=1):
# def annotate(frame, mapping, num_annotations, knot_only=False, offset=1, bias_knot=3):
    # knot_only = True:  means only record the under, over crossings
    # knot_only = False:  means record annotations for full rope
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    pixels = []
    if knot_only:
        pull, hold, _ = find_knot(50)
        indices = list(range(pull-offset, pull+offset+1)) + list(range(hold-offset, hold+offset+1))
        knot_indices = indices
        bias_knot = 1 # no bias
    else:
        indices = list(range(50))
        pull, hold, _ = find_knot(50)
        offset = 3
        knot_indices = list(range(pull-offset, pull+offset+1)) + list(range(hold-offset, hold+offset+1))
        # for _ in range(bias_knot):
        #     indices.extend(knot_indices)
    for i in indices:
        cyl = get_piece("Cylinder", i if i != 0 else -1)
        cyl_verts = list(cyl.data.vertices)
        step_size = len(indices)*len(cyl_verts)//num_annotations
        if i in knot_indices:
            step_size = int(step_size/bias_knot)
        # else:
        #     step_size = int(step_size*bias_knot/2)
        vertex_coords = [cyl.matrix_world @ v.co for v in cyl_verts][np.random.randint(step_size)::step_size]
        # vertex_coords = [cyl.matrix_world @ v.co for v in cyl_verts][::step_size]
        for i in range(len(vertex_coords)):
            v = vertex_coords[i]
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
            pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
            pixels.append([pixel])
    mapping[frame] = pixels

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

def find_knot(num_segments, chain=False, depth_thresh=0.4, idx_thresh=3, pull_offset=3):

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
            action_vec = [7*dx, 7*dy, 6] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
    return -1, None, [0,0,0] # Didn't find a pull/hold

def center_camera(randomize=True):
    # move camera to above knot location as crop
    knot_pull, knot_hold, _ = find_knot(50)
    if knot_pull is -1 or knot_hold is None:
        return
    hold_cyl = get_piece("Cylinder", -1 if knot_hold == 0 else knot_hold)
    pull_cyl = get_piece("Cylinder", -1 if knot_pull == 0 else knot_pull)
    hold_loc = hold_cyl.matrix_world.translation
    pull_loc = pull_cyl.matrix_world.translation
    # pull_loc = hold_cyl.matrix_world.translation
    camera_x = (hold_loc[0] + pull_loc[0])/2
    camera_y = (hold_loc[1] + pull_loc[1])/2
    camera_z = 1
    # camera_z = 28
    offset = 0.1
    dx = np.random.uniform(-offset, offset) if randomize else 0
    dy = np.random.uniform(-offset, offset) if randomize else 0
    dz = np.random.uniform(-0.2, 0) if randomize else 0
    bpy.context.scene.camera.rotation_euler = (0, 0, 0)

    # reset camera location:
    bpy.context.scene.camera.location = (camera_x+dx, camera_y+dy, camera_z+dz)
    return

def render_frame(frame, render_offset=0, step=2, num_annotations=400, filename="%06d_rgb.png", folder="images", annot=True, mapping=None):
    # Renders a single frame in a sequence (if frame%step == 0)
    frame -= render_offset
    center_camera()
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

def tie_knot(params, chain=False, render=False):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)
    for i in range(last+1):
        obj = get_piece(piece, i if i != 0 else -1)
        take_action(obj, 1, (0,0,0), animate=(i==0 or i==last))

    # Wrap endpoint one circularly around endpoint 2
    take_action(end2, 80, (10,0,0))
    take_action(end1, 80, (-15,5,0))
    take_action(end1, 120, (-1,-7,0))
    take_action(end1, 150, (3,0,-4))
    take_action(end1, 170, (0,2.5,0))

    # Thread endpoint 1 through the loop (downward)
    take_action(end1, 180, (0,0,-2))

    # Pull to tighten knot
    take_action(end1, 200, (5,0,2))
    take_action(end2, 200, (0,0,0))
    take_action(end1, 230, (7,0,5))
    take_action(end2, 230, (-7,0,0))

    # Now, we "drop" the rope; no longer animated and will move only based on rigid body physics
    toggle_animation(end1, 240, False)
    toggle_animation(end2, 240, False)

    ## Reidemeister
    for step in range(1, 350):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step)
    return 350

def reidemeister(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1
    end1 = get_piece(piece, -1)
    end2 = get_piece(piece, last)

    middle_frame = start_frame+25
    end_frame = start_frame+50
    end1_first = random.random() > 0.5
    if end1_first:
        take_action(end1, middle_frame, (11-end1.matrix_world.translation[0],0,0))
    else:
        take_action(end2, middle_frame, (-9-end2.matrix_world.translation[0],0,0))
    for step in range(start_frame, middle_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
    if end1_first:
        take_action(end2, end_frame, (-9-end2.matrix_world.translation[0],0,0))
    else:
        take_action(end1, end_frame, (11-end1.matrix_world.translation[0],0,0))

    # Drop the ends
    toggle_animation(end1, middle_frame, False)
    toggle_animation(end2, end_frame, False)

    for step in range(middle_frame, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
    return end_frame

def random_loosen(params, start_frame, render=False, render_offset=0, annot=True, mapping=None):

    piece = "Cylinder"
    last = params["num_segments"]-1

    pick, hold, _ = find_knot(params["num_segments"])
    if random.random() < 0.5:
        pick = random.choice(range(10, 40))
    pull_cyl = get_piece(piece, pick)
    hold_cyl = get_piece(piece, hold)

    dx = np.random.uniform(0,1)*random.choice((-1,1))
    dy = np.random.uniform(0,1)*random.choice((-1,1))
    #dz = np.random.uniform(0.5,1)
    # dz = np.random.uniform(0.75, 2.25)
    dz = np.random.uniform(0.75,1.5)

    mid_frame = start_frame + 50
    end_frame = start_frame + 100

    take_action(hold_cyl, mid_frame, (0,0,0))
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)

    take_action(pull_cyl, mid_frame, (dx,dy,dz))
    toggle_animation(pull_cyl, mid_frame, False)
    toggle_animation(hold_cyl, mid_frame, False)
    for step in range(start_frame + 10, end_frame):
        bpy.context.scene.frame_set(step)
        if render:
            render_frame(step, render_offset=render_offset, annot=annot, mapping=mapping)
    return end_frame

def generate_dataset(params, iters=1, chain=False, render=False):

    set_animation_settings(7000)
    piece = "Cylinder"
    last = params["num_segments"]-1
    mapping = {}

    knot_end_frame = tie_knot(params, render=False)

    reid_start = knot_end_frame
    for i in range(iters):
    # NOTE: each iteration renders 75 images, ~45 is about 3500 images for generating a training dset
        reid_end_frame = reidemeister(params, reid_start, render=render, render_offset=knot_end_frame, mapping=mapping)
        reid_start = random_loosen(params, reid_end_frame, render=render, render_offset=knot_end_frame, mapping=mapping)

    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(mapping, outfile, sort_keys=True, indent=2)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    generate_dataset(params, iters=40, render=True)
    # generate_dataset(params, render=True)
