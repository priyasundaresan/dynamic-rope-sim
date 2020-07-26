import bpy
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('policies')
from untangle_utils import *
from knots import *
from rigidbody_rope import *

from sklearn.neighbors import NearestNeighbors

from yumi_gripper import *

# Load policies
from oracle import Oracle
#from hierarchical_descriptors import Hierarchical
#from baseline import Heuristic
#from random_action import RandomAction
#from hierarchical_keypoints import Hierarchical_kp

def run_untangling_rollout(policy, params):
    set_animation_settings(15000)
    #piece = "Cylinder"
    #last = params["num_segments"]-1

    knot_end_frame = 0
    num_knots = len(params["knots"])
    if num_knots == 1:
        if params["knots"][0] == "pretzel":
            knot_end_frame = tie_pretzel_knot(params, render=False)
        elif params["knots"][0] == "fig8":
            knot_end_frame = tie_figure_eight(params, render=False)
    elif num_knots == 2:
        if "pretzel" in params["knots"] and not "fig8" in params["knots"]:
            knot_end_frame = tie_double_pretzel(params, render=True)
        elif "fig8" in params["knots"] and not "pretzel" in params["knots"]:
            raise Exception("Double Figure 8 config not yet supported")
        else:
            raise Exception("Figure 8 and Pretzel config not yet supported")
    else:
        raise Exception("More than 2 knot configs not yet supported")

    knot_end_frame = random_perturb(knot_end_frame, params)
    render_offset = knot_end_frame
    render_frame(knot_end_frame, render_offset=render_offset, step=1)

    num_actions = 0

    reid_end = policy.reidemeister(knot_end_frame, render=True, render_offset=render_offset)
    num_actions += 1
    undo_end_frame = reid_end

    bbox, _ = policy.bbox_untangle(undo_end_frame, render_offset=render_offset)
    while bbox is not None:
        undone = False
        ### TESTING 
        undo_end, pull, hold, action_vec = policy.undo(undo_end_frame, render=True, render_offset=render_offset)
        undone = policy.policy_undone_check(undo_end, pull, hold, action_vec, render_offset=render_offset)
        undo_end_frame = undo_end
        num_actions += 1
        ### 
        # i = 0
        # #while not undone and i < 10:
        # while not undone and i < 2:
        #     try: # if rope goes out of frame, take a reid move
        #         undo_end, pull, hold, action_vec = policy.undo(undo_end_frame, render=True, render_offset=render_offset)
        #         undone = policy.policy_undone_check(undo_end, pull, hold, action_vec, render_offset=render_offset)
        #         undo_end_frame = undo_end
        #         num_actions += 1
        #     except:
        #         undo_end_frame = policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)
        #         num_actions += 1
        #     i += 1
        #     if num_actions == 29:
        #         undo_end_frame = policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)
        #         num_actions += 1
        #         return
        # undo_end_frame = policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)
        # num_actions += 1
        # bbox, _ = policy.bbox_untangle(undo_end_frame, render_offset=render_offset)
        return #TESTING

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_capsule_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)

    use_grippers = True

    if use_grippers:
        print("MAKING GRIPPER")
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

        policy = Oracle(params, use_grippers)
    else:
        policy = Oracle(params)
        # policy = Hierarchical(path_to_refs, DESCRIPTOR_DIR, BBOX_DIR, params)
        # policy = Heuristic(path_to_refs, BBOX_DIR, params)
        # policy = RandomAction(path_to_refs, BBOX_DIR, params)
        # policy = Hierarchical_kp(path_to_refs, KP_DIR, BBOX_DIR, params)

    run_untangling_rollout(policy, params)
