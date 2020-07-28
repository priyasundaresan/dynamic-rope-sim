import bpy
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('policies')
from untangle_utils import *
from knots import *
from rigidbody_rope import *

# Load policies
from multi_head import MultiHead

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
    undo_end_frame = knot_end_frame
    num_actions = 0

    terminate = False
    while not terminate and num_actions < 30:
        reid = policy.reid_check(undo_end_frame, render_offset=render_offset)
        if reid:
            undo_end_frame = policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)
        else:
            undo_end_frame = policy.undo(undo_end_frame, render=True, render_offset=render_offset)
        num_actions += 1
        terminate = policy.terminate_check(undo_end_frame, render_offset=render_offset)


if __name__ == '__main__':

    if not os.path.exists("./preds"):
        os.makedirs('./preds')
    else:
        os.system('rm -r ./preds')
        os.makedirs('./preds')

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)

    BASE_DIR = os.getcwd()
    DESCRIPTOR_DIR = os.path.join(BASE_DIR, 'dense_correspondence')
    KP_DIR = os.path.join(BASE_DIR, 'keypoints_dir')
    BBOX_DIR = os.path.join(BASE_DIR, 'mrcnn_bbox', 'networks')
    MH_DIR = os.path.join(BASE_DIR, 'keypoints_cls')
    path_to_refs = os.path.join(BASE_DIR, 'references', params["texture"])

    policy = MultiHead(path_to_refs, MH_DIR, params)

    clear_scene()
    make_capsule_rope(params)
    if not params["texture"] == "capsule":
        rig_rope(params, braid=params["texture"]=="braid")
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    run_untangling_rollout(policy, params)
