import bpy
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('policies')
from untangle_utils import *
from knots import *
from rigidbody_rope import *
from oracle import Oracle

def run_untangling_rollout(policy, params):
    set_animation_settings(7000)
    #piece = "Cylinder"
    #last = params["num_segments"]-1

    knot_end_frame = tie_pretzel_knot(params, render=False)
    knot_end_frame = random_perturb(knot_end_frame, params)
    render_offset = knot_end_frame
    render_frame(knot_end_frame, render_offset=render_offset, step=1)

    reid_end = policy.reidemeister(knot_end_frame, render=True, render_offset=render_offset)
    undo_end_frame = reid_end

    while not policy.undone_check():
        undo_end_frame, pull, hold, action_vec = policy.undo(undo_end_frame, render=True, render_offset=render_offset)
    policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    policy = Oracle(params)
    clear_scene()
    make_capsule_rope(params)
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    run_untangling_rollout(policy, params)
