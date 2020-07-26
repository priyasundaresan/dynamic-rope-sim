import bpy
import numpy as np
import sys
from untangle_utils import *
from render import find_knot
from render_bbox import find_knot_cylinders
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))
from yumi_gripper import *

class Oracle(object):

    def __init__(self, params, use_grippers):
        self.action_count = 0
        self.max_actions = 10
        self.rope_length = params["num_segments"]
        self.num_knots = len(params["knots"])
        self.use_grippers = use_grippers

    def bbox_untangle(self, start_frame, render_offset=0):
        return find_knot(self.rope_length)[-1]  == [0,0,0], None

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, prev_action_vec, render_offset=0):
        if self.action_count > self.max_actions or find_knot(self.rope_length)[-1]  == [0,0,0]:
            return True
        end2_idx = self.rope_length-1
        end1_idx = -1
        ret = undone_check(start_frame, prev_pull, prev_hold, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)
        if ret:
            self.num_knots -= 1
        return ret

    def undo(self, start_frame, render=False, render_offset=0):
        idx_lists = find_knot_cylinders(self.rope_length, num_knots=self.num_knots)
        if self.num_knots > 1:
            idx_list1, idx_list2 = idx_lists
            knot_idx_list = idx_list1 if min(idx_list1) < min(idx_list2) else idx_list2 # find the right most knot
        else:
            knot_idx_list = idx_lists[0]
        pull_idx, hold_idx, action_vec = find_knot(self.rope_length, knot_idx=knot_idx_list)
        action_vec /= np.linalg.norm(action_vec)
        print(action_vec)
        end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, self.use_grippers, render=render, render_offset=render_offset)
        pull_pixel, hold_pixel = cyl_to_pixels([pull_idx, hold_idx])
        self.action_count += 1
        return end_frame, pull_pixel[0], hold_pixel[0], action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):

        middle_frame = reidemeister_right(start_frame, -1, self.rope_length-1, self.use_grippers, render=render, render_offset=render_offset)
        end_frame = reidemeister_left(middle_frame, -1, self.rope_length-1, self.use_grippers, render=render, render_offset=render_offset)
        self.action_count += 2
        return end_frame
