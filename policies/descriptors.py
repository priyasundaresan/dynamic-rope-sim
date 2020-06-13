import bpy
import numpy as np
import sys
from untangle_utils import *
from render import find_knot
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))

class Hierarchical(object):
    def __init__(self, params):
        pass

    def undone_check(self):
        pass

    def undo(self, start_frame, render=False, render_offset=0):
        pass
    
    def reidemeister(self, start_frame, render=False, render_offset=0):
        pass
