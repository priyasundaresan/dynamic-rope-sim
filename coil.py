import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

def coil_test(params, chain=False):

    # Allow endpoints to be keyframe-animated
    if not chain:
        end1 = bpy.data.objects['Cylinder']
    else:
        end1 = bpy.data.objects['Torus']
    end1.rigid_body.enabled = False
    end1.rigid_body.kinematic = True
    end1.keyframe_insert(data_path="location", frame=1)

    anim_start = 120
    anim_end = 600
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

    # Helix equation:
    # x = rcos(t), y = rsin(t), z = ct
    r = 1.15
    c = -0.75
    start_height = 19
    t0 = 0
    tn = 10*np.pi
    timesteps = 50
    for t in np.linspace(t0, tn, timesteps):
        x = r*np.cos(t)
        y = r*np.sin(t)
        z = c*t + start_height
        end1.location = x,y,z
        end1.keyframe_insert(data_path="location", frame=anim_start+(t+1)*(float(scene.frame_end-anim_start)/timesteps))

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    make_rope(params)
    make_table(params)
    coil_test(params, chain=False)
