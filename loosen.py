import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

def find_knot_cluster(params, chain=False):
    # this returns a pick and hold point for a rope with a single knot by clustering
    # the depths of the cylinders to find two points
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1

    depths = {}
    for i in range(last):
        piece_name = '%s.%03d'%(piece, i) if i != 0 else piece
        depths[str(i)] = bpy.data.objects[piece_name].matrix_world.translation[2]

    occlusions = []
    for d in depths.keys():
        # threshold = np.mean([h for h in depths.values()])
        heights = [h for h in depths.values()]
        threshold = max(heights) - (max(heights) - np.mean(heights))/4
        if depths[d] > threshold:
            occlusions.append(int(d))

    clus = KMeans(n_clusters=2)
    clus.fit(np.array(occlusions).reshape((-1, 1)))
    labels = clus.labels_

    pick_and_hold = []
    for l in np.unique(labels):
        l_pieces = [occlusions[i] for i in range(len(labels)) if labels[i] == l]
        l_depths = [depths[str(i)] for i in l_pieces]
        for piece_index in l_pieces:
            if depths[str(piece_index)] == np.max(l_depths):
                pick_and_hold.append(piece_index)
    hold = max(pick_and_hold)
    pick = min(pick_and_hold)
    print(pick, hold)
    return pick, hold

def find_knot(params, chain=False):
    # this returns a pick and hold point for a rope with a single knot by following
    # the rope to find all cylinders in the knot, then pick the 1/4 and 3/4th points
    # as the pick and hold
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1

    knot_radius = 2
    depths = {}
    highest_cylinder = None
    highest_z = -100000
    for i in range(last):
        piece_name = '%s.%03d'%(piece, i) if i != 0 else piece
        depth = bpy.data.objects[piece_name].matrix_world.translation[2]
        depths[str(i)] = depth
        if depth > highest_z:
            highest_z = depth
            highest_cylinder = i

    highest_name = '%s.%03d'%(piece, highest_cylinder) if highest_cylinder != 0 else piece
    highest_cyl_x = bpy.data.objects[highest_name].matrix_world.translation[0]
    highest_cyl_y = bpy.data.objects[highest_name].matrix_world.translation[1]
    knot_cylinders = []
    for d in depths.keys():
        piece_name = '%s.%03d'%(piece, int(d)) if int(d) != 0 else piece
        x = bpy.data.objects[piece_name].matrix_world.translation[0]
        y = bpy.data.objects[piece_name].matrix_world.translation[1]
        if ((x - highest_cyl_x)**2 + (y - highest_cyl_y)**2)**0.5 < knot_radius:
            knot_cylinders.append(int(d))

    index1 = int(len(knot_cylinders)/4)
    index2 = int(3*len(knot_cylinders)/4)
    pick_and_hold = [knot_cylinders[index1], knot_cylinders[index2]]
    hold = max(pick_and_hold)
    pick = min(pick_and_hold)
    print(pick, hold)
    return pick, hold

def knot_test(params, chain=False):
    piece = "Torus" if chain else "Cylinder"
    last = 2**(params["chain_len"]+1)-1 if chain else params["num_segments"]-1
    anim_end = 800
    scene = bpy.context.scene
    scene.frame_end = anim_end
    scene.rigidbody_world.point_cache.frame_end = anim_end

    end1 = bpy.data.objects[piece]
    end2 = bpy.data.objects['%s.%03d'%(piece, last)]

    for i in range(last):
        if i != 0:
            bpy.data.objects['%s.%03d'%(piece, i)].keyframe_insert(data_path="rigid_body.kinematic", frame=1)
            bpy.data.objects['%s.%03d'%(piece, i)].keyframe_insert(data_path="location", frame=1)
    # Set up cylinders that will be used to undo a crossing
    # pull_cyl = bpy.data.objects['%s.015'%(piece)]
    # # pull_cyl = bpy.data.objects['%s.%03d'%(piece, pick)]
    # pull_cyl.rigid_body.kinematic = False
    # pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # pull_cyl.keyframe_insert(data_path="location", frame=1)
    #
    # hold_cyl = bpy.data.objects['%s.026'%(piece)]
    # # hold_cyl = bpy.data.objects['%s.%03d'%(piece, hold)]
    # hold_cyl.rigid_body.kinematic = False
    # hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # hold_cyl.keyframe_insert(data_path="location", frame=1)

    # Allow endpoints to be keyframe-animated at the start
    end1.rigid_body.kinematic = True # This means end1 is manually animable
    end2.rigid_body.kinematic = True
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=1)

    # Pin the two endpoints initially
    end1.keyframe_insert(data_path="location", frame=1)
    end2.keyframe_insert(data_path="location", frame=1)

    # Wrap endpoint one circularly around endpoint 2
    end2.location[0] += 10
    end1.location[0] -= 15
    end1.location[1] += 5
    end1.keyframe_insert(data_path="location", frame=80)
    end2.keyframe_insert(data_path="location", frame=80)
    end1.location[0] -= 1
    end1.location[1] -= 7
    end1.keyframe_insert(data_path="location", frame=120)
    end1.location[0] += 3
    end1.location[2] -= 4
    end1.keyframe_insert(data_path="location", frame=150)
    end1.location[1] += 2.5
    end1.keyframe_insert(data_path="location", frame=170)

    # Thread endpoint 1 through the loop (downward)
    end1.location[2] -= 2
    end1.keyframe_insert(data_path="location", frame=180)

    # Pull to tighten knot
    end1.location[0] += 5
    end1.location[2] += 2
    end1.keyframe_insert(data_path="location", frame=200)
    end2.keyframe_insert(data_path="location", frame=200)

    end1.location[2] += 5
    end1.location[0] += 7
    end2.location[0] -= 7
    end1.keyframe_insert(data_path="location", frame=230)
    end2.keyframe_insert(data_path="location", frame=230)

    # Now, we "drop" the rope; no longer animated and will move only based on rigid body physics
    end1.rigid_body.kinematic = False # This means end1 is not animable, will move based on physics solver
    end2.rigid_body.kinematic = False
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=240)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=240)

    # Now, I want to try re-picking up the end of the rope after it's been dropped
    # Turns out this is a little tricky because we need to know where end1 ended up after the drop happened
    # Otherwise, when you just translate end1 and keyframe it, it thinks its still at its pre-drop position
    # so end1 first quickly jumps from its post-drop position to pre-drop pose and then does the motion
    # This is the "snapping" effect where it quickly jumps to an outdated location (like the softbody!!)
    # We really just want it to move from its post-drop position to the translation

    # Workaround: I pick  a frame when I think the rope has settled, get the updated location of end1, and
    # set it back to kinematic (controllable by animation)

    # Note: I step through the animation up to frame 350 cuz that's about when the rope settles after being dropped
    # I had to use end1.matrix_world.translation to get the updated world coordinate of end1 as the sim progresses
    # because end1.location does NOT give the up-to-date location taking into account physics

    for step in range(1, 351):
        bpy.context.scene.frame_set(step)
        #print(end1.matrix_world.translation) # does update properly :)
        #print(end1.location) # does NOT update properly

    # Reidemeister
    end1.rigid_body.kinematic = True
    # Next line is critical - without it, the rope "snaps" back to starting position at frame 1 because its location is not up to date with how the simulation progressed after the drop; try uncommmenting to see what I mean
    end1.location = end1.matrix_world.translation
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=350)
    end1.keyframe_insert(data_path="location", frame=350)
    end1.location[0] = 9
    end1.keyframe_insert(data_path="location", frame=375)

    end2.rigid_body.kinematic = True
    end2.location = end2.matrix_world.translation
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=375)
    end2.keyframe_insert(data_path="location", frame=375)
    end2.location[0] = -9
    end2.keyframe_insert(data_path="location", frame=400)

    end1.rigid_body.kinematic = False
    end2.rigid_body.kinematic = False
    end1.keyframe_insert(data_path="rigid_body.kinematic",frame=400)
    end2.keyframe_insert(data_path="rigid_body.kinematic",frame=400)

    for step in range(351, 401):
        bpy.context.scene.frame_set(step)

    pick, hold = find_knot(params, chain=chain)
    # pull_cyl = bpy.data.objects['%s.015'%(piece)]
    pull_cyl = bpy.data.objects['%s.%03d'%(piece, pick)]
    pull_cyl.rigid_body.kinematic = False
    # pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # pull_cyl.keyframe_insert(data_path="location", frame=1)

    # hold_cyl = bpy.data.objects['%s.026'%(piece)]
    hold_cyl = bpy.data.objects['%s.%03d'%(piece, hold)]
    hold_cyl.rigid_body.kinematic = False
    # hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=1)
    # hold_cyl.keyframe_insert(data_path="location", frame=1)

    # Undoing
    hold_cyl.rigid_body.kinematic = True
    hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=400)
    hold_cyl.location = hold_cyl.matrix_world.translation
    # We want the hold cylinder to stay in place during the pull
    hold_cyl.keyframe_insert(data_path="location", frame=400)
    hold_cyl.keyframe_insert(data_path="location", frame=500)

    # Doing this because if I try to keyframe the location of hold and pull at frame 400, it causes "snapping"
    for step in range(401, 410):
        bpy.context.scene.frame_set(step)

    pull_cyl.rigid_body.kinematic = True
    pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=410)
    pull_cyl.location = pull_cyl.matrix_world.translation
    pull_cyl.keyframe_insert(data_path="location", frame=410)
    # Pull
    pull_cyl.location[2] += 3
    pull_cyl.location[0] -= 6
    pull_cyl.location[1] -= 2
    pull_cyl.keyframe_insert(data_path="location", frame=500)

    # Release both pull, hold
    pull_cyl.rigid_body.kinematic = False
    pull_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=500)
    hold_cyl.rigid_body.kinematic = False
    hold_cyl.keyframe_insert(data_path="rigid_body.kinematic",frame=500)


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
    knot_test(params)
