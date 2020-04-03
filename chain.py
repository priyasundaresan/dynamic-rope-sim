import bpy
import numpy as np

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from rigidbody_rope import *

def make_chain(params):
    '''
    Join multiple toruses together to make chain shape (http://jayanam.com/chains-with-blender-tutorial/)
    TODOS: fix torus vertex selection to make oval shape
           fix ARRAY selection after making empty mesh
    '''
    # hacky fix from https://www.reddit.com/r/blenderhelp/comments/dnb56f/rendering_python_script_in_background/
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.screen.screen_full_area(override)
                break
    scale = params["chain_scale"]
    chain_len = params["chain_len"] # chain length doubles for every 1 of chain_len
    center_start = (1.8*scale*(2**(chain_len+1))/2) + 2
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), major_radius=1*scale, minor_radius=0.25*scale, abso_major_rad=1.25*scale, abso_minor_rad=0.75*scale)
    # bpy.ops.object.editmode_toggle()
    link = bpy.context.active_object
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    verts_to_move = [0] + list(range(12, 288))
    for i in verts_to_move:
        link.data.vertices[i].select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    # bpy.ops.transform.translate(value=(0, 1, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.transform.translate(value=(0*scale, 1*scale, 0*scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.editmode_toggle()

    first_link = bpy.context.active_object
    link_len = 1.9 * scale

    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, link_len, 0), "orient_type":'LOCAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'LOCAL', "constraint_axis":(False, True, False), "mirror":True, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":False, "use_accurate":False})
    bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

    for i in range(chain_len):
        bpy.ops.object.select_all(action='SELECT')
        move_len = 2**i * link_len * 2
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, move_len, 0), "orient_type":'LOCAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'LOCAL', "constraint_axis":(False, True, False), "mirror":True, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":False, "use_accurate":False})

    bpy.ops.object.select_all(action='SELECT')
    # first_link.select_set(True)
    # bpy.context.scene.context = 'PHYSICS'
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.object_settings_copy()

    bpy.ops.transform.translate(value=(0, -center_start, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 1000

    # bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0*scale, 1.8*scale, 0*scale))
    # bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    #
    # bpy.context.view_layer.objects.active = self.rope
    # bpy.ops.object.modifier_add(type='ARRAY')
    # bpy.context.object.modifiers["Array"].use_relative_offset = False
    # bpy.context.object.modifiers["Array"].use_object_offset = True
    # bpy.context.object.modifiers["Array"].offset_object = bpy.data.objects["Empty"]
    # self.rope_asymm.modifiers["Array"].count = 100
    # bpy.ops.object.modifier_add(type='CURVE')
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.join()

def chain_knot_test(params):
    # Makes a knot by a hardcoded trajectory
    # Press Spacebar once the Blender script loads to run the animation
    end2 = bpy.data.objects['Torus']
    end1 = bpy.data.objects['Torus.%03d'%((2**(params["chain_len"]+1))-1)]

    # Allow endpoints to be keyframe-animated
    end1.rigid_body.enabled = False
    end1.rigid_body.kinematic = True
    end2.rigid_body.enabled = False
    end2.rigid_body.kinematic = True

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

    end1.location[0] += 7
    end1.location[2] += 5
    end2.location[0] -= 7
    end1.keyframe_insert(data_path="location", frame=230)
    end2.keyframe_insert(data_path="location", frame=230)



if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    #make_rope(params)
    make_chain(params)
    make_table(params)
    chain_knot_test(params)
    #knot_test(params)
    # knot_test(params, chain=True)
    # coil_test(params)
    # coil_test(params, chain=True)
