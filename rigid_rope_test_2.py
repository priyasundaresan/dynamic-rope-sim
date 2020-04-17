import bpy
from math import pi, radians
from mathutils import Matrix

def make_rope():
    num_segments = 75
    radius = 0.1
    separation = radius*1.1
    link_mass = 0.05
    link_friction = 1.5
    twist_stiffness = 100
    twist_damping = 10
    bend_stiffness = 0
    bend_damping = 5

    num_joints = int(radius/separation)*2+1
    loc0 = -1

    bpy.ops.import_mesh.stl(filepath="capsule_12_8_1_2.stl")
    link0 = bpy.context.object
    link0.name = "link_0"
    bpy.ops.transform.resize(value=(radius, radius, radius))
    link0.rotation_euler = (0, pi/2, 0)
    link0.location = (loc0, 0, 0)
    bpy.ops.rigidbody.object_add()
    link0.rigid_body.mass = link_mass
    link0.rigid_body.friction = link_friction
    link0.rigid_body.collision_shape = 'CAPSULE'
    
    links = [link0]
    for i in range(1,num_segments):
        linki = link0.copy()
        linki.data = link0.data.copy()
        linki.name = "link_" + str(i)
        linki.location = (loc0 + i*separation, 0, 0)
        bpy.context.collection.objects.link(linki)
        links.append(linki)

        bpy.ops.object.empty_add(type='ARROWS', radius=radius*2, location=(loc0 + (i-0.5)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-1) + ':' + str(i)
        rbc = joint.rigid_body_constraint
        rbc.object1 = links[i-1]
        rbc.object2 = links[i]
        rbc.use_limit_lin_x = True
        rbc.use_limit_lin_y = True
        rbc.use_limit_lin_z = True
        rbc.limit_lin_x_lower = 0
        rbc.limit_lin_x_upper = 0
        rbc.limit_lin_y_lower = 0
        rbc.limit_lin_y_upper = 0
        rbc.limit_lin_z_lower = 0
        rbc.limit_lin_z_upper = 0
        if twist_stiffness > 0 or twist_damping > 0:
            rbc.use_spring_ang_x = True
            rbc.spring_stiffness_ang_x = twist_stiffness
            rbc.spring_damping_ang_x = twist_damping
        if bend_stiffness > 0 or bend_damping > 0:
            rbc.use_spring_ang_y = True
            rbc.use_spring_ang_z = True
            rbc.spring_stiffness_ang_y = bend_stiffness
            rbc.spring_stiffness_ang_z = bend_stiffness
            rbc.spring_damping_ang_y = bend_damping
            rbc.spring_damping_ang_z = bend_damping

    for i in range(2, num_segments):
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=radius*1.5, location=(loc0 + (i-1)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-2) + ':' + str(i)
        joint.rigid_body_constraint.object1 = links[i-2]
        joint.rigid_body_constraint.object2 = links[i]

    return links

if "__main__" == __name__:
    frame_end = 300
    bpy.ops.object.delete(use_global=False)
    bpy.ops.rigidbody.world_add()
    bpy.context.scene.rigidbody_world.steps_per_second = 1000
    bpy.context.scene.rigidbody_world.solver_iterations = 100
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end
    links = make_rope()
    
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0,0,-5))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 0.7

    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, rotation=(pi/2, 0,  0), location=(2.75, 0, -1))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 0.7

    links[0].rigid_body.kinematic = True
    for i in range(1,frame_end):
        bpy.context.scene.frame_current = i
        links[0].rotation_euler = (links[0].rotation_euler.to_matrix() @ Matrix.Rotation(radians(5), 3, 'Z')).to_euler()
        links[0].keyframe_insert(data_path="rotation_euler", frame=i)

    bpy.context.scene.frame_current = 1


    
