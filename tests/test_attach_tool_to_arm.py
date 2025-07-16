import genesis as gs
import numpy as np

# import argparse
from genesis.engine.entities import RigidEntity
import torch
from PIL import Image


def main():
    from repairs_components.logic.tools.screwdriver import Screwdriver
    from repairs_components.logic.tools.tool import attach_tool_to_arm

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--vis", action="store_true", default=False)
    # parser.add_argument("-c", "--cpu", action="store_true", default=False)
    # args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    cube_2: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.4, 0.2, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    camera = scene.add_camera(
        pos=(1.0, 2.5, 3.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
    )
    scene.build()
    camera.start_recording()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.5]),
        quat=np.array([0, 1, 0, 0]),  # downward facing orientation.
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,  # 1s duration
    )
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        scene.step()
        camera.render()

    # allow robot to reach the last waypoint
    for i in range(100):
        scene.step()
        camera.render()

    print("end_effector.get_pos(): ", end_effector.get_pos())
    print("end_effector.get_quat(): ", end_effector.get_quat())

    # tested func.
    attach_tool_to_arm(scene, cube, franka, Screwdriver(), None)
    rgb, _, _, _ = camera.render()
    Image.fromarray(rgb).save("cube_tool.png")

    # camera.stop_recording(
    #     save_to_filename=f"/workspace/RepairsComponents-v0/video_reposition_debug.mp4",
    #     fps=60,
    # )

    assert torch.isclose(
        cube.get_pos(), torch.tensor([0.65, 0.0, 0.5 - 0.3]), atol=0.05
    ).all(), f"Cube pos expected to be [0.65, 0.0, 0.5 - 0.3], got {cube.get_pos()}"

    # move to
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.65, 0.5]),  # rotate 90 deg
        quat=np.array([0, 1, 0, 0]),  # downward facing orientation.
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,  # 1s duration
    )
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        scene.step()
        camera.render()

    # allow robot to reach the last waypoint
    for i in range(100):
        # note: control_dofs_position is set implicitly, so this reaches
        # the last waypoint under the last configuration.
        scene.step()
        camera.render()

    camera.stop_recording(
        save_to_filename=f"/workspace/RepairsComponents-v0/video_reposition_debug.mp4",
        fps=60,
    )

    assert torch.isclose(
        cube.get_pos(), torch.tensor([0.0, 0.65, 0.5 - 0.3]), atol=0.05
    ).all(), f"Cube pos expected to be [0.0, 0.65, 0.5 - 0.3], got {cube.get_pos()}"


if __name__ == "__main__":
    main()
