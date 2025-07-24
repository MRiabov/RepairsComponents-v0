from typing_extensions import deprecated
from genesis.engine.entities.rigid_entity import RigidEntity
import torch
import genesis as gs
from genesis.vis.camera import Camera
from torch.nn import functional as F


@deprecated("Batch motion planning was merged in genesis!")
def execute_straight_line_trajectory(
    franka: RigidEntity,
    scene: gs.Scene,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    gripper_force: torch.Tensor,
    render: bool,
    keypoint_distance=0.1,
    num_steps_between_keypoints=10,
):
    """
    Execute a straight-line trajectory for a robot arm in Cartesian space.

    This implements a simple motion planning approach:
    1. Creates a straight line from current position to target in Cartesian space
    2. Generates evenly spaced keypoints along that line
    3. Computes IK for each keypoint
    4. Interpolates in joint space between keypoints

    Args:
        franka: The Franka robot entity
        scene: The simulation scene
        pos: Target position tensor with shape (num_envs, 3)
        quat: Target orientation quaternion tensor with shape (num_envs, 4)
        gripper_force: Force in newtons applied to gripper fingers.
        render: Whether to render the trajectory to store for visualization. Will not affect training, but will slow it down.
        keypoint_distance: Distance between keypoints in meters (default: 0.1m or 10cm)
        num_steps_between_keypoints: Number of interpolation steps between keypoints

    Returns:
        None: The function directly executes the trajectory in the simulation
    """

    assert gripper_force.shape[-1] == 2, (
        f"Gripper force shape must be (num_envs, 2) or (2,). Currently: {gripper_force.shape}"
    )
    assert target_pos.shape[-1] == 3, (
        f"Target position shape must be (num_envs, 3) or (3,). Currently: {target_pos.shape}"
    )
    assert target_quat.shape[-1] == 4, (
        f"Target quaternion shape must be (num_envs, 4) or (4,). Currently: {target_quat.shape}"
    )
    device = target_pos.device
    # normalize quat in case it wasn't already
    target_quat = F.normalize(target_quat, dim=-1, eps=1e-6)

    current_end_effector_pos = torch.tensor(franka.get_link("hand").pos, device=device)
    current_end_effector_quat = torch.tensor(
        franka.get_link("hand").quat, device=device
    )

    alpha = torch.linspace(0, 1, num_steps_between_keypoints + 1, device=device)

    for a in alpha:
        # print(
        #     "target_pos.shape",
        #     target_pos.shape,
        #     "target_quat.shape",
        #     target_quat.shape,
        #     "current_end_effector_pos.shape",
        #     current_end_effector_pos.shape,
        #     "current_end_effector_quat.shape",
        #     current_end_effector_quat.shape,
        # )
        keypoint_ik = franka.inverse_kinematics(  # wait, why would I want to do it here? isn't IK done once?
            link=franka.get_link("hand"),
            pos=(
                current_end_effector_pos + (target_pos - current_end_effector_pos) * a
            ),  # [None]
            quat=(
                current_end_effector_quat
                + (target_quat - current_end_effector_quat) * a
            ),  # [None]  # expand dim.# it was a bug somewhere in tests.
            # init_qpos=current_state,
        )
        # print("keypoint_ik.shape", keypoint_ik.shape)
        franka.control_dofs_position(keypoint_ik)
        franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
        scene.step(update_visualizer=render, refresh_visualizer=render)

        if render:
            cameras = scene.visualizer.cameras
            for camera in cameras:
                camera.render()

    # let dry-run for 10 steps.
    for _ in range(10):  # note was 100, but reduced to 10 for debug.
        scene.step(
            update_visualizer=render, refresh_visualizer=render
        )  # let it actually run.
        franka.control_dofs_position(keypoint_ik)  # set at the last point.
        franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
        if render:
            cameras = scene.visualizer.cameras
            for camera in cameras:
                camera.render()

    # print(franka.get_links_pos()[:, 7])  # hand.


def execute_planned_path(
    franka: RigidEntity,
    scene: gs.Scene,
    target_hand_pos: torch.Tensor,
    target_hand_quat: torch.Tensor,
    gripper_force: torch.Tensor,
    render: bool,
    keypoint_distance=0.1,
    num_steps_between_keypoints=10,
):
    hand = franka.get_link("hand")
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=hand,
        pos=target_hand_pos,
        quat=target_hand_quat,
    )
    # gripper open pos
    qpos[-2:] = gripper_force
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=100)
    # 1s duration
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step(update_visualizer=render, refresh_visualizer=render)
        if render:
            cameras = scene.visualizer.cameras
            for camera in cameras:
                camera.render()

    # dry run not added (?)

    if render:
        cameras = scene.visualizer.cameras
        for camera in cameras:
            camera.stop_recording(
                save_to_filename="test_tool_genesis_bug_repro.mp4",
            )


# attempt to convert to ti kernel (failed - ti kernels don't support python/torch operations.)
# def execute_straight_line_trajectory(
#     franka: RigidEntity,
#     scene: gs.Scene,
#     target_pos: torch.Tensor,
#     target_quat: torch.Tensor,
#     gripper_force: torch.Tensor,
#     render: bool,
#     keypoint_distance=0.1,
#     num_steps_between_keypoints=10,
# ):
#     """
#     Execute a straight-line trajectory for a robot arm in Cartesian space.

#     This implements a simple motion planning approach:
#     1. Creates a straight line from current position to target in Cartesian space
#     2. Generates evenly spaced keypoints along that line
#     3. Computes IK for each keypoint
#     4. Interpolates in joint space between keypoints

#     Args:
#         franka: The Franka robot entity
#         scene: The simulation scene
#         pos: Target position tensor with shape (num_envs, 3)
#         quat: Target orientation quaternion tensor with shape (num_envs, 4)
#         gripper_force: Force in newtons applied to gripper fingers.
#         render: Whether to render the trajectory to store for visualization. Will not affect training, but will slow it down.
#         keypoint_distance: Distance between keypoints in meters (default: 0.1m or 10cm)
#         num_steps_between_keypoints: Number of interpolation steps between keypoints

#     Returns:
#         None: The function directly executes the trajectory in the simulation
#     """

#     assert gripper_force.shape[-1] == 2, (
#         f"Gripper force shape must be (num_envs, 2) or (2,). Currently: {gripper_force.shape}"
#     )
#     assert target_pos.shape[-1] == 3, (
#         f"Target position shape must be (num_envs, 3) or (3,). Currently: {target_pos.shape}"
#     )
#     assert target_quat.shape[-1] == 4, (
#         f"Target quaternion shape must be (num_envs, 4) or (4,). Currently: {target_quat.shape}"
#     )
#     device = target_pos.device

#     current_end_effector_pos = torch.tensor(franka.get_link("hand").pos, device=device)
#     current_end_effector_quat = torch.tensor(
#         franka.get_link("hand").quat, device=device
#     )

#     alpha = torch.linspace(0, 1, num_steps_between_keypoints + 1)

#     # Precompute interpolated positions and quaternions for each alpha step
#     precomputed_pos = torch.stack(
#         [
#             current_end_effector_pos + (target_pos[0] - current_end_effector_pos) * a
#             for a in alpha
#         ]
#     )  # shape: (num_steps_between_keypoints + 1, 3)
#     precomputed_quat = torch.stack(
#         [
#             current_end_effector_quat + (target_quat[0] - current_end_effector_quat) * a
#             for a in alpha
#         ]
#     )  # shape: (num_steps_between_keypoints + 1, 4)

#     precomputed_pos_field = ti.Vector.field(
#         3, dtype=ti.f32, shape=(num_steps_between_keypoints + 1,)
#     )
#     precomputed_quat_field = ti.Vector.field(
#         4, dtype=ti.f32, shape=(num_steps_between_keypoints + 1,)
#     )
#     precomputed_pos_field.from_torch(precomputed_pos)
#     precomputed_quat_field.from_torch(precomputed_quat)

#     @ti.func
#     def ik_step(precomputed_pos: ti.template(), precomputed_quat: ti.template()):
#         keypoint_ik = franka.inverse_kinematics(
#             link=franka.get_link("hand"), pos=precomputed_pos, quat=precomputed_quat
#         )
#         franka.control_dofs_position(keypoint_ik)
#         franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
#         scene.step(update_visualizer=render, refresh_visualizer=render)

#         if render:
#             cameras = scene.visualizer.cameras
#             for camera in cameras:
#                 camera.render()

#     @ti.func
#     def dry_run_step(target_pos: ti.template(), target_quat: ti.template()):
#         scene.step(
#             update_visualizer=render, refresh_visualizer=render
#         )  # let it actually run.
#         franka.control_dofs_position(target_pos)  # set at the last point.
#         franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
#         if render:
#             cameras = scene.visualizer.cameras
#             for camera in cameras:
#                 camera.render()

#     @ti.kernel
#     def execute_ik(
#         precomputed_pos_field: ti.template(), precomputed_quat_field: ti.template()
#     ):
#         for i in range(num_steps_between_keypoints + 1):
#             ik_step(precomputed_pos_field[i], precomputed_quat_field[i])

#         # let dry-run for 80 steps.
#         for _ in range(100):
#             dry_run_step(precomputed_pos_field[-1], precomputed_quat_field[-1])

#     execute_ik(precomputed_pos_field, precomputed_quat_field)

#     # print(franka.get_links_pos()[:, 7])  # hand.
