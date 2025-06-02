from genesis.engine.entities.rigid_entity import RigidEntity
import torch
import genesis as gs
from genesis.vis.camera import Camera


def execute_straight_line_trajectory(
    franka: RigidEntity,
    scene: gs.Scene,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    gripper_force: torch.Tensor,
    keypoint_distance=0.1,
    num_steps_between_keypoints=10,
    camera: Camera | None = None,
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
        keypoint_distance: Distance between keypoints in meters (default: 0.1m or 10cm)
        num_steps_between_keypoints: Number of interpolation steps between keypoints

    Returns:
        None: The function directly executes the trajectory in the simulation
    """
    device = target_pos.device

    current_end_effector_pos = torch.tensor(franka.get_link("hand").pos, device=device)
    current_end_effector_quat = torch.tensor(
        franka.get_link("hand").quat, device=device
    )

    alpha = torch.linspace(0, 1, num_steps_between_keypoints + 1, device=device)

    for a in alpha:
        keypoint_ik = franka.inverse_kinematics(  # wait, why would I want to do it here? isn't IK done once?
            link=franka.get_link("hand"),
            pos=(
                current_end_effector_pos + (target_pos - current_end_effector_pos) * a
            )[None],
            quat=(
                current_end_effector_quat
                + (target_quat - current_end_effector_quat) * a
            )[None],  # expand dim.
            # init_qpos=current_state,
        )
        franka.control_dofs_position(keypoint_ik)
        franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
        scene.step()
        if camera is not None:
            camera.render()

    # let dry-run for 80 steps.
    for _ in range(100):
        scene.step()  # let it actually run.
        franka.control_dofs_position(keypoint_ik)  # set at the last point.
        franka.control_dofs_force(gripper_force, dofs_idx_local=[7, 8])
        if camera is not None:
            camera.render()

    print(franka.get_links_pos()[:, 7])  # hand.
