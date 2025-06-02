import torch
import genesis as gs
from genesis.vis.camera import Camera


def execute_straight_line_trajectory(
    franka,
    scene,
    pos,
    quat,
    gripper,
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
        gripper: Gripper command tensor with shape (num_envs,) - 0 for close, 1 for open
        keypoint_distance: Distance between keypoints in meters (default: 0.1m or 10cm)
        num_steps_between_keypoints: Number of interpolation steps between keypoints

    Returns:
        None: The function directly executes the trajectory in the simulation
    """
    device = pos.device

    # Get the current joint positions
    current_state = franka.get_dofs_position()
    end_effector = franka.get_link("hand")

    current_qpos = current_state.clone()

    # Get the current end-effector pose
    current_pos = end_effector.pos
    current_quat = end_effector.quat

    # Target end-effector pose
    target_pos = pos
    target_quat = quat

    # Calculate the distance between current and target positions
    distance = torch.norm(target_pos - torch.from_numpy(current_pos).cuda())

    # Calculate number of keypoints needed
    num_keypoints = max(int(distance / keypoint_distance) + 1, 2)

    # Generate evenly spaced keypoints along the straight line
    alphas = torch.linspace(0, 1, num_keypoints, device=device)

    # Initialize list to store joint configurations for keypoints
    keypoint_qpos = []

    # Compute IK for each keypoint
    prev_qpos = current_qpos
    for alpha in alphas:
        # Interpolate position
        keypoint_pos = torch.from_numpy(current_pos).cuda() + alpha * (
            target_pos - torch.from_numpy(current_pos).cuda()
        )

        # Interpolate orientation (simple linear interpolation - could use slerp for better results)
        keypoint_quat = (1 - alpha) * torch.from_numpy(
            current_quat
        ).cuda() + alpha * target_quat
        keypoint_quat = keypoint_quat / torch.norm(
            keypoint_quat
        )  # Normalize quaternion

        # Compute IK for all environments (but we only care about the current one)
        keypoint_ik = franka.inverse_kinematics(
            link=end_effector,
            pos=keypoint_pos,
            quat=keypoint_quat,
            init_qpos=prev_qpos,
        )

        # Store the solution and use it as initial guess for next keypoint
        keypoint_qpos.append(keypoint_ik)
        prev_qpos = keypoint_ik

    # Set gripper position for the final keypoint
    if gripper > 0.5:  # Open
        keypoint_qpos[-1][-2:] = 0.04
    else:  # Close
        keypoint_qpos[-1][-2:] = 0.0

    # Execute the trajectory by interpolating between keypoints
    for i in range(len(keypoint_qpos) - 1):
        start_qpos = keypoint_qpos[i]
        end_qpos = keypoint_qpos[i + 1]

        # Linear interpolation between keypoints
        for t in range(num_steps_between_keypoints):
            beta = t / num_steps_between_keypoints
            interp_qpos = (1 - beta) * start_qpos + beta * end_qpos

            # Control the robot - need to handle per-environment control
            # Control the position for all environments
            franka.control_dofs_position(position=interp_qpos)
            scene.step()
            if camera is not None:
                camera.render()


def plan_linear_trajectory(
    franka,
    end_effector,
    current_qpos,
    target_pos,
    target_quat,
    gripper_cmd,
    keypoint_distance=0.1,
    num_envs=1,
    device=None,
):
    """
    Plan a straight-line trajectory without executing it.

    This function calculates the trajectory but doesn't execute it, allowing for testing
    or deferred execution.

    Args:
        franka: The Franka robot entity
        end_effector: The end effector link
        current_qpos: Current joint positions
        target_pos: Target position tensor
        target_quat: Target orientation quaternion tensor
        gripper_cmd: Gripper command (0 for close, 1 for open)
        keypoint_distance: Distance between keypoints in meters (default: 0.1m or 10cm)
        num_envs: Number of environments
        device: Torch device

    Returns:
        List of joint positions defining the trajectory
    """
    if device is None:
        device = target_pos.device

    # Get the current end-effector pose
    all_envs_qpos = torch.zeros((num_envs, current_qpos.shape[0]), device=device)
    all_envs_qpos[0] = current_qpos

    all_pos, all_quat = franka.forward_kinematics(link=end_effector, qpos=all_envs_qpos)
    current_pos = all_pos[0]
    current_quat = all_quat[0]

    # Calculate the distance
    distance = torch.norm(target_pos - current_pos)

    # Calculate number of keypoints needed
    num_keypoints = max(int(distance / keypoint_distance) + 1, 2)

    # Generate evenly spaced keypoints
    alphas = torch.linspace(0, 1, num_keypoints, device=device)

    # Compute IK for each keypoint
    keypoint_qpos = []
    prev_qpos = current_qpos

    for alpha in alphas:
        # Interpolate position
        keypoint_pos = current_pos + alpha * (target_pos - current_pos)

        # Interpolate orientation
        keypoint_quat = (1 - alpha) * current_quat + alpha * target_quat
        keypoint_quat = keypoint_quat / torch.norm(keypoint_quat)

        # Create batch tensors
        batch_pos = torch.zeros((num_envs, 3), device=device)
        batch_quat = torch.zeros((num_envs, 4), device=device)
        batch_init_qpos = torch.zeros((num_envs, prev_qpos.shape[0]), device=device)

        batch_pos[0] = keypoint_pos
        batch_quat[0] = keypoint_quat
        batch_init_qpos[0] = prev_qpos

        # Compute IK
        keypoint_ik = franka.inverse_kinematics(
            link=end_effector, pos=batch_pos, quat=batch_quat, init_qpos=batch_init_qpos
        )

        keypoint_ik = keypoint_ik[0]
        keypoint_qpos.append(keypoint_ik)
        prev_qpos = keypoint_ik

    # Set gripper position for the final keypoint
    if gripper_cmd > 0.5:  # Open
        keypoint_qpos[-1][-2:] = 0.04
    else:  # Close
        keypoint_qpos[-1][-2:] = 0.0

    return keypoint_qpos
