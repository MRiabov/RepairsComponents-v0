import torch
import genesis as gs
from repairs_components.training_utils.motion_planning import (
    execute_straight_line_trajectory,
)
import numpy as np
import time


def test_motion_planning():
    """
    Test the motion planning module with a simple straight-line trajectory.
    """
    print("Testing motion planning module...")

    gs.init()
    # Create a simulation scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2), show_viewer=False
    )

    # Add a Franka robot to the scene
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
    )

    # Create a camera for rendering
    cam = scene.add_camera(pos=(1.5, 1.5, 3), lookat=(0, 0, 1), res=(640, 480))

    scene.build(n_envs=1)

    # Start recording
    cam.start_recording()

    # Allow the scene to initialize
    for _ in range(20):
        scene.step()
        cam.render()

    # Define target positions for the end effector
    # We'll create a square path for the end effector to follow
    device = gs.device
    num_envs = 1  # We're testing with a single environment

    # Define height for the trajectory
    height = 0.4  # Ensure it's at a good height for the arm to reach

    # Define the square trajectory points using the coordinates suggested
    # Square centered around the robot with corners at (±0.75, ±0.75)
    positions = [
        torch.tensor(
            [0.4, 0.0, height], device=device
        ),  # Start position (slightly in front of robot)
        torch.tensor([-0.75, -0.75, height], device=device),  # Bottom left corner
        torch.tensor([-0.75, -0.75, height], device=device),  # Bottom left corner
        torch.tensor([-0.75, -0.75, height], device=device),  # Bottom left corner
        torch.tensor([-0.75, -0.75, height], device=device),  # Bottom left corner
        torch.tensor([-0.75, -0.75, height], device=device),  # Bottom left corner
    ]  # do not move, but move only gripper.
    gripper_forces = torch.tensor(
        [[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1], [1, 1]]
    )

    # Define a constant orientation (pointing downward)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # w, x, y, z

    # Execute motion planning for each segment of the square
    print("Executing square trajectory...")

    # Initialize current position to the first point
    current_pos = positions[0].clone()

    # Create batched tensors for position and orientation
    pos_batch = torch.zeros((num_envs, 3), device=device)
    quat_batch = torch.zeros((num_envs, 4), device=device)

    for target_pos, force in zip(positions, gripper_forces):
        # Set the target position and orientation
        pos_batch = target_pos
        quat_batch = orientation  # still always downward

        # Update current position for the next iteration
        current_pos.copy_(target_pos)

        print(f"Moving to point {target_pos}")

        # Execute the trajectory
        execute_straight_line_trajectory(
            franka=franka,
            scene=scene,
            target_pos=pos_batch,
            target_quat=quat_batch,
            gripper_force=force[None],
            keypoint_distance=0.05,  # 5cm between keypoints for smoother motion
            num_steps_between_keypoints=20,
            camera=cam,
        )

    # Stop recording and save video
    cam.stop_recording(
        save_to_filename="/workspace/RepairsComponents-v0/motion_planning_test_gripper.mp4",
        fps=50,
    )

    print(
        "Motion planning test completed. Video saved to motion_planning_test_gripper.mp4"
    )


if __name__ == "__main__":
    test_motion_planning()
