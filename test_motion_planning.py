import torch
import genesis as gs
from repairs_components.training_utils.motion_planning import (
    execute_straight_line_trajectory,
    plan_linear_trajectory,
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
    cam = scene.add_camera(pos=(0.0, 0.0, 2.5), lookat=(0, 0, 0.0), res=(640, 480))

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
        torch.tensor([0.75, -0.75, height], device=device),  # Bottom right corner
        torch.tensor([0.75, 0.75, height], device=device),  # Top right corner
        torch.tensor([-0.75, 0.75, height], device=device),  # Top left corner
        torch.tensor(
            [-0.75, -0.75, height], device=device
        ),  # Back to bottom left (complete the square)
    ]

    # Define a constant orientation (pointing downward)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # w, x, y, z

    # Get the end effector link
    end_effector = franka.get_link("hand")

    # Execute motion planning for each segment of the square
    print("Executing square trajectory...")

    for i in range(len(positions) - 1):
        # Create batched tensors for position and orientation
        pos_batch = torch.zeros((num_envs, 3), device=device)
        quat_batch = torch.zeros((num_envs, 4), device=device)
        gripper_batch = torch.zeros(num_envs, device=device)

        # Set the target position and orientation
        pos_batch[0] = positions[i + 1]
        quat_batch[0] = orientation

        print(f"Moving to point {i + 1}: {positions[i + 1]}")

        # Execute the trajectory
        execute_straight_line_trajectory(
            franka=franka,
            scene=scene,
            pos=pos_batch,
            quat=quat_batch,
            gripper=gripper_batch,
            keypoint_distance=0.05,  # 5cm between keypoints for smoother motion
            num_steps_between_keypoints=20,
            camera=cam,
        )

        # Render a frame after reaching the target
        cam.render()

    # Additional cooldown steps to allow robot to settle
    for _ in range(20):
        scene.step()
        cam.render()

    # Stop recording and save video
    cam.stop_recording(
        save_to_filename="/workspace/RepairsComponents-v0/motion_planning_test.mp4",
        fps=50,
    )

    print("Motion planning test completed. Video saved to motion_planning_test.mp4")


if __name__ == "__main__":
    test_motion_planning()
