import logging
import torch
import genesis as gs
from repairs_components.training_utils.gym_env import RepairsEnv
from repairs_components.processing.tasks import AssembleTask
from examples.box_to_pos_task import MoveBoxSetup
from repairs_components.geometry.base_env.tooling_stand_plate import render_and_save


def main():
    # Initialize Genesis
    gs.init(backend=gs.cuda)

    # Create task and environment setup
    task = AssembleTask()
    env_setup = MoveBoxSetup()

    # Environment configuration
    env_cfg = {
        "num_actions": 9,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_force_left, gripper_force_right]
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -2.0,
            "joint5": 0.0,
            "joint6": 2.0,
            "joint7": 0.79,  # no "hand" here? there definitely was hand.
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
    }

    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
        "res": (640, 480),
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
    }

    command_cfg = {}

    # Create gym environment
    print("Creating gym environment...")
    env = RepairsEnv(
        env_setup=env_setup,
        tasks=[task],
        num_envs=2,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        num_scenes_per_task=1,
    )

    print("Environment created successfully!")

    # Reset the environment
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial diff count: {info['initial_diff_count']}")

    # Generate a sample action
    print("Generating sample action...")
    action = torch.zeros((2, 9), device=gs.device)  # [batch_size, action_dim]

    # Position at center of workspace, slightly elevated
    action[:, 0:3] = torch.tensor([0.0, 0.0, 0.3], device=gs.device)

    # Quaternion for downward-facing end effector [w, x, y, z]
    action[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)

    # Open gripper
    action[:, 7:8] = 1.0  # N of force.

    # Execute a few steps
    print("Executing steps...")
    for i in range(3):
        print(f"Step {i + 1}")
        obs, reward, done, info = env.step(action)
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Total diff left: {info['total_diff_left']}")

    # Render and save final state
    print("Rendering final state...")
    render_and_save(env.desired_scene, env.cameras[0], env.cameras[1])
    print("Rendering saved to the 'renders' directory.")

    print("Test completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    b123d_logger = logging.getLogger("build123d")
    b123d_logger.disabled = True

    main()
