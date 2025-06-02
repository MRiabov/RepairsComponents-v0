import logging
import torch
import genesis as gs
import random
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
        "num_actions": 8,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper]
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
            "joint7": 0.79,
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
    }

    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
    }

    command_cfg = {}

    # Create gym environment
    print("Creating gym environment...")
    env = RepairsEnv(
        env_setup=env_setup,
        tasks=[task],
        num_envs=1,
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

    # Run random policy for 200 steps
    print("Running random policy for 200 steps...")

    total_reward = 0.0
    best_reward_step = -1
    best_reward = float("-inf")

    # Range for random actions
    pos_range = 0.3  # Position range in meters (±0.3m from center)

    # Record video of the random policy
    env.cameras[0].start_recording()

    # Run the random policy
    for step in range(200):
        # Generate random action
        action = torch.zeros((1, 8), device=gs.device)  # [batch_size, action_dim]

        # Random position within workspace bounds
        action[:, 0] = torch.tensor(
            random.uniform(-pos_range, pos_range), device=gs.device
        )  # x
        action[:, 1] = torch.tensor(
            random.uniform(-pos_range, pos_range), device=gs.device
        )  # y
        action[:, 2] = torch.tensor(
            random.uniform(0.1, 0.5), device=gs.device
        )  # z (keep above table)

        # Random quaternion (simplified to just rotate around z-axis)
        # Keep gripper pointed downward but with random rotation around z
        angle = random.uniform(0, 6.28)  # Random angle in radians
        action[:, 3] = torch.tensor(0.707, device=gs.device)  # w
        action[:, 4] = torch.tensor(0.0, device=gs.device)  # x
        action[:, 5] = torch.tensor(0.0, device=gs.device)  # y
        action[:, 6] = torch.tensor(0.707, device=gs.device)  # z

        # Random gripper command (0 = closed, 1 = open)
        action[:, 7] = torch.tensor(random.choice([0.0, 1.0]), device=gs.device)

        # Execute action
        obs, reward, done, info = env.step(action)

        # Track rewards
        curr_reward = reward.mean().item()
        total_reward += curr_reward

        if curr_reward > best_reward:
            best_reward = curr_reward
            best_reward_step = step

        # Log progress
        if step % 20 == 0:
            print(
                f"Step {step}: Reward = {curr_reward:.4f}, Total Reward = {total_reward:.4f}"
            )
            print(f"  Total diff left: {info['total_diff_left']}")

        # Early stopping if task is completed
        if done.all():
            print(f"All environments completed at step {step}!")
            break

    # Stop recording and save video
    env.cameras[0].stop_recording(
        save_to_filename="/workspace/RepairsComponents-v0/random_policy.mp4", fps=60
    )

    # Print final statistics
    print("\nRandom Policy Statistics:")
    print(f"Total steps executed: {step + 1}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Best reward: {best_reward:.4f} at step {best_reward_step}")
    print(f"Final diff left: {info['total_diff_left']}")

    # Render and save final state
    print("Rendering final state...")
    if (
        hasattr(env, "desired_scene")
        and hasattr(env, "cameras")
        and len(env.cameras) >= 2
    ):
        render_and_save(env.desired_scene, env.cameras[0], env.cameras[1])
        print("Final state rendering saved to the 'renders' directory.")

    print("Random policy test completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    b123d_logger = logging.getLogger("build123d")
    b123d_logger.disabled = True

    main()
