import random
import torch
import genesis as gs
import gymnasium as gym
from genesis.engine.entities import RigidEntity
import numpy as np
from typing import List, Any

from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_sim_step import step_repairs
from repairs_components.training_utils.final_reward_calc import (
    calculate_reward_and_done,
)
from repairs_components.training_utils.motion_planning import (
    execute_straight_line_trajectory,
)

# todo: change to progressive reward calc once it's ready ^
from build123d import Compound
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.processing.voxel_export import export_voxel_grid
from repairs_components.processing.scene_creation_funnel import (
    create_random_scenes,
    generate_scene_meshes,
)
from repairs_components.processing.tasks import Task, AssembleTask


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RepairsEnv(gym.Env):
    def __init__(
        self,
        env_setup: EnvSetup,
        tasks: List[Task],
        batch_dim: int,
        # Batch dim number of parallel environments to simulate
        # and the num_scenes per task is the feed-in repetitions of training pipeline.
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
        num_scenes_per_task: int = 1,
    ):
        """Initialize the Repairs environment.

        Args:
            env_setup: Environment setup instance that defines the base environment
            tasks: List of tasks to train on (e.g. AssembleTask)
            num_envs: Number of parallel environments to simulate
            env_cfg: Configuration dictionary containing environment parameters
            obs_cfg: Configuration for observations
            reward_cfg: Configuration for reward function
            command_cfg: Configuration for command inputs
            show_viewer: Whether to render the simulation viewer
            num_scenes_per_task: Number of random scenes to generate per task
        """
        # Store basic environment parameters
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.device = gs.device
        self.dt = env_cfg.get("dt", 0.02)  # Default to 50Hz if not specified
        self.tasks = tasks
        self.env_setup = env_setup
        self.num_scenes_per_task = num_scenes_per_task

        # Store configuration dictionaries
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.batch_dim = batch_dim

        # ===== Scene Setup =====
        # Create simulation scene with specified timestep and substeps
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            show_viewer=False,
        )

        # if scene meshes don't exist yet, create them now.
        generate_scene_meshes()

        # Create random scenes using the scene creation funnel
        (
            self.desired_scene,
            self.cameras,
            self.gs_entities,
            self.starting_sim_states,
            self.desired_sim_states,
            self.voxel_grids_initial,
            self.voxel_grids_desired,
        ) = create_random_scenes(
            self.scene,
            env_setup,
            tasks,
            num_scenes_per_task=num_scenes_per_task,
            batch_dim=self.batch_dim,
        )  # NOTE: create random scenes must have one scene per one env setup. This is for batching. But they will be alternated.
        # TODO: add mechanism that would recreate random scenes whenever the scene was called too many times.
        # for now just work in one env setup.
        self.called_times_this_batch = 0

        # Store the initial difference count for reward calculation
        self.current_sim_state = self.starting_sim_states[
            0
        ]  # Initialize with the first state
        self.desired_state = self.desired_sim_states[
            0
        ]  # Initialize with the first desired state
        self.diff, self.initial_diff_count = self.current_sim_state.diff(
            self.desired_state
        )

        # Map for entity names to their gs.Entity objects
        self.entity_name_map = {
            name: entity for name, entity in self.gs_entities.items()
        }

        self.franka = self.entity_name_map["franka@control"]
        # ===== Robot Configuration =====
        # Setup joint names and their corresponding DOF indices
        self.joint_names = env_cfg["joint_names"]
        self.dof_idx = [
            self.franka.get_joint(name).dof_start for name in self.joint_names
        ]

        # Set default joint positions from config
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in self.joint_names],
            device=self.device,
        )
        self.data

        # ===== Control Parameters =====
        # Set PD control gains (tuned for Franka Emika Panda)
        # These values are robot-specific and affect the stiffness and damping
        # of the robot's joints during control
        self.franka.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )

        # Set force limits for each joint (in Nm for rotational joints, N for prismatic)
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # Initialize environment to starting state
        self.reset()

    def step(self, action: torch.Tensor):
        """Execute one step in the environment.

        Args:
            action: Action tensor with shape (num_envs, action_dim)
                   Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, gripper]
                   where gripper is 0 for close, 1 for open

        Returns:
            obs: Observation tensor
            reward: Reward tensor
            done: Done tensor
            info: Additional information dictionary
        """
        # Extract position and orientation from action
        pos = action[:, :3]  # Position: [x, y, z]
        quat = action[:, 3:7]  # Quaternion: [w, x, y, z]
        gripper_force = action[:, 7:9]  # two gripper forces (grip push in/out)
        # print("gripper_force.shape", gripper_force.shape)
        # Execute the motion planning trajectory using our dedicated module
        execute_straight_line_trajectory(
            franka=self.franka,
            scene=self.scene,
            target_pos=pos,
            target_quat=quat,
            gripper_force=gripper_force,
            keypoint_distance=0.1,  # 10cm as suggested
            num_steps_between_keypoints=10,
        )

        # Update the current simulation state based on the scene
        success, total_diff_left, self.current_sim_state, diff = step_repairs(
            self.scene,
            action,
            self.entity_name_map,
            self.current_sim_state,
            self.desired_state,
        )

        # Capture observations from cameras
        obs = []
        for camera in self.cameras:
            rgb, depth, _segmentation, normal = camera.render(
                rgb=True, depth=True, segmentation=False, normal=True
            )
            # print(rgb.shape)
            # note: for whichever reason, in batch dim of 1, the cameras don't return batch shape. So I'd expand.
            rgb = np.expand_dims(rgb, (0))
            depth = np.expand_dims(depth, (0))[:, :, :, None]
            normal = np.expand_dims(normal, (0))
            assert all(lambda a: a.ndim == 4 for a in (rgb, depth, normal)), (
                "Too many dims found."
            )

            # Combine all camera observations
            camera_obs = torch.tensor(
                np.concatenate([rgb, depth, normal], axis=-1),
                device=self.device,
            )  # Combine along channel dimension
            obs.append(camera_obs)

        # Stack all camera observations
        video_obs = torch.stack(
            obs, dim=1
        )  # Shape: [num_envs, num_cameras, height, width, channels]

        # Compute reward based on progress toward the goal
        reward, done = calculate_reward_and_done(
            self.current_sim_state,
            self.desired_state,
            self.initial_diff_count,
        )

        # Additional info for debugging
        info = {"diff": diff, "total_diff_left": total_diff_left, "success": success}

        return video_obs, reward, done, info

    def reset_idx(
        self, training_batch: list[tuple[RepairsSimState, RepairsSimState]], envs_idx
    ):
        """Reset specific environments to their initial state.

        Args:
            envs_idx: Indices of environments to reset
        """
        if len(envs_idx) > 0:
            # Reset robot joint positions to default
            dof_pos = self.default_dof_pos.expand(len(envs_idx), -1)
            self.franka.set_dofs_position(
                position=dof_pos,
                # dofs_idx_local=self.dof_idx,
                envs_idx=envs_idx,
            )

            # For each environment, assign a random scene configuration
            for i, env_idx in enumerate(envs_idx):
                # Choose a random starting state from our pregenerated states
                random_scene_idx = torch.randint(
                    0, len(self.starting_sim_states), (1,), device=self.device
                ).item()

                # Set the current and desired states for this environment
                if env_idx == 0:  # Only update the tracked state for the first env
                    self.current_sim_state = self.starting_sim_states[random_scene_idx]
                    self.desired_state = self.desired_sim_states[random_scene_idx]

                    # Calculate initial difference for reward calculation
                    diff, self.initial_diff_count = self.current_sim_state.diff(
                        self.desired_state
                    )

                # TODO: replace with scene_creation_funnel.move_entities_to_pos()
                # Set positions of all objects in this environment
                for name, entity in self.gs_entities.items():
                    if name.endswith(
                        "@control"
                    ):  # don't explicitly reset franka arm...
                        # maybe it's worth returning franka explicitly instead.
                        continue
                    # Get position from the simulation state
                    pos = (
                        torch.tensor(
                            [
                                self.starting_sim_states[
                                    random_scene_idx
                                ].physical_state.positions[name]
                            ]
                        )
                        / 100
                    )  # Convert from cm to meters
                    # TODO !! this has to be generated with batching. for now, just expand the dim of pos.
                    # generated specifically with task initial state perturbation (!or loaded from a dataloader.)
                    # Set the position for this entity in this environment
                    entity.set_pos(
                        pos, envs_idx=torch.tensor([env_idx], device=self.device)
                    )

            # Update visual states to show the new positions
            self.scene.visualizer.update_visual_states()

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments to initial state.

        Returns:
            obs: Initial observation after reset
            info: Additional information
        """
        # Reset all environments
        idxs = torch.arange(self.batch_dim - 1, device=self.device)
        self.reset_idx(idxs)

        # Get initial observations
        obs = []
        for camera in self.cameras:
            rgb, depth, _segmentation, normal = camera.render(
                rgb=True, depth=True, normal=True
            )
            print("rgb shape", str(rgb.shape))
            print("depth shape", str(depth.shape))

            # note: for whichever reason, in batch dim of 1, the cameras don't return batch shape. So I'd expand.
            rgb = np.expand_dims(rgb, (0))
            depth = np.expand_dims(depth, (0))[:, :, :, None]
            normal = np.expand_dims(normal, (0))
            assert all(lambda a: a.ndim == 4 for a in (rgb, depth, normal)), (
                "Too many dims found."
            )

            camera_obs = obs_to_int8(rgb, depth, normal)  # type: ignore

            obs.append(camera_obs)

        # Stack all camera observations
        obs = torch.stack(obs, dim=1)
        # why no batch dim?

        info = {"initial_diff_count": self.initial_diff_count}
        return obs, info


def obs_to_int8(rgb: np.ndarray, depth: np.ndarray, normal: np.ndarray):
    # Normalize and convert to uint8
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    normal_normalized = (normal * 0.5 + 0.5) * 255
    normal_uint8 = normal_normalized.astype(np.uint8)
    return torch.from_numpy(
        np.concatenate([rgb_uint8, depth_uint8, normal_uint8], axis=-1)
    ).cuda()  # why would I convert it to torch here anyway? well, anyway.
