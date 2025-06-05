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
from torch.utils.data import DataLoader
from repairs_components.processing.scene_creation_funnel import move_entities_to_pos
from repairs_components.training_utils.multienv_dataloader import RepairsEnvDataLoader


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
        concurrent_scenes: int = 1,
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
        self.batch_dim = batch_dim  # ML batch dim
        self.concurrent_scenes = concurrent_scenes  # todo implement concurrent scenes.
        # genesis batch dim = batch_dim // concurrent_scenes

        # concurrent scene tuples
        # #(scene, gs_entities, cameras, starting_state, desired_state, voxel_grids_initial, voxel_grids_desired, initial_diff, initial_diff_count)
        self.concurrent_scene_tuples = [None] * concurrent_scenes

        # ===== Scene Setup =====
        # Create simulation scene with specified timestep and substeps
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            show_viewer=False,
        )

        # if scene meshes don't exist yet, create them now.
        generate_scene_meshes()

        self.env_dataloader = RepairsEnvDataLoader(
            scenes=[self.scene],
            env_setups=[self.env_setup],
            tasks=self.tasks,
            batch_dim=self.batch_dim,
            num_scenes_per_task=self.num_scenes_per_task,
        )

        # TODO: add mechanism that would recreate random scenes whenever the scene was called too many times.
        # for now just work in one env setup.

        # # Map for entity names to their gs.Entity objects
        # self.entity_name_map = {
        #     name: entity for name, entity in self.gs_entities.items()
        # }

        self.franka: RigidEntity = self.gs_entities["franka@control"]
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
        assert action.shape == (self.batch_dim, self.num_actions), (
            "Action must have shape (batch_dim, action_dim)"
        )
        action_by_scenes = action.reshape(
            self.concurrent_scenes,
            self.batch_dim // self.concurrent_scenes,  # "genesis" batch dim
            self.num_actions,
        )
        # step through different, concurrent scenes.
        for scene_idx in range(self.concurrent_scenes):
            (
                scene,
                batch_starting_state,
                batch_desired,
                vox_init,
                vox_des,
                initial_diff,
                initial_diff_count,
            ) = self.concurrent_scene_tuples[scene_idx]
            # Extract position and orientation from action
            pos = action_by_scenes[scene_idx, :, :3]  # Position: [x, y, z]
            quat = action_by_scenes[scene_idx, :, 3:7]  # Quaternion: [w, x, y, z]
            gripper_force = action_by_scenes[
                scene_idx, :, 7:9
            ]  # two gripper forces (grip push in/out)
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
            scene = self.scenes[scene_idx]
            gs_entities = gs_entities_by_scene[scene_idx]
            current_sim_state = self.current_sim_states[scene_idx]
            desired_state = self.desired_states[scene_idx]

            # Update the current simulation state based on the scene
            success, total_diff_left, current_sim_state, diff = step_repairs(
                scene,
                action[i],
                gs_entities,
                current_sim_state,
                desired_state,
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
                current_sim_state,
                desired_state,
                initial_diff_count,
            )

            # Additional info for debugging
            info = {
                "diff": diff,
                "total_diff_left": total_diff_left,
                "success": success,
            }

        return video_obs, reward, done, info

    def reset_idx(self, envs_idx: torch.Tensor):
        """Reset specific environments to their initial state.

        Args:
            envs_idx: Indices of environments to reset
        """
        if len(envs_idx) > 0:
            # Reset robot joint positions to default
            dof_pos = self.default_dof_pos.expand(envs_idx.shape[0], -1)
            self.franka.set_dofs_position(
                position=dof_pos,
                # dofs_idx_local=self.dof_idx,
                envs_idx=envs_idx,
            )
            for env_i in envs_idx:
                # for each environment, get its bucket of dataloaded states.
                # get_batch returns: (scene, batch_start, batch_desired, vox_init, vox_des, initial_diff, initial_diff_count)
                batch_tuple = self.env_dataloader.get_batch(env_i)

                # Update the concurrent scene tuples memory with the new batch
                # Store the tuple exactly as returned by get_batch to match step method expectations
                (
                    scene,
                    batch_starting_state,
                    batch_desired_state,
                    vox_init,
                    vox_des,
                    initial_diff,
                    initial_diff_count,
                ) = batch_tuple

                # Store the complete tuple matching the format expected by step method
                self.concurrent_scene_tuples[env_i] = (
                    scene,
                    batch_starting_state,
                    batch_desired_state,
                    vox_init,
                    vox_des,
                    initial_diff,
                    initial_diff_count,
                )

                # Move entities to their starting positions
                move_entities_to_pos(scene, batch_starting_state)

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
