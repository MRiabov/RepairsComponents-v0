import random
from genesis.vis.visualizer import Camera
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
    create_env_configs,
    generate_scene_meshes,
)
from repairs_components.processing.tasks import Task, AssembleTask
from torch.utils.data import DataLoader
from repairs_components.processing.scene_creation_funnel import (
    move_entities_to_pos,
    initialize_and_build_scene,
)
from repairs_components.training_utils.multienv_dataloader import RepairsEnvDataLoader
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
    merge_scene_configs_at_idx,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RepairsEnv(gym.Env):
    def __init__(
        self,
        env_setups: List[EnvSetup],
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

        # Using dataclass to store concurrent scene data for better organization and clarity
        self.concurrent_scenes_data: list[ConcurrentSceneData] = []

        scenes = []
        task = tasks[0]  # any task will suffice for init.

        # if scene meshes don't exist yet, create them now.
        generate_scene_meshes()

        init_generate_per_scene = env_cfg["dataloader_settings"]["prefetch_memory_size"]
        init_generate_per_scene = torch.full(
            (concurrent_scenes,), init_generate_per_scene
        )

        partial_env_configs = create_env_configs(
            env_setups, tasks, init_generate_per_scene
        )[0]  # for init I need only one. (it will be discarded later)

        # scene init setup # note: technically this should be the Dataloader worker init fn.
        for scene_idx in range(concurrent_scenes):
            scene = gs.Scene(  # empty scene
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                show_viewer=False,
            )
            scenes.append(scene)

            scene, cameras, gs_entities, franka = initialize_and_build_scene(
                scene,  # build scene.
                env_setups[scene_idx].desired_state_geom(),
                partial_env_configs.desired_state,
                self.batch_dim,
            )

            # Using dataclass to store concurrent scene data for better organization and clarity
            self.concurrent_scenes_data.append(
                ConcurrentSceneData(
                    scene=scene,
                    gs_entities=gs_entities,
                    cameras=tuple(cameras),
                    current_state=partial_env_configs.current_state,
                    desired_state=partial_env_configs.desired_state,
                    vox_init=partial_env_configs.vox_init,
                    vox_des=partial_env_configs.vox_des,
                    initial_diffs=partial_env_configs.initial_diffs,
                    initial_diff_counts=partial_env_configs.initial_diff_counts,
                    scene_id=scene_idx,
                )
            )
        # create the dataloader to update scenes.
        self.env_dataloader = RepairsEnvDataLoader(
            scenes=scenes,
            env_setups=env_setups,
            tasks=tasks,
            batch_dim=self.batch_dim,
            num_scenes_per_task=self.num_scenes_per_task,
        )

        # Set default joint positions from config
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            device=self.device,
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
            # Get data for this concurrent scene
            scene_data = self.concurrent_scenes_data[scene_idx]

            # Extract position and orientation from action
            pos = action_by_scenes[scene_idx, :, :3]  # Position: [x, y, z]
            quat = action_by_scenes[scene_idx, :, 3:7]  # Quaternion: [w, x, y, z]
            gripper_force = action_by_scenes[
                scene_idx, :, 7:9
            ]  # two gripper forces (grip push in/out)
            # print("gripper_force.shape", gripper_force.shape)
            # Execute the motion planning trajectory using our dedicated module
            execute_straight_line_trajectory(
                franka=scene_data.gs_entities["franka@control"],
                scene=scene_data.scene,
                target_pos=pos,
                target_quat=quat,
                gripper_force=gripper_force,
                keypoint_distance=0.1,  # 10cm as suggested
                num_steps_between_keypoints=10,
            )

            # Update the current simulation state based on the scene
            success, total_diff_left, current_sim_state, diff = step_repairs(
                scene_data.scene,
                action_by_scenes[scene_idx],
                scene_data.gs_entities,
                scene_data.current_state,
                scene_data.desired_state,
            )

            # Update the scene data with the new state
            scene_data.current_state = current_sim_state

            video_obs = _render_all_cameras(scene_data.cameras)

            # Compute reward based on progress toward the goal
            reward, done = calculate_reward_and_done(scene_data)

            # reset at done
            self.reset_idx(done)

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

        # FIXME: envs_idx or scene_idx? the dataloader returns for the whole scene. and I don't need it.

        assert len(envs_idx) > 0, "Can't reset 0 environments"

        # get from which scene(s) we are resetting
        scene_idx = envs_idx // self.batch_dim
        unique_scene_idx, counts = torch.unique(scene_idx, return_counts=True)

        # update the scene
        # Reset robot joint positions to default
        dof_pos = self.default_dof_pos.expand(envs_idx.shape[0], -1)

        for scene_id in unique_scene_idx:
            # get the number of environments to reset
            num_envs_to_reset = counts[scene_id]
            envs_idx_to_reset_this_scene = torch.nonzero(scene_idx == scene_id).squeeze(
                1
            )

            self.concurrent_scenes_data[scene_id].gs_entities[
                "franka@control"
            ].set_dofs_position(
                position=dof_pos,
                envs_idx=envs_idx_to_reset_this_scene,
            )

            reset_scene_data: ConcurrentSceneData = (
                self.env_dataloader.get_processed_data(
                    scene_id.item(), get_count=num_envs_to_reset.item()
                )
            )

            # Create ConcurrentSceneData instance
            new_scene_data = ConcurrentSceneData(
                scene=reset_scene_data.scene,
                gs_entities=reset_scene_data.gs_entities,
                cameras=reset_scene_data.cameras,
                current_state=reset_scene_data.current_state,
                desired_state=reset_scene_data.desired_state,
                vox_init=reset_scene_data.vox_init,
                vox_des=reset_scene_data.vox_des,
                initial_diffs=reset_scene_data.initial_diffs,
                initial_diff_counts=reset_scene_data.initial_diff_counts,
                scene_id=reset_scene_data.scene_id,
            )
            self.concurrent_scenes_data[scene_id] = new_scene_data

            # Move entities to their starting positions
            move_entities_to_pos(
                new_scene_data.gs_entities, new_scene_data.current_state
            )

            # Update visual states to show the new positions
            self.concurrent_scenes_data[
                scene_id
            ].scene.visualizer.update_visual_states()

        return self.concurrent_scenes_data

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments to initial state.

        Returns:
            obs: Initial observation after reset
            info: Additional information
        """
        # Reset all environments
        idxs = torch.arange(self.batch_dim - 1, device=self.device)
        self.reset_idx(idxs)

        # Get initial observations from all concurrent scenes
        all_env_obs = []

        # Process each environment
        for scene_idx in range(self.concurrent_scenes):
            # Get the scene data for this environment
            scene_data = self.concurrent_scenes_data[scene_idx]

            # Extract cameras from scene data
            cameras = scene_data.cameras
            video_obs = _render_all_cameras(cameras)

            all_env_obs.append(video_obs)

        return torch.stack(all_env_obs, dim=0), {}


def _render_all_cameras(cameras: list[Camera]):
    # Process each camera in the scene
    env_obs = []
    for camera in cameras:
        rgb, depth, _segmentation, normal = camera.render(
            rgb=True, depth=True, normal=True
        )

        # note: for whichever reason, in batch dim of 1, the cameras don't return batch shape. So I'd expand.
        rgb = np.expand_dims(rgb, (0))
        depth = np.expand_dims(depth, (0))[:, :, :, None]
        normal = np.expand_dims(normal, (0))
        assert all(lambda a: a.ndim == 4 for a in (rgb, depth, normal)), (
            "Too many dims found."
        )

        # Process camera observation
        camera_obs = obs_to_int8(rgb, depth, normal)  # type: ignore
        env_obs.append(camera_obs)

    # Stack all cameras for this environment
    video_obs = torch.stack(env_obs, dim=1)
    return video_obs


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
