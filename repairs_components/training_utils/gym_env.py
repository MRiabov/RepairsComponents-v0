import time
from functools import partial
from typing import List

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from genesis.vis.visualizer import Camera
from igl.helpers import os
from torch_geometric.data import Batch

from repairs_components.processing.scene_creation_funnel import (
    create_env_configs,
    generate_scene_meshes,
    initialize_and_build_scene,
    move_entities_to_pos,
)
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_scene_configs_at_idx,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.failure_modes import out_of_bounds
from repairs_components.training_utils.motion_planning import (
    execute_straight_line_trajectory,
)
from repairs_components.training_utils.multienv_dataloader import RepairsEnvDataLoader
from repairs_components.training_utils.progressive_reward_calc import (
    RewardHistory,
    calculate_done,
)
from repairs_components.training_utils.save import optional_save
from repairs_sim_step import step_repairs


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RepairsEnv(gym.Env):
    def __init__(
        self,
        env_setups: List[EnvSetup],
        tasks: List[Task],
        ml_batch_dim: int,
        # Batch dim number of parallel environments to simulate (ml batch dim)
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
            ml_batch_dim: Batch dim number of parallel environments to simulate (ml batch dim)
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
        self.env_setups = env_setups
        self.num_scenes_per_task = num_scenes_per_task
        self.min_bounds = torch.tensor(env_cfg["min_bounds"], device=self.device)
        self.max_bounds = torch.tensor(env_cfg["max_bounds"], device=self.device)

        # Store configuration dictionaries
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.ml_batch_dim = ml_batch_dim  # ML batch dim. I want to make it explicit. (incl because I've failed on this).
        self.per_scene_batch_dim = ml_batch_dim // concurrent_scenes
        self.concurrent_scenes = concurrent_scenes  # todo implement concurrent scenes.
        # genesis batch dim = batch_dim // concurrent_scenes

        # save config
        if self.env_cfg["save_obs"]:
            os.makedirs(self.env_cfg["save_obs"]["path"], exist_ok=True)
            self.save_video: bool = self.env_cfg["save_obs"]["video"]
            self.save_voxel: bool = self.env_cfg["save_obs"]["voxel"]
            self.save_electronic_graph: bool = self.env_cfg["save_obs"][
                "electronic_graph"
            ]
            self.save_path: str = self.env_cfg["save_obs"]["path"]
            self.save_any: bool = (
                self.save_video or self.save_voxel or self.save_electronic_graph
            )
            self.partial_save = partial(  # don't mess the `step()` code.
                optional_save,
                save_any=self.save_any,
                save_path=self.save_path,
                save_image=self.save_video,
                save_voxel=self.save_voxel,
                save_state=self.save_electronic_graph,
            )

        # Using dataclass to store concurrent scene data for better organization and clarity
        self.concurrent_scenes_data: list[ConcurrentSceneData] = []

        self.scenes = []
        task = tasks[0]  # any task will suffice for init.

        # if scene meshes don't exist yet, create them now.
        generate_scene_meshes()

        prefetch_memory_size = env_cfg["dataloader_settings"]["prefetch_memory_size"]
        init_generate_per_scene = torch.full((concurrent_scenes,), prefetch_memory_size)

        partial_env_configs = create_env_configs(
            env_setups,
            tasks,
            torch.tensor(
                [self.per_scene_batch_dim] * concurrent_scenes, dtype=torch.int16
            ),
        )[0]  # note: all of them are discarded, but just for the sake of init.

        # scene init setup # note: technically this should be the Dataloader worker init fn.
        for scene_idx in range(concurrent_scenes):
            scene = gs.Scene(  # empty scene
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                show_viewer=False,
                vis_options=gs.options.VisOptions(env_separate_rigid=True),  # type: ignore
            )
            self.scenes.append(scene)

            scene, cameras, gs_entities, franka = initialize_and_build_scene(
                scene,  # build scene.
                env_setups[scene_idx].desired_state_geom(),
                partial_env_configs.desired_state,
                self.per_scene_batch_dim,
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
                    reward_history=RewardHistory(self.per_scene_batch_dim),
                    batch_dim=self.per_scene_batch_dim,
                    step_count=torch.zeros(self.per_scene_batch_dim, dtype=torch.int),
                )
            )
        # create the dataloader to update scenes.
        self.env_dataloader = RepairsEnvDataLoader(
            scenes=self.scenes,
            env_setups=env_setups,
            tasks=tasks,
            batch_dim=self.ml_batch_dim,  # ml batch dim.
            prefetch_memory_size=prefetch_memory_size,
        )
        print("before populate_async")
        # populate the dataloader with initial configs
        self.env_dataloader.populate_async(
            init_generate_per_scene.to(dtype=torch.int16)
        )
        print("after populate_async")
        # Set default joint positions from config
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            device=self.device,
        )
        self.last_step_time = time.perf_counter()

        # Initialize environment to starting state
        self.reset()

        print("Repairs environment initialized")

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
        assert action.shape == (self.ml_batch_dim, self.num_actions), (
            "Action must have shape (batch_dim, action_dim)"
        )
        action_by_scenes = action.reshape(
            self.concurrent_scenes,
            self.ml_batch_dim // self.concurrent_scenes,  # "genesis" batch dim
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

            # voxel_init, voxel_des, video_obs, graph_obs, graph_des = (
            #     self._observe_scene(scene_data)
            # )

            # Compute reward based on progress toward the goal
            dones = calculate_done(scene_data)  # note: pretty expensive.
            rewards = scene_data.reward_history.calculate_reward_this_timestep(
                scene_data
            )

            # # save the step information if necessary.
            # self.partial_save(
            #     sim_state=scene_data.current_state,
            #     obs_image=video_obs,
            #     voxel_grids_initial=voxel_init,
            #     voxel_grids_desired=voxel_des,
            #     # NOTE: ideally save voxel only after reset.
            # )

            # check if any entity is out of bounds
            out_of_bounds_fail = out_of_bounds(
                min=self.min_bounds,
                max=self.max_bounds,
                gs_entities=self.concurrent_scenes_data[scene_idx].gs_entities,
            )

            reset_envs = dones | out_of_bounds_fail  # or any other failure mode.

            # reset at done
            if reset_envs.any():
                self.reset_idx(reset_envs.nonzero().squeeze(1))

            # Additional info for debugging
            info = {
                "diff": diff,
                "total_diff_left": total_diff_left,
                "success": success,
                "out_of_bounds": out_of_bounds_fail,
            }

        voxel_init, voxel_des, video_obs, graph_obs, graph_des = (
            self._observe_all_scenes()
        )
        print(
            "step happened. Time elapsed:",
            time.perf_counter() - self.last_step_time,
            "Rewards:",
            rewards.mean().item(),
            "out_of_bounds_fail:",
            out_of_bounds_fail.float().mean().item(),
        )
        self.last_step_time = time.perf_counter()

        return (
            voxel_init,
            voxel_des,
            video_obs,
            graph_obs,
            graph_des,
            rewards,
            dones,
            info,
        )

    def reset_idx(self, envs_idx: torch.IntTensor):
        """Reset specific environments to their initial state.

        Args:
            envs_idx: Indices of environments to reset
        """
        assert envs_idx.shape[0] > 0, "Can't reset 0 environments."
        assert (envs_idx >= 0).all(), "Can't reset negative environments"
        assert (envs_idx <= self.ml_batch_dim).all(), (
            "Can't reset environments out of bounds"
        )

        # get from which scene(s) we are resetting
        num_env_per_scene = self.ml_batch_dim // self.concurrent_scenes
        scene_idx = envs_idx // num_env_per_scene
        env_in_scene_idx = envs_idx % num_env_per_scene
        env_per_scene_idx = env_in_scene_idx.reshape(self.concurrent_scenes, -1)
        unique_scene_idx, counts = torch.unique(scene_idx, return_counts=True)

        # update the scene
        # Reset robot joint positions to default
        dof_pos = self.default_dof_pos.expand(envs_idx.shape[0], -1)

        # get data for the entire batch.
        reset_scene_data: list[ConcurrentSceneData | None] = (  # none if counts were 0.
            self.env_dataloader.get_processed_data(counts.to(dtype=torch.int16))
        )
        # print("reset_scene_data", reset_scene_data)

        for scene_id in range(len(reset_scene_data)):
            reset_scene = reset_scene_data[scene_id]
            reset_env_ids_this_scene = env_per_scene_idx[scene_id]
            if reset_scene is None:
                continue

            self.concurrent_scenes_data[scene_id].gs_entities[
                "franka@control"
            ].set_dofs_position(position=dof_pos, envs_idx=reset_env_ids_this_scene)

            # Create a mask for which environments to reset in this scene
            reset_mask = torch.zeros(
                self.per_scene_batch_dim, dtype=torch.bool, device=self.device
            )
            reset_mask[reset_env_ids_this_scene] = True

            # Merge states, only updating the environments that need to be reset
            updated_scene_data = merge_scene_configs_at_idx(
                self.concurrent_scenes_data[scene_id], reset_scene, reset_mask
            )

            # merge the
            self.concurrent_scenes_data[scene_id] = updated_scene_data

            # Move entities to their starting positions
            move_entities_to_pos(
                updated_scene_data.gs_entities, updated_scene_data.current_state
            )

            # Update visual states to show the new positions
            self.concurrent_scenes_data[
                scene_id
            ].scene.visualizer.update_visual_states()

        return self.concurrent_scenes_data

    def reset(self):
        """Reset all environments to initial state.

        Returns:
            obs: Initial observation after reset
            info: Additional information
        """
        # Reset all environments # note: it was self.batch_dim - 1, but I don't think this is ri
        idxs = torch.arange(self.ml_batch_dim, device=self.device)
        self.reset_idx(idxs)

        # observe all environments (ideally done in paralel.)
        voxel_init, voxel_des, video_obs, graph_obs, graph_des = (
            self._observe_all_scenes()
        )

        return voxel_init, voxel_des, video_obs, graph_obs, graph_des

    def _observe_scene(self, scene_data: ConcurrentSceneData):
        # Process a single environment
        # Get the scene data for this environment

        # Extract cameras from scene data
        cameras = scene_data.cameras
        video_obs = _render_all_cameras(cameras)

        # get voxel obs
        sparse_voxel_init = scene_data.vox_init
        sparse_voxel_des = scene_data.vox_des

        # get graph obs
        graph_obs = [
            state.graph for state in scene_data.current_state.electronics_state
        ]
        graph_des = [
            state.graph for state in scene_data.desired_state.electronics_state
        ]  # return them as lists because they will be stacked to batch as global env.

        return sparse_voxel_init, sparse_voxel_des, video_obs, graph_obs, graph_des

    def _observe_all_scenes(self):
        """A helper to merge all voxel and graph observations."""
        all_voxel_init = []
        all_voxel_des = []
        all_video_obs = []
        all_graph_obs = []
        all_graph_des = []
        for scene_data in self.concurrent_scenes_data:
            voxel_init, voxel_des, video_obs, graph_obs, graph_des = (
                self._observe_scene(scene_data)
            )
            all_voxel_init.append(voxel_init)
            all_voxel_des.append(voxel_des)
            all_video_obs.append(video_obs)
            all_graph_obs.extend(graph_obs)
            all_graph_des.extend(graph_des)

        voxel_init = torch.concat(all_voxel_init, dim=0)
        voxel_des = torch.concat(all_voxel_des, dim=0)
        graph_obs = Batch.from_data_list(all_graph_obs)
        graph_des = Batch.from_data_list(all_graph_des)
        video_obs = torch.cat(
            all_video_obs, dim=0
        )  # cat, not stack because it's already batched
        video_obs = video_obs.permute(0, 1, 4, 2, 3)  # to torch format
        
        return voxel_init, voxel_des, video_obs, graph_obs, graph_des


def _render_all_cameras(cameras: list[Camera]):
    # Process each camera in the scene
    env_obs = []
    for camera in cameras:
        rgb, depth, _segmentation, normal = camera.render(
            rgb=True, depth=True, normal=True
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
    depth_uint8 = np.expand_dims(depth_uint8, axis=-1)
    normal_normalized = (normal * 0.5 + 0.5) * 255
    normal_uint8 = normal_normalized.astype(np.uint8)
    return torch.from_numpy(
        np.concatenate([rgb_uint8, depth_uint8, normal_uint8], axis=-1)
    ).cuda()  # why would I convert it to torch here anyway? well, anyway.
