import pathlib
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
from repairs_components.processing.translation import (
    create_constraints_based_on_graph,
    reset_constraints,
)
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_scene_configs_at_idx,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.failure_modes import out_of_bounds
from repairs_components.training_utils.motion_planning import (
    execute_straight_line_trajectory,
)
from repairs_components.save_and_load.multienv_dataloader import (
    RepairsEnvDataLoader,
)
from repairs_components.save_and_load.offline_data_creation import create_data
from repairs_components.save_and_load.offline_dataloading import (
    check_if_data_exists,
    get_scene_mesh_file_names,
)

from repairs_components.training_utils.progressive_reward_calc import (
    RewardHistory,
    calculate_done,
)
from repairs_components.save_and_load.online_save import optional_save
from repairs_sim_step import step_repairs
from pathlib import Path
import shutil


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
        io_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        num_scenes_per_task: int = 1,
        # concurrent_scenes: int = 1,
        use_offline_dataset: bool = False,
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
        self.io_cfg = io_cfg
        self.reward_cfg = reward_cfg
        self.ml_batch_dim = ml_batch_dim  # ML batch dim. I want to make it explicit. (incl because I've failed on this).
        self.concurrent_scenes = len(
            io_cfg["env_setup_ids"]
        )  # todo implement concurrent scenes.
        self.env_setup_ids = torch.tensor(io_cfg["env_setup_ids"])
        self.per_scene_batch_dim = ml_batch_dim // self.concurrent_scenes

        # genesis batch dim = batch_dim // concurrent_scenes

        base_dir = Path(io_cfg["data_dir"])
        if io_cfg["force_recreate_data"]:
            print(
                "Warning: removing and recreating all data for force_recreate_data=True."
            )
            if base_dir.exists():
                # can't remove dir if it's not empty, so rmtree.
                shutil.rmtree(base_dir)

        base_dir.mkdir(exist_ok=True, parents=True)
        (base_dir / "graphs").mkdir(exist_ok=True, parents=True)
        (base_dir / "voxels").mkdir(exist_ok=True)
        (base_dir / "meshes").mkdir(exist_ok=True)
        debug_render_dir = base_dir / "debug_render"
        debug_render_dir.mkdir(exist_ok=True)
        for scene_id in io_cfg["env_setup_ids"]:
            (base_dir / f"scene_{scene_id}").mkdir(exist_ok=True)
        use_random_textures = obs_cfg["use_random_textures"]

        # save config
        if io_cfg["save_obs"]:
            os.makedirs(io_cfg["save_obs"]["path"], exist_ok=True)
            self.save_video: bool = io_cfg["save_obs"]["video"]
            self.save_voxel: bool = io_cfg["save_obs"]["voxel"]
            self.save_electronic_graph: bool = io_cfg["save_obs"]["electronic_graph"]
            self.save_path: str = io_cfg["save_obs"][
                "path"
            ]  # path for online obs, not data generation!
            self.save_any: bool = (
                self.save_video or self.save_voxel or self.save_electronic_graph
            )
            self.partial_save = partial(  # don't mess the `step()` code.
                optional_save,
                save_any=self.save_any,
                save_path=self.save_path,
                save_video=self.save_video,
                save_video_every_steps=io_cfg["save_obs"]["new_video_every"],
                video_len=io_cfg["save_obs"]["video_len"],
                save_voxel=self.save_voxel,
                save_state=self.save_electronic_graph,
            )

        # Using dataclass to store concurrent scene data for better organization and clarity
        self.concurrent_scenes_data: list[ConcurrentSceneData] = []

        task = tasks[0]  # any task will suffice for init.

        # -- data pregeneration --

        self.env_setup_ids = torch.tensor(io_cfg["env_setup_ids"])

        # minimum amount of configs to generate
        generate_number_of_configs_per_scene = io_cfg[
            "generate_number_of_configs_per_scene"
        ]
        generate_number_of_configs_per_scene = torch.full(
            (self.concurrent_scenes,),
            generate_number_of_configs_per_scene,
            dtype=torch.int16,
        )

        data_gen_start_time = time.time()
        data_already_exists = check_if_data_exists(
            self.env_setup_ids.tolist(), base_dir, generate_number_of_configs_per_scene
        )

        if not data_already_exists or io_cfg["force_recreate_data"]:
            # if scene meshes don't exist yet, create them now.
            generate_scene_meshes(base_dir=Path(self.io_cfg["data_dir"]))
            if not data_already_exists and not io_cfg["force_recreate_data"]:
                print(
                    "Data was not found to exist but force_recreate_data was not called. Still, recreating..."
                )
                # TODO if ever necessary, do conditional generation - don't regenerate what was already generated.
            create_data(
                scene_setups=env_setups,
                tasks=tasks,
                scene_idx=self.env_setup_ids,
                num_configs_to_generate_per_scene=generate_number_of_configs_per_scene,
                base_dir=base_dir,
            )
        else:
            print("Existing data found. Skipping data generation.")
        # -- dataloader (offline) --
        # init dataloader
        # note: will take some time to load.
        start_dataloader_load = time.time()
        self.env_dataloader = RepairsEnvDataLoader(
            env_setup_ids=self.env_setup_ids.tolist(),
            online=False,
            offline_data_dir=base_dir,
        )
        print(
            f"Offline dataloader loaded in {time.time() - start_dataloader_load:.2f} seconds."
        )
        in_memory = torch.full(
            (self.concurrent_scenes,),
            io_cfg["dataloader_settings"]["prefetch_memory_size"],
            dtype=torch.int16,
        )
        # self.env_dataloader.populate_async(in_memory)  # is it still necessary?
        partial_env_configs = self.env_dataloader.get_processed_data(
            torch.full(
                (self.concurrent_scenes,), self.per_scene_batch_dim, dtype=torch.int16
            )
        )  # get a batch of configs size prefetch_memory_size

        # scene init setup # note: technically this should be the Dataloader worker init fn.
        mesh_file_names = get_scene_mesh_file_names(
            self.env_setup_ids.tolist(), base_dir, append_path=True
        )

        self.concurrent_scenes_data = []

        for scene_id in range(len(self.env_setup_ids)):
            # NOTE: scene_id is not the same as env_setup_id!
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2, ),
                show_viewer=False,
                vis_options=gs.options.VisOptions(
                    env_separate_rigid=True,
                    shadow=True,
                ),  # type: ignore
                rigid_options=gs.options.RigidOptions(max_dynamic_constraints=128),
                # note^: max_dynamic constraints is 8 by default. 128 is too low too.
                # show_FPS=False,
            )

            scene, gs_entities = initialize_and_build_scene(
                scene,
                partial_env_configs[scene_id].desired_state,
                mesh_file_names[scene_id],
                self.per_scene_batch_dim,
                random_textures=use_random_textures,
                base_dir=base_dir,
                scene_id=scene_id,
            )

            # Using dataclass to store concurrent scene data for better organization
            scene_data = ConcurrentSceneData(
                scene=scene,
                gs_entities=gs_entities,
                init_state=partial_env_configs[scene_id].init_state,
                current_state=partial_env_configs[scene_id].current_state,
                desired_state=partial_env_configs[scene_id].desired_state,
                vox_init=partial_env_configs[scene_id].vox_init,
                vox_des=partial_env_configs[scene_id].vox_des,
                initial_diffs=partial_env_configs[scene_id].initial_diffs,
                initial_diff_counts=partial_env_configs[scene_id].initial_diff_counts,
                scene_id=scene_id,
                reward_history=RewardHistory(self.per_scene_batch_dim),
                batch_dim=self.per_scene_batch_dim,
                step_count=torch.zeros(self.per_scene_batch_dim, dtype=torch.int),
                task_ids=partial_env_configs[scene_id].task_ids,
            )

            # store built scene and data
            self.concurrent_scenes_data.append(scene_data)

        print(f"Data generation took {time.time() - data_gen_start_time} seconds")

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            device=self.device,
        )

        self.last_step_time = time.perf_counter()

        self.reset()
        # debug render all envs
        import cv2

        for scene_data in self.concurrent_scenes_data:
            debug_render = []
            for cam in scene_data.scene.visualizer.cameras:
                render_out = cam.render(rgb=True)
                rgb = render_out[0][0]  # from tuple and 1st in batch.
                debug_render.append(rgb)

            for i, rgb_img in enumerate(debug_render):
                cv2.imwrite(
                    str(
                        debug_render_dir / f"debug_render_{scene_data.scene_id}_{i}.png"
                    ),
                    rgb_img,
                )

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
        reset_by_scenes = torch.zeros(
            self.concurrent_scenes,
            self.ml_batch_dim // self.concurrent_scenes,
            dtype=torch.bool,
        )
        dones_by_scenes = torch.zeros(
            self.concurrent_scenes,
            self.ml_batch_dim // self.concurrent_scenes,
            dtype=torch.bool,
        )
        rewards_by_scenes = torch.zeros(
            self.concurrent_scenes,
            self.ml_batch_dim // self.concurrent_scenes,
            dtype=torch.float,
        )
        # step through different, concurrent scenes.
        for scene_id in range(self.concurrent_scenes):
            # Get data for this concurrent scene
            scene_data = self.concurrent_scenes_data[scene_id]

            # Extract position and orientation from action
            pos = action_by_scenes[scene_id, :, :3]  # Position: [x, y, z]
            quat = action_by_scenes[scene_id, :, 3:7]  # Quaternion: [w, x, y, z]
            gripper_force = action_by_scenes[
                scene_id, :, 7:9
            ]  # two gripper forces (grip push in/out)

            # Execute the motion planning trajectory using our dedicated module
            motion_planning_time = time.perf_counter()
            execute_straight_line_trajectory(
                franka=scene_data.gs_entities["franka@control"],
                scene=scene_data.scene,
                target_pos=pos,
                target_quat=quat,
                gripper_force=gripper_force,
                render=True,
                keypoint_distance=0.1,  # 10cm as suggested
                num_steps_between_keypoints=10,
            )
            print(
                "Motion planning and exec time:",
                time.perf_counter() - motion_planning_time,
            )

            # Update the current simulation state based on the scene
            success, total_diff_left, current_sim_state, diff = step_repairs(
                scene_data.scene,
                action_by_scenes[scene_id],
                scene_data.gs_entities,
                scene_data.current_state,
                scene_data.desired_state,
            )

            # Update the scene data with the new state
            scene_data.current_state = current_sim_state
            self.concurrent_scenes_data[scene_id] = scene_data

            # Compute reward based on progress toward the goal
            dones = calculate_done(scene_data)  # note: pretty expensive.
            rewards = scene_data.reward_history.calculate_reward_this_timestep(
                scene_data
            )

            # check if any entity is out of bounds
            out_of_bounds_fail = out_of_bounds(
                min=self.min_bounds,
                max=self.max_bounds,
                gs_entities=self.concurrent_scenes_data[scene_id].gs_entities,
            )

            reset_envs = dones | out_of_bounds_fail  # or any other failure mode.
            reset_by_scenes[scene_id] = reset_envs
            dones_by_scenes[scene_id] = dones
            rewards_by_scenes[scene_id] = rewards

        # reset at done
        if reset_by_scenes.any():
            self.reset_idx(reset_by_scenes.flatten().nonzero().squeeze(1))

        # Additional info for debugging
        info = {
            "diff": diff,
            "total_diff_left": total_diff_left,
            "success": success,
            "out_of_bounds": out_of_bounds_fail,
        }  # FIXME: does not account for multiple scenes, only for the last one executed.

        (
            voxel_init,
            voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
        ) = self._observe_all_scenes()
        print(
            f"step happened. Time elapsed: {str(time.perf_counter() - self.last_step_time)}s",
            f"Rewards: {rewards.mean().item()}",
            f"out_of_bounds_fail: {out_of_bounds_fail.float().mean().item()}",
        )
        self.last_step_time = time.perf_counter()

        return (
            voxel_init,
            voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
            rewards_by_scenes.flatten(),
            dones_by_scenes.flatten(),
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
        # env_per_scene_idx = env_in_scene_idx.reshape(self.concurrent_scenes, -1)
        # ^ not true because env_per_scene_idx is not a full scene id tensor, only partial. (won't work with e.g. scene_idx=[0])
        unique_scene_idx, counts = torch.unique(scene_idx, return_counts=True)

        # pad scene counts with zeros for absent in this batch envs
        counts_full = torch.zeros(
            self.concurrent_scenes, dtype=torch.int64, device=self.device
        )
        counts_full[unique_scene_idx] = counts

        # get data for the entire batch.
        reset_scene_data: list[ConcurrentSceneData | None] = (  # none if counts were 0.
            self.env_dataloader.get_processed_data(counts_full.to(dtype=torch.int16))
        )

        for scene_id in range(len(reset_scene_data)):  # iter over scene data
            reset_scene = reset_scene_data[scene_id]
            reset_env_ids_this_scene = env_in_scene_idx[scene_idx == scene_id]
            # ^ get reset_env_ids_this_scene as equality of scene_idx to scene_id.
            if reset_scene is None:
                assert reset_env_ids_this_scene.shape[0] == 0, (
                    "reset_env_ids_this_scene should be empty if reset_scene is None"
                )
                continue  # if no reset data was necessary, skip

            # self.concurrent_scenes_data[scene_id].scene.reset(
            #     envs_idx=reset_env_ids_this_scene
            # ) #FIXME: commented out, possibly is very bad!

            # update the scene
            # Reset robot joint positions to default
            dof_pos = self.default_dof_pos.expand(reset_scene.batch_dim, -1)

            self.concurrent_scenes_data[scene_id].scene.step()
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

            # # reset the constraints # FIXME: only on certain envs.
            # reset_constraints(updated_scene_data.scene)

            # Move entities to their starting positions
            move_entities_to_pos(
                updated_scene_data.gs_entities, updated_scene_data.current_state
            )

            create_constraints_based_on_graph(
                updated_scene_data.current_state,
                updated_scene_data.gs_entities,
                updated_scene_data.scene,
                env_idx=reset_env_ids_this_scene,
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
        (
            voxel_init,
            voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
        ) = self._observe_all_scenes()

        return (
            voxel_init,
            voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
        )

    def _observe_scene(self, scene_data: ConcurrentSceneData):
        # Process a single environment
        # Get the scene data for this environment

        # Extract cameras from scene data
        cameras = scene_data.scene.visualizer.cameras
        video_obs = _render_all_cameras(cameras)

        # get voxel obs
        sparse_voxel_init = scene_data.vox_init
        sparse_voxel_des = scene_data.vox_des

        # TODO partial export of graphs (avoid export when unnecessary.)
        # get graph obs
        elec_graph_init = [
            state.export_graph() for state in scene_data.current_state.electronics_state
        ]
        elec_graph_des = [
            state.export_graph() for state in scene_data.desired_state.electronics_state
        ]  # return them as lists because they will be stacked to batch as global env.
        mech_graph_init = [
            state.export_graph() for state in scene_data.current_state.physical_state
        ]
        mech_graph_des = [
            state.export_graph() for state in scene_data.desired_state.physical_state
        ]

        return (
            sparse_voxel_init,
            sparse_voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
        )

    def _observe_all_scenes(self):
        """A helper to merge all voxel and graph observations."""

        all_voxel_init = []
        all_voxel_des = []
        all_video_obs = []
        all_mech_graph_init = []
        all_mech_graph_des = []
        all_elec_graph_init = []
        all_elec_graph_des = []
        for scene_data in self.concurrent_scenes_data:
            (
                voxel_init,
                voxel_des,
                video_obs,
                mech_graph_init,
                mech_graph_des,
                elec_graph_init,
                elec_graph_des,
            ) = self._observe_scene(scene_data)
            all_voxel_init.append(voxel_init)
            all_voxel_des.append(voxel_des)
            all_video_obs.append(video_obs)
            all_mech_graph_init.extend(mech_graph_init)
            all_mech_graph_des.extend(mech_graph_des)
            all_elec_graph_init.extend(elec_graph_init)
            all_elec_graph_des.extend(elec_graph_des)

        voxel_init = torch.concat(all_voxel_init, dim=0)
        voxel_des = torch.concat(all_voxel_des, dim=0)
        mech_graph_init = Batch.from_data_list(
            all_mech_graph_init, follow_batch=["global_feat"]
        )
        mech_graph_des = Batch.from_data_list(
            all_mech_graph_des, follow_batch=["global_feat"]
        )
        elec_graph_init = Batch.from_data_list(all_elec_graph_init)
        elec_graph_des = Batch.from_data_list(all_elec_graph_des)

        # # add num_feat to batches (necessary for scatter_add.):
        # mech_graph_init.global_feat_batch = num_feat_to_batch(
        #     mech_graph_init.global_feat_count
        # )
        # elec_graph_init.global_feat_batch = num_feat_to_batch(
        #     elec_graph_init.global_feat_count
        # )
        # mech_graph_des.global_feat_batch = num_feat_to_batch(
        #     mech_graph_des.global_feat_count
        # )
        # elec_graph_des.global_feat_batch = num_feat_to_batch(
        #     elec_graph_des.global_feat_count
        # )

        video_obs = torch.cat(
            all_video_obs, dim=0
        )  # cat, not stack because it's already batched
        video_obs = video_obs.permute(0, 1, 4, 2, 3)  # to torch format

        return (
            voxel_init,
            voxel_des,
            video_obs,
            mech_graph_init,
            mech_graph_des,
            elec_graph_init,
            elec_graph_des,
        )


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


# def num_feat_to_batch(num_feat: torch.Tensor):
#     return torch.repeat_interleave(torch.arange(num_feat.shape[0]), num_feat)
