import math

import genesis as gs
import torch
from torch.utils.data import IterableDataset, get_worker_info

from repairs_components.processing.scene_creation_funnel import (
    create_random_scenes,
    merge_global_states,
    move_entities_to_pos,
)
from repairs_components.processing.tasks import Task
from repairs_components.processing.translation import translate_to_genesis_scene
from repairs_components.training_utils.env_setup import EnvSetup


class RepairsEnvDataset(IterableDataset):
    def __init__(
        self,
        scenes: list[gs.Scene],  # len(scenes) == num_workers
        env_setups: list[EnvSetup],
        tasks: list[Task],
        batch_dim: int = 128,
        num_scenes_per_task: int = 128,
        batches_in_memory_per_scene: int = 8,
    ):
        super().__init__()
        self.scenes = scenes
        self.env_setups = env_setups
        self.tasks = tasks
        self.batch_dim = batch_dim
        self.num_scenes_per_task = num_scenes_per_task
        self.batches_in_memory_per_scene = batches_in_memory_per_scene

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Running in “single-process” mode: treat as worker 0
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Enforce that DataLoader(num_workers) == len(self.scenes)
        assert num_workers == len(self.scenes), (
            f"Expected num_workers={len(self.scenes)}, but got {num_workers}."
        )

        # Identify which scene this worker is responsible for:
        scene_idx = worker_id
        scene_template = self.scenes[scene_idx]
        env_setup = self.env_setups[scene_idx % len(self.env_setups)]

        # Generate one “chunk” of data for this scene:
        (
            _,
            _,
            _,
            starting_states,
            desired_states,
            voxel_grids_initial,
            voxel_grids_desired,
        ) = create_random_scenes(
            empty_scene=scene_template,
            env_setup=env_setup,
            tasks=self.tasks,
            batch_dim=self.batch_dim,
            num_scenes_per_task=self.num_scenes_per_task,
        )

        

        num_batches = len(starting_states) // self.batch_dim
        # Grab “initial_entities” once so you can reset the scene before each batch if needed:
        initial_entities = starting_states[0].entities

        for i in range(num_batches):
            start = i * self.batch_dim
            end = (i + 1) * self.batch_dim

            batch_start = merge_global_states(starting_states[start:end])
            batch_desired = merge_global_states(desired_states[start:end])
            vox_init = voxel_grids_initial[start:end]
            vox_des = voxel_grids_desired[start:end]

            # (Re‐initialize the scene’s object positions before yielding a new training sample)
            scene = move_entities_to_pos(scene_template, batch_start)

            # Yield a tuple of (scene, batch_start, batch_desired, voxel_init, voxel_des)
            yield (scene, batch_start, batch_desired, vox_init, vox_des)
