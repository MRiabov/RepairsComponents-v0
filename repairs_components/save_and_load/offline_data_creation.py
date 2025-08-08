import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.processing.tasks import Task
from repairs_components.save_and_load.multienv_dataloader import RepairsEnvDataLoader


def create_data(
    scene_setups: list[EnvSetup],
    tasks: list[Task],
    scene_idx: torch.Tensor,
    num_configs_to_generate_per_scene: int | torch.Tensor,
    base_dir: Path,
):
    assert len(scene_idx) == len(scene_setups), (
        "Len of scene_idx and scene_setups must match."
    )
    assert all(scene.validate() for scene in scene_setups), "All scenes must be valid."
    if isinstance(num_configs_to_generate_per_scene, int):
        num_configs_to_generate_per_scene = torch.full(
            (len(scene_setups),), num_configs_to_generate_per_scene, dtype=torch.int16
        )

    # use online multienv dataloader to create data.
    # note: meshes will be saved to disk automatically if save_to_disk=True
    data_batches, mesh_file_names = RepairsEnvDataLoader(
        online=True,
        env_setups=scene_setups,
        tasks_to_generate=tasks,
        save_to_disk=True,
        offline_data_dir=base_dir,
    ).generate_sequential(num_configs_to_generate_per_scene)
    # debug
    assert mesh_file_names is not None, "Mesh file names must be provided"
    assert data_batches is not None, "Data batches must be provided"
    # /
    # create the (scene) data.
    for scene_id, scene_data in enumerate(data_batches):
        save_concurrent_scene_metadata(
            scene_data,
            base_dir,
            scene_id,
            mesh_file_name_mapping=mesh_file_names[scene_id],
            scene_setups=scene_setups,
        )


def save_concurrent_scene_metadata(
    data: ConcurrentSceneData,
    base_dir: Path,
    scene_id: int,
    mesh_file_name_mapping: dict[str, str],
    scene_setups: list[EnvSetup],
    env_idx: list[int] | None = None,
):
    """Save a ConcurrentSceneData instance to disk.

    scene_idx and env_idx are names to which save graph and voxel under."""
    if data.gs_entities is not None:  # not true during initial config gen
        assert len(data.gs_entities) == len(mesh_file_name_mapping), (
            f"Attempted to export meshes with gs_entities and mesh_file_name_mapping which do not match.\n"
            f"gs_entities: {data.gs_entities.keys()}\n"
            f"mesh_file_name_mapping: {mesh_file_name_mapping.keys()}"
        )
        assert set(data.gs_entities.keys()) == set(mesh_file_name_mapping.keys()), (
            f"Attempted to export meshes with gs_entities and mesh_file_name_mapping which do not match.\n"
            f"gs_entities: {data.gs_entities.keys()}\n"
            f"mesh_file_name_mapping: {mesh_file_name_mapping.keys()}"
        )
    assert isinstance(mesh_file_name_mapping, dict), (
        "mesh_file_name_mapping must be a dict"
    )
    if env_idx is None:
        env_idx = list(range(data.batch_dim))
    scene_dir = base_dir / f"scene_{scene_id}"
    os.makedirs(scene_dir, exist_ok=True)

    # states and voxel grids are already saved.

    # hole - persist tensors
    torch.save(data.starting_hole_positions, scene_dir / "starting_hole_positions.pt")
    torch.save(data.starting_hole_quats, scene_dir / "starting_hole_quats.pt")
    torch.save(data.hole_depth, scene_dir / "hole_depth.pt")
    torch.save(data.part_hole_batch, scene_dir / "part_hole_batch.pt")
    torch.save(data.hole_is_through, scene_dir / "hole_is_through.pt")

    # save metadata
    metadata = {
        "scene_id": scene_id,
        "task_ids": data.task_ids.tolist(),
        # note: graphs paths can now be recovered by get_graph_save_paths.
        "mesh_file_names": mesh_file_name_mapping,
        "electronics_indices": data.current_state.electronics_state[0].indices,
        "mechanical_indices": data.current_state.physical_state.body_indices,
        "count_generated_envs": data.batch_dim,
        "env_setup_name": scene_setups[scene_id].__class__.__name__,
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # initial diffs are to be recomputed;
    # cameras are to be loaded in gym_env.step (or elsewhere)
    # the exact scene is to be loaded elsewhere
    # except for voxels, current and desired state there is nothing to save.
