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
    if isinstance(num_configs_to_generate_per_scene, int):
        num_configs_to_generate_per_scene = torch.full(
            (len(scene_setups),), num_configs_to_generate_per_scene
        )

    # use online multienv dataloader to create data.
    data_batches = RepairsEnvDataLoader(
        online=True, env_setups=scene_setups, tasks_to_generate=tasks
    ).get_processed_data(num_configs_to_generate_per_scene)
    # create the (scene) data.
    for scene_data in data_batches:
        save_concurrent_scene_data(scene_data, base_dir, scene_idx.item())


def save_sparse_tensor(tensor: Tensor, file_path: Path):
    """Utility method to save sparse tensors to disk."""  # deprecated?
    torch.save(tensor, file_path)


def save_concurrent_scene_data(
    data: ConcurrentSceneData,
    base_dir: Path,
    scene_idx: int,
    env_idx: list[int] | None = None,
):
    """Save a ConcurrentSceneData instance to disk.

    scene_idx and env_idx are names to which save graph and voxel under."""

    if env_idx is None:
        env_idx = list(range(data.batch_dim))
    scene_dir = base_dir / f"scene_{scene_idx}"
    os.makedirs(scene_dir, exist_ok=True)

    # Save states
    mechanical_name_mapping, electronics_name_mapping = data.current_state.save(
        base_dir, scene_idx, torch.tensor(env_idx)
    )
    _, _ = data.desired_state.save(base_dir, scene_idx, torch.tensor(env_idx))

    # Save voxel grids
    save_sparse_tensor(data.vox_init, scene_dir / "vox_init.npz")
    save_sparse_tensor(data.vox_des, scene_dir / "vox_des.npz")

    # save metadata
    metadata = {
        "scene_id": scene_idx,
        "electronics_graph_id_to_name": electronics_name_mapping,
        "mechanical_graph_id_to_name_mapping": mechanical_name_mapping,
        "task_ids": data.task_ids.tolist(),
        "mesh_file_names": data.mesh_file_names,
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # initial diffs are to be recomputed;
    # cameras are to be loaded in gym_env.step (or elsewhere)
    # the exact scene is to be loaded elsewhere
    # except for voxels, current and desired state there is nothing to save.
