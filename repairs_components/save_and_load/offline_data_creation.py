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

def save_sparse_tensor(tensor: Tensor, file_path: Path):
    """Utility method to save sparse tensors to disk."""  # deprecated?
    torch.save(tensor, file_path)


def save_concurrent_scene_data(
    data: ConcurrentSceneData, base_dir: Path, scene_idx: int, env_idx: list[int]
):
    """Save a ConcurrentSceneData instance to disk and return metadata.

    scene_idx and env_idx are names to which save graph and voxel under."""
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
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # initial diffs are to be recomputed;
    # cameras are to be loaded in gym_env.step (or elsewhere)
    # the exact scene is to be loaded elsewhere
    # except for voxels, current and desired state there is nothing to save.
