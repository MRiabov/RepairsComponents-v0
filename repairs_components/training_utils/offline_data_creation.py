import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)


def save_env_configs_to_disk_and_save(
    env_setups: List[Any],
    tasks: List[Any],
    output_dir: Union[str, Path],
    num_configs_per_scene: int = 10,
) -> None:
    """
    Generate and save environment configurations to disk.
    We are saving:
    1. step files (in create_env_configs)
    2. voxel grids (in save_concurrent_scene_data)
    3. graphs of electronics, mechanical and possibly other connections
    4. scene metadata. Scene metadata also holds keys for graphs (!) (naming).
    this over the desired and current state."""
    from repairs_components.processing.scene_creation_funnel import create_env_configs

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate configs
    num_configs = torch.tensor([num_configs_per_scene] * len(env_setups))
    scene_configs = create_env_configs(env_setups, tasks, num_configs)

    # Save metadata
    metadata = {
        "num_scenes": len(scene_configs),
        "num_configs_per_scene": num_configs_per_scene,
    }

    # Save each scene
    for i, scene_config in enumerate(scene_configs):
        assert scene_config is not None, "Shouldn't save no configs."

        save_concurrent_scene_data(
            scene_config,
            output_dir,
            i,
            list(range(num_configs_per_scene)),
        )

    # Save metadata file
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_sparse_tensor(tensor: Tensor, file_path: Path):
    """Convert a sparse tensor to a serializable dictionary."""
    torch.save(tensor, file_path)


def save_concurrent_scene_data(
    data: ConcurrentSceneData, base_dir: Path, scene_idx: int, env_idx: list[int]
) -> Dict[str, Any]:
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
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # initial diffs are to be recomputed;
    # cameras are to be loaded in gym_env.step (or elsewhere)
    # the exact scene is to be loaded elsewhere
    # except for voxels, current and desired state there is nothing to save.
