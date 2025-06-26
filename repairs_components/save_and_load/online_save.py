from typing import Optional
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from repairs_components.training_utils.sim_state_global import RepairsSimState

# Executor for asynchronous saving of tensors
env_executor = ThreadPoolExecutor(max_workers=4)


def _save_tensor(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)


def optional_save(
    save_any: bool,
    save_voxel: bool = False,
    save_image: bool = False,
    save_state: bool = False,
    sim_state: Optional[RepairsSimState] = None,
    obs_image: Optional[torch.Tensor] = None,
    voxel_grids_initial: Optional[torch.Tensor] = None,
    voxel_grids_desired: Optional[torch.Tensor] = None,
    save_path: str = "./dataset/render",
):
    """Optionally save the state to a JSON file, and the observation image."""

    if save_any:
        if save_state:
            sim_state.save(save_path, scene_id, init=True)  # saves graphs too
        if save_image:
            assert obs_image is not None, "obs_image must be provided to save_image"
            img_path = os.path.join(save_path, "obs_image.pt")
            env_executor.submit(_save_tensor, obs_image, img_path)

        if save_voxel:
            # note: voxels are already saved to temppaths, by the way.
            assert voxel_grids_initial is not None, (
                "voxel_grids_initial must be provided to save_voxel"
            )
            init_path = os.path.join(save_path, "voxel_initial.pt")
            env_executor.submit(_save_tensor, voxel_grids_initial, init_path)
            assert voxel_grids_desired is not None, (
                "voxel_grids_desired must be provided to save_voxel"
            )
            des_path = os.path.join(save_path, "voxel_desired.pt")
            env_executor.submit(_save_tensor, voxel_grids_desired, des_path)
