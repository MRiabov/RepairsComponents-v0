from typing import Optional
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from repairs_components.training_utils.sim_state_global import RepairsSimState
from genesis.vis.visualizer import Camera

# Executor for asynchronous saving of tensors
env_executor = ThreadPoolExecutor(max_workers=4)
# note: can render async too.


def _save_tensor(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)


def optional_save(
    save_any: bool,
    save_voxel: bool = False,
    save_image: bool = False,
    save_state: bool = False,
    save_video: bool = False,
    video_cams: list[Camera] | None = None,
    save_video_every_steps=-1,
    video_len=-1,
    current_step=-1,
    sim_state: Optional[RepairsSimState] = None,
    obs_image: Optional[torch.Tensor] = None,
    voxel_grids_initial: Optional[torch.Tensor] = None,
    voxel_grids_desired: Optional[torch.Tensor] = None,
    save_path: str = "./data/obs/video",
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
        if save_video:
            assert (
                video_cams is not None
                and video_len > 0
                and save_video_every_steps > 0
                and current_step != -1
            ), (
                "video_cams, video_len, current_step and save_video_every_steps must be provided to save_video, got: "
                f"video_cams={video_cams}, video_len={video_len}, save_video_every_steps={save_video_every_steps}, current_step={current_step}"
            )
            assert save_video_every_steps > video_len, (
                "save_video_every_steps must be greater than video_len"
            )  # save video every save_video_every_steps [1000] steps with len video_len [50].
            if current_step % save_video_every_steps == 0:
                for cam in video_cams:
                    cam.start_recording()
                    # note: can render async too, see gs examples.
            if current_step % save_video_every_steps == video_len - 1:
                for cam_id, cam in enumerate(video_cams):
                    video_idx = (
                        current_step // save_video_every_steps
                    ) * save_video_every_steps
                    cam.stop_recording(
                        f"{save_path}/video_{video_idx:04d}_cam{cam_id}.mp4"
                    )
