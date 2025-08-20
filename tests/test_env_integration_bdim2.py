import sys
from pathlib import Path

import os
import pytest
import torch

from genesis import gs  # type: ignore


# Local examples and env
from examples.ten_holes_14 import TenHoles  # type: ignore
from repairs_components.training_utils.gym_env import RepairsEnv
from repairs_components.processing.tasks import AssembleTask

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ENV_INTEGRATION") != "1",
    reason="Set RUN_ENV_INTEGRATION=1 to enable this slow integration test.",
)


def _minimal_configs(data_dir: Path):
    # Environment configuration (mirrors cpu benchmark defaults, trimmed)
    env_cfg = {
        "num_actions": 10,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, grip_L, grip_R, pick_tool]
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -2.0,
            "joint5": 0.0,
            "joint6": 2.0,
            "joint7": 0.79,
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
        "min_bounds": (-0.6, -0.7, -0.1),
        "max_bounds": (0.5, 0.5, 2.0),
    }

    # Small observation resolution for CI speed
    obs_cfg = {
        "num_obs": 3,
        "res": (64, 64),
        "use_random_textures": False,
    }

    # Simple reward config
    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,
    }

    # IO/data configuration (store under tmp path)
    (data_dir / "obs").mkdir(parents=True, exist_ok=True)
    io_cfg = {
        "generate_number_of_configs_per_scene": 1,
        "dataloader_settings": {"prefetch_memory_size": 1},
        "data_dir": str(data_dir),
        "save_obs": {
            "video": False,
            "new_video_every": 1000,
            "video_len": 10,
            "voxel": False,
            "electronic_graph": False,
            "mechanics_graph": False,
            "path": f"{data_dir}/obs/",
            "show_fps": False,
        },
        "force_recreate_data": False,
        "env_setup_ids": [0],
        "show_fps": False,
        "show_viewer": False,
    }

    # Command bounds
    device = torch.device("cpu")
    command_cfg = {
        "min_bounds": torch.tensor(
            [
                *(-0.8, -0.8, 0.0),
                *(-1.0, -1.0, -1.0, -1.0),
                *(0.0, 0.0),
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        "max_bounds": torch.tensor(
            [
                *(0.8, 0.8, 1.0),
                *(1.0, 1.0, 1.0, 1.0),
                *(1.0, 1.0),
                1.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    }

    # One setup, one task
    env_setups = [TenHoles()]
    tasks = [AssembleTask()]

    return env_setups, tasks, env_cfg, obs_cfg, io_cfg, reward_cfg, command_cfg


@pytest.mark.timeout(180)
def test_env_integration_bdim2_runs_two_steps(tmp_path: Path):
    """Smoke test: construct env with batch dim 2 and run two steps successfully."""
    # Initialize Genesis on CPU, headless
    gs.init(backend=gs.cpu, logging_level="warning", performance_mode=True)

    env_setups, tasks, env_cfg, obs_cfg, io_cfg, reward_cfg, command_cfg = (
        _minimal_configs(tmp_path / "repairs-data-ci")
    )

    ml_batch_dim = 2
    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        ml_batch_dim=ml_batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        io_cfg=io_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        num_scenes_per_task=1,
    )

    # Reset and get initial observation
    (
        voxel_init,
        voxel_des,
        video_obs,
        mech_graph_init,
        mech_graph_des,
        elec_graph_init,
        elec_graph_des,
    ) = env.reset()

    # Basic assertions on observation batch dims
    assert isinstance(video_obs, torch.Tensor)
    assert video_obs.ndim == 5  # (B, num_obs, C, H, W)
    assert video_obs.shape[0] == ml_batch_dim

    # Prepare actions within command bounds
    low = command_cfg["min_bounds"].to(dtype=torch.float32)
    high = command_cfg["max_bounds"].to(dtype=torch.float32)

    def rand_action():
        return (
            torch.rand((ml_batch_dim, env_cfg["num_actions"]), dtype=torch.float32)
            * (high - low)
            + low
        )

    # Step 1
    (
        voxel_init,
        voxel_des,
        video_obs,
        mech_graph_init,
        mech_graph_des,
        elec_graph_init,
        elec_graph_des,
        rewards,
        dones,
        info,
    ) = env.step(rand_action())

    assert isinstance(rewards, torch.Tensor) and rewards.shape == (ml_batch_dim,)
    assert isinstance(dones, torch.Tensor) and dones.shape == (ml_batch_dim,)
    assert isinstance(video_obs, torch.Tensor) and video_obs.shape[0] == ml_batch_dim

    # Step 2
    (
        voxel_init,
        voxel_des,
        video_obs,
        mech_graph_init,
        mech_graph_des,
        elec_graph_init,
        elec_graph_des,
        rewards,
        dones,
        info,
    ) = env.step(rand_action())

    # Final assertions after two steps
    assert rewards.shape == (ml_batch_dim,)
    assert dones.shape == (ml_batch_dim,)
    assert video_obs.shape[0] == ml_batch_dim
