import os
import sys
import time
import cProfile
import pstats
from pathlib import Path

import torch
import pytest

# Ensure Repairs-v0 is on the import path for examples/*, matching SAC defaults
REPAIRS_V0_ROOT = "/home/maksym/Work/Repairs-v0"
if REPAIRS_V0_ROOT not in sys.path:
    sys.path.append(REPAIRS_V0_ROOT)

# Genesis init
try:
    from genesis import gs
except Exception as e:  # pragma: no cover
    pytest.skip(f"Genesis not available: {e}", allow_module_level=True)

# Imports from both repos
from examples.ten_holes_14 import TenHoles  # type: ignore
from repairs_components.training_utils.gym_env import RepairsEnv
from repairs_components.processing.tasks import AssembleTask, DisassembleTask


def get_default_configs(debug: bool = True):
    """
    Mirror the defaults from neural_nets/sac_repairs_torch.py (__main__ block),
    but tuned for CPU benchmarking (debug=True => smaller obs).
    """
    # Tasks and env setups (SAC defaults)
    tasks = [AssembleTask(), DisassembleTask()]
    env_setups = [TenHoles()]

    # Env cfg (SAC defaults)
    env_cfg = {
        "num_actions": 10,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_left, gripper_right, pick_up_tool]
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
        "max_bounds": (0.5, 0.5, 2),
    }

    # Observation cfg (SAC defaults: 256x256; debug uses 64x64)
    obs_cfg = {
        "num_obs": 3,
        "res": (64, 64) if debug else (256, 256),
        "use_random_textures": False,
    }

    # Reward cfg (SAC defaults)
    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,
    }

    # IO/data cfg (SAC defaults)
    data_dir = "/home/maksym/Work/repairs-data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    io_cfg = {
        "generate_number_of_configs_per_scene": 8 if debug else 164,
        "dataloader_settings": {
            "prefetch_memory_size": 4 if debug else 512,
        },
        "data_dir": data_dir,
        "save_obs": {
            "video": False,
            "new_video_every": 1000,
            "video_len": 50,
            "voxel": False,
            "electronic_graph": False,
            "mechanics_graph": False,
            "path": f"{data_dir}/obs/",
            "show_fps": True,
        },
        "force_recreate_data": False,
        "env_setup_ids": [0],
        "show_fps": True,
    }

    # Command bounds (SAC defaults)
    device = torch.device("cpu")
    command_cfg = {
        "min_bounds": torch.tensor(
            [
                *(-0.8, -0.8, 0),
                *(-1.0, -1.0, -1.0, -1.0),
                *(0.0, 0.0),
                0.0,
            ],
            dtype=torch.float,
            device=device,
        ),
        "max_bounds": torch.tensor(
            [
                *(0.8, 0.8, 1.0),
                *(1.0, 1.0, 1.0, 1.0),
                *(1.0, 1.0),
                1.0,
            ],
            dtype=torch.float,
            device=device,
        ),
    }

    # Batch size (SAC uses 128 non-debug; use modest batch for CPU throughput)
    ml_batch_dim = 64 if debug else 128

    return (
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        io_cfg,
        reward_cfg,
        command_cfg,
        ml_batch_dim,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.backends.mkldnn.is_available(), reason="Expecting CPU with decent perf"
)
def test_env_benchmark_cpu():
    # Init Genesis on CPU (no viewer)
    gs.init(backend=gs.cpu, logging_level="info", performance_mode=True)

    (
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        io_cfg,
        reward_cfg,
        command_cfg,
        ml_batch_dim,
    ) = get_default_configs(debug=True)

    # Create environment
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

    # Reset and warm up
    env.reset()
    action_low = command_cfg["min_bounds"].to(torch.bfloat16)
    action_high = command_cfg["max_bounds"].to(torch.bfloat16)

    steps_warmup = 20
    for _ in range(steps_warmup):
        actions = (
            torch.rand((ml_batch_dim, env_cfg["num_actions"]), dtype=torch.bfloat16)
            * (action_high - action_low)
            + action_low
        )
        env.step(actions)

    # Profile + time
    steps_bench = 200
    # pr = cProfile.Profile()
    # pr.enable()

    t0 = time.perf_counter()
    for _ in range(steps_bench):
        actions = (
            torch.rand((ml_batch_dim, env_cfg["num_actions"]), dtype=torch.bfloat16)
            * (action_high - action_low)
            + action_low
        )
        env.step(actions)
    elapsed = time.perf_counter() - t0

    # pr.disable()

    total_frames = steps_bench * ml_batch_dim
    fps = total_frames / max(elapsed, 1e-9)
    print(f"Env benchmark: {total_frames} frames in {elapsed:.3f}s => {fps:.1f} FPS")

    # Print top hotspots
    # stats = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    # stats.print_stats(25)

    # Expect at least 500 FPS total on CPU
    assert fps >= 500.0, f"Expected >= 500 FPS on CPU, got {fps:.1f}"


# NOTE: my code runs at 27.54 FPS on 64 envs (0.42 FPS per env)
if __name__ == "__main__":
    test_env_benchmark_cpu()
