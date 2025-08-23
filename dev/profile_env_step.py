import sys
import time
from pathlib import Path

import torch

# Ensure local examples/ is importable for TenHoles
from examples.ten_holes_14 import TenHoles  # type: ignore

from repairs_components.processing.tasks import AssembleTask, DisassembleTask
from repairs_components.training_utils.gym_env import RepairsEnv


def get_default_configs(debug: bool = True):
    """
    Mirror tests/test_env_benchmark_cpu.py defaults, with explicit
    num_steps_per_action and profiling flags.
    """
    tasks = [AssembleTask(), DisassembleTask()]
    env_setups = [TenHoles()]

    env_cfg = {
        "num_actions": 10,
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
        "num_steps_per_action": 3,  # crucial for motion planning stepping
        "profile": True,
        "verbose": False,
    }

    obs_cfg = {
        "num_obs": 3,
        "res": (64, 64) if debug else (256, 256),
        "use_random_textures": False,
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,
    }

    data_dir = "/home/maksym/Work/repairs-data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    io_cfg = {
        "generate_number_of_configs_per_scene": 8 if debug else 164,
        "dataloader_settings": {"prefetch_memory_size": 4 if debug else 512},
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
        "show_viewer": False,
    }

    device = torch.device("cpu")
    command_cfg = {
        "min_bounds": torch.tensor(
            [
                *(-0.8, -0.8, 0.0),
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

    ml_batch_dim = 1

    return env_setups, tasks, env_cfg, obs_cfg, io_cfg, reward_cfg, command_cfg, ml_batch_dim


def main():
    try:
        import genesis as gs
    except Exception as e:  # pragma: no cover
        print(f"Genesis not available: {e}")
        sys.exit(1)

    # Init Genesis CPU, no viewer
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

    env.reset()

    action_low = command_cfg["min_bounds"].to(torch.bfloat16)
    action_high = command_cfg["max_bounds"].to(torch.bfloat16)

    def sample_actions():
        return (
            torch.rand((ml_batch_dim, env_cfg["num_actions"]), dtype=torch.bfloat16)
            * (action_high - action_low)
            + action_low
        )

    # Warmup
    for _ in range(20):
        env.step(sample_actions())

    # Bench + aggregate profiling
    steps_bench = 100
    sums = None
    count = 0
    t0 = time.perf_counter()
    for _ in range(steps_bench):
        out = env.step(sample_actions())
        info = out[9]
        prof = info.get("profile") if isinstance(info, dict) else None
        if prof:
            if sums is None:
                sums = {k: float(v) for k, v in prof.items()}
            else:
                for k, v in prof.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
            count += 1
    elapsed = time.perf_counter() - t0

    total_frames = steps_bench * ml_batch_dim
    fps = total_frames / max(elapsed, 1e-9)
    print(f"Benchmark: {total_frames} frames in {elapsed:.3f}s => {fps:.1f} FPS")

    if sums and count > 0:
        print("\nPer-step mean timings (s):")
        means = {k: v / count for k, v in sums.items()}
        total = means.get("total_step_s", sum(means.values()))
        for k in sorted(means.keys()):
            pct = (means[k] / total * 100.0) if total > 0 else 0.0
            print(f"  {k:20s} {means[k]:.6f} s  ({pct:5.1f}%)")

        # Top 3 contributors
        top = sorted(((k, v) for k, v in means.items() if k != "total_step_s"), key=lambda kv: kv[1], reverse=True)[:3]
        print("\nTop contributors:")
        for k, v in top:
            pct = (v / total * 100.0) if total > 0 else 0.0
            print(f"  {k:20s} {v:.6f} s  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
