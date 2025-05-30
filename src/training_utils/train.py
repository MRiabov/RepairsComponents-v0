import argparse
import os
import pickle
import shutil
from importlib import metadata

from training_utils.env_setup import EnvSetup

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from src.training_utils.env import RepairsEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    # Franka arm configuration
    env_cfg = {
        "num_actions": 7,
        "default_joint_angles": {f"panda_joint{i + 1}": 0.0 for i in range(7)},
        "joint_names": [f"panda_joint{i + 1}" for i in range(7)],
        "kp": 100.0,
        "kd": 2.0,
        "termination_if_joint_exceed": 2.5,
        "episode_length_s": 10.0,
        "action_scale": 1.0,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
    }
    obs_cfg = {
        "num_obs": 14,
        "obs_scales": {"dof_pos": 1.0, "dof_vel": 0.1},
    }
    reward_cfg = {
        "tracking_sigma": 0.1,
        "reward_scales": {"tracking_joint_pos": -1.0, "action_rate": -0.01},
    }
    command_cfg = {"num_commands": 0}
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = RepairsEnv(
        env_setup=EnvSetup(),  # TODO: a mechanism for picking the envs at random. Or having them precompiled.
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    # runner.learn(
    #     num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    # )


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/franka_train.py
"""
