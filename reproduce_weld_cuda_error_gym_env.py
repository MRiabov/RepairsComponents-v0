import os

# NOTE: are these configs what throws?
os.environ["PYOPENGL_PLATFORM"] = "egl"  # 7.11 disabled this... who knows.
os.environ["EGL_PLATFORM"] = "surfaceless"

import torch
import genesis as gs
from examples.clamp_plates_49 import ClampPlates
from repairs_components.geometry.fasteners import Fastener
from repairs_components.processing.scene_creation_funnel import move_entities_to_pos
from repairs_components.processing.tasks import AssembleTask
from repairs_components.processing.translation import create_constraints_based_on_graph
from repairs_components.training_utils.sim_state_global import RepairsSimState
from genesis.engine.entities import RigidEntity
from repairs_components.training_utils.gym_env import RepairsEnv


# initialize configs

# Initialize Genesis
gs.init(
    backend=gs.cuda,
    logging_level="warning",  # logging_level="debug",
    # performance_mode=True,
    # debug=True,
)
# Create task and environment setup
tasks = [AssembleTask()]
env_setups = [ClampPlates()]
assert len(env_setups) == 1, "Only one environment setup is supported for now."
# TODO robot selection (franka/humanoid)
ml_batch_dim = 8

debug = True  # True
force_recreate_data = False  # True
# Note: set force_recreate_data to True after non-debug runs to remove large config files.

# Environment configuration
env_cfg = {
    "num_actions": 10,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_force_left, gripper_force_right, pick_up_tool]
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
        "joint7": 0.79,  # no "hand" here? there definitely was hand.
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    },
    "min_bounds": (-0.6, -0.7, -0.1),
    "max_bounds": (0.5, 0.5, 2),
}

obs_cfg = {
    "num_obs": 3,  # RGB, depth, segmentation
    "res": (256, 256) if not debug else (64, 64),
    "use_random_textures": False,
}

reward_cfg = {
    "success_reward": 10.0,
    "progress_reward_scale": 1.0,
    "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
}

io_cfg = {
    "generate_number_of_configs_per_scene": 164
    if not debug
    else 8,  # note: strange shape to debug
    "dataloader_settings": {
        "prefetch_memory_size": 512 if not debug else 4  # 256 environments per scene.
    },  # note^ 4 is for faster env spinup.
    "data_dir": "/workspace/data",
    "save_obs": {
        # "video": True,
        # "voxel": True,
        # "electronic_graph": True,
        # "path": "./obs/",
        "video": False,  # not flooding the disk..
        "new_video_every": 1000,
        "video_len": 50,
        "voxel": False,
        "electronic_graph": False,
        "mechanics_graph": False,
        "path": "/workspace/data/obs/",
    },
    "force_recreate_data": force_recreate_data,
    "env_setup_ids": list(range(len(env_setups))),
    "show_fps": True,
}

command_cfg = {
    "min_bounds": [
        *(-0.5, -0.5, 0),  # XYZ position min
        *(0, 0, 0, 0),  # Quaternion components (w,x,y,z) min
        *(0.0, 0.0),  # Gripper control min
        0.0,  # tool control min (0.5 denotes pick up tool)
    ],
    "max_bounds": [
        *(0.5, 0.5, 0.8),  # XYZ position max
        # ^note: xyz is dep
        *(1.0, 1.0, 1.0, 1.0),
        *(1.0, 1.0),  # Quaternion components (w,x,y,z) max
        1.0,  # tool control max (0.5 denotes pick up tool)
    ],
}
# initialize env

gym_env = RepairsEnv(
    env_setups, tasks, ml_batch_dim, env_cfg, obs_cfg, io_cfg, reward_cfg, command_cfg
)

for i in range(10):
    rand_act = torch.rand((ml_batch_dim, env_cfg["num_actions"])) * (
        torch.tensor(command_cfg["max_bounds"])
        - torch.tensor(command_cfg["min_bounds"])
    ) + torch.tensor(command_cfg["min_bounds"])
    gym_env.step(action=rand_act)

    # gym_env.reset()

    # next: I'll have to
