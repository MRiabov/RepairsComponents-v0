import argparse
import time
from pathlib import Path
from typing import Tuple

# Genesis imports (match style used in gym_env)
import genesis as gs
import numpy as np
import torch

from repairs_components.processing.geom_utils import euler_deg_to_quat_wxyz
from repairs_components.processing.tasks import AssembleTask, DisassembleTask
from repairs_components.training_utils.gym_env import RepairsEnv

# Use the existing keyboard listener
from repairs_components.training_utils.keyboard_teleop import KeyboardDevice

# Utilities


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Create quaternion (w,x,y,z) for rotation about Z axis by yaw radians."""
    # Use euler->quat util (degrees) with only yaw nonzero for robustness
    yaw_deg = torch.rad2deg(yaw)
    zeros = torch.zeros_like(yaw_deg)
    quat = euler_deg_to_quat_wxyz(torch.stack((zeros, zeros, yaw_deg), dim=-1))
    return quat


def get_default_configs(debug: bool = True):
    """
    Minimal configs adapted from tests/test_env_benchmark_cpu.py and
    neural_nets/sac_repairs_torch.py. Small obs if debug.
    """
    tasks = [AssembleTask(), DisassembleTask()]

    # Ensure we import TenHoles from THIS repo's examples directory (avoid picking the one in Repairs-v0).
    # Insert the local examples/ to sys.path and import the module directly.
    import sys
    from pathlib import Path as _Path

    _examples_dir = _Path(__file__).resolve().parents[2] / "examples"
    if str(_examples_dir) not in sys.path:
        sys.path.insert(0, str(_examples_dir))
    from ten_holes_14 import TenHoles  # type: ignore

    # ^bad code above.
    env_setups = [TenHoles()]

    env_cfg = {
        "num_actions": 10,  # [x,y,z, qw,qx,qy,qz, fastener_pick/release, screw_in/out, tool_pick/release]
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
        "num_steps_per_action": 3,
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
        "generate_number_of_configs_per_scene": 25,
        "dataloader_settings": {"prefetch_memory_size": 4},
        "data_dir": data_dir,
        "save_obs": {
            "video": True,
            "new_video_every": 1000,
            "video_len": 50,
            "voxel": True,
            "electronic_graph": True,
            "mechanics_graph": True,
            "path": f"{data_dir}/obs/",
            "show_fps": True,
        },
        "force_recreate_data": False,
        "env_setup_ids": [0],  # single concurrent scene
        "show_fps": True,
        "show_viewer": True,
        "run_in_thread": False,  # temporary fix for genesis visualizer to work.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # modest batch by default for interactive
    ml_batch_dim = 1

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


def build_action_from_keys(
    pressed_keys: set,
    pos: torch.Tensor,
    yaw: torch.Tensor,
    command_min: torch.Tensor,
    command_max: torch.Tensor,
    dpos: float = 0.01,
    dyaw: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update pos/yaw from keys and return a single action vector [10].
    - Arrow keys: XY
    - n/m: Z up/down
    - j/k: yaw +/-
    - f/g: fastener pick/release (actions[7] -> 1.0 / 0.0; neutral 0.5)
    - z/x: screw in/out (actions[8] -> 1.0 / 0.0; neutral 0.5)
    - t/y: tool pick/release (actions[9] -> 1.0 / 0.0; neutral 0.5)
    """
    from pynput import keyboard

    # Clone to avoid in-place on caller
    pos = pos.clone()
    yaw = yaw.clone()

    # Movement
    if keyboard.Key.up in pressed_keys:
        pos[0] -= dpos
    if keyboard.Key.down in pressed_keys:
        pos[0] += dpos
    if keyboard.Key.right in pressed_keys:
        pos[1] += dpos
    if keyboard.Key.left in pressed_keys:
        pos[1] -= dpos
    if keyboard.KeyCode.from_char("n") in pressed_keys:
        pos[2] += dpos
    if keyboard.KeyCode.from_char("m") in pressed_keys:
        pos[2] -= dpos

    # Rotation
    if keyboard.KeyCode.from_char("j") in pressed_keys:
        yaw += dyaw
    if keyboard.KeyCode.from_char("k") in pressed_keys:
        yaw -= dyaw

    # Clamp position to command bounds (first 3 entries)
    pos = torch.clamp(pos, command_min[:3], command_max[:3])

    # Orientation: build quaternion from yaw around Z
    quat = yaw_to_quat(yaw)

    # Discrete-ish action heads with neutral midpoint 0.5
    # 7: fastener pick/release, 8: screw in/out, 9: tool pick/release
    fastener_act = torch.tensor(0.5, device=pos.device, dtype=pos.dtype)
    screw_act = torch.tensor(0.5, device=pos.device, dtype=pos.dtype)
    tool_act = torch.tensor(0.5, device=pos.device, dtype=pos.dtype)

    # Keys mapping:
    # f: fastener pick up (->1.0), g: fastener release (->0.0)
    if keyboard.KeyCode.from_char("f") in pressed_keys:
        fastener_act = torch.tensor(1.0, device=pos.device, dtype=pos.dtype)
    if keyboard.KeyCode.from_char("g") in pressed_keys:
        fastener_act = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    # z: screw in (->1.0), x: screw out (->0.0)
    if keyboard.KeyCode.from_char("z") in pressed_keys:
        screw_act = torch.tensor(1.0, device=pos.device, dtype=pos.dtype)
    if keyboard.KeyCode.from_char("x") in pressed_keys:
        screw_act = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    # t: pick up tool (->1.0), y: release tool (->0.0)
    if keyboard.KeyCode.from_char("t") in pressed_keys:
        tool_act = torch.tensor(1.0, device=pos.device, dtype=pos.dtype)
    if keyboard.KeyCode.from_char("y") in pressed_keys:
        tool_act = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    action = torch.cat(
        (pos, quat, fastener_act.view(1), screw_act.view(1), tool_act.view(1))
    )
    # Ensure within min/max
    action = torch.max(action, command_min.to(action.dtype))
    action = torch.min(action, command_max.to(action.dtype))

    return action, yaw


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for RepairsEnv"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use smaller observations and lower offline data gen",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="ML batch dimension (default from config)",
    )
    parser.add_argument(
        "--hz", type=float, default=50.0, help="Target control frequency"
    )
    parser.add_argument("--no-viewer", action="store_true", help="Force viewer off")
    args = parser.parse_args()

    # Init Genesis (no viewer). Honor available backend.
    gs.init(
        backend=gs.cuda if torch.cuda.is_available() else gs.cpu,
        logging_level="debug",
    )

    (
        env_setups,
        tasks,
        env_cfg,
        obs_cfg,
        io_cfg,
        reward_cfg,
        command_cfg,
        ml_batch_dim_default,
    ) = get_default_configs(debug=bool(args.debug))
    ml_batch_dim = ml_batch_dim_default

    # Respect CLI no-viewer override
    if args.no_viewer:
        io_cfg["show_viewer"] = False

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

    # Reset once to build scenes and get to a consistent state
    env.reset()

    # Keyboard device
    kb = KeyboardDevice()
    kb.start()

    device = env.device
    dtype = torch.bfloat16

    # Initialize target state near center of command bounds
    cmin = command_cfg["min_bounds"].to(device=device, dtype=torch.float)
    cmax = command_cfg["max_bounds"].to(device=device, dtype=torch.float)

    pos = ((cmin[:3] + cmax[:3]) * 0.5).to(dtype)
    yaw = torch.tensor(0.0, device=device, dtype=dtype)

    print("Keyboard Teleop Controls:")
    print(
        "Arrows: XY | n/m: Z up/down | j/k: yaw +/- | f/g: fastener pick/release | z/x: screw in/out | t/y: tool pick/release | u: reset | esc: quit"
    )

    period = 1.0 / max(args.hz, 1e-3)
    stop = False
    try:
        while not stop:
            t0 = time.perf_counter()
            pressed = kb.get_cmd().copy()
            # Debug: print currently pressed keys (helps verify if viewer steals focus)
            if pressed:
                print("Pressed:", ", ".join(sorted(str(k) for k in pressed)))

            # Reset and quit
            from pynput import keyboard

            if keyboard.Key.esc in pressed:
                stop = True

            if keyboard.KeyCode.from_char("u") in pressed:
                env.reset()
                # re-center
                pos = ((cmin[:3] + cmax[:3]) * 0.5).to(dtype)
                yaw = torch.tensor(0.0, device=device, dtype=dtype)

            # Build single action from keys
            action_single, yaw = build_action_from_keys(
                pressed_keys=pressed,
                pos=pos,
                yaw=yaw,
                command_min=cmin,
                command_max=cmax,
                dpos=0.01,
                dyaw=0.05,
            )
            # Keep pos in caller updated
            pos = action_single[:3]
            # Debug: show desired target position
            if pressed:
                p = action_single[:3]
                print(
                    f"Desired pos: [{float(p[0]):.3f}, {float(p[1]):.3f}, {float(p[2]):.3f}]"
                )

            # Batch it
            action_batch = (
                action_single.to(dtype).to(device).unsqueeze(0).repeat(ml_batch_dim, 1)
            )

            # Step env
            (
                voxel_init_obs,
                voxel_des_obs,
                video_obs,
                mech_graph_init_obs,
                mech_graph_des_obs,
                elec_graph_init_obs,
                elec_graph_des_obs,
                rewards,
                dones,
                info,
            ) = env.step(action_batch)

            # Simple HUD
            r_mean = (
                float(rewards.mean().detach().cpu())
                if isinstance(rewards, torch.Tensor)
                else float(np.mean(rewards))
            )
            print(f"reward_mean={r_mean:.3f}\r", end="")

            # Sleep to maintain loop rate
            elapsed = time.perf_counter() - t0
            to_sleep = max(0.0, period - elapsed)
            time.sleep(to_sleep)
    finally:
        kb.stop()
        print("\nTeleop finished.")


if __name__ == "__main__":
    main()
