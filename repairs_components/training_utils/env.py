import random
import torch
import genesis as gs
import gym
from genesis.engine.entities import RigidEntity
import numpy as np

from repairs_sim_step import step_repairs
from training_utils.translation import (
    translate_genesis_to_python,
    translate_to_genesis_scene,
)
from training_utils.reward_calc import calculate_reward
from build123d import Compound
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.processing.voxel_export import export_voxel_grid


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RepairsEnv(gym.Env):
    def __init__(
        self,
        env_setup: EnvSetup,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
    ):
        """Initialize the Repairs environment.

        Args:
            num_envs: Number of parallel environments to simulate
            env_cfg: Configuration dictionary containing environment parameters
            obs_cfg: Configuration for observations
            reward_cfg: Configuration for reward function
            command_cfg: Configuration for command inputs
            show_viewer: Whether to render the simulation viewer
        """
        # Store basic environment parameters
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.device = gs.device
        self.dt = env_cfg.get("dt", 0.02)  # Default to 50Hz if not specified

        # Store configuration dictionaries
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        # ===== Scene Setup =====
        # Create simulation scene with specified timestep and substeps
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            show_viewer=show_viewer,
        )

        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Add Franka Emika Panda robot arm from MJCF file
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )

        # Add custom components to the scene
        initial_b123d_assembly, self.current_sim_state, aux, self.cameras = (
            env_setup.starting_state(self.scene)
        )
        desired_b123d_assembly, self.desired_state, _ = env_setup.desired_state(
            self.scene
        )
        self.initial_voxel_grid = export_voxel_grid(
            initial_b123d_assembly, voxel_size=1
        )
        self.desired_voxel_grid = export_voxel_grid(
            desired_b123d_assembly, voxel_size=1
        )

        self.scene, self.hex_to_name = translate_to_genesis_scene(
            self.scene, initial_b123d_assembly, self.current_sim_state, aux
        )  # how to randomize?

        # Finalize scene construction for all environments
        self.scene.build(n_envs=self.num_envs)

        # ===== Robot Configuration =====
        # Setup joint names and their corresponding DOF indices
        self.joint_names = env_cfg["joint_names"]
        self.dof_idx = [
            self.franka.get_joint(name).dof_start for name in self.joint_names
        ]

        # Set default joint positions from config
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in self.joint_names],
            device=self.device,
        )

        # ===== Control Parameters =====
        # Set PD control gains (tuned for Franka Emika Panda)
        # These values are robot-specific and affect the stiffness and damping
        # of the robot's joints during control
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )

        # Set force limits for each joint (in Nm for rotational joints, N for prismatic)
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # Initialize environment to starting state
        self.reset()

    def step(self, action: torch.Tensor):
        success, total_diff_left, self.current_sim_state, diff = step_repairs(
            self.scene,
            action,
            self.hex_to_name,
            self.current_sim_state,
            self.desired_state,
        )

        # instead of per-pos RL, do inverse kinematics approach.
        # get pos and quat from action
        pos = action[:3]
        quat = action[3:]

        # get the end-effector link
        end_effector = self.franka.get_link("hand")

        # move to pre-grasp pose
        qpos = self.franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=quat,
        )
        # gripper open pos
        qpos[-2:] = 0.04  # open... but how to give the control to the model?
        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=100,
        )
        # execute the planned path
        for waypoint in path:
            self.franka.control_dofs_position(waypoint)
            self.scene.step()

        # allow robot to reach the last waypoint
        for _ in range(100):
            self.scene.step()

        # compute obs
        obs = []
        for camera in self.cameras:
            rgb, depth, segmentation, normal = (
                camera.render(depth=True, segmentation=True, normal=True),
            )
            cat_obs = torch.stack((rgb, depth, segmentation, normal))
            cat_obs = obs.append(cat_obs)
        obs = torch.stack(obs)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # compute reward
        reward = calculate_reward(
            self.current_sim_state,
            self.desired_state,
            self.initial_diff_count,
        )

        # False is for terminated/truncated.
        return obs, reward, done, False, {"diff": diff}

    def reset_idx(self, envs_idx):
        if len(envs_idx) > 0:
            dof_pos = self.default_dof_pos.expand(len(envs_idx), -1)
            self.franka.set_dofs_position(
                position=dof_pos,
                dofs_idx_local=self.dof_idx,
                zero_velocity=True,
                envs_idx=envs_idx,
            )
            env_id = torch.randint(0, len(envs_idx), (1,), device=self.device)
            # reset sim env state, repopulate the environment
            diff, self.initial_diff_count = self.current_sim_state.diff(
                self.desired_state
            )

    def reset(self):
        # reset all envs
        idxs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(idxs)
        return None, None
