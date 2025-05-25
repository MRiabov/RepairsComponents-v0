import torch
import genesis as gs
import gym
from genesis.engine.entities import RigidEntity
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RepairsEnv(gym.Env):
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False
    ):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.device = gs.device
        self.dt = env_cfg.get("dt", 0.02)
        # store configs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            show_viewer=show_viewer,
        )
        # add ground
        self.scene.add_entity(gs.morphs.Plane())
        # add Franka arm
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )

        # build scene
        self.scene.build(n_envs=self.num_envs)
        # joint setup
        self.joint_names = env_cfg["joint_names"]
        self.dof_idx = [
            self.franka.get_joint(name).dof_start for name in self.joint_names
        ]
        # kp = env_cfg.get("kp", 100.0)
        # kd = env_cfg.get("kd", 2.0)
        # self.franka.set_dofs_kp([kp] * self.num_actions, self.dof_idx)
        # self.franka.set_dofs_kv([kd] * self.num_actions, self.dof_idx)
        # # default positions
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in self.joint_names],
            device=self.device,
        )

        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # buffers # necessary because real-life robots have a ~1-step delay
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)
        self.reset_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.extras = {"observations": {"critic": self.obs_buf}, "episode": {}}
        # initial reset
        self.reset()

    def step(self, actions, env_electronics_actions):
        # # apply position control
        # target = actions * self.env_cfg.get("action_scale", 1.0)
        # self.franka.control_dofs_position(target, self.dof_idx)
        # # step sim
        # self.scene.step()
        # # read state
        # self.dof_pos[:] = self.franka.get_dofs_position(self.dof_idx)
        # self.dof_vel[:] = self.franka.get_dofs_velocity(self.dof_idx)
        # # form obs
        # self.obs_buf = torch.cat([self.dof_pos, self.dof_vel], dim=-1)
        # # stub reward and done
        # self.rew_buf[:] = 0.0
        # self.reset_buf[:] = False

        # instead of per-pos RL, do inverse kinematics approach.
        # get pos and quat from action
        pos = actions[:3]
        quat = actions[3:]

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
            self.scene.step()  # It was possible to set more steps to execute per step, it's faster.

        # allow robot to reach the last waypoint
        for i in range(100):  # was 100
            self.scene.step()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) > 0:
            self.dof_pos[envs_idx] = self.default_dof_pos
            self.dof_vel[envs_idx] = 0.0
            self.franka.set_dofs_position(
                position=self.dof_pos[envs_idx],
                dofs_idx_local=self.dof_idx,
                zero_velocity=True,
                envs_idx=envs_idx,
            )

    def reset(self):
        # reset all envs
        idxs = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[:] = True
        self.reset_idx(idxs)
        # initial observation
        self.obs_buf = torch.cat([self.dof_pos, self.dof_vel], dim=-1)
        return self.obs_buf, None
