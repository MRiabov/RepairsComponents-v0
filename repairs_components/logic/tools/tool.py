from abc import abstractmethod
from genesis import gs
from genesis.engine.entities import RigidEntity
import torch
from repairs_components.geometry.base import Component
from dataclasses import dataclass
from enum import Enum

attachment_link_name = "attachment_link"


class ToolsEnum(Enum):
    """A class useful for saving and reconstructing only."""

    GRIPPER = 0
    SCREWDRIVER = 1


@dataclass
class Tool(Component):
    name: str
    action_shape: int = 2
    active: bool = False

    @abstractmethod
    def step(self, action: torch.Tensor, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_from_grip_link(self):
        """Distance from the base link of the tool necessary to grip the tool."""
        raise NotImplementedError


def attach_tool_to_arm(
    scene: gs.Scene, tool: RigidEntity, arm: RigidEntity, env_idx: torch.Tensor
):
    device = env_idx.device
    # TODO assertion of similar orientaion and close position.
    tool_base_link = torch.tensor(tool.base_link.idx, device=device)
    arm_hand_link = torch.tensor(arm.get_link("hand").idx, device=device)

    arm_hand_pos = arm.get_pos(envs_idx=env_idx)
    arm_hand_quat = arm.get_quat(envs_idx=env_idx)

    # set the tool attachment link to the same position as the arm hand link
    tool.set_pos(arm_hand_pos, env_idx)  # I hope it works...
    tool.set_quat(arm_hand_quat, env_idx)

    # debuge
    assert (
        not torch.isnan(tool_base_link).any()
        and tool_base_link >= 0
        and tool_base_link <= scene.rigid_solver.n_links
    )
    assert (
        not torch.isnan(arm_hand_link).any()
        and arm_hand_link >= 0
        and arm_hand_link <= scene.rigid_solver.n_links
    )
    assert (
        not torch.isnan(env_idx).any()
        and (env_idx >= 0).all()
        and (env_idx < scene.n_envs).all()
    )

    scene.sim.rigid_solver.add_weld_constraint(
        tool_base_link.unsqueeze(0), arm_hand_link.unsqueeze(0), env_idx
    )


def detach_tool_from_arm(
    scene: gs.Scene, tool: RigidEntity, arm: RigidEntity, env_idx: torch.Tensor
):
    device = env_idx.device
    tool_base_link = torch.tensor(tool.base_link.idx, device=device)
    arm_hand_link = torch.tensor(arm.get_link("hand").idx, device=device)
    scene.sim.rigid_solver.delete_weld_constraint(
        tool_base_link.unsqueeze(0), arm_hand_link.unsqueeze(0), env_idx
    )
