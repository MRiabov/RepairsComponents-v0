from abc import abstractmethod
from genesis import gs
from genesis.engine.entities import RigidEntity
import numpy as np
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
    # TODO assertion of similar orientaion and close position.
    tool_base_link = np.array(tool.base_link.idx_local)
    arm_hand_link = np.array(arm.get_link("hand").idx_local)

    arm_hand_pos = arm.get_pos(envs_idx=env_idx)
    arm_hand_quat = arm.get_quat(envs_idx=env_idx)

    # set the tool attachment link to the same position as the arm hand link
    tool.set_pos(arm_hand_pos, env_idx)  # I hope it works...
    tool.set_quat(arm_hand_quat, env_idx)
    scene.sim.rigid_solver.add_weld_constraint(
        np.expand_dims(tool_base_link, 0), np.expand_dims(arm_hand_link, 0), env_idx
    )


def detach_tool_from_arm(
    scene: gs.Scene, tool: RigidEntity, arm: RigidEntity, env_idx: torch.Tensor
):
    tool_base_link = np.array(tool.get_link(attachment_link_name).idx_local)
    arm_hand_link = np.array(arm.get_link("hand").idx_local)
    scene.sim.rigid_solver.delete_weld_constraint(
        tool_base_link, arm_hand_link, env_idx
    )
