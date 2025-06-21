from abc import abstractmethod
from genesis import gs
from genesis.engine.entities import RigidEntity
import numpy as np
from repairs_components.geometry.base import Component
from dataclasses import dataclass
from enum import Enum


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
    def step(self, action: np.ndarray, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError


def attach_tool_to_arm(scene: gs.Scene, tool: RigidEntity, arm: RigidEntity):
    # TODO assertion of similar orientaion and close position.
    tool_base_link = np.array(tool.base_link.idx)
    arm_hand_link = np.array(arm.get_link("hand").idx)
    scene.sim.rigid_solver.add_weld_constraint(
        tool_base_link, arm_hand_link
    )  # works is genesis's examples.


def detach_tool_from_arm(scene: gs.Scene, tool: RigidEntity, arm: RigidEntity):
    tool_base_link = np.array(tool.base_link.idx)
    arm_hand_link = np.array(arm.get_link("hand").idx)
    scene.sim.rigid_solver.delete_weld_constraint(tool_base_link, arm_hand_link)
