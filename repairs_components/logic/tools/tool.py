from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch
from genesis import gs
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink

from repairs_components.geometry.base import Component

if TYPE_CHECKING:
    from repairs_components.training_utils.sim_state_global import (
        RepairsSimInfo,
        RepairsSimState,
    )
    from repairs_components.logic.tools.tools_state import ToolInfo, ToolState

attachment_link_name = "attachment_link"


class ToolsEnum(Enum):
    """A class useful for saving and reconstructing only."""

    GRIPPER = 0
    SCREWDRIVER = 1
    MULTIMETER = 2


@dataclass
class Tool(Component):
    @property
    @abstractmethod
    def id(self):
        """ID of a tool as per ToolsEnum"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: torch.Tensor, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_from_grip_link(self):
        """Distance from the base link of the tool necessary to grip the tool."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tool_grip_position(self):
        """Grip position that the tool can be gripped with relative to base link."""
        raise NotImplementedError

    @abstractmethod
    def on_tool_release(self):
        """Called when the tool is released."""
        raise NotImplementedError


def attach_tool_to_arm(
    scene: gs.Scene,
    sim_state: RepairsSimState,
    sim_info: RepairsSimInfo,
    env_idx: torch.Tensor,
):
    from repairs_components.processing.geom_utils import get_connector_pos

    assert (sim_state.tool_state.tool_ids[env_idx] != ToolsEnum.GRIPPER.value).all(), (
        "Can not attach a gripper - it is always attached."
    )

    # TODO assertion of similar orientaion and close position. # maybe it should be done via ompl?
    tool_base_link = sim_info.tool_info.tool_base_link_idx[
        sim_state.tool_state.tool_ids[env_idx]
    ]
    arm_hand_link = sim_info.tool_info.tool_base_link_idx[ToolsEnum.GRIPPER.value]
    arm_hand_pos = sim_state.tool_state.pos[env_idx, ToolsEnum.GRIPPER.value]  # [b,3]
    arm_hand_quat = sim_state.tool_state.quat[env_idx, ToolsEnum.GRIPPER.value]  # [b,4]

    # darn, I'll need to get tool_grip_position based on tool_state_to_update.tool_ids somehow.
    # Lazy import to avoid circular dependency

    ids_index = sim_state.tool_state.tool_ids[env_idx]
    rel_grip_offsets = sim_info.tool_info.TOOLS_GRIPPER_POS[ids_index]  # [k,3]
    tool_grip_pos = get_connector_pos(
        arm_hand_pos, arm_hand_quat, rel_grip_offsets
    )  # place tool base so that hand aligns with tool grip: pos = hand_pos + R_hand*grip

    # FIXME: the tool is not repositioned to the entity, for whichever reason.

    # set the tool attachment link to the same position as the arm hand link
    scene.rigid_solver.set_base_links_pos(tool_grip_pos, tool_base_link, env_idx)
    scene.rigid_solver.set_base_links_quat(arm_hand_quat, tool_base_link, env_idx)

    scene.sim.rigid_solver.add_weld_constraint(tool_base_link, arm_hand_link, env_idx)
    # Keep the existing tool id as provided by caller/test; do not overwrite here.


def detach_tool_from_arm(
    scene: gs.Scene,
    sim_state: RepairsSimState,
    sim_info: RepairsSimInfo,
    tool_entity: RigidEntity,
    arm_hand_link: RigidLink,
    tool_state_to_update: ToolState,
    env_idx: torch.Tensor,  # [B]
):
    # assert env_idx.shape[0] == 1, "Only one environment is supported for now."
    # ^ this will become valid if I have more tools.
    # batching is non-trivial on constraint add/remove.
    rigid_solver = scene.sim.rigid_solver
    tool_base_link = tool_entity.base_link.idx
    arm_hand_link = arm_hand_link.idx
    rigid_solver.delete_weld_constraint(tool_base_link, arm_hand_link, env_idx)
    # Drop fasteners if present (we use -1 sentinel for "none").
    fastener_ids = tool_state_to_update.screwdriver_tc.picked_up_fastener_id  # [B]
    fasteners_present = fastener_ids[env_idx] >= 0  # [k]
    if fasteners_present.any():
        valid_envs = env_idx[fasteners_present]
        valid_fastener_ids = fastener_ids[valid_envs]
        fastener_base_links = sim_info.physical_info.fastener_base_link_idx[
            valid_fastener_ids
        ]
        rigid_solver.delete_weld_constraint(
            tool_base_link,
            fastener_base_links,
            valid_envs,
        )
        tool_state_to_update.screwdriver_tc.on_tool_release(valid_envs)
        # for screwdriver:
