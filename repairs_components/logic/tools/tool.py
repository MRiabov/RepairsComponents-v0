from abc import abstractmethod
from genesis import gs
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink
import torch
from repairs_components.geometry.base import Component
from dataclasses import dataclass
from enum import Enum

attachment_link_name = "attachment_link"


class ToolsEnum(Enum):
    """A class useful for saving and reconstructing only."""

    GRIPPER = 0
    SCREWDRIVER = 1
    MULTIMETER = 2


@dataclass
class Tool(Component):
    id: int
    action_shape: int = 2  # unused.
    # active: bool = False # deprecated

    @abstractmethod
    def step(self, action: torch.Tensor, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError

    @staticmethod  # kind of abstract method.
    def dist_from_grip_link():
        """Distance from the base link of the tool necessary to grip the tool."""
        raise NotImplementedError

    @staticmethod  # kind of abstract method.
    def tool_grip_position():
        """Grip position that the tool can be gripped with relative to base link."""
        raise NotImplementedError

    @abstractmethod
    def on_tool_release(self):
        """Called when the tool is released."""
        raise NotImplementedError


def attach_tool_to_arm(
    scene: gs.Scene,
    tool_entity: RigidEntity,
    arm_hand_link: RigidLink,
    tool: Tool,
    env_idx: torch.Tensor,
):
    from repairs_components.processing.geom_utils import get_connector_pos
    from repairs_components.logic.tools.gripper import Gripper

    assert not isinstance(tool, Gripper), (
        "Can not attach a gripper - it is always attached."
    )

    # TODO assertion of similar orientaion and close position. # maybe it should be done via ompl?
    tool_base_link = tool_entity.base_link.idx

    arm_hand_pos = arm_hand_link.get_pos(env_idx)  # [b,1,3]
    arm_hand_quat = arm_hand_link.get_quat(env_idx)  # [b,1,4]

    tool_grip_pos = get_connector_pos(
        arm_hand_pos.squeeze(1),  # [b,3]
        arm_hand_quat.squeeze(1),  # [b,4]
        tool.tool_grip_position().unsqueeze(0),
    )  # place tool base so that hand aligns with tool grip: pos = hand_pos + R_hand*grip

    # FIXME: the tool is not repositioned to the entity, for whichever reason.

    arm_hand_quat = arm_hand_quat.squeeze(1)

    # set the tool attachment link to the same position as the arm hand link
    tool_entity.set_pos(tool_grip_pos, env_idx)
    tool_entity.set_quat(arm_hand_quat, env_idx)  #

    scene.sim.rigid_solver.add_weld_constraint(
        tool_base_link, arm_hand_link.idx, env_idx
    )


def detach_tool_from_arm(
    scene: gs.Scene,
    tool_entity: RigidEntity,
    arm_hand_link: RigidLink,
    gs_entities: dict[str, RigidEntity],
    tool_state_to_update: list[Tool],
    env_idx: torch.Tensor,
):
    # assert env_idx.shape[0] == 1, "Only one environment is supported for now."
    # ^ this will become valid if I have more tools.
    # batching is non-trivial on constraint add/remove.
    rigid_solver = scene.sim.rigid_solver
    tool_base_link = tool_entity.base_link.idx
    arm_hand_link = arm_hand_link.idx
    rigid_solver.delete_weld_constraint(tool_base_link, arm_hand_link, env_idx)
    # drop fasteners if present.
    for tool in tool_state_to_update:
        # that's only for a screwdriver, but could do.
        if tool.picked_up_fastener_name is not None:
            fastener_entity = gs_entities[tool.picked_up_fastener_name]
            rigid_solver.delete_weld_constraint(
                tool_base_link, fastener_entity.base_link.idx, env_idx
            )
        tool.on_tool_release()
        # for screwdriver:
