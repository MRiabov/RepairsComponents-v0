from dataclasses import dataclass, field

import numpy as np
import torch

from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import ToolsEnum
from repairs_components.training_utils.sim_state import SimState


@dataclass
class ToolInfo:
    TOOLS_GRIPPER_POS: torch.Tensor = field(
        default_factory=lambda: torch.stack(
            [tool.tool_grip_position for tool in [Gripper(), Screwdriver()]]
        )
    )
    TOOLS_DIST_FROM_GRIP_LINK: torch.Tensor = field(
        default_factory=lambda: torch.stack(
            [tool.dist_from_grip_link for tool in [Gripper(), Screwdriver()]]
        )
    )
    tool_base_link_idx: torch.Tensor = field(
        default_factory=lambda: torch.empty(
            (len(ToolsEnum.__members__)), dtype=torch.int32
        )
    )  # note: there may be an issue if there are multiple tools to grip e.g. in a multimeter.
    """Base link indices for each tool, aligned with tool_ids' integer ids.
    Gripper (at [0]), has its link set as franka's hand."""


class ToolState(SimState):
    # note: default bdim = (), so no batch. this is expected.
    tool_ids: torch.Tensor = field(
        default_factory=lambda: torch.tensor(ToolsEnum.GRIPPER.value)
    )
    # NOTE: will be renamed to tool_ids (already was), leaving now for backward compatibility.
    gripper_tc: Gripper = field(default_factory=Gripper)
    "Gripper tool state which has `nan` on all values if it's not picked up. '_tc' because of tensorclass"
    screwdriver_tc: Screwdriver = field(default_factory=Screwdriver)
    "Screwdriver tool state which has `nan` on all values if it's not picked up. '_tc' because of tensorclass"

    def diff(self, other: "ToolState") -> tuple[dict[str, np.ndarray], int]:
        """Compute differences in tool state between two states."""
        raise NotImplementedError(
            "No need to call tool state diff, it shouldn't be in reward."
        )

    @staticmethod
    def rebuild_from_saved(current_tool_id: torch.Tensor) -> "ToolState":
        # def rebuild_from_saved(current_tool_id: int) -> "ToolState": # was.
        # if current_tool_id == ToolsEnum.GRIPPER.value:
        #     return ToolState(current_tool=Gripper())
        # elif current_tool_id == ToolsEnum.SCREWDRIVER.value:
        #     return ToolState(current_tool=Screwdriver())
        # else:
        #     raise ValueError(f"Invalid tool id: {current_tool_id}")

        tool_state = ToolState(device=current_tool_id.device).expand(
            current_tool_id.shape[0], -1
        )  # as before, it is unpopulated.
        tool_state.tool_ids = current_tool_id
        return tool_state
