import torch
from repairs_components.training_utils.sim_state import SimState
import numpy as np
from dataclasses import dataclass, field
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import ToolsEnum, Tool
from tensordict import TensorClass


class ToolState(TensorClass, SimState):
    tool_ids = torch.tensor(ToolsEnum.GRIPPER.value)  # scalar to be batched.
    # NOTE: will be renamed to tool_ids (already was), leaving now for backward compatibility.
    gripper_tc = Gripper()
    screwdriver_tc = Screwdriver()

    def diff(self, other: "ToolState") -> tuple[dict[str, np.ndarray], int]:
        """Compute differences in tool state between two states."""
        # if self.current_tool.name != other.current_tool.name:
        #     return {"current_tool": self.current_tool.name}, 1
        # return {}, 0
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

        return ToolState(tool_ids=current_tool_id)  # as before, it is unpopulated.
