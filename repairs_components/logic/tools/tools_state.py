from repairs_components.training_utils.sim_state import SimState
import numpy as np
from repairs_components.logic.tools.tool import Tool, ToolsEnum
from dataclasses import dataclass
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import Screwdriver


@dataclass
class ToolState(SimState):
    current_tool: Tool = Gripper()
    current_tool_id: int = ToolsEnum.GRIPPER.value
    all_tools = {  # for convenience only (for now)
        "gripper": Gripper(),
        "screwdriver": Screwdriver(),
    }

    def diff(self, other: "ToolState") -> tuple[dict[str, np.ndarray], int]:
        """Compute differences in tool state between two states."""
        # if self.current_tool.name != other.current_tool.name:
        #     return {"current_tool": self.current_tool.name}, 1
        # return {}, 0
        raise NotImplementedError(
            "No need to call tool state diff, it shouldn't be in reward."
        )

    @staticmethod
    def rebuild_from_saved(current_tool_id: int) -> "ToolState":
        if current_tool_id == ToolsEnum.GRIPPER.value:
            return ToolState(current_tool=Gripper(), current_tool_id=current_tool_id)
        elif current_tool_id == ToolsEnum.SCREWDRIVER.value:
            return ToolState(
                current_tool=Screwdriver(), current_tool_id=current_tool_id
            )
        else:
            raise ValueError(f"Invalid tool id: {current_tool_id}")
