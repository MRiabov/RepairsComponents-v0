from repairs_components.training_utils.sim_state import SimState
import numpy as np
from dataclasses import dataclass
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import ToolsEnum, Tool


@dataclass
class ToolState(SimState):    

    current_tool: Tool = Gripper()

    @property
    def current_tool_id(self):
        return self.current_tool.id

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
            return ToolState(current_tool=Gripper())
        elif current_tool_id == ToolsEnum.SCREWDRIVER.value:
            return ToolState(current_tool=Screwdriver())
        else:
            raise ValueError(f"Invalid tool id: {current_tool_id}")
