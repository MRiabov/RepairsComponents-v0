from repairs_components.training_utils.sim_state import SimState
import numpy as np
from repairs_components.logic.tools.tool import Tool


class ToolState(SimState):
    current_tool: Tool

    def diff(self, other: "ToolState") -> tuple[dict[str, np.ndarray], int]:
        """Compute differences in tool state between two states."""
        # if self.current_tool.name != other.current_tool.name:
        #     return {"current_tool": self.current_tool.name}, 1
        # return {}, 0
        raise NotImplementedError(
            "No need to call tool state diff, it shouldn't be in reward."
        )
