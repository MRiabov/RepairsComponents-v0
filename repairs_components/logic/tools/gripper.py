import numpy as np
from repairs_components.logic.tools.tool import Tool, ToolsEnum


class Gripper(Tool):  # tensorclass by inheritance!
    @property
    def id(self):
        return ToolsEnum.GRIPPER.value

    def step(self, action: np.ndarray, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError

    def bd_geometry(self):
        raise NotImplementedError(
            "Unnecessary bd geometry for gripper tool, use MJCF instead."
        )

    @property
    def dist_from_grip_link(self):
        raise NotImplementedError("Unnecessary for gripper tool.")

    @property
    def tool_grip_position(self):
        raise NotImplementedError("Unnecessary for gripper tool.")

    def on_tool_release(self):
        raise NotImplementedError("Unnecessary for gripper tool.")
