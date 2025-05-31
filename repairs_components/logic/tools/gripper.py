from dataclasses import dataclass
import numpy as np
from repairs_components.logic.tools.tool import Tool


@dataclass
class Gripper(Tool):
    name: str = "gripper"
    action_shape: int = 2
    active: bool = True

    def step(self, action: np.ndarray, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError

    def bd_geometry(self):
        raise NotImplementedError(
            "Unnecessary bd geometry for gripper tool, use MJCF instead."
        )
