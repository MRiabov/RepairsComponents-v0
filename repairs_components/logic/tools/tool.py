from abc import abstractmethod
import numpy as np
from repairs_components.geometry.base import Component
from dataclasses import dataclass


@dataclass
class Tool(Component):
    name: str
    action_shape: int = 2
    active: bool = False

    @abstractmethod
    def step(self, action: np.ndarray, state: dict):
        """Step the tool with the given action."""
        raise NotImplementedError
