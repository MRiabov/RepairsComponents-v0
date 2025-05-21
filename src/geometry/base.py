"""Base component class and core functionality for repair components.

This module provides base classes for creating components that can be used with the Genesis
physics simulator. Genesis supports both MJCF (MuJoCo) and UDRF (Unified Robot Description Format)
for model definition.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

import numpy as np
import genesis

if TYPE_CHECKING:
    from genesis.sim import Model, Data


class Component(ABC):
    """Base class for all repair components.

    This class defines the common interface and functionality for all repair components.
    """

    def __init__(self, name: str = None):
        """Initialize the component.

        Args:
            name: Optional name for the component. If None, a default name will be generated.
        """
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self._model = None
        self._data = None
        self._initialized = False

    @abstractmethod
    def to_mjcf(self) -> str:
        """Convert the component to MJCF XML string.

        Returns:
            str: MJCF XML string representation of the component.
        """
        pass

    def attach_to_model(self, model: "Model", data: "Data" = None) -> None:
        """Attach the component to a Genesis model.

        Args:
            model: The Genesis model to attach to.
            data: Optional Genesis data structure. If not provided, it will be created.
        """
        self._model = model
        if data is not None:
            self._data = data
        else:
            self._data = genesis.Data(model)
        self._initialized = True

    def step(self) -> None:
        """Perform a simulation step for this component.

        This method can be overridden by subclasses to implement component-specific
        behavior that needs to be updated each simulation step.
        """
        if not self._initialized:
            raise RuntimeError("Component must be attached to a model before stepping")

    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the component.

        Returns:
            Dict containing the component's state information.
        """
        return {"name": self.name, "initialized": self._initialized}

    def reset(self) -> None:
        """Reset the component to its initial state."""
        if self._data is not None:
            self._data.reset()
