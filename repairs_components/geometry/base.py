"""Base component class and core functionality for repair components.

This module provides base classes for creating components that can be used with the Genesis
physics simulator. Genesis supports both MJCF (MuJoCo) and UDRF (Unified Robot Description Format)
for model definition.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

import numpy as np
import genesis
from build123d import Part, Compound

if TYPE_CHECKING:
    from genesis.sim import Model, Data


class Component(ABC):
    """Base class for all repair components.

    This class defines the common interface and functionality for all repair components.
    """

    # required, but not always.
    def get_mjcf(self) -> str:
        """Get the MJCF representation of the component."""
        raise NotImplementedError

    @abstractmethod  # actually required
    def bd_geometry(self) -> Part | Compound:
        """Get the build123d geometry of the component."""
        raise NotImplementedError
